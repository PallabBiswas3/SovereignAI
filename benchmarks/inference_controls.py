"""Strict, host-local controlled inference protocol. No subprocess or shell hooks."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
import statistics
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import psutil

SCHEMA = 3
LIMITATIONS = (
    "Host available RAM and summed provider process-tree RSS sampled with psutil; "
    "RSS can double-count shared pages; short peaks/short-lived children can be missed. "
    "No GPU/VRAM measurement. WSL reports guest RAM, not Windows physical RAM. "
    "Cold means verified model unload (Ollama) or process-tree restart (vLLM), "
    "not OS page-cache eviction. Dedicated, otherwise idle instances are required."
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def positive_int(value):
    return type(value) is int and value > 0


def harness_hash():
    return digest({p.name: hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob('inference_*.py'))})


def host_identity():
    cpu = platform.processor() or platform.machine()
    if sys.platform.startswith('linux'):
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    elif sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    return {"os": platform.platform(), "cpu": cpu,
            "logical_cpus": psutil.cpu_count(), "ram_total_bytes": psutil.virtual_memory().total,
            "ram_available_bytes": psutil.virtual_memory().available,
            "python": platform.python_version(), "harness_sha256": harness_hash()}


def load_fixture(path):
    fixture = json.loads(path.read_text(encoding='utf-8'))
    if fixture.get('schema_version') != 1:
        raise ValueError('Unsupported quality fixture schema')
    if fixture.get('provenance') not in {'synthetic', 'real'}:
        raise ValueError('Explicit fixture provenance required')
    if fixture.get('review_status') not in {'unreviewed', 'human_reviewed'}:
        raise ValueError('Explicit quality review status required')
    if not isinstance(fixture.get('fixture_id'), str) or not fixture['fixture_id'].strip():
        raise ValueError('Fixture ID required')
    evidence = fixture.get('evidence')
    facts = fixture.get('expected_facts')
    if not isinstance(evidence, dict) or not evidence or not all(
            re.fullmatch(r'E\d+', k) and isinstance(v, str) and v.strip()
            for k, v in evidence.items()):
        raise ValueError('Evidence IDs and texts required')
    if not isinstance(facts, list) or not facts:
        raise ValueError('Expected facts required')
    ids = []
    for fact in facts:
        if not isinstance(fact.get('id'), str) or not fact['id'].strip():
            raise ValueError('Fact ID required')
        ids.append(fact['id'])
        if not fact.get('statement') or not fact.get('match_terms') or not all(
                isinstance(t, str) and t.strip() for t in fact['match_terms']):
            raise ValueError('Fact statement and nonempty lexical match terms required')
        if not fact.get('citations') or not set(fact['citations']) <= evidence.keys():
            raise ValueError('Expected citation refers to missing evidence')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate fact IDs')
    if fixture['review_status'] == 'human_reviewed':
        review = fixture.get('review') or {}
        if not all(isinstance(review.get(k), str) and review[k].strip()
                   for k in ('reviewer', 'reviewed_at', 'method')):
            raise ValueError('Human review requires real reviewer/date/method metadata')
    return fixture


def quality_labels(fixture, answer):
    cited = set(re.findall(r'\[(E\d+)\]', answer))
    expected = {c for f in fixture['expected_facts'] for c in f['citations']}
    return {'method': 'lexical_fact_and_citation_proxy_v1',
            'fixture_sha256': digest(fixture), 'review_status': fixture['review_status'],
            'provenance': fixture['provenance'],
            'facts': {f['id']: all(t.casefold() in answer.casefold() for t in f['match_terms'])
                      for f in fixture['expected_facts']},
            'citation_precision': len(cited & expected) / max(1, len(cited)),
            'citation_recall': len(cited & expected) / len(expected),
            'limitation': 'Text matching cannot adjudicate entailment, negation or citation placement.'}


def schedule(providers, contexts, budgets, repetitions, seed):
    """Every Cartesian condition exactly once in each independently shuffled block."""
    conditions = [dict(provider=p, experiment=e, prompt_token_budget=b,
                       concurrency=c, prefix_reuse=reuse, state=state)
                  for p in providers
                  for e, sizes in [('context_length', contexts), ('evidence_budget', budgets)]
                  for b in sizes for c in (1, 2, 4) for reuse in (False, True)
                  for state in ('cold', 'warm') if state == 'warm' or not reuse]
    rng = random.Random(seed)
    result = []
    for repetition in range(1, repetitions + 1):
        block = conditions.copy()
        rng.shuffle(block)
        for condition in block:
            result.append(condition | {'repetition': repetition, 'batch_id': len(result)})
    return result


def process_for_endpoint(endpoint, provider):
    """Discover an actual local listening process, never trust an operator PID alone."""
    parsed = urlsplit(endpoint)
    if parsed.hostname not in {'127.0.0.1', '::1'}:
        raise ValueError('Strict protocol must run on the provider host using literal loopback')
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    candidates = {c.pid for c in psutil.net_connections(kind='tcp')
                  if c.status == psutil.CONN_LISTEN and c.laddr.port == port and c.pid}
    if len(candidates) != 1:
        raise ValueError('Cannot uniquely identify provider listener; check permissions/port')
    process = psutil.Process(candidates.pop())
    command = process.cmdline()
    if provider not in ' '.join(command).lower():
        raise ValueError('Listener is not the requested provider')
    return process


def process_tree(process):
    return [{'pid': p.pid, 'created': p.create_time()} for p in [process, *process.children(recursive=True)]]


def tree_gone(tree):
    for item in tree:
        try:
            if psutil.Process(item['pid']).create_time() == item['created']:
                return False
        except psutil.NoSuchProcess:
            pass
    return True


def restart_verified(old_tree, process, requested_at):
    return bool(old_tree) and tree_gone(old_tree) and process.create_time() >= requested_at


async def request(client, method, url, **kwargs):
    response = await client.request(method, url, **kwargs)
    response.raise_for_status()
    return response.json()


_FILE_HASH_CACHE = {}


def file_hash(path):
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _FILE_HASH_CACHE:
        value = hashlib.sha256()
        with path.open('rb') as file:
            for block in iter(lambda: file.read(1024 * 1024), b''):
                value.update(block)
        _FILE_HASH_CACHE[key] = value.hexdigest()
    return _FILE_HASH_CACHE[key]


def option(command, name):
    for index, item in enumerate(command):
        if item == name and index + 1 < len(command):
            return command[index + 1]
        if item.startswith(name + '='):
            return item.split('=', 1)[1]
    return None


async def identity(client, provider, endpoint, model, snapshot=None):
    process = process_for_endpoint(endpoint, provider)
    result = {'provider': provider, 'model': model, 'host': host_identity(),
              'pid': process.pid, 'created': process.create_time(),
              'process_tree': process_tree(process), 'command': process.cmdline()}
    if provider == 'ollama':
        version = await request(client, 'GET', endpoint + '/api/version')
        tags = await request(client, 'GET', endpoint + '/api/tags')
        show = await request(client, 'POST', endpoint + '/api/show', json={'model': model})
        matches = [x for x in tags['models'] if model in (x.get('name'), x.get('model'))]
        if len(matches) != 1:
            raise ValueError('Ollama tag identity is ambiguous or missing')
        result['template_sha256'] = digest(show.get('template'))
        result['system_sha256'] = digest(show.get('system'))
        result['model_parameters'] = show.get('parameters')
        result['runtime_environment'] = {k: process.environ().get(k) for k in (
            'OLLAMA_NUM_PARALLEL', 'OLLAMA_MAX_LOADED_MODELS', 'OLLAMA_CONTEXT_LENGTH',
            'OLLAMA_FLASH_ATTENTION', 'OLLAMA_KV_CACHE_TYPE', 'CUDA_VISIBLE_DEVICES')}
        result.update(version=version.get('version'), digest=matches[0].get('digest'),
                      format=show.get('details', {}).get('format'),
                      quantization=show.get('details', {}).get('quantization_level'))
        if not all(result.get(k) for k in ('version', 'digest', 'format', 'quantization')):
            raise ValueError('Incomplete Ollama runtime/model identity')
        if not re.fullmatch(r'(sha256:)?[a-f0-9]{64}', result['digest']):
            raise ValueError('Invalid Ollama model digest')
    else:
        # Installed-package versions are valid only in the server interpreter environment.
        if Path(process.exe()).resolve() != Path(sys.executable).resolve():
            raise ValueError('Run the harness with the vLLM server Python interpreter')
        command = process.cmdline()
        # Virtualenvs may resolve to the same base binary. Check invocation/environment too.
        executable = Path(command[0]).absolute()
        if executable.name.startswith('python') and executable != Path(sys.executable).absolute():
            raise ValueError('vLLM Python invocation differs from harness environment')
        if snapshot is None or not snapshot.is_dir():
            raise ValueError('--model-snapshot must identify the local pinned vLLM snapshot')
        snapshot = snapshot.absolute()
        if str(snapshot) not in command and option(command, '--model') != str(snapshot):
            raise ValueError('vLLM must be started with the exact local snapshot path')
        revision = snapshot.name
        if not re.fullmatch(r'[a-f0-9]{40}', revision):
            raise ValueError('Snapshot directory must be an immutable 40-hex revision')
        environment = process.environ()
        for key in ('VIRTUAL_ENV', 'PYTHONPATH'):
            if environment.get(key, '') != os.environ.get(key, ''):
                raise ValueError(f'vLLM and harness {key} environments differ')
        if any(option(command, flag) for flag in ('--tokenizer', '--lora-modules', '--hf-overrides')):
            raise ValueError('Custom tokenizer/adapter/config overrides are outside the pinned protocol')
        versions = {k: importlib.metadata.version(k) for k in ('vllm', 'torch', 'transformers')}
        served = await request(client, 'GET', endpoint + '/models')
        server_version = await request(client, 'GET', endpoint.removesuffix('/v1') + '/version')
        if server_version.get('version') != versions['vllm']:
            raise ValueError('vLLM server and collector package versions differ')
        if model not in [item['id'] for item in served['data']]:
            raise ValueError('Requested model not served')
        dtype = option(command, '--dtype')
        if not dtype or dtype == 'auto':
            raise ValueError('Explicit --dtype required on vLLM server')
        files = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in snapshot.rglob('*.json') if p.is_file()}
        if 'config.json' not in files or 'tokenizer_config.json' not in files:
            raise ValueError('Snapshot missing model/tokenizer configuration')
        weights = sorted(snapshot.glob('*.safetensors'))
        if not weights:
            raise ValueError('Snapshot contains no safetensors weights')
        weight_hashes = {p.name: file_hash(p) for p in weights}
        result['runtime_environment'] = {k: environment.get(k) for k in (
            'VLLM_CPU_KVCACHE_SPACE', 'VLLM_CPU_OMP_THREADS_BIND', 'OMP_NUM_THREADS',
            'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')}
        result.update(weight_sha256=weight_hashes, versions=versions, python=platform.python_version(), revision=revision,
                      snapshot=str(snapshot), config_hashes=files, dtype=dtype,
                      format='HuggingFace local snapshot', quantization=option(command, '--quantization') or 'none')
    result['verified'] = True
    return result


class MemorySampler:
    def __init__(self, process, interval=0.05):
        self.process = process
        self.birth = process.create_time()
        self.interval = interval
        self.samples = []
        self.errors = []
        self.stop = asyncio.Event()

    def sample(self):
        try:
            if self.process.create_time() != self.birth or not self.process.is_running():
                raise ValueError('Provider process replaced during sampling')
            members = [self.process, *self.process.children(recursive=True)]
            self.samples.append({'monotonic_s': time.perf_counter(),
                                 'host_available_bytes': psutil.virtual_memory().available,
                                 'provider_tree_rss_bytes': sum(p.memory_info().rss for p in members),
                                 'pids': [p.pid for p in members]})
        except (psutil.Error, ValueError) as exc:
            self.errors.append(str(exc))

    async def run(self):
        while not self.stop.is_set():
            self.sample()
            try:
                await asyncio.wait_for(self.stop.wait(), self.interval)
            except asyncio.TimeoutError:
                pass

    def result(self):
        return {'method': 'psutil_host_available_and_process_tree_rss', 'interval_s': self.interval,
                'limitations': LIMITATIONS, 'samples': self.samples, 'errors': self.errors,
                'complete': len(self.samples) >= 2 and not self.errors,
                'peak_sampled_rss_bytes': max((s['provider_tree_rss_bytes'] for s in self.samples), default=None),
                'minimum_available_bytes': min((s['host_available_bytes'] for s in self.samples), default=None)}


async def reset_ollama(client, endpoint, model, expected_digest, timeout):
    unloaded = await request(client, 'POST', endpoint + '/api/generate',
                             json={'model': model, 'keep_alive': 0, 'stream': False})
    if unloaded.get('done') is not True:
        raise ValueError('Ollama did not acknowledge unload')
    deadline = time.monotonic() + timeout
    while True:
        resident = await request(client, 'GET', endpoint + '/api/ps')
        if not isinstance(resident.get('models'), list):
            raise ValueError('Ollama residency API unsupported/malformed')
        if not any(expected_digest == m.get('digest') or model in (m.get('name'), m.get('model'))
                   for m in resident['models']):
            return {'verified': True, 'method': 'ollama_unload_and_api_ps_absence',
                    'verified_at': time.time(), 'digest': expected_digest,
                    'residency': resident, 'scope': 'model residency; OS page cache untouched'}
        if time.monotonic() >= deadline:
            raise ValueError('Ollama model remains resident after unload')
        await asyncio.sleep(0.1)


async def reset_vllm(endpoint, old_tree, timeout):
    requested = time.time()
    print('Restart the dedicated vLLM server NOW using the same pinned launch command. '
          'The harness waits for the old process tree to exit and a new listener.', flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            process = process_for_endpoint(endpoint, 'vllm')
            if restart_verified(old_tree, process, requested):
                return {'verified': True, 'method': 'external_process_tree_restart',
                        'requested_at': requested, 'verified_at': time.time(),
                        'old_tree': old_tree, 'new_tree': process_tree(process),
                        'scope': 'process restart; OS page cache untouched'}
        except (psutil.Error, ValueError):
            pass
        await asyncio.sleep(0.5)
    raise ValueError('External vLLM restart not verified; no cold measurement allowed')


async def count_prompt(client, provider, endpoint, model, prompt):
    if provider == 'vllm':
        data = await request(client, 'POST', endpoint.removesuffix('/v1') + '/tokenize', json={
            'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}})
        count = data.get('count')
    else:
        # Untimed calibration; never mistaken for a cold or measured sample.
        data = await request(client, 'POST', endpoint + '/api/generate', json={
            'model': model, 'prompt': prompt, 'stream': False, 'think': False,
            'keep_alive': '15m', 'options': {'temperature': 0, 'num_predict': 1}})
        if data.get('done') is not True:
            raise ValueError('Incomplete Ollama token calibration')
        count = data.get('prompt_eval_count')
    if not positive_int(count):
        raise ValueError('Provider did not report a valid full prompt-token count')
    return count


async def fit_prompt(base, budget, counter):
    """Keep all facts/citations; fill with neutral padding up to a real token ceiling."""
    count = await counter(base)
    if count > budget:
        raise ValueError(f'Full fixture/instructions require {count} prompt tokens; budget is {budget}')
    best = (base, count)
    low, high = 0, budget
    while low <= high:
        middle = (low + high) // 2
        prompt = base + ' context' * middle
        count = await counter(prompt)
        if count <= budget:
            best = (prompt, count)
            low = middle + 1
        else:
            high = middle - 1
    return best


def condition_key(condition):
    return tuple(condition[k] for k in ('provider', 'experiment', 'prompt_token_budget',
                                       'concurrency', 'prefix_reuse', 'state', 'repetition'))


def identity_complete(item, provider):
    if not item or item.get('verified') is not True:
        return False
    host = item.get('host', {})
    if not all(host.get(k) for k in ('os', 'cpu', 'python', 'ram_available_bytes', 'harness_sha256')):
        return False
    if provider == 'ollama':
        return all(item.get(k) for k in ('version', 'digest', 'format', 'quantization'))
    return (all(item.get('versions', {}).get(k) for k in ('vllm', 'torch', 'transformers'))
            and all(item.get(k) for k in ('python', 'revision', 'snapshot', 'dtype', 'weight_sha256')))


def memory_complete(memory):
    samples = memory.get('samples', [])
    return (memory.get('complete') is True and not memory.get('errors') and
            memory.get('method') == 'psutil_host_available_and_process_tree_rss' and
            memory.get('interval_s', 0) > 0 and len(samples) >= 2 and
            all(type(s.get('host_available_bytes')) is int and s['host_available_bytes'] >= 0 and
                positive_int(s.get('provider_tree_rss_bytes')) for s in samples))


def completion_checks(report):
    batches = report['batches']
    planned = report['schedule']
    config = report['config']
    required = schedule(report['providers'], config['contexts'], config['budgets'],
                        config['repetitions'], config['seed'])
    expected = Counter(condition_key(c) for c in required)
    plan_valid = Counter(condition_key(c) for c in planned) == expected and config['repetitions'] >= 5
    actual = Counter(condition_key(b['condition']) for b in batches if not b.get('error'))
    measured = [row for b in batches for row in b.get('measurements', [])]
    consistent = all(len({digest(stable_identity(b['identity'])) for b in batches
                          if b['condition']['provider'] == provider and b.get('identity')}) == 1
                     for provider in report['providers'])
    return {
        'identity_consistency': consistent,
        'both_providers': set(report['providers']) == {'ollama', 'vllm'},
        'identity': bool(batches) and all(identity_complete(b.get('identity'), b['condition']['provider']) and
                identity_complete(b.get('identity_after'), b['condition']['provider']) for b in batches),
        'cold_state': bool(batches) and all(b.get('reset', {}).get('verified')
                        for b in batches if b['condition']['state'] == 'cold'),
        'warmups': bool(batches) and all(
            len(b.get('warmups', [])) == b['condition']['concurrency'] and
            all(not r.get('error') and r.get('budget_verified') for r in b['warmups'])
            for b in batches if b['condition']['state'] == 'warm'),
        'memory': bool(batches) and all(memory_complete(b.get('memory', {})) for b in batches),
        'required_samples': plan_valid and expected == actual and bool(planned) and all(
            len(b.get('measurements', [])) == b['condition']['concurrency'] and
            all(not r.get('error') for r in b['measurements']) for b in batches),
        'token_accounting': bool(measured) and all(r.get('budget_verified') is True and
            positive_int(r.get('prompt_tokens')) and positive_int(r.get('completion_tokens')) and
            r['prompt_tokens'] == r.get('calibrated_prompt_tokens') and
            r['prompt_tokens'] <= r.get('prompt_token_budget', 0) for r in measured),
        'quality_labels': report['fixture']['provenance'] == 'real' and
                          report['fixture']['review_status'] == 'human_reviewed' and
                          all(report['fixture'].get('review', {}).get(k) for k in
                              ('reviewer', 'reviewed_at', 'method')) and
                          bool(measured) and all(r.get('quality', {}).get('fixture_sha256') == digest(report['fixture'])
                              and set(r['quality'].get('facts', {})) ==
                              {f['id'] for f in report['fixture']['expected_facts']} for r in measured),
        'no_failures': not report.get('errors') and all(not b.get('error') for b in batches),
    }


def write_report(output, report):
    output.mkdir(parents=True, exist_ok=True)
    report['completion_checks'] = completion_checks(report)
    report['groups'] = {}
    for batch in report['batches']:
        if batch.get('error'):
            continue
        key = '|'.join(map(str, condition_key(batch['condition'])[:-1]))
        group = report['groups'].setdefault(key, {'batches': 0, 'samples': 0,
            'ttft_ms': [], 'latency_ms': [], 'batch_throughput_tok_s': []})
        group['batches'] += 1
        for row in batch.get('measurements', []):
            if row.get('error'):
                continue
            group['samples'] += 1
            for metric, field in [('ttft_ms', 'ttft_ms'), ('latency_ms', 'total_ms')]:
                if row.get(field) is not None:
                    group[metric].append(row[field])
        if batch.get('aggregate_completion_tokens_per_second') is not None:
            group['batch_throughput_tok_s'].append(batch['aggregate_completion_tokens_per_second'])
    for group in report['groups'].values():
        for metric in ('ttft_ms', 'latency_ms', 'batch_throughput_tok_s'):
            values = group.pop(metric)
            group[metric + '_p50'] = statistics.median(values) if values else None
    report['group_key'] = 'provider|experiment|prompt_token_budget|concurrency|prefix_reuse|state'
    report['quality_interpretation'] = 'Completion means required evidence collected, not human adjudication of generated answers or proven factual accuracy.'
    report['experiment_complete'] = all(report['completion_checks'].values())
    report['missing_controls'] = [k for k, v in report['completion_checks'].items() if not v]
    temporary = output / 'summary.json.tmp'
    rows = [dict(row, condition=b['condition'], record_type=phase)
            for b in report['batches'] for phase in ('warmups', 'measurements') for row in b.get(phase, [])]
    (output / 'raw.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    report['runs'] = len(rows)
    report['warmup_runs'] = sum(r['record_type'] == 'warmups' for r in rows)
    report['measured_successful'] = sum(r['record_type'] == 'measurements' and not r.get('error') for r in rows)
    report['failed_requests'] = sum(bool(r.get('error')) for r in rows)
    # Counts are computed from persisted raw observations, never from planned requests.
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(output / 'summary.json')
    import csv
    with (output / 'raw.csv').open('w', newline='', encoding='utf-8') as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()} for row in rows)


def stable_identity(item):
    return {k: v for k, v in item.items() if k not in
            {'host', 'pid', 'created', 'process_tree'}}


async def run(args, legacy):
    if args.repetitions < 5 or args.timeout <= 0 or args.reset_timeout <= 0:
        raise ValueError('At least five repetitions and positive timeouts required')
    if len(set(args.providers)) != len(args.providers) or not args.providers:
        raise ValueError('Unique providers required')
    for sizes in (args.context_lengths, args.evidence_budgets):
        if not sizes or any(not positive_int(b) for b in sizes) or len(set(sizes)) != len(sizes):
            raise ValueError('Unique positive token budgets required')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory must be empty; existing evidence cannot be overwritten')
    fixture = load_fixture(args.quality_fixture)
    configured = {p: (legacy.validate_endpoint(getattr(args, p + '_url')), getattr(args, p + '_model'))
                  for p in args.providers}
    for endpoint, _ in configured.values():
        if urlsplit(endpoint).hostname not in {'127.0.0.1', '::1'}:
            raise ValueError('Run each provider harness on its own host using loopback')
    config = {'contexts': args.context_lengths, 'budgets': args.evidence_budgets,
              'repetitions': args.repetitions, 'seed': args.seed,
              'fixture_hash': digest(fixture), 'harness_sha256': harness_hash(),
              'max_output_tokens': 256, 'temperature': 0, 'thinking': False}
    report = {'schema_version': SCHEMA, 'protocol': 'controlled', 'providers': args.providers,
              'config': config, 'fixture': fixture, 'limitations': LIMITATIONS,
              'comparison_scope': 'runtime + weight format + host; not isolated runtime performance',
              'budget_unit': 'provider-verified full prompt tokens including template; ceiling, not exact length',
              'legacy_target_tokens': 'compatibility alias for prompt_token_budget in v3 only; v2 was words',
              'schedule': schedule(args.providers, args.context_lengths, args.evidence_budgets,
                                   args.repetitions, args.seed), 'batches': [], 'errors': []}
    write_report(args.output, report)
    if getattr(args, 'plan_only', False):
        print(json.dumps({'batches': len(report['schedule']),
                          'missing_controls': report['missing_controls']}, indent=2))
        return 3
    if fixture['review_status'] != 'human_reviewed' or fixture['provenance'] != 'real':
        report['errors'].append('Final runs require a real human-reviewed fixture; use --plan-only for the synthetic template')
        write_report(args.output, report)
        return 3
    first_identities = {}
    async with httpx.AsyncClient(timeout=args.timeout, follow_redirects=False, trust_env=False) as client:
        for condition in report['schedule']:
            provider = condition['provider']
            endpoint, model = configured[provider]
            batch = {'condition': condition, 'warmups': [], 'measurements': []}
            report['batches'].append(batch)
            try:
                before = await identity(client, provider, endpoint, model, args.model_snapshot)
                if provider in first_identities and stable_identity(before) != first_identities[provider]:
                    raise ValueError('Runtime/model identity drifted across experiment batches')
                first_identities.setdefault(provider, stable_identity(before))
                # Prefix nonce at START prevents accidental full-prefix reuse across batches/lanes.
                prompts = []
                for lane in range(condition['concurrency']):
                    prompt_pair = []
                    for phase in ('warmup', 'measurement'):
                        nonce = digest([args.seed, condition['batch_id'], lane,
                                        'reuse' if condition['prefix_reuse'] else phase])[:24]
                        base = (f'{nonce}\nUse only the evidence. State the Pump-102 finding and bounded '
                                'next step with bracketed citations. Abstain on unsupported claims.\n' +
                                '\n'.join(f'[{k}] {v}' for k, v in fixture['evidence'].items()))
                        prompt, count = await fit_prompt(base, condition['prompt_token_budget'],
                            lambda value: count_prompt(client, provider, endpoint, model, value))
                        prompt_pair.append((prompt, count))
                    prompts.append(prompt_pair)
                # Calibration may load/cache the model. Reset AFTER calibration for ALL batches.
                if provider == 'ollama':
                    reset = await reset_ollama(client, endpoint, model, before['digest'], args.reset_timeout)
                else:
                    tree = process_tree(process_for_endpoint(endpoint, provider))
                    reset = await reset_vllm(endpoint, tree, args.reset_timeout)
                batch['reset'] = reset
                current = await identity(client, provider, endpoint, model, args.model_snapshot)
                if stable_identity(current) != stable_identity(before):
                    raise ValueError('Runtime/model identity changed during reset')
                batch['identity'] = current

                async def observe_lane(lane, phase):
                    prompt, count = prompts[lane][0 if phase == 'warmup' else 1]
                    row = await legacy.observe(provider, endpoint, model, prompt, condition['experiment'],
                        condition['prompt_token_budget'], condition['concurrency'],
                        condition['repetition'], args.timeout)
                    from dataclasses import asdict
                    value = asdict(row)
                    value.update(lane=lane, phase=phase, prompt=prompt,
                                 prompt_token_budget=condition['prompt_token_budget'],
                                 calibrated_prompt_tokens=count,
                                 budget_verified=positive_int(row.prompt_tokens) and
                                    row.prompt_tokens == count and row.prompt_tokens <= condition['prompt_token_budget'],
                                 quality=quality_labels(fixture, row.answer or ''))
                    # Never call legacy hard-coded quality scores adjudicated observations.
                    for key in ('groundedness', 'citation_precision', 'citation_recall'):
                        value.pop(key, None)
                    return value

                if condition['state'] == 'warm':
                    batch['warmups'] = await asyncio.gather(*[
                        observe_lane(lane, 'warmup') for lane in range(condition['concurrency'])])
                    if any(r['error'] or not r['budget_verified'] for r in batch['warmups']):
                        raise ValueError('Warmup failed or token accounting disagreed; measurement suppressed')
                # Cold batches have no primed prefix and are never warmed before measurement.
                batch['effective_prefix_reuse'] = condition['prefix_reuse'] and condition['state'] == 'warm'
                sampler = MemorySampler(process_for_endpoint(endpoint, provider))
                sampler.sample()
                task = asyncio.create_task(sampler.run())
                started = time.perf_counter()
                try:
                    batch['measurements'] = await asyncio.gather(*[
                        observe_lane(lane, condition['state']) for lane in range(condition['concurrency'])])
                finally:
                    batch['wall_seconds'] = time.perf_counter() - started
                    sampler.stop.set()
                    await task
                    sampler.sample()
                    batch['memory'] = sampler.result()
                if not batch['memory']['complete']:
                    raise ValueError('Provider/host memory sampling incomplete')
                after = await identity(client, provider, endpoint, model, args.model_snapshot)
                if stable_identity(after) != stable_identity(current) or (
                        after['pid'], after['created']) != (current['pid'], current['created']):
                    raise ValueError('Identity/process changed during measurement')
                batch['identity_after'] = after
                if any(r['error'] or not r['budget_verified'] for r in batch['measurements']):
                    raise ValueError('Measured request failed or token count differs from calibration')
                counts = [r['completion_tokens'] for r in batch['measurements']]
                batch['aggregate_completion_tokens_per_second'] = (
                    sum(counts) / batch['wall_seconds'] if all(positive_int(n) for n in counts) else None)
            except Exception as exc:
                batch['error'] = f'{type(exc).__name__}: {exc}'
                report['errors'].append(batch['error'])
                write_report(args.output, report)
                print(batch['error'], file=sys.stderr)
                return 2
            write_report(args.output, report)
            print(f"Completed batch {len(report['batches'])}/{len(report['schedule'])}", flush=True)
    return 0 if report['experiment_complete'] else 3


def merge(paths, output):
    reports = [json.loads((path / 'summary.json').read_text(encoding='utf-8')) for path in paths]
    if output.exists() and any(output.iterdir()):
        raise ValueError('Merge output must be empty')
    if not reports or any(r.get('schema_version') != SCHEMA for r in reports):
        raise ValueError('Only v3 controlled evidence can be merged')
    first = reports[0]
    providers = [p for r in reports for p in r['providers']]
    if len(set(providers)) != len(providers) or set(providers) != {'ollama', 'vllm'}:
        raise ValueError('Merge requires exactly one run for each provider')
    if any(r['config'] != first['config'] or r['fixture'] != first['fixture'] for r in reports):
        raise ValueError('Protocol, fixture and harness identities must match')
    merged = dict(first, providers=providers,
                  schedule=[c for r in reports for c in r['schedule']],
                  batches=[b for r in reports for b in r['batches']],
                  errors=[e for r in reports for e in r['errors']],
                  source_reports=[{'path': str(p), 'sha256': hashlib.sha256(
                      (p / 'summary.json').read_bytes()).hexdigest()} for p in paths])
    write_report(output, merged)
    return 0 if merged['experiment_complete'] else 3
