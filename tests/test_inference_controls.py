"""Deterministic synthetic control tests, never live model performance evidence."""
import asyncio
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import httpx
import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import inference_controls as controls
from benchmarks import inference_tradeoff as legacy

FIXTURE = Path(__file__).resolve().parents[1] / 'benchmarks/fixtures/pump102_quality_v1.json'


def fixture():
    return controls.load_fixture(FIXTURE)


def reviewed():
    # Test-only metadata; NEVER written as project evidence or represented as human work.
    result = fixture()
    result.update(provenance='real', review_status='human_reviewed',
                  review={'reviewer': 'TEST DOUBLE', 'reviewed_at': 'TEST', 'method': 'TEST'})
    return result


def fake_identity(provider='ollama'):
    return {'verified': True, 'digest': 'a' * 64, 'pid': 10, 'created': 1,
            'host': {'os': 'TEST', 'cpu': 'TEST', 'python': 'TEST', 'ram_available_bytes': 1,
                     'harness_sha256': 'TEST'}, 'version': 'TEST', 'format': 'TEST',
            'quantization': 'TEST', 'versions': dict(vllm='TEST', torch='TEST', transformers='TEST'),
            'python': 'TEST', 'revision': 'TEST', 'snapshot': 'TEST', 'dtype': 'TEST',
            'weight_sha256': {'TEST': 'TEST'}}


def report():
    plan = controls.schedule(['vllm', 'ollama'], [256], [512], 5, 102)
    batches = []
    for condition in plan:
        row = {'error': None, 'budget_verified': True, 'prompt_tokens': 200,
               'calibrated_prompt_tokens': 200, 'prompt_token_budget': condition['prompt_token_budget'],
               'completion_tokens': 10, 'quality': controls.quality_labels(reviewed(), '')}
        batches.append({'condition': condition, 'identity': fake_identity(),
                        'identity_after': fake_identity(), 'reset': {'verified': True},
                        'memory': {'complete': True, 'method': 'psutil_host_available_and_process_tree_rss',
                                   'interval_s': .05, 'errors': [], 'samples': [
                                       {'host_available_bytes': 100, 'provider_tree_rss_bytes': 10},
                                       {'host_available_bytes': 100, 'provider_tree_rss_bytes': 10}]},
                        'warmups': [dict(row) for _ in range(condition['concurrency'])] if condition['state'] == 'warm' else [],
                        'measurements': [dict(row) for _ in range(condition['concurrency'])]})
    return {'schema_version': 3, 'providers': ['vllm', 'ollama'], 'fixture': reviewed(),
            'config': {'contexts': [256], 'budgets': [512], 'repetitions': 5, 'seed': 102},
            'schedule': plan, 'batches': batches, 'errors': []}


def test_schedule_balanced_randomized_reproducible():
    plan = controls.schedule(['vllm', 'ollama'], [256, 512], [256], 5, 102)
    assert plan == controls.schedule(['vllm', 'ollama'], [256, 512], [256], 5, 102)
    assert plan != controls.schedule(['vllm', 'ollama'], [256, 512], [256], 5, 103)
    assert set(Counter(controls.condition_key(c)[:-1] for c in plan).values()) == {5}
    assert {c['concurrency'] for c in plan} == {1, 2, 4}
    assert not any(c['prefix_reuse'] for c in plan if c['state'] == 'cold')
    assert {c['prefix_reuse'] for c in plan if c['state'] == 'warm'} == {True, False}
    for repetition in range(1, 6):
        block = [c for c in plan if c['repetition'] == repetition]
        assert len(block) == len({controls.condition_key(c) for c in block}) == 54


@pytest.mark.parametrize('mutation', [
    lambda r: r['batches'][0].pop('identity'),
    lambda r: r['batches'][0].pop('identity_after'),
    lambda r: next(b for b in r['batches'] if b['condition']['state'] == 'cold').pop('reset'),
    lambda r: r['batches'][0].pop('memory'),
    lambda r: r['batches'][0]['memory'].pop('samples'),
    lambda r: r['batches'][0]['identity'].pop('host'),
    lambda r: next(b for b in r['batches'] if b['condition']['provider'] == 'vllm')['identity'].pop('versions'),
    lambda r: r['batches'].pop(),
    lambda r: r['batches'][0]['measurements'].pop(),
    lambda r: r['batches'][0]['measurements'][0].update(prompt_tokens=None),
    lambda r: r['batches'][0]['measurements'][0].update(prompt_tokens=9000),
    lambda r: r['batches'][0]['measurements'][0].update(completion_tokens=None),
    lambda r: r['batches'][0]['measurements'][0].pop('quality'),
    lambda r: r['fixture'].update(review_status='unreviewed'),
    lambda r: r['fixture'].update(provenance='synthetic'),
    lambda r: r['errors'].append('failure'),
    lambda r: r['providers'].pop(),
    lambda r: r['schedule'].pop(),
    lambda r: r['batches'].append(copy.deepcopy(r['batches'][0])),
    lambda r: r['config'].update(repetitions=4),
])
def test_each_missing_control_blocks_completion(mutation):
    value = report()
    assert all(controls.completion_checks(value).values())
    mutation(value)
    assert not all(controls.completion_checks(value).values())


def test_removing_cold_from_both_plan_and_results_does_not_bypass_gate():
    value = report()
    value['schedule'] = [c for c in value['schedule'] if c['state'] == 'warm']
    value['batches'] = [b for b in value['batches'] if b['condition']['state'] == 'warm']
    assert not controls.completion_checks(value)['required_samples']


def test_fixture_is_explicitly_synthetic_and_unreviewed():
    value = fixture()
    assert value['provenance'] == 'synthetic'
    assert value['review_status'] == 'unreviewed'
    labels = controls.quality_labels(value, '8.2 [E1] [E999]')
    assert labels['citation_precision'] == .5
    assert labels['citation_recall'] == .25


@pytest.mark.parametrize('mutation', [
    lambda f: f.update(schema_version=2),
    lambda f: f.update(review_status='human_reviewed'),
    lambda f: f.pop('provenance'),
    lambda f: f['expected_facts'][0].update(citations=['E999']),
    lambda f: f['expected_facts'][0].update(match_terms=[]),
    lambda f: f['expected_facts'].append(f['expected_facts'][0]),
])
def test_invalid_fixture_rejected(tmp_path, mutation):
    value = fixture()
    mutation(value)
    path = tmp_path / 'fixture.json'
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, TypeError)):
        controls.load_fixture(path)


def test_budget_counts_tokens_not_words_and_retains_full_evidence():
    async def counter(prompt):
        return len(prompt.encode()) + 10  # Deliberately unlike whitespace words.
    prompt, count = asyncio.run(controls.fit_prompt('[E1] 数字 8.2', 100, counter))
    assert prompt.startswith('[E1] 数字 8.2')
    assert count == len(prompt.encode()) + 10 <= 100
    with pytest.raises(ValueError, match='require'):
        asyncio.run(controls.fit_prompt('[E1] essential evidence', 5, counter))


@pytest.mark.parametrize('provider', ['vllm', 'ollama'])
def test_provider_token_count_includes_template_and_never_falls_back(provider):
    async def run():
        def handler(request):
            body = json.loads(request.content)
            if provider == 'vllm':
                assert body['add_generation_prompt'] is True
                assert body['chat_template_kwargs'] == {'enable_thinking': False}
                return httpx.Response(200, json={'count': 20})
            assert body['options']['num_predict'] == 1
            return httpx.Response(200, json={'done': True, 'prompt_eval_count': 20})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await controls.count_prompt(client, provider, 'http://127.0.0.1', 'model', 'x') == 20
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={'done': True}))) as client:
            with pytest.raises(ValueError, match='count'):
                await controls.count_prompt(client, provider, 'http://127.0.0.1', 'model', 'x')
    asyncio.run(run())


@pytest.mark.parametrize('resident', [[], [{'name': 'qwen3:0.6b', 'digest': 'abc'}], None])
def test_ollama_cold_requires_successful_unload_and_residency_absence(resident):
    async def run():
        requests = []
        def handler(request):
            requests.append(request)
            if request.method == 'POST':
                assert json.loads(request.content) == {'model': 'qwen3:0.6b', 'keep_alive': 0, 'stream': False}
                return httpx.Response(200, json={'done': True})
            return httpx.Response(200, json={'models': resident})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            if resident == []:
                result = await controls.reset_ollama(client, 'http://127.0.0.1', 'qwen3:0.6b', 'abc', 0)
                assert result['verified'] is True
            else:
                with pytest.raises(ValueError):
                    await controls.reset_ollama(client, 'http://127.0.0.1', 'qwen3:0.6b', 'abc', 0)
        assert len(requests) == 2
    asyncio.run(run())


def test_vllm_restart_requires_old_tree_exit_and_new_birth(monkeypatch):
    process = SimpleNamespace(create_time=lambda: 200)
    monkeypatch.setattr(controls, 'tree_gone', lambda tree: False)
    assert not controls.restart_verified([{'pid': 1}], process, 100)
    monkeypatch.setattr(controls, 'tree_gone', lambda tree: True)
    assert controls.restart_verified([{'pid': 1}], process, 100)
    assert not controls.restart_verified([{'pid': 1}], process, 201)
    assert not controls.restart_verified([], process, 100)


def test_tree_pid_reuse_is_not_old_process(monkeypatch):
    monkeypatch.setattr(controls.psutil, 'Process', lambda pid: SimpleNamespace(create_time=lambda: 200))
    assert controls.tree_gone([{'pid': 10, 'created': 100}])
    assert not controls.tree_gone([{'pid': 10, 'created': 200}])


def test_memory_samples_process_tree_and_declares_limits():
    child = SimpleNamespace(pid=11, memory_info=lambda: SimpleNamespace(rss=100))
    process = SimpleNamespace(pid=10, create_time=lambda: 1, is_running=lambda: True,
                              children=lambda recursive: [child],
                              memory_info=lambda: SimpleNamespace(rss=200))
    sampler = controls.MemorySampler(process)
    sampler.sample()
    sampler.sample()
    result = sampler.result()
    assert result['complete']
    assert result['peak_sampled_rss_bytes'] == 300
    assert result['minimum_available_bytes'] > 0
    assert 'No GPU' in result['limitations']


def test_memory_missing_permission_never_substitutes_zero():
    class Process:
        def create_time(self):
            return 1
        def is_running(self):
            return True
        def children(self, recursive):
            raise psutil.AccessDenied()
    sampler = controls.MemorySampler(Process())
    sampler.sample()
    result = sampler.result()
    assert not result['complete'] and result['errors']
    assert result['peak_sampled_rss_bytes'] is None


def test_host_identity():
    value = controls.host_identity()
    assert all(value.get(k) for k in ['os', 'cpu', 'python', 'ram_available_bytes', 'harness_sha256'])
    assert len(value['harness_sha256']) == 64


def test_remote_process_identity_is_rejected():
    with pytest.raises(ValueError, match='host'):
        controls.process_for_endpoint('http://192.168.0.2:8001', 'vllm')


def test_ollama_identity_uses_live_api(monkeypatch):
    process = SimpleNamespace(pid=123, create_time=lambda: 10, cmdline=lambda: ['ollama', 'serve'], environ=lambda: {})
    monkeypatch.setattr(controls, 'process_for_endpoint', lambda *a: process)
    monkeypatch.setattr(controls, 'process_tree', lambda p: [{'pid': 123, 'created': 10}])
    async def run():
        data = {'/api/version': {'version': 'TEST'}, '/api/tags': {'models': [
            {'name': 'model', 'digest': 'a' * 64}]},
            '/api/show': {'details': {'format': 'gguf', 'quantization_level': 'Q4_K_M'}}}
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=data[r.url.path]))) as client:
            value = await controls.identity(client, 'ollama', 'http://127.0.0.1', 'model')
            assert value['verified'] and value['digest'] == 'a' * 64
            data['/api/show'] = {'details': {}}
            with pytest.raises(ValueError, match='Incomplete'):
                await controls.identity(client, 'ollama', 'http://127.0.0.1', 'model')
    asyncio.run(run())


def test_plan_only_has_no_runtime_calls(tmp_path, monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail('No provider request allowed')
    monkeypatch.setattr(controls, 'identity', forbidden)
    args = SimpleNamespace(repetitions=5, timeout=1, reset_timeout=1, providers=['ollama'],
        context_lengths=[256], evidence_budgets=[512], output=tmp_path / 'plan', seed=102,
        quality_fixture=FIXTURE, ollama_url='http://127.0.0.1:11434', ollama_model='qwen3:0.6b',
        plan_only=True)
    assert asyncio.run(controls.run(args, legacy)) == 3
    data = json.loads((args.output / 'summary.json').read_text())
    assert not data['experiment_complete'] and not data['batches']
    assert len(data['schedule']) == 90
    with pytest.raises(ValueError, match='empty'):
        asyncio.run(controls.run(args, legacy))


def test_unreviewed_fixture_blocks_expensive_run(tmp_path):
    args = SimpleNamespace(repetitions=5, timeout=1, reset_timeout=1, providers=['ollama'],
        context_lengths=[256], evidence_budgets=[512], output=tmp_path / 'run', seed=102,
        quality_fixture=FIXTURE, ollama_url='http://127.0.0.1:11434', ollama_model='qwen3:0.6b')
    assert asyncio.run(controls.run(args, legacy)) == 3
    data = json.loads((args.output / 'summary.json').read_text())
    assert not data['batches'] and data['errors']


def test_merge_preserves_provenance_and_requires_matching_config(tmp_path):
    original = report()
    paths = []
    for provider in original['providers']:
        value = copy.deepcopy(original)
        value['providers'] = [provider]
        value['schedule'] = [c for c in value['schedule'] if c['provider'] == provider]
        value['batches'] = [b for b in value['batches'] if b['condition']['provider'] == provider]
        path = tmp_path / provider
        controls.write_report(path, value)
        paths.append(path)
    assert controls.merge(paths, tmp_path / 'merged') == 0
    merged = json.loads((tmp_path / 'merged/summary.json').read_text())
    assert merged['experiment_complete']
    assert len(merged['source_reports']) == 2
    with pytest.raises(ValueError, match='exactly one'):
        controls.merge([paths[0], paths[0]], tmp_path / 'bad')
    value['config']['seed'] = 103
    controls.write_report(paths[1], value)
    with pytest.raises(ValueError, match='identities'):
        controls.merge(paths, tmp_path / 'mismatch')


def test_vllm_identity_requires_snapshot_versions_and_server_match(tmp_path, monkeypatch):
    snapshot = tmp_path / ('a' * 40)
    snapshot.mkdir()
    for name in ('config.json', 'tokenizer_config.json'):
        (snapshot / name).write_text('{}')
    (snapshot / 'model.safetensors').write_bytes(b'TEST ONLY')
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
               '--model', str(snapshot), '--dtype', 'bfloat16']
    process = SimpleNamespace(pid=123, create_time=lambda: 10, cmdline=lambda: command,
                              exe=lambda: sys.executable, environ=lambda: dict(controls.os.environ))
    monkeypatch.setattr(controls, 'process_for_endpoint', lambda *a: process)
    monkeypatch.setattr(controls, 'process_tree', lambda p: [{'pid': 123, 'created': 10}])
    monkeypatch.setattr(controls.importlib.metadata, 'version', lambda name: 'TEST')
    async def run():
        data = {'/v1/models': {'data': [{'id': 'model'}]}, '/version': {'version': 'TEST'}}
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=data[r.url.path]))) as client:
            value = await controls.identity(client, 'vllm', 'http://127.0.0.1/v1', 'model', snapshot)
            assert value['versions'] == dict(vllm='TEST', torch='TEST', transformers='TEST')
            assert value['revision'] == 'a' * 40
            assert value['weight_sha256']['model.safetensors']
            data['/version']['version'] = 'MISMATCH'
            with pytest.raises(ValueError, match='versions differ'):
                await controls.identity(client, 'vllm', 'http://127.0.0.1/v1', 'model', snapshot)
    asyncio.run(run())


@pytest.mark.parametrize('warmup_fails', [False, True])
def test_runner_cold_warm_concurrency_and_token_provenance(tmp_path, monkeypatch, warmup_fails):
    path = tmp_path / 'reviewed-test-only.json'
    path.write_text(json.dumps(reviewed()))
    condition = {'provider': 'ollama', 'experiment': 'evidence_budget', 'prompt_token_budget': 512,
                 'concurrency': 2, 'prefix_reuse': True, 'state': 'warm', 'repetition': 1, 'batch_id': 0}
    cold = dict(condition, state='cold', prefix_reuse=False, batch_id=1)
    monkeypatch.setattr(controls, 'schedule', lambda *a: [condition, cold])
    calls, events = [], []
    ident = fake_identity()
    async def identity(*args):
        return dict(ident)
    async def reset(*args):
        events.append('reset')
        return {'verified': True}
    async def count(*args):
        events.append('calibrate')
        return len(args[-1].split()) + 10
    async def observe(provider, endpoint, model, prompt, experiment, target, concurrency, repetition, timeout):
        calls.append(prompt)
        events.append('observe')
        return legacy.Observation(experiment, provider, model, target, concurrency, repetition,
            1, 10, len(prompt.split()) + 10, 2, 200,
            answer='8.2 [E1]', error='TEST failure' if warmup_fails else None)
    process = SimpleNamespace(pid=10, create_time=lambda: 1, is_running=lambda: True,
                              children=lambda recursive: [],
                              memory_info=lambda: SimpleNamespace(rss=200))
    monkeypatch.setattr(controls, 'identity', identity)
    monkeypatch.setattr(controls, 'reset_ollama', reset)
    monkeypatch.setattr(controls, 'count_prompt', count)
    monkeypatch.setattr(controls, 'process_for_endpoint', lambda *a: process)
    monkeypatch.setattr(legacy, 'observe', observe)
    args = SimpleNamespace(repetitions=5, timeout=1, reset_timeout=1, providers=['ollama'],
        context_lengths=[256], evidence_budgets=[512], output=tmp_path / 'run', seed=102,
        quality_fixture=path, ollama_url='http://127.0.0.1:11434', ollama_model='model', model_snapshot=None)
    code = asyncio.run(controls.run(args, legacy))
    data = json.loads((args.output / 'summary.json').read_text())
    assert not data['experiment_complete']  # Only one provider, not a final comparison.
    assert events.index('calibrate') < events.index('reset') < events.index('observe')
    if warmup_fails:
        assert code == 2 and len(calls) == 2
        assert not data['batches'][0]['measurements']
    else:
        assert code == 3 and len(calls) == 6
        assert calls[:2] == calls[2:4]  # same lane prefix reused across warm-up and measurement
        assert calls[0] != calls[1]     # independent lane prefixes
        assert not data['batches'][1]['warmups']
        assert data['batches'][0]['memory']['complete']
        assert data['batches'][0]['aggregate_completion_tokens_per_second'] > 0
        assert all(r['budget_verified'] for b in data['batches'] for r in b['measurements'])
        raw = json.loads((args.output / 'raw.json').read_text())
        assert len(raw) == 6


def test_reset_failure_never_emits_cold_row(tmp_path, monkeypatch):
    path = tmp_path / 'reviewed-test-only.json'
    path.write_text(json.dumps(reviewed()))
    condition = {'provider': 'ollama', 'experiment': 'evidence_budget', 'prompt_token_budget': 512,
                 'concurrency': 1, 'prefix_reuse': False, 'state': 'cold', 'repetition': 1, 'batch_id': 0}
    monkeypatch.setattr(controls, 'schedule', lambda *a: [condition])
    async def identity(*args):
        return {'verified': True, 'digest': 'a' * 64}
    async def reset(*args):
        raise ValueError('TEST reset not verified')
    async def count(*args):
        return len(args[-1].split())
    async def forbidden(*args):
        pytest.fail('Generation forbidden without verified reset')
    monkeypatch.setattr(controls, 'identity', identity)
    monkeypatch.setattr(controls, 'reset_ollama', reset)
    monkeypatch.setattr(controls, 'count_prompt', count)
    monkeypatch.setattr(legacy, 'observe', forbidden)
    args = SimpleNamespace(repetitions=5, timeout=1, reset_timeout=1, providers=['ollama'],
        context_lengths=[256], evidence_budgets=[512], output=tmp_path / 'run', seed=102,
        quality_fixture=path, ollama_url='http://127.0.0.1:11434', ollama_model='model', model_snapshot=None)
    assert asyncio.run(controls.run(args, legacy)) == 2
    data = json.loads((args.output / 'summary.json').read_text())
    assert not data['batches'][0]['measurements']
    assert not data['completion_checks']['cold_state']
