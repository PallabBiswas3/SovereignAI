# Completed Dataset Evaluation

Dataset: `controlplane_batch_generation_completed.zip`

Evaluation date: 2026-08-25

## Executive assessment

The supplied pack is structurally complete and useful for hackathon evaluation. The integrated ControlPlane is substantially safer than the deterministic-only configuration: on the 2,000-case test split, its unsafe-release rate falls from **21.05% to 0.67%**. The cost is a more conservative user experience, with **50.29% over-intervention**, and much higher CPU latency.

This is a good hackathon result, but it is not production-ready evidence. All labels declare `annotation_status: synthetic_candidate_unreviewed`, and source-family templates overlap across development, validation, and test splits.

## Dataset integrity

The attached validator passed every structural and verbatim-generation invariant.

| Check | Result |
|---|---:|
| Batches | 100 |
| Records | 15,000 |
| Unique record IDs | 15,000 |
| Unique seeds | 100 |
| Development | 2,000 |
| Validation | 1,000 |
| Test | 2,000 |
| Stress | 10,000 |

The requested distributions are exact: profiles are 35% customer support, 35% internal assistant, and 30% regulated decision support; difficulty is 30% easy, 40% medium, and 30% hard; actions are 25% allow, 20% warning, 15% redact, 30% review, and 10% block.

No exact prompt, response, or interaction duplicates were found within a split. However, source families are not isolated:

| Split pair | Shared source families |
|---|---:|
| Development / validation | 529 |
| Development / test | 754 |
| Validation / test | 530 |

Consequently, the test split measures robustness to generated variations more than generalization to genuinely unseen scenario families.

## Validation results (1,000 cases)

| Metric | Deterministic only | Integrated DeBERTa NLI |
|---|---:|---:|
| Action accuracy | 71.80% | 68.70% |
| Unsafe-release rate | 17.45% | **0.67%** |
| Over-intervention rate | 13.33% | 49.02% |
| Human-review rate | 16.90% | 23.30% |
| Hallucination F1 | 71.72% | 62.63% |
| Privacy F1 | 96.01% | 96.01% |
| Median latency | 9.2 ms | 151.5 ms |
| p95 latency | 22.3 ms | 760.4 ms |

## Frozen test results (2,000 cases)

| Metric | Deterministic only | Integrated DeBERTa NLI |
|---|---:|---:|
| Action accuracy | 70.05% | 68.00% |
| Unsafe-release rate | 21.05% | **0.67%** |
| Over-intervention rate | 16.37% | 50.29% |
| Human-review rate | 16.95% | 25.30% |
| Hallucination precision | 57.94% | 42.19% |
| Hallucination recall | 82.12% | 99.87% |
| Hallucination F1 | 67.95% | 59.32% |
| Privacy F1 | 93.61% | 93.61% |
| Median latency | 9.1 ms | 247.3 ms |
| p95 latency | 20.0 ms | 1,267.6 ms |
| p99 latency | 27.7 ms | 1,952.3 ms |

The integrated system predicted only 265 unconditional allows, versus 513 in the gold labels. It changed many gold `allow` and `human_review` cases into `allow_with_warning`. That explains why ordinary accuracy slightly decreases even while unsafe release improves dramatically.

### Integrated test results by difficulty

| Difficulty | Cases | Action accuracy |
|---|---:|---:|
| Easy | 591 | 68.70% |
| Medium | 807 | 69.02% |
| Hard | 602 | 65.95% |

### Integrated test results by profile

| Profile | Cases | Action accuracy |
|---|---:|---:|
| Customer support | 661 | 60.67% |
| Internal assistant | 709 | 63.33% |
| Regulated decision support | 630 | 80.95% |

## Capability findings

Strong areas:

- Privacy has perfect precision and 87.99% recall. Aadhaar, bearer tokens, email, payment cards, phone numbers, private keys, and SSNs each score 100% on their represented test cases.
- Grounded value contradictions score **96.79% F1**.
- Claim contradiction recall is 98.72%.
- The integrated path nearly eliminates unsafe unconditional releases on this synthetic set.

Weak areas:

- API-key recall is 0% for the 79 generated values. The pack uses strings such as `sk_test_SYNTHETIC_00004803_NONFUNCTIONAL`, which do not match the current credential pattern.
- Policy / compounding-conversation-risk recall is 0% for 172 tagged cases. Some receive the correct final review action for another reason, but the policy taxonomy is missing.
- Bias recall is only 15.07%. Explicit stereotypes have 0% subtype recall, and protected-attribute adverse action recall is 17.44%.
- Unknown-claim detection is sensitive but noisy: 78.69% recall and 28.95% precision.
- `unsupported_entity_claim` produces 870 false-positive subtype findings even though that subtype is absent from the gold test labels. This is a major source of over-intervention and needs label/rule reconciliation.

## Stress results

The 10,000-case stress split was used as a request source. A 5,000-request deterministic load and a bounded 300-request integrated load were run.

| Configuration | Requests | Concurrency | Errors | Throughput | p50 | p95 | Budget exceeded |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deterministic/control path | 5,000 | 25 | 0 | 59.22 req/s | 150 ms | 1,960 ms | 8.22% |
| Integrated NLI path | 300 | 4 | 0 | 3.53 req/s | 780 ms | 3,907 ms | 35.00% |

The deep path is CPU-bound on this machine. It should be an adaptive escalation path for uncertain or high-risk cases, not a mandatory stage for every request. A production service should use worker isolation, batching where supported, bounded queues, timeouts, and load shedding.

## Decision

For a hackathon demonstration, this result is credible and visually compelling if presented honestly:

1. Lead with the unsafe-release reduction from 21.05% to 0.67%.
2. Show the tradeoff: action accuracy is 68.00% and over-intervention is 50.29%.
3. Demonstrate that deterministic checks stay fast while deeper verification is selectively invoked.
4. Do not describe the synthetic test split as a final production benchmark.

Before a production claim, source-isolate the split, have humans adjudicate hidden-test labels, broaden bias/policy/credential parsing, calibrate unknown-claim thresholds, and repeat the frozen comparison across the larger NLI and LLM judges.

## Reproduction commands

```powershell
# Deterministic evaluation
.\.venv\Scripts\python.exe scripts\evaluate_controlplane.py `
  --scenarios data\controlplane\generated\completed_pack_v1\controlplane_batch_generation\test.jsonl `
  --output results\controlplane\completed_pack_test_lightweight.json

# Integrated evaluation
$env:CONTROLPLANE_ADAPTIVE_VERIFICATION='1'
$env:CONTROLPLANE_NLI_MODEL='cross-encoder/nli-deberta-v3-small'
.\.venv\Scripts\python.exe scripts\evaluate_controlplane.py `
  --scenarios data\controlplane\generated\completed_pack_v1\controlplane_batch_generation\test.jsonl `
  --output results\controlplane\completed_pack_test_integrated.json

# Concurrent stress run
.\.venv\Scripts\python.exe scripts\stress_controlplane.py `
  --scenarios data\controlplane\generated\completed_pack_v1\controlplane_batch_generation\stress.jsonl `
  --requests 5000 --concurrency 25 --warmup 50 `
  --output results\controlplane\completed_pack_stress_lightweight.json
```

Raw result files:

- `results/controlplane/completed_pack_validation_lightweight.json`
- `results/controlplane/completed_pack_validation_integrated.json`
- `results/controlplane/completed_pack_test_lightweight.json`
- `results/controlplane/completed_pack_test_integrated.json`
- `results/controlplane/completed_pack_stress_lightweight.json`
- `results/controlplane/completed_pack_stress_integrated.json`
