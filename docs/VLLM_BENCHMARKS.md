# vLLM and Ollama inference experiments

The research question is not whether vLLM is integrated. It is: **what evidence
budget lies on the latency/grounding Pareto frontier on the target local host?**

The executable harness covers context length → TTFT, evidence tokens → TTFT and
quality, repeated-prefix cache behavior, concurrency 1/2/4, and the same workload
on Ollama and vLLM.

```powershell
python benchmarks\inference_tradeoff.py `
  --vllm-model Qwen/Qwen3-0.6B `
  --ollama-model qwen3:0.6b `
  --output experiments\results\pump102_cpu_001
```

Use the same model weights and quantization where the runtimes permit it;
otherwise label the comparison as runtime-plus-format rather than runtime-only.
Perform one cold run and at least five warm repetitions for reportable results.
Record model identity, hardware, cache state, and RAM. `raw.json`/`raw.csv` are
the evidence; `summary.json` is derived. No headline number is invented here.

Select Pareto-optimal budgets by minimizing TTFT while maximizing citation recall
and groundedness. A point is dominated when another is no slower and better
grounded, or no less grounded and faster.
