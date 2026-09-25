# Experiments

Committed experiment protocols live in `docs/VLLM_BENCHMARKS.md` and
`docs/GRAPHRAG_EVALUATION.md`. Generated observations go in `results/` and must
never be described as production measurements unless they were collected on the
documented target hardware with all services healthy.

The principal research question is the Pareto boundary between evidence budget
and grounded answer quality: reduce evidence until TTFT improves, but stop before
citation recall or groundedness materially degrades.
