# ControlPlane synthetic development evaluation v1

`evaluation_v1.jsonl` is a synthetic, hand-specified development set for checking
the combined safety contract. It is not an independent or publication-grade
benchmark.

The set covers:

- supported, contradicted, unknown, mixed, irrelevant, and conflicting factual evidence;
- numeric, date, entity, and no-context claims;
- output and input privacy risks, including negative controls;
- explicit stereotypes, adverse actions, and counterfactual inconsistency;
- conversation compounding, cross-risk precedence, and consequential policies;
- the three policy profiles and all five enforcement actions that are reachable in
  the present policy configurations.

The existing evaluator reads `expected_action`, `expected_categories`, and the
optional `expected_subtypes`. Richer `gold_claims`, `gold_privacy_spans`, and
`gold_bias` annotations are included for the future integrated AdaptiveFact runner.

Run it with:

```bash
python scripts/evaluate_controlplane.py \
  --scenarios data/controlplane/evaluation_v1.jsonl \
  --output results/controlplane/evaluation_v1.json
```

Interpretation rules:

1. Do not tune detectors or policies on this file and then report its score as test
   accuracy. It is a development set.
2. Preserve a separate source-isolated holdout that developers do not inspect.
3. Have at least two reviewers independently label factual status, evidence,
   importance, risk categories, and expected action for the final benchmark.
4. Report action and category metrics together with claim-level factuality,
   unsafe-release rate, supported-answer retention, human-review burden, latency,
   and cost.
5. Record the dataset version, configuration, model versions, and Git commit for
   every run.
