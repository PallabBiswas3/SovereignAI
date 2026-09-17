# Generated results

This directory is for local run output and logs. Generated contents are ignored
by Git; this guide is the only tracked file.

- `controlplane/` — gateway evaluations, stress tests, audit events, and feedback.
- `model_comparison/` — frozen-set model comparison reports.
- `phase0/` through `phase7/` — AdaptiveFact research experiment outputs.
- `training/<run-name>/` — fine-tuning reports, metrics, confusion matrices, and logs.

Do not store trained model weights here. Put downloaded checkpoints in
`models/nli/<run-name>/` and keep their matching report in `results/training/`.
