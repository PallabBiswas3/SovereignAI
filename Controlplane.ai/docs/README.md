# Documentation index

Start with the root [`README.md`](../README.md) for setup and quick commands.
Read [`OVERALL_ARCHITECTURE_AND_REVIEW.md`](../OVERALL_ARCHITECTURE_AND_REVIEW.md)
for the complete problem statement, end-to-end architecture, component
responsibilities, and current measured results.

## Architecture

- [`architecture/CONTROLPLANE.md`](architecture/CONTROLPLANE.md) — product design and policy gateway.
- [`architecture/ARCHITECTURE_REVIEW.md`](architecture/ARCHITECTURE_REVIEW.md) — strengths, limitations, and evidence safeguards.
- [`architecture/AGENTIC_VERIFICATION.md`](architecture/AGENTIC_VERIFICATION.md) — bounded deep-verification agent.
- [`architecture/AUTOMATIC_REMEDIATION.md`](architecture/AUTOMATIC_REMEDIATION.md) — safe release, correction, review, and blocking behavior.

## Evaluation

- [`evaluation/COMPLETED_DATASET_EVALUATION.md`](evaluation/COMPLETED_DATASET_EVALUATION.md) — 15,000-case pack results and reproduction commands.
- [`evaluation/VALIDATION_REPORT.md`](evaluation/VALIDATION_REPORT.md) — controlled validation and stress-test interpretation.

## Research history

- [`research/PHASES.md`](research/PHASES.md) — phase index.
- `research/PHASE1_2.md` through `research/PHASE6_1.md` — detailed AdaptiveFact experiments.

## Training

- [`training/KAGGLE_TRAINING_GUIDE.md`](training/KAGGLE_TRAINING_GUIDE.md) — prepared NLI data, exact Kaggle workflow, and where to save reports and models.

Documentation is separated from generated files. Training/evaluation logs go
under `results/`; downloaded model checkpoints go under `models/nli/`; datasets
and distributable training bundles go under `data/controlplane/`.
