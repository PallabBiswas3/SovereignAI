# ControlPlane.ai Independent Batch Generation Pack

This package instantiates the supplied dataset-generation prompt into 100 independent prompt files.

## Batch plan

| Split | Batches | Records/batch | Total | Seed range |
|---|---:|---:|---:|---:|
| development | 20 | 100 | 2,000 | 41001–41020 |
| validation | 10 | 100 | 1,000 | 42001–42010 |
| test | 20 | 100 | 2,000 | 43001–43020 |
| stress | 50 | 200 | 10,000 | 44001–44050 |
| **Total** | **100** |  | **15,000** |  |

Each prompt already contains its concrete COUNT, SPLIT, BATCH_ID, and SEED.

Example:
- Split: development
- Batch ID: dev-001
- Seed: 41001
- Count: 100

## Files

- `batch_manifest.csv` — one row per generation job.
- `batch_manifest.json` — same manifest as JSON.
- `prompts/` — 100 fully instantiated generation prompts.
- `outputs/` — save each model response here using the filename in the manifest.
- `validate_outputs.py` — checks counts, JSON validity, unique IDs, split/batch consistency,
  allowed enums, claim/evidence verbatim constraints, and reports major dataset distributions.
- `merge_outputs.py` — merges completed batch files into split-level JSONL files and
  `all_15000.jsonl`.

## Generation workflow

1. Open `batch_manifest.csv`.
2. For each row, send the corresponding `prompt_file` to your chosen generator.
3. Save raw JSONL exactly to the row's `output_file`.
4. Do not wrap model output in Markdown or a JSON array.
5. After all batches are generated, run:

```bash
python validate_outputs.py
```

6. Only if validation passes, merge:

```bash
python merge_outputs.py
```

## Important independence rule

Do not reuse output from one batch as context for another batch. Each batch should be a
fresh generation request using only its own prompt. The seed and batch ID are unique,
so the generator is explicitly instructed to produce an independent set.

## Recommended quality gate

Structural validation is necessary but not sufficient. After generation:
- inspect a random sample from every split;
- check label/action consistency manually;
- run your ControlPlane system on the held-out validation and test splits;
- tune only on development;
- never tune thresholds or prompts against the final test set;
- use `stress` for robustness/load/adversarial evaluation rather than model selection.
