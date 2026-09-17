# ControlPlane NLI fine-tuning data v1

This is the 600-row synthetic pilot dataset prepared for fine-tuning
`cross-encoder/nli-deberta-v3-small`. It is useful for pipeline validation, but
its labels are still `synthetic_candidate_unreviewed` and are not production
gold labels.

## Folders

- `raw/` — six original generated JSONL batches; treat these as immutable inputs.
- `processed/` — validated/cleaned 500-row train and 100-row validation splits,
  CSV inspection copies, the merged dataset, and `dataset_summary.json`.
- `artifacts/` — the ready-to-upload Kaggle zip and its unpacked bundle.

Rebuild the processed data from the repository root with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_nli_finetuning_data.py
```

Training instructions are in
[`docs/training/KAGGLE_TRAINING_GUIDE.md`](../../../docs/training/KAGGLE_TRAINING_GUIDE.md).
After a Kaggle run, keep the small JSON training report under
`results/training/<run-name>/` and extract the downloaded checkpoint under
`models/nli/<run-name>/`.
