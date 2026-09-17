# Kaggle training guide: ControlPlane DeBERTa NLI pilot

## What is prepared

The six supplied batches were validated, cleaned, merged, and split by
`source_family_id`.

| File | Purpose | Rows |
|---|---|---:|
| `processed/train.jsonl` | Gradient updates | 500 |
| `processed/validation.jsonl` | Early stopping and model selection | 100 |
| `processed/all_600_cleaned.jsonl` | Audited merged copy | 600 |
| `processed/train.csv` | Human inspection / alternative loaders | 500 |
| `processed/validation.csv` | Human inspection / alternative loaders | 100 |
| `processed/dataset_summary.json` | Counts, corrections, and audit notes | - |

There is no source-family overlap between training and validation. Five
batch-6 rows had `label=1` and entailment evidence/rationales but incorrectly
said `label_text=neutral`; the cleaned files consistently mark them as
entailment. No rows were rejected.

All annotations remain `synthetic_candidate_unreviewed`. This is a pilot
fine-tuning dataset, not a final gold benchmark.

## 1. Upload the prepared bundle

Upload
`data/controlplane/nli_finetune_v1/artifacts/controlplane-nli-kaggle-ready.zip`
to a new private Kaggle Dataset.
Suggested dataset title:

```text
controlplane-nli-v1-pilot
```

Do not make it public unless you have reviewed every record and intend to
publish it.

## 2. Create the Kaggle notebook

1. Open Kaggle.
2. Select **Create > New Notebook**.
3. Open **Notebook options**.
4. Select a GPU accelerator.
5. Turn Internet on so Hugging Face can download the base checkpoint.
6. Select **Add Input** and attach the private dataset uploaded in step 1.

The exact `/kaggle/input/...` directory name depends on the dataset slug shown
by Kaggle.

## 3. Verify the GPU

Run this as the first notebook cell:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

assert torch.cuda.is_available(), "Enable a GPU accelerator before training."
```

## 4. Install training dependencies

Run:

```python
!pip install -q "transformers>=4.45,<6" datasets accelerate scikit-learn sentencepiece
```

If Kaggle requests a kernel restart after installation, restart the session and
rerun the GPU cell before continuing.

## 5. Locate the uploaded files

Run:

```python
from pathlib import Path
import shutil
import zipfile

work = Path("/kaggle/working/controlplane_nli")
work.mkdir(parents=True, exist_ok=True)

input_root = Path("/kaggle/input")
zip_matches = list(input_root.rglob("controlplane-nli-kaggle-ready.zip"))

if len(zip_matches) == 1:
    with zipfile.ZipFile(zip_matches[0]) as archive:
        archive.extractall(work)
    source = zip_matches[0]
else:
    # Kaggle often extracts an uploaded dataset automatically, so the zip may
    # not appear in /kaggle/input. Locate the training script instead.
    script_matches = list(input_root.rglob("train_deberta_nli_kaggle.py"))
    candidate_dirs = sorted({path.parent for path in script_matches})
    candidate_dirs = [
        path for path in candidate_dirs
        if (path / "train.jsonl").is_file()
        and (path / "validation.jsonl").is_file()
        and (path / "dataset_summary.json").is_file()
    ]
    assert len(candidate_dirs) == 1, (
        f"Expected one extracted ControlPlane dataset, found: {candidate_dirs}"
    )
    source = candidate_dirs[0]
    for name in (
        "train.jsonl",
        "validation.jsonl",
        "dataset_summary.json",
        "train_deberta_nli_kaggle.py",
    ):
        shutil.copy2(source / name, work / name)

print("Source:", source)
print("Files:")
for path in sorted(work.iterdir()):
    print(" -", path.name)
```

Expected files include:

```text
train.jsonl
validation.jsonl
dataset_summary.json
train_deberta_nli_kaggle.py
```

## 6. Confirm the data before training

Run:

```python
import json
from collections import Counter
from pathlib import Path

work = Path("/kaggle/working/controlplane_nli")

def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

train_rows = load_jsonl(work / "train.jsonl")
validation_rows = load_jsonl(work / "validation.jsonl")

print("Training rows:", len(train_rows))
print("Validation rows:", len(validation_rows))
print("Training labels:", Counter(row["label_text"] for row in train_rows))
print("Validation labels:", Counter(row["label_text"] for row in validation_rows))

train_families = {row["source_family_id"] for row in train_rows}
validation_families = {row["source_family_id"] for row in validation_rows}
print("Family overlap:", len(train_families & validation_families))

assert len(train_rows) == 500
assert len(validation_rows) == 100
assert not train_families & validation_families
```

## 7. Run the pilot fine-tuning

Run:

```python
!python /kaggle/working/controlplane_nli/train_deberta_nli_kaggle.py \
  --data-dir /kaggle/working/controlplane_nli \
  --output-dir /kaggle/working/controlplane-deberta-v3-small-v1 \
  --model-name cross-encoder/nli-deberta-v3-small \
  --max-length 256 \
  --epochs 6 \
  --learning-rate 1e-5 \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --gradient-accumulation-steps 2 \
  --freeze-encoder-layers 3 \
  --early-stopping-patience 2 \
  --seed 42
```

For this 500-row pilot, the embedding table and the lowest three transformer
layers remain frozen. This reduces overfitting and GPU memory use while training
the upper transformer layers and classification head.

If GPU memory is exhausted, reduce `--train-batch-size` to `4` and increase
`--gradient-accumulation-steps` to `4`.

## 8. Read the result

Run:

```python
import json
from pathlib import Path

report_path = Path("/kaggle/working/controlplane-deberta-v3-small-v1/training_report.json")
report = json.loads(report_path.read_text())

print("BASE MODEL")
for key, value in report["baseline_metrics"].items():
    if isinstance(value, (int, float)):
        print(f"{key}: {value:.4f}")

print("\nFINE-TUNED MODEL")
for key, value in report["finetuned_metrics"].items():
    if isinstance(value, (int, float)):
        print(f"{key}: {value:.4f}")
```

Focus on these comparisons:

| Metric | Desired direction |
|---|---|
| `macro_f1` | Higher |
| `f1_contradiction` | Higher |
| `precision_contradiction` | Higher |
| `recall_neutral` | Higher |
| `supported_retention` | Higher |
| `unsafe_release_rate` | Lower |
| `over_intervention_rate` | Lower |

Do not accept a model merely because overall accuracy increases. An increase in
unsafe release is a safety regression.

Because validation contains only 100 synthetic records, small metric changes
are unstable. The purpose of this run is to establish whether fine-tuning works
and whether the direction is promising.

## 9. Download the model

The training script creates:

```text
/kaggle/working/controlplane-deberta-v3-small-v1/
├── final_model/
├── training_report.json
└── controlplane-deberta-v3-small-v1.zip
```

Open the Kaggle **Output** panel and download:

```text
controlplane-deberta-v3-small-v1.zip
training_report.json
```

Keep the report with the model so its dataset and training configuration remain
traceable.

## 10. Connect the downloaded model to ControlPlane

Extract the downloaded model inside the project, for example:

```text
models/nli/controlplane-deberta-v3-small-v1/
```

That directory must directly contain files such as `config.json`, tokenizer
files, and `model.safetensors`.

Also save `training_report.json` at:

```text
results/training/controlplane-deberta-v3-small-v1/training_report.json
```

This keeps deployable model files separate from run logs and metrics.

Then run:

```powershell
$env:CONTROLPLANE_ADAPTIVE_VERIFICATION='1'
$env:CONTROLPLANE_NLI_MODEL='models/nli/controlplane-deberta-v3-small-v1'
.\.venv\Scripts\python.exe scripts\evaluate_controlplane.py `
  --scenarios data\controlplane\generated\completed_pack_v1\controlplane_batch_generation\test.jsonl `
  --output results\controlplane\finetuned_test_integrated.json
```

Compare the result with:

```text
results/controlplane/completed_pack_test_integrated.json
```

The current integrated baseline has:

```text
action accuracy:        68.00%
unsafe-release rate:     0.67%
over-intervention rate: 50.29%
```

The fine-tuned model should reduce over-intervention and improve supported
retention without making the unsafe-release rate worse.

## 11. What to do after the pilot

If the pilot improves the intended metrics:

1. Review ambiguous training labels.
2. Expand to at least 5,000 source-diverse claim pairs.
3. Create a larger source-isolated validation set.
4. Retune the NLI decision thresholds.
5. Evaluate once on a new human-reviewed hidden test.

If it does not improve, inspect its confusion matrix and label errors before
generating thousands of additional synthetic examples.
