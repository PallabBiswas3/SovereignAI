#!/usr/bin/env python3
"""Fine-tune the ControlPlane DeBERTa NLI checkpoint on Kaggle."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


ID_TO_LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL_TO_ID = {value: key for key, value in ID_TO_LABEL.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--model-name",
        default="cross-encoder/nli-deberta-v3-small",
    )
    parser.add_argument("--output-dir", default="/kaggle/working/controlplane-deberta-v3-small-v1")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=6.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--freeze-encoder-layers", type=int, default=3)
    parser.add_argument(
        "--train-embeddings",
        action="store_true",
        help="Train the large embedding table. It is frozen by default for this small pilot dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def metric_dict(eval_prediction) -> dict[str, float]:
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    labels = np.asarray(eval_prediction.label_ids)
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1, 2],
        zero_division=0,
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    non_supported = labels != LABEL_TO_ID["entailment"]
    supported = labels == LABEL_TO_ID["entailment"]
    unsafe_release = np.logical_and(
        non_supported,
        predictions == LABEL_TO_ID["entailment"],
    ).sum() / max(1, non_supported.sum())
    supported_retention = np.logical_and(
        supported,
        predictions == LABEL_TO_ID["entailment"],
    ).sum() / max(1, supported.sum())
    over_intervention = np.logical_and(
        supported,
        predictions != LABEL_TO_ID["entailment"],
    ).sum() / max(1, supported.sum())

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(macro_f1),
        "unsafe_release_rate": float(unsafe_release),
        "supported_retention": float(supported_retention),
        "over_intervention_rate": float(over_intervention),
    }
    for index, name in ID_TO_LABEL.items():
        metrics[f"precision_{name}"] = float(precision[index])
        metrics[f"recall_{name}"] = float(recall[index])
        metrics[f"f1_{name}"] = float(f1[index])
        metrics[f"support_{name}"] = float(support[index])
    return metrics


def freeze_for_small_pilot(model, *, encoder_layers: int, train_embeddings: bool) -> dict[str, int]:
    if not train_embeddings:
        for parameter in model.deberta.embeddings.parameters():
            parameter.requires_grad = False
    layers = model.deberta.encoder.layer
    freeze_count = min(max(0, encoder_layers), len(layers))
    for layer in layers[:freeze_count]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    return {
        "frozen_encoder_layers": freeze_count,
        "embeddings_trainable": int(train_embeddings),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def training_arguments(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 10,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "fp16": bool(torch.cuda.is_available()),
        "dataloader_num_workers": 2,
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    validation_path = data_dir / "validation.jsonl"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {validation_path}; check --data-dir"
        )

    raw = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(validation_path)},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=args.max_length,
        )

    removable = [column for column in raw["train"].column_names if column != "label"]
    tokenized = raw.map(tokenize, batched=True, remove_columns=removable)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    parameter_summary = freeze_for_small_pilot(
        model,
        encoder_layers=args.freeze_encoder_layers,
        train_embeddings=args.train_embeddings,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer_kwargs = {
        "model": model,
        "args": training_arguments(args),
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": data_collator,
        "compute_metrics": metric_dict,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    baseline_metrics = trainer.evaluate(metric_key_prefix="baseline")
    train_result = trainer.train()
    final_metrics = trainer.evaluate(metric_key_prefix="finetuned")

    output_dir = Path(args.output_dir)
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    run_report = {
        "model_name": args.model_name,
        "data_dir": str(data_dir),
        "train_rows": len(raw["train"]),
        "validation_rows": len(raw["validation"]),
        "label_mapping": LABEL_TO_ID,
        "parameters": parameter_summary,
        "baseline_metrics": baseline_metrics,
        "training_metrics": train_result.metrics,
        "finetuned_metrics": final_metrics,
        "arguments": vars(args),
        "warning": (
            "This run uses synthetic_candidate_unreviewed labels and is a pilot, "
            "not a production-quality model evaluation."
        ),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(run_report, indent=2, default=float),
        encoding="utf-8",
    )
    archive = shutil.make_archive(
        str(output_dir / "controlplane-deberta-v3-small-v1"),
        "zip",
        root_dir=final_model_dir,
    )
    print(json.dumps(run_report, indent=2, default=float))
    print(f"\nSaved model: {final_model_dir}")
    print(f"Download archive: {archive}")


if __name__ == "__main__":
    main()
