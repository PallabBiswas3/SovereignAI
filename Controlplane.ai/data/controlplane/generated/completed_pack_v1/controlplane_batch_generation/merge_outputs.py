#!/usr/bin/env python3
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "batch_manifest.csv").open(encoding="utf-8")))

for split in ["development", "validation", "test", "stress"]:
    out = ROOT / f"{split}.jsonl"
    with out.open("w", encoding="utf-8") as dst:
        for row in rows:
            if row["split"] != split:
                continue
            src = ROOT / row["output_file"]
            if not src.exists():
                raise FileNotFoundError(f"Missing {src}")
            text = src.read_text(encoding="utf-8")
            dst.write(text)
            if text and not text.endswith("\n"):
                dst.write("\n")
    print(f"Wrote {out.name}")

all_out = ROOT / "all_15000.jsonl"
with all_out.open("w", encoding="utf-8") as dst:
    for split in ["development", "validation", "test", "stress"]:
        src = ROOT / f"{split}.jsonl"
        dst.write(src.read_text(encoding="utf-8"))
print(f"Wrote {all_out.name}")
