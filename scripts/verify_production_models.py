"""Verify locked bytes and run tiny offline model-load smoke checks on CPU."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.monitoring.model_manifest import load_lock, verify_directory


def main():
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(2)
    checks = {}
    for name, model in load_lock(ROOT / "deploy/model-lock.json").items():
        path = ROOT / "deploy/models" / name
        checked = verify_directory(path, model)
        if not checked["valid"]:
            print(json.dumps({"model": name, **checked}))
            return 1
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        loader = AutoModel if name == "all-MiniLM-L6-v2" else AutoModelForSequenceClassification
        loaded = loader.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_safetensors=True)
        loaded.eval()
        tokens = tokenizer("This is a synthetic model load check.", return_tensors="pt")
        with torch.inference_mode():
            result = loaded(**tokens)
        tensor = result.last_hidden_state if name == "all-MiniLM-L6-v2" else result.logits
        checks[name] = {"revision": model["revision"], "finite_output": bool(torch.isfinite(tensor).all()), "shape": list(tensor.shape)}
        print(json.dumps({"model": name, **checks[name]}), flush=True)
        del tokenizer, loaded, result, tensor, tokens
        gc.collect()
    return 0 if all(value["finite_output"] for value in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
