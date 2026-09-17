# Model artifacts

Lightweight manifests, thresholds, and calibrator metadata can live in their
phase folders. Large binary weights are ignored by Git.

Place a downloaded NLI checkpoint at:

```text
models/nli/<run-name>/
```

The run directory should directly contain `config.json`, tokenizer files, and
the model weights. Keep its training/evaluation report separately at
`results/training/<run-name>/` so logs do not mix with deployable artifacts.
