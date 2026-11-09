# Offline Kaggle CPU execution

The entry notebook uses the maintained wheel, the official competition snapshot and a predeclared baseline. It does not require access to private tracking infrastructure, does not submit a score automatically and does not assume cross-version joblib compatibility.

```bash
uv sync --locked --group dev --group data
.venv/bin/python scripts/export_kaggle.py --output .local/kaggle-code-v1
```

The fresh staging directory contains a flat `code-dataset/` with wheel, MIT license, JSON configs, dataset metadata and a source receipt; `kernel/` contains the thin notebook and private CPU/offline metadata. No competition rows or trained models are staged.

Authenticate locally, then create the code dataset **privately** (the CLI default) with `--keep-tabular`, so JSON bytes are preserved. Use the same option for later dataset versions. Wait until the dataset is ready before pushing the notebook. Verify the actual attached dataset version and wheel/config checksums; similarly named assets are not interchangeable. The Kaggle username here is `cttsai`, distinct from the GitHub owner.

```bash
.venv/bin/kaggle datasets create --path .local/kaggle-code-v1/code-dataset --keep-tabular
.venv/bin/kaggle kernels push --path .local/kaggle-code-v1/kernel
.venv/bin/kaggle kernels status cttsai/kaggle-mercedes-benz-solution-mlops
```

The canonical title matches the kernel slug. The notebook accepts the three official CSVs or ZIP archives, copies only the expected inputs to its writable working directory, validates them, fits the fixed recipe, reloads every model and regenerates all 4,209 official test predictions. It binds outputs to input, wheel, config and producer-manifest hashes. The producer manifest is never patched after training; code-asset provenance is retained separately.

Retain the exact executed notebook version and downloaded `run-bundle.zip`, `code-asset.json`, logs and `submission.csv` privately. Validate hashes and same-environment model reconstruction before importing completed artifacts to an approved shared tracker. Native and hosted runs are separate experiments; compare their recorded input/fold/source identities and package versions before comparing scores.

The validation record distinguishes a prepared entry point, mocked control-flow checks and an actual completed hosted run. No fixture or prepared notebook is presented as full hosted reproduction.
