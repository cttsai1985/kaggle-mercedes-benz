# CPU reproduction and delivery

Install Python 3.12 and uv 0.9.24. Run from the repository root:

```bash
uv sync --locked --group dev --group lab --group data --group tracking
export MERCEDES_DATA_DIR=/absolute/path/outside/git/mercedes-benz
.venv/bin/kaggle auth login
.venv/bin/python scripts/download_data.py --data-dir "$MERCEDES_DATA_DIR"
.venv/bin/mercedes validate --data-dir "$MERCEDES_DATA_DIR"
.venv/bin/mercedes train --data-dir "$MERCEDES_DATA_DIR" \
  --config configs/baseline.json --run-dir .local/runs/baseline
.venv/bin/mercedes evaluate --data-dir "$MERCEDES_DATA_DIR" --run-dir .local/runs/baseline
.venv/bin/mercedes predict --run-dir .local/runs/baseline \
  --data "$MERCEDES_DATA_DIR/test.csv" --output .local/predictions.csv
```

Kaggle access requires your own credentials and competition access. Credentials are captured privately by the helper and never printed. Existing archives are verified and reused without contacting Kaggle. Raw data must be outside the checkout or inside ignored `.local/`.

The fixed baseline has five shuffled folds, split seed 904, model seeds 170904/170905, Ridge and two 160-tree estimators per fold/seed, at most four CPU threads. The full recorded run fitted 30 pipelines in 32.60 seconds; observed time is machine-specific, not a repeated latency benchmark. No target rows are filtered. Every new training invocation requires a new directory.

`verify` checks completed artifact identity; `evaluate` also reloads every model and reconstructs OOF/test predictions against the exact training snapshot. `analyze` includes retained error/fold diagnostics. `predict` accepts the fitted feature order, rejects training-ID overlap and writes an output/input/manifest checksum receipt. Joblib artifacts are for trusted models in a compatible scientific environment; cross-version binary portability is not claimed.

## Controlled representation comparison

```bash
.venv/bin/mercedes train --data-dir "$MERCEDES_DATA_DIR" \
  --config configs/projection.json --run-dir .local/runs/projection
.venv/bin/mercedes evaluate --data-dir "$MERCEDES_DATA_DIR" --run-dir .local/runs/projection
.venv/bin/python scripts/analyze_comparison.py \
  --baseline .local/runs/baseline --projection .local/runs/projection \
  --data-dir "$MERCEDES_DATA_DIR" --output .local/comparison.json
```

This appends 16 SVD coordinates, retaining original features; it is a representation comparison, not compression or new information. All other scientific settings match. [Observed effects and limits](modeling-study.md).

## Docker JupyterLab and dev container

```bash
.venv/bin/python scripts/create_lab_token.py
docker compose up --build -d lab
```

Open http://127.0.0.1:8893/lab with the token in ignored `.local/jupyter_token`. Do not publish the token. The container runs as `researcher`, mounts data read-only, writes experiment outputs and imports the checkout at `/workspace/src`. The dev container uses `/workspace` and a writable `/opt/venv`.

The runtime image's entry point is `mercedes`; the Lab target's entry point is Jupyter. Use `--entrypoint mercedes` when executing the CLI from a Lab image. Configs are available at `/opt/mercedes/configs`.

## Existing shared services

The platform must provision `kaggle/mercedes-benz` with artifact location `mlflow-artifacts:/projects/mercedes-benz`. Training does not require a server.

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5500
.venv/bin/mercedes track --run-dir .local/runs/baseline --tracking-uri "$MLFLOW_TRACKING_URI"
```

Import requires completed, consistent local artifacts. It verifies all remote bytes before marking delivery finished. Repeating a completed import rechecks those bytes and reuses the remote run. Interrupted/failed imports require `--resume` after the operator confirms the previous importer stopped. Conflicting manifests or duplicate logical identities fail closed. The journal records attempts; the logical experiment remains unchanged. This is single-writer operator recovery, not a distributed locking guarantee.

Host and container endpoints differ. The optional `docker/compose.mlops.yaml` joins an explicitly approved existing client network and uses `http://mlflow:5000`; set `MLOPS_CLIENT_NETWORK` appropriately. A loopback-published host endpoint is not automatically reachable through a container gateway. Optional `docker/compose.seaweed.yaml` exposes an existing client mount read-only at `/artifacts`; it does not deploy or administer storage. Clients use the MLflow proxy without database/S3 credentials.
