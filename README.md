# Mercedes-Benz Regression Pipeline

A CPU regression pipeline that makes category handling, ensemble selection and error accounting explicit—and verifies that delivered predictions come from the evaluated models.

This revisits Chia-Ta Tsai's 2017 [Mercedes-Benz Greener Manufacturing](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing) scripts as an ML engineering / MLOps consumer case study. The task predicts test-bench duration from anonymized vehicle features, using the competition's R² metric. The implementation keeps conventional Ridge, Random Forest and Extra Trees models and a predeclared equal mean.

## Why category handling is a delivery contract

The official inputs contain 4,209 training and 4,209 test observations, with 376 predictors. **24 test rows contain a category absent from the full training input.** Fitting a vocabulary on combined train/test data would conceal that delivery condition. Each fold instead fits its own encoder and records how unknown categories are transformed; the same fitted preprocessing travels with its saved model. Unlabeled test rows do not establish prediction quality on those new categories.

## Why a regression score needs its population

The target has a long upper tail. A single maximum contributes about 4% of total target variation. That fact alone does not identify a model's largest error: the analysis separately measures retained OOF residuals, fold denominators and pooled R². All official training rows remain in evaluation. A model score, a mean fold score and a retrospectively filtered score are different claims.

## Recorded evidence

The fixed baseline reaches **0.54849 pooled OOF R²**, with RMSE **8.51883**. Adding 16 SVD coordinates changes blend R² to **0.55007** and slightly worsens MAE; effects differ across estimators. Baseline Random Forest scores **0.55527**, stronger than either blend. The declared recipe is retained rather than optimizing weights against the reported OOF labels. [Controlled comparison](docs/modeling-study.md).

All **60 saved pipelines** across the two recipes reconstruct their retained OOF and test predictions within the declared tolerance. Native artifacts are also verified through Docker and shared MLflow storage. [Validation receipt](docs/validation.json) · [Executed walkthrough](notebooks/01-method-and-rerun.ipynb).

The historical account's best public score **0.55833** corresponds to approximately **top 43.49%** of the dated public leaderboard snapshot. The modern native figures above are OOF results, with no inferred leaderboard percentile. [Score context and limits](docs/results.md).

## From evaluated experiment to delivered predictions

Training fits vocabulary and transforms inside each fold, and the saved pipeline owns that state at inference. Verification checks input/schema/model identities; evaluation independently reconstructs the predictions. A local completed run survives tracking unavailability: explicit import recovery reuses its logical identity and checks remote artifact bytes before completion. [Consumer contract](docs/consumer-contract.md) · [Architecture](docs/architecture.md).

The shared platform owns service operations. This repository consumes those services and remains runnable offline; there is no production manufacturing service or automatic retraining claim. The original source and attribution remain in [legacy/](legacy/README.md). [Changes from 2017](docs/migration.md).

## Reproduce on CPU

```bash
uv sync --locked --group dev --group lab --group data --group tracking
export MERCEDES_DATA_DIR=/absolute/path/outside/git/mercedes-benz
.venv/bin/mercedes validate --data-dir "$MERCEDES_DATA_DIR"
.venv/bin/mercedes train --data-dir "$MERCEDES_DATA_DIR" \
  --config configs/baseline.json --run-dir .local/runs/baseline
.venv/bin/mercedes evaluate --data-dir "$MERCEDES_DATA_DIR" \
  --run-dir .local/runs/baseline
```

Use authorized Kaggle inputs and a new run directory. Model configurations cap CPU threads at four. [The runbook](docs/reproduction.md) covers Kaggle CLI download, the matched projection recipe, model reload, tracking and prediction.

## Docker, dev container and Kaggle

```bash
.venv/bin/python scripts/create_lab_token.py
docker compose up --build -d lab
```

JupyterLab is available at http://127.0.0.1:8893/lab with the ignored `.local/jupyter_token`. Data is mounted read-only. The dev container uses the same image and imports the checkout from `/workspace`. [Runtime instructions](docs/reproduction.md).

The [offline Kaggle entry](notebooks/kaggle/rerun.ipynb) installs a verified code wheel and trains the same package against the attached competition input, retaining actual hosted versions and a portable run bundle. [Kaggle delivery](docs/kaggle.md).

`src/mercedes/` owns the implementation; `configs/` fixes recipes; `notebooks/` explains and executes them; `docs/` retains results and decisions; `tests/` checks contracts; `docker/` owns client environments. Data, weights, private row-level predictions and credentials stay outside Git.

## License

Source code retains the [MIT License](LICENSE). Competition data follows its own terms.
