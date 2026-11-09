# Mercedes-Benz: Test-Bench Duration

A useful test-time estimate can help a manufacturing team plan around vehicle complexity. The [Mercedes-Benz challenge](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing) asks for testing-duration predictions from anonymized configuration features. This solution combines fold-fitted categorical preprocessing with Ridge, Random Forest and Extra Trees models in a CPU workflow.

The fixed blend reaches **0.54849 OOF R²**. Appending SVD coordinates raises it to **0.55007**, but Extra Trees benefits while Random Forest worsens; the baseline Random Forest remains stronger than either blend at **0.55527**. Estimator-specific representation matters more here than adding complexity to every model.

The [problem and value discussion](docs/problem-and-value.md) connects duration estimates to capacity planning, long-delay risk and the operating measures needed to establish a business benefit. The implementation retains fitted preprocessing and all **60 pipelines** across the two experiments so their predictions can be independently reconstructed.

[Controlled comparison](docs/modeling-study.md) · [Results](docs/results.md) · [Run the solution](#reproduce-on-cpu)

<!-- portfolio-results:start -->
## Results

**Public leaderboard — R² ↑.**

| Model / submission | Public score | Public reference rank / teams | Top % | Reference ties |
|---|---:|---:|---:|---:|
| Historical submission 5080178 (2017-07-08) | 0.55833 | 1,663 / 3,824 | **43.49%** | 1 |
| Historical submission 5094866 (2017-07-10) | 0.55631 | 1,884–1,885 / 3,824 | **49.27%** | 2 |
| Historical submission 5322692 (2017-09-05) | 0.55576 | 1,948 / 3,824 | **50.94%** | 0 |

Reference: **2026-11-09T04:12:42Z**, **3,824 unique scored teams**. Ranks are score-equivalent comparisons with the dated public export, not awarded final ranks. Lower Top % is better. A zero tie count denotes an insertion position; ties occupy the shown interval.

**OOF/holdout rank and Top %: not applicable. Private leaderboard rank and Top %: unavailable without a matching private reference population.** Internal validation and private scores remain separately reported in [detailed results](docs/results.md). [Reference data and calculation](docs/leaderboard-context.json).
<!-- portfolio-results:end -->

## Which delays dominate the error?

The worst 211 of 4,209 retained OOF predictions contribute 63.69% of squared error; 207 underestimate the duration. Almost the same examples recur across training seeds. Repeated feature configurations are not enriched in that cohort, so they do not explain its concentration. This points to a controlled investigation of long-delay underprediction before changing ensemble weights. [Hard-case analysis, competing explanations and next experiment](docs/error-analysis.md).

## Models and results

The fixed baseline reaches **0.54849 pooled OOF R²**, with RMSE **8.51883**. Adding 16 SVD coordinates changes blend R² to **0.55007** and slightly worsens MAE; effects differ across estimators. Baseline Random Forest scores **0.55527**, stronger than either blend. The declared recipe is retained rather than optimizing weights against the reported OOF labels. [Controlled comparison](docs/modeling-study.md).

All **60 saved pipelines** across the two recipes reconstruct their retained OOF and test predictions within the declared tolerance. Native artifacts are also verified through Docker and shared MLflow storage. A separate offline Kaggle CPU run reaches **0.54850 OOF R²**, reloads all 30 hosted models and delivers its verified bundle to the same tracker. [Validation receipt](docs/validation.json) · [Executed walkthrough](notebooks/01-method-and-rerun.ipynb).

The historical account's best public score **0.55833** corresponds to approximately **top 43.49%** of the dated public leaderboard snapshot. The modern native figures above are OOF results, with no inferred leaderboard percentile. [Score context and limits](docs/results.md).

## Handling unseen configurations

The official inputs contain 4,209 training and 4,209 test observations, with 376 predictors. **24 test rows contain a category absent from the full training input.** Fitting a vocabulary on combined train/test data would conceal that delivery condition. Each fold instead fits its own encoder and records how unknown categories are transformed; the same fitted preprocessing travels with its saved model. Unlabeled test rows do not establish prediction quality on those new categories.

## Use the saved model

The [model packaging and serving guide](docs/serving.md) explains the actual input/output contract, export from saved artifacts and the shared HTTP parity check. A canary release retains the tested source, dependency lock and aggregate verification evidence; private model weights and request rows are not release assets.

[Real-model canary verification](docs/canary-validation.json) records the source, package/runtime identity, native-to-HTTP parity and rejected invalid request.

## Understanding long-delay errors

The target has a long upper tail. A single maximum contributes about 4% of total target variation. That fact alone does not identify a model's largest error: the analysis separately measures retained OOF residuals, fold denominators and pooled R². All official training rows remain in evaluation. A model score, a mean fold score and a retrospectively filtered score are different claims.

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

## From saved models to predictions

Training fits vocabulary and transforms inside each fold, and the saved pipeline owns that state at inference. Verification checks input/schema/model identities; evaluation independently reconstructs the predictions. A local completed run survives tracking unavailability: explicit import recovery reuses its logical identity and checks remote artifact bytes before completion. [Consumer contract](docs/consumer-contract.md) · [Architecture](docs/architecture.md).

The shared platform owns service operations. This repository consumes those services and remains runnable offline; there is no production manufacturing service or automatic retraining claim. The original source and attribution remain in [legacy/](legacy/README.md). [Changes from 2017](docs/migration.md).

## License

Source code retains the [MIT License](LICENSE). Competition data follows its own terms.
