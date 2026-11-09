# Mercedes-Benz Regression Pipeline

A CPU regression pipeline that makes category handling, ensemble selection and error accounting explicit—and verifies that delivered predictions come from the evaluated models.

This revisits Chia-Ta Tsai's 2017 [Mercedes-Benz Greener Manufacturing](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing) scripts as an ML engineering / MLOps consumer case study. The task predicts test-bench duration from anonymized vehicle features, using the competition's R² metric. The implementation keeps conventional Ridge, Random Forest and Extra Trees models and a predeclared equal mean.

## Why category handling is a delivery contract

The official inputs contain 4,209 training and 4,209 test observations, with 376 predictors. **24 test rows contain a category absent from the full training input.** Fitting a vocabulary on combined train/test data would conceal that delivery condition. Each fold instead fits its own encoder and records how unknown categories are transformed; the same fitted preprocessing travels with its saved model. Unlabeled test rows do not establish prediction quality on those new categories.

## Why a regression score needs its population

The target has a long upper tail. A single maximum contributes about 4% of total target variation. That fact alone does not identify a model's largest error: the analysis separately measures retained OOF residuals, fold denominators and pooled R². All official training rows remain in evaluation. A model score, a mean fold score and a retrospectively filtered score are different claims.

Recorded real-data results, verification receipts and the executed walkthrough are being assembled in this branch. The original files remain in [legacy/](legacy/README.md); the maintained workflow uses current APIs rather than emulating removed parameters.

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

Use authorized Kaggle inputs and a new run directory. Model configurations cap CPU threads at four. Data, weights, private row-level predictions and credentials stay outside Git.

## License

Source code retains the [MIT License](LICENSE). Competition data follows its own terms.
