# Projection helps one estimator; the blend does not establish a universal gain

**Appending SVD coordinates changes which estimator benefits.** The equal-mean R² rises from **0.548490 to 0.550070**, but the matched family results tell a different story: ExtraTrees gains **0.015935**, Random Forest loses **0.004531**, and Ridge is nearly unchanged. ExtraTrees improves in all ten seed/fold pairs; the blend improves in only three of five ensemble folds, and its MAE slightly worsens. Appending coordinates retains every original input: it changes the representation available to the estimator, without adding information or compressing features. The experiment does not isolate the mechanism. Keep the baseline frozen; estimator-specific projection is a next hypothesis, not a measured replacement. Baseline Random Forest still exceeds both blends.

**The weakest fold needs error accounting before a modeling response.** Baseline fold R² ranges from **0.409596 to 0.620809**. The maximum-target observation contributes **9.79%** of total squared error and **33.63%** of the weakest fold's error. The ten largest residuals contribute **19.91%** overall. This explains part of the spread without establishing that any label is wrong or any error unavoidable. Retain all **4,209** rows and report pooled R², fold results and error concentration together. Neither trimming difficult targets nor searching blend weights on these same OOF labels would supply fresh validation evidence.

## Controlled results

Each family averages its two seeds; the blend gives each family and seed equal weight.

| Predictor | Baseline R² | Appended SVD R² | Paired Δ R² |
|---|---:|---:|---:|
| Ridge | 0.528588 | 0.528534 | −0.000054 |
| Random Forest | 0.555267 | 0.550736 | −0.004531 |
| ExtraTrees | 0.488273 | 0.504208 | +0.015935 |
| Predeclared equal mean | 0.548490 | 0.550070 | +0.001580 |

The blend's paired seed gains are **+0.001396** and **+0.001720**. RMSE changes **8.518833 → 8.503915**; MAE changes **5.404546 → 5.406060**. No winner or weights were selected from these results.

Both recipes use five identical shuffled folds (seed 904), model seeds 170904/170905, training-fold-only preprocessing, and the same Ridge/160-tree budgets. Only the descriptive name and **0 versus 16 appended SVD coordinates** differ. Baseline Ridge predictions are identical across seeds. With SVD, seed variation also includes the fitted projection; it is not purely tree-training variation. Two seeds on one split plan support descriptive comparisons, not confidence intervals.

Pooled R² is `1 − total SSE / global SST`; mean fold R² averages ratios with different fold denominators. They are **0.548490** and **0.553212** for the baseline. The [aggregate receipt](modeling-study.json) retains every estimator/seed/fold metric, matched delta, and SSE/SST reconciliation. The largest 43 targets account for **32.14%** of baseline SSE; this is a post-fit diagnostic and all remain in evaluation.

## Evidence and reproduction

The independent analyzer checks identical input/split/source identities, the controlled configuration difference, artifact hashes, exact ID/target coverage, declared folds and equal-mean arithmetic. Direct NumPy calculations reconcile **8,692 numeric values**, with maximum absolute error **5.82 × 10⁻¹¹**. Saved-model reconstruction is a separate check:

```bash
.venv/bin/mercedes evaluate --run-dir .local/runs/baseline --data-dir "$MERCEDES_DATA_DIR"
.venv/bin/mercedes evaluate --run-dir .local/runs/projection --data-dir "$MERCEDES_DATA_DIR"
.venv/bin/python scripts/analyze_comparison.py \
  --baseline .local/runs/baseline --projection .local/runs/projection \
  --data-dir "$MERCEDES_DATA_DIR" --output docs/modeling-study.json
```

**MEASURED:** 30 fitted fold models per recipe, four CPU threads, **32.60 s** baseline and **42.67 s** projection. These are observed local run times, not a repeated latency benchmark. Inputs, row IDs, predictions and weights remain outside Git.

The legacy scripts already fit their decomposition, embedding and clustering steps inside outer folds. Their distinct limitations are a train-plus-test category vocabulary, self-including training median encodings, unordered feature sets and blend-weight selection on the reported OOF labels. This modernization changes several choices; its scores are not a controlled improvement over the legacy recipe. Random record folds also do not establish future-production or unseen-configuration performance. No leaderboard result is inferred.
