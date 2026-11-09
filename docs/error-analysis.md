# The largest errors consistently underestimate longer tests

**The highest-error 211 of 4,209 OOF records account for 63.69% of total squared error; 207 of those durations are underestimated.** The same pattern survives the two recorded model seeds: their separately selected hard cohorts share 210 of 211 records. This directs the next modeling question toward large underestimates, rather than expecting another seed to fix the difficult observations.

The cohort is fixed as the largest **ceiling(5% × rows)** squared residuals of the retained baseline ensemble, with stable retained-row order breaking ties. It is selected using observed errors for diagnosis. Every original record remains in the reported R², RMSE and MAE.

## What concentrates the error

| Diagnostic | Observed value |
|---|---:|
| Reconciled pooled OOF R² / RMSE / MAE | 0.548490 / 8.518833 / 5.404546 |
| Hard cohort | 211 records; 63.69% of squared error |
| Underestimated / overestimated within that cohort | 207 / 4 |
| Mean observed / predicted duration within that cohort | 127.21 / 101.09 seconds |
| Smallest absolute error selected into the cohort | 15.94 seconds |
| Hard cases in the observed target upper decile | 157 / 211 |
| Shared hard cases between the two individual seeds | 210; Jaccard overlap 0.991 |

The cohort is not confined to the weakest fold: the five folds contain **44, 33, 42, 43 and 49** hard records. Fold 4 contributes 29.11% of global squared error, but removing or redesigning that fold alone would not resolve the broad pattern. The [earlier study](modeling-study.md) separately accounts for its largest individual residual and the controlled projection comparison.

The two seed ensembles differ by only **0.124 seconds on average on the hard cohort**, compared with that cohort's **26.84-second MAE**. This is descriptive evidence that the observed errors persist across this narrow seed variation. Shared folds and a deterministic Ridge component mean it is not an independent generalization test.

## Check a competing explanation

Identical observed feature vectors occur in **515 records (12.24%)**, and **26 hard cases (12.32% of the cohort)** have repeated features. Their nearly equal representation supplies no evidence that feature repetition alone concentrates the difficult cases.

All repeated-feature records belong to groups with differing observed targets; the largest within-pattern duration range is **58.56 seconds**. This raises a question about omitted process context or outcome variation. It does not establish irreducible noise, bad labels, the same physical vehicle, or leakage. The remaining **185 hard records** do not have an exact feature repeat in this snapshot.

## The next controlled question

A useful next experiment would ask whether a declared model or representation reduces large underestimates **while retaining full-population accuracy**. Compare the frozen blend and its Random Forest component on the same new, predeclared evaluation plan; retain per-fold effects and report signed error as well as R²/MAE. Do not optimize blend weights against the hard cases already examined here.

For a planning use case, an independently calibrated upper-duration estimate is another testable idea. It needs a declared coverage/underestimation objective, calibration data separate from final evaluation, and time/process-version identifiers where available. Neither the necessary calibration nor a process intervention has been performed in this analysis.

## Reproduce the diagnostic

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/analyze_hard_cases.py \
  --run-dir .local/runs/baseline --data-dir "$MERCEDES_DATA_DIR" \
  --output docs/error-analysis.json
```

The analyzer checks input/artifact hashes, exact raw/OOF/fold identities, raw targets and declared member prediction columns. It independently reconstructs the equal family/seed mean and reconciles pooled metrics before computing the cohort. The [aggregate receipt](error-analysis.json) includes provenance and all fold totals, without IDs or raw rows. No fitting, inference, filtering, relabelling, threshold tuning or promotion occurs.
