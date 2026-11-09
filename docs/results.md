# Measured results and their scope

The native full-data baseline fits 30 fold pipelines across five fixed partitions, two model seeds and three conventional families. It retains all 4,209 training rows and predicts all 4,209 test IDs. The producer is clean source commit `6530d167ca38262c1e2e1a22ffb301db14a31b2e`.

| Native pooled OOF result | R² | RMSE | MAE |
|---|---:|---:|---:|
| Predeclared equal mean | 0.548490 | 8.518833 | 5.404546 |
| Same recipe +16 appended SVD coordinates | 0.550070 | 8.503915 | 5.406060 |
| Baseline Random Forest component | 0.555267 | 8.454658 | 5.369755 |

The baseline is not changed after observing the stronger component. Appended coordinates have different effects across estimators; a small blend R² gain does not imply a gain for every family, fold or metric. [Full controlled comparison](modeling-study.md).

All 60 pipelines independently reload and reconstruct their retained OOF/test predictions. Maximum observed numerical difference is below 1e-13 against a declared 1e-10 tolerance. An independent NumPy analysis reconciles 8,692 metric/denominator values, with maximum absolute difference 5.82e-11. These checks establish consistency of the recorded artifacts, not future-production generalization.

Baseline fold R² ranges 0.409596–0.620809. One maximum-target observation contributes 9.79% of global squared error and 33.63% within the weakest fold. All rows remain evaluated. Pooled R² 0.548490 and mean fold R² 0.553212 use different denominators; neither is substituted for the other.

## Historical public-leaderboard context

| Authenticated historical submission | Public R² | Private R² | Public score-equivalent reference position |
|---|---:|---:|---:|
| Best recorded public weighted ensemble, 8 July 2017 | 0.55833 | 0.54776 | 1,663 / 3,824, approximately top 43.49% |
| `subm2_mean.csv`, 10 July 2017 | 0.55631 | 0.54957 | 1,884 / 3,824, approximately top 49.27% |
| Later weighted ensemble, 5 September 2017 | 0.55576 | 0.54683 | 1,948 / 3,824, approximately top 50.94% |

The reference is the dated official **public** leaderboard export recorded in [leaderboard-context.json](leaderboard-context.json). Positions use exact displayed-score comparisons and the unique scored-team population in that snapshot, rather than an assumed historical participant count. They are score-equivalent placements, not newly awarded ranks or private percentiles. Submission metadata does not prove an exact preserved source revision produced a historical checkpoint.

The modern native run has OOF evaluation only at this point; no modern scored competition submission or leaderboard percentile is inferred. CPU runtime was 32.60 seconds for the baseline and 42.67 seconds for the projection recipe, each capped at four threads. These are single observed local runtimes.

## Runtime and delivery evidence

Docker independently reconstructs the real baseline and enforces authenticated Jupyter API access. Native baseline and projection models are delivered through shared MLflow/SeaweedFS with full artifact-byte verification. Detailed receipts and the hosted-runtime status are retained in the validation record; offline Kaggle execution is a separate experiment, with its actual scientific environment recorded.

## Completed offline Kaggle run

Private notebook version 1 completed full-data CPU training with internet and accelerators disabled. It achieved pooled OOF R² **0.548500**, RMSE **8.518738** and MAE **5.404427**. All 30 models reloaded in their producing environment, with maximum prediction difference 8.53e-14; all 4,209 test IDs match the official submission order. The downloaded bundle, artifact hashes and independently recomputed OOF metrics were verified before shared import.

The code package, official inputs, configuration and fold assignment match the native baseline. Hosted Python 3.13.18 / scikit-learn 1.6.1 differ from native Python 3.12.16 / scikit-learn 1.9.1; the small score difference is recorded without claiming a diagnosed cause or bitwise equivalence. Native models were verified in Docker; hosted models were verified on Kaggle. The hosted bundle also reached FINISHED only after all 37 artifacts were read back from shared MLflow storage. [Hosted receipt](hosted-validation.json). No modern scored competition submission was made.
