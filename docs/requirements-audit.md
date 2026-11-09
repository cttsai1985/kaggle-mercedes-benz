# Requirements audit

## Presentation update — standard 1.1

The current primary display pairs **Public LB** and **Private LB** scores with
**Public Top %**. Rank/team counts and ties remain in detailed source evidence;
they are no longer repeated in the README table or GitHub description. Private
scores do not inherit public percentiles. This supersedes the canary-2 presentation
format reviewed below; its historical regression and source-preservation review
remain dated evidence. Local reproducibility may use scientific functions, a
notebook, script or existing thin command entry point; no bespoke CLI is required.
The existing canary-2 releases and completed merge/branch cleanup remain intact.
The original profile/standard are retained at `v0.2.0-canary.2`; the current
profile receipt records the latest approved compact description.


Audited **2026-11-09** against [portfolio standard 1.1](portfolio-standard.md).
The standard is the accumulated acceptance contract. **Pass** below refers to the
cited scope; **Open** is accepted work that has not been waived. Historical model,
evaluation and canary receipts keep their original source identities. The current
editorial/identity correction does not make them fresh executions.

## Ranking regression and remediation

The historical public-score/Top-% paragraph occurred on README **line 21** at
`a962d2dd596fede27083ace4267862cc72e6d904` and **line 17** in the merged canary
state, `52006c5efbf549bf2be26b0c361e33cdda0a097b`. It was **not pushed farther
down**. The problem was incomplete primary context: the leading model summary
highlighted OOF R², while its opening did not pair the historical public benchmark
with a reference rank and team denominator. The domain-focused GitHub description
also dropped score/rank/Top-% context during the branding change.

The correction restores primary **Results** immediately after the opening, with
all three retained historical public submissions, R² direction, matched reference
positions/N, Top %, tie counts and snapshot. The live description was read back
with historical **0.55833: 1,663/3,824 (Top 43.49%)**, the dated public reference,
and an explicit statement that modern baseline **OOF R² 0.54849 has no leaderboard
rank**. This is a score-equivalent public comparison, not an awarded rank.

The other retained public scores **0.55631** and **0.55576** remain in the primary
table with their own comparisons. OOF/holdout ranking is not applicable; private
ranking is unavailable without a matching private reference population. See
[reference evidence](leaderboard-context.json) and [detailed results](results.md).

## Acceptance matrix

| ID | Applicability | Status and project evidence |
|---|---|---|
| R01 | All accumulated portfolio requirements apply. | **Pass for the review artifact:** this R01–R26 matrix references the versioned [standard](portfolio-standard.md), records scope and retains explicit open items. Correction release closure is not assumed. |
| R02 | A usable duration-prediction solution is the unit of presentation. | **Pass:** README opening → primary Results → model comparison, unseen-category handling, runnable CPU workflow and saved-model use. [Problem/value](problem-and-value.md), [runbook](reproduction.md) and [serving](serving.md) explain how the output can be inspected and consumed. |
| R03 | Verified compact presentation | Primary README and description retain public/private scores and public Top %. Rank/N and ties stay in linked detailed evidence. |
| R04 | Verified source scope | The metric/direction and submission are identified; private scores are paired with the same public submission. Detailed evidence retains reference counts, ties and snapshot. |
| R05 | Official awards, reference positions and evaluation populations must remain separate. | **Pass:** primary Results labels score-equivalent public comparisons and marks OOF/holdout rank non-applicable. Historical private values remain separate and receive no invented private percentile. The modern model does not inherit a historical submission's rank. |
| R06 | Modern representation/ensemble changes require measured decisions. | **Pass:** [controlled study](modeling-study.md) retains matched baseline/SVD results, family-specific effects, both seeds and every fold. Extra Trees improves while Random Forest worsens; the small blend gain does not justify a universal improvement or post-hoc weight search. |
| R07 | Hard cases must connect model inputs/representation with prediction failure. | **Pass for retained-OOF diagnosis:** [error analysis](error-analysis.md) follows anonymized configuration inputs, fixed fold pipelines and equal-mean predictions into a fixed high-squared-error cohort, signed underestimation and repeated-feature context. [Aggregate evidence](error-analysis.json) reconciles the full population. Physical causes and missing process variables remain hypotheses. |
| R08 | Trustworthy regression evaluation and explicit limitations apply. | **Pass within recorded scope:** [validation](validation.json) and [study](modeling-study.md) retain source/input/split hashes, all 4,209 OOF rows, fold-local transforms, both model seeds, saved-model reconstruction and independent metric checks. Random record folds do not establish future-period or unseen-configuration performance; Ridge's repeated seed is not stochastic evidence. |
| R09 | Hard-case selection must not alter the evaluation or claim an unbiased gain. | **Pass:** the ceiling-top-5% squared-residual cohort is retrospective. All rows remain in R²/RMSE/MAE; no outlier dropping, relabelling, blend-weight tuning, threshold selection or promotion is performed. [Diagnostic limits](error-analysis.md). |
| R10 | The evaluated model ensemble and preprocessing must survive reload. | **Pass for the declared baseline:** [serving guide](serving.md) and adapter package all **30** fold/family/seed pipelines with ordered schema, fold-fitted encoders and equal-mean aggregation. Original experiment and lock identities remain in [validation](validation.json) and [canary](canary-validation.json). |
| R11 | Actual native→package→fresh-workspace HTTP inference must be tested. | **Pass for canary.1 scope:** [receipt](canary-validation.json) records four real-input predictions, maximum absolute difference **2.842170943040401e-14** at tolerance **1e-10**, invalid-request HTTP **400**, digest checks and cleanup. Production ingress and a serving-container image were not verified by that test. |
| R12 | The inference boundary must be explicit and reproducible. | **Pass:** [serving](serving.md) declares ordered anonymized configuration features, excluding ID and target. Each saved pipeline owns its fitted encoding and numeric preprocessing. This is configuration-to-duration inference, not a raw factory-event or scheduling service. |
| R13 | Local scripts, Docker/dev container and offline Kaggle paths apply. | **Preserved with recorded evidence:** [reproduction](reproduction.md), [Kaggle guide](kaggle.md), executed notebooks, [Docker/native receipt](validation.json) and [actual hosted receipt](hosted-validation.json). No fresh training, Lab authentication or Kaggle run is claimed for the correction. |
| R14 | Infrastructure dependency is client→platform, with no client catalog. | **Pass:** [consumer contract](consumer-contract.md) keeps project/experiment/artifact names, scientific acceptance and model package ownership here. Generic shared interfaces do not need a source change for this client. |
| R15 | Shared infrastructure lifecycle belongs to its operator, not this model repo. | **Delegated applicability:** the client exports its model and supplies its environment/request contract. The shared [requirements](https://github.com/cttsai1985/ml-infrastructure/blob/main/docs/requirements.md) own lightweight `./infra` deployment, state reuse, identity and diagnostics. No local copy of the service stack is required. |
| R16 | Branding must preserve existing services and retained identities. | **Pass for correction scope:** edits target presentation/metadata and add diagnostics; they do not migrate volumes, ports or experiments. [Canary evidence](canary-validation.json) explicitly records no shared-service modification. This is not a fresh host-wide service inventory. |
| R17 | Raw inputs, row outputs, weights and credentials must remain private. | **Pass for inspected Git/release scope:** no tracked `.joblib`/`.pt` weights or raw train/test/OOF/submission payloads were found. [Canary.1 release assets](https://github.com/cttsai1985/kaggle-mercedes-benz-solution/releases/tag/v0.2.0-canary.1) contain package/source, lock, checksums and aggregate documents; private assets are explicitly excluded. |
| R18 | Release/evaluation provenance must stay exact. | **Pass for retained evidence; correction checks need closure:** canary export source `907438c508637c2f97da86f9a5bc0da2d1845b7a` is recorded clean with lock/package/runtime hashes. The new [diagnostic](error-analysis.json) has its own analyzer/input/OOF identities. Metadata version `0.2.0.dev2` is a correction candidate, not a rewritten `0.2.0.dev1` execution. Final-source checks remain distinct from fresh HTTP/training runs. |
| R19 | The client participates in the coordinated canary requirement. | **Published client evidence:** `v0.2.0-canary.1` is a GitHub prerelease at `52006c5efbf549bf2be26b0c361e33cdda0a097b`. **Open for this correction:** finish its merge/check/release review and coordinated portfolio closure. A canary prerelease does not mean production traffic splitting. |
| R20 | Credible naming, solution-first prose and attribution apply. | **Pass for current presentation:** canonical name is `kaggle-mercedes-benz-solution`; opening leads the actual modeling task and measured effects. MIT attribution and original `legacy/` source remain. The precise preservation comparison is below. |
| R21 | Business relevance must use primary sources and distinguish prediction from intervention. | **Pass:** [problem/value](problem-and-value.md) cites the original challenge and Mercedes-Benz's dated MO360 context. Capacity planning is a proposed use; no queue, throughput, energy or carbon improvement is claimed as measured by this repository. |
| R22 | Production batch and MLflow-based online serving remain accepted targets. | **Open P1:** the model package/interface has canary evidence. Full batch and endpoint operation remains subject to the shared [requirements](https://github.com/cttsai1985/ml-infrastructure/blob/main/docs/requirements.md), rather than reducing the product target to a demonstration. |
| R23 | Production hosting, ownership, capacity and recovery need explicit decisions/evidence. | **Open P1:** target SLOs and budgets have not been invented. Appropriate TLS/auth, observability/alert delivery, off-host backup/restore and rollout/rollback still require selected deployment decisions and tested acceptance. |
| R24 | Delivery must preserve input identity, safe recovery and gate semantics. | **Preserved; operational completion open P1:** [consumer contract](consumer-contract.md) and tracking tests cover completed local experiments, checksum verification, explicit same-identity recovery and idempotence. Shared batch gates/retries have their own infrastructure evidence; this does not claim this benchmark was scheduled through Prefect. |
| R25 | Retained errors should guide specific next experiments. | **Implemented diagnostic; hypotheses open P1:** [hard-case analysis](error-analysis.md) quantifies tail underestimation, seed persistence and competing repeated-feature context. New predeclared evaluation, calibrated upper-duration estimates and time/process-version analysis are proposals; none is reported as a completed gain. |
| R26 | Canonical URLs, export/package metadata, releases and safe branch hygiene apply. | **In progress until correction merge:** current solution URLs and metadata are being aligned; prior immutable receipts retain their identities. Release checks and deletion of only verified merged branches remain owner actions; active/default/unmerged work must be retained. |

## Scientific-preservation comparison

The audit compared exact file bytes through `git show <commit>:<path>`:

1. **Pre-canary → merged canary/editorial:**
   `a962d2dd596fede27083ace4267862cc72e6d904` →
   `52006c5efbf549bf2be26b0c361e33cdda0a097b`.
2. **Merged canary/editorial → canary-2 correction working tree.**

These enumerated files are byte-identical across both comparisons:

| Scope | Checked paths |
|---|---|
| Controlled comparison | `docs/modeling-study.md`, `docs/modeling-study.json` |
| Recorded native/hosted/data evidence | `docs/validation.json`, `docs/hosted-validation.json`, `docs/data-audit.json` |
| Notebook content and retained outputs | `notebooks/01-method-and-rerun.ipynb`, `notebooks/kaggle/rerun.ipynb` |
| Training/evaluation and delivery implementation | `src/mercedes/pipeline.py`, `src/mercedes/tracking.py`, `src/mercedes/import_support.py` |
| Scientific recipes | `configs/baseline.json`, `configs/projection.json`, `configs/smoke.json` |

`docs/canary-validation.json`, `docs/serving.md` and `src/mercedes/serving.py` were
added during canary work and remain byte-identical from the merged canary to the
canary-2 correction. Inspectable SHA-256 anchors are:

- Modeling-study JSON: `03aa8c4403159c105f5927d30b198d9d56a5d7847936da405f0ba0d63fa49659`.
- Canary receipt: `6281d5cfc3bae318623fa7d650204cc3a91865b3b41ca1ace1f145996e7c3722`.
- Training/evaluation module: `4a40bee7f567dae5e793b6ca7f7e7d9ad0d0efdda78c9688561acd27da6262f9`.

`docs/results.md` matched the pre-canary and merged-canary states, but **is not
byte-identical to the canary-2 correction**, which adds explicit public-ranking
context. README, version/URL/export metadata, portfolio-standard/check files and
new error-analysis files are also outside the preserved set. The correction bumps
package/lock metadata to `0.2.0.dev2`; earlier receipts retain `0.2.0.dev1` and
their recorded source. New diagnostics supplement the existing all-row metrics;
`LICENSE` and every tracked `legacy/` file were also compared across both
transitions and matched. This audit is not a comparison of every untracked private
file or a fresh training reproduction.

The published canary asset inventory was read through GitHub during this audit; release archives were not newly downloaded or unpacked. No new training or HTTP canary was executed by this requirements review.
