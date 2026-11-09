# Planning test-bench capacity from vehicle configurations

The [competition brief](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing) asks how configuration-dependent testing can become more efficient without compromising standards. The modeling task estimates test duration. Translating that estimate into shorter queues, higher utilization or lower energy use requires an additional operating decision and a measured intervention.

A plausible use is capacity planning: compare expected workloads, flag configurations likely to overrun and give planners a consistent estimate before testing. These are proposed uses. The anonymized benchmark does not identify which production decisions were made or measure their consequences.

## What the solution teaches about that decision

The baseline blend reaches pooled OOF R² 0.548490. Adding 16 SVD coordinates raises it to 0.550070, but the gain varies by estimator: ExtraTrees improves while Random Forest worsens, and baseline Random Forest remains stronger than either blend. This argues for testing the additional representation by model family rather than assuming more components yield a better deployable solution. [Controlled comparison](modeling-study.md).

The largest-target record contributes 33.63% of squared error in the weakest fold. That makes long-delay behavior a planning question, not just a detail behind the mean score. All records remain evaluated. The audit also finds 24 test rows with previously unseen categories; inference therefore needs explicit category handling, and unseen configurations need their own performance validation.

## Score, operating KPI and business effect

| Level | Measure | Current evidence |
|---|---|---|
| Predictive accuracy | Pooled/fold R², RMSE, MAE and tail-error contribution | Measured in this repository |
| Planning quality | Duration bias and error by configuration/time period; costly underestimation rate | Requires representative operational data and an agreed decision policy |
| Process outcome | Queue delay, test-bench utilization, work in process, on-time completion | Requires scheduling or shadow-planning evaluation |
| Business effect | Avoided overtime, usable throughput or energy per completed test | Hypothesis; needs observed intervention and cost/energy inputs |

The [MO360 production release](https://media.mercedes-benz.com/en/article/abc9fd0b-edf2-4edd-901a-8bd5a294804a) describes production teams working with quality status and operating KPIs. That is relevant context for connecting a model to a decision, rather than presenting an isolated score. It does not establish a relationship between this solution and Mercedes-Benz's operating systems.

A minimal pilot would use time-ordered held-out production periods, retain identifiers needed for configuration and process-version analysis, compare against the planner's current estimate, and evaluate the same decision rule with and without model support. Safety and quality standards remain fixed. Record actual queue and energy outcomes before attributing savings or emissions reductions.

The serving contract packages each fold's encoder and fitted estimator, preserves input order and unknown-category behavior, and verifies that API predictions match the evaluated ensemble. This makes the estimate callable by a planning application. Whether that application improves the production process remains a separate, measurable outcome.
