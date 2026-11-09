# The MLOps consumer contract

The engineering question is whether delivered predictions can be traced to the evaluated pipelines, including their fitted preprocessing and input population. The model is a conventional workload; shared service names alone do not establish operational correctness.

## A new category must have a defined meaning

There are 24 test observations with a category absent from full training. Fold-local training can encounter additional held-out-only levels. This supports an explicit unknown-category policy and a saved encoder at the prediction boundary. Contract tests prove held-out-only values do not enter fitted categories, while prediction remains defined. Unlabeled test inputs establish coverage, not predictive quality.

## A score must retain its population and recipe

Every official training row is retained, including the largest target/error. The manifest binds actual fold IDs and input hashes; evaluation reconstructs the same population rather than merely rerunning a seed. Pooled R², per-fold scores and MAE retain their distinct definitions. The fixed ensemble's score is reported even where a component is stronger; OOF labels do not select new weights.

## A completed model can survive failed delivery

The local run bundle remains the source evidence when a tracking service is unavailable. A failed transfer receives an attempt journal and failed status; explicit recovery reuses its logical/remote identity and reads back artifact hashes before completion. Tests inject partial transfer and verify same-run recovery. Native real model artifacts have also been imported and read back through the existing shared proxy. This is a consumer-delivery guarantee, not server uptime or distributed locking.

This repository owns project ID `mercedes-benz`, tracking experiment `kaggle/mercedes-benz` and artifact namespace `projects/mercedes-benz`. These existing identities remain unchanged. An authorized operator provisions the experiment through MLflow using this client-declared configuration; the importer requires it to exist and never chooses a fallback experiment. The operator checks the intended artifact route before provisioning. Shared services do not need a catalog entry or source-code change for this solution. Run identities and source/input/model fingerprints remain valid across local and offline Kaggle execution.

## Online serving handoff

The solution owns its inference interface and model acceptance criteria. A serving release must package a supported MLflow Model with an input/output signature, compatible dependencies and all fitted preprocessing. For Mercedes-Benz, that includes ordered input columns, fold-fitted encoders, unknown-category handling and the declared ensemble. A raw checkpoint or an artifact uploaded to MLflow is not by itself an MLflow serving package; packaging and serving validation remain implementation work in this repository.

The handoff supplies a verified immutable model version or image digest, prediction checks, resource requirements and declared evaluation gates. The shared platform owns deployment, access control, health/latency/error monitoring, traffic switching and rollback. A registry alias may select a release candidate, but a running release must record the resolved version; changing an alias does not establish that serving processes have reloaded it. The solution links to the [shared infrastructure requirements](https://github.com/cttsai1985/ml-infrastructure/blob/main/docs/requirements.md); the infrastructure does not import this repository or maintain its project identity.

There is no production manufacturing deployment, live drift/label feedback or automatic retraining claim. The shared platform's multi-stage batch fault drill uses Iceberg; it is not presented as this model's scheduled execution. [Runbook](reproduction.md) · [Measured study](modeling-study.md).
