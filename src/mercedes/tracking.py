"""Verified delivery into a preprovisioned experiment, independent of model fitting."""

import json
import uuid
from pathlib import Path
from urllib.parse import urlparse

from mercedes import PROJECT_ID, REPOSITORY
from mercedes.import_support import import_verified
from mercedes.pipeline import sha256, verify_run


def track(run_dir, tracking_uri, *, resume=False):
    from mlflow import MlflowClient

    if not tracking_uri or urlparse(tracking_uri).username:
        raise ValueError(
            "Supply an approved endpoint; use runtime authentication, not URI credentials"
        )
    run_dir = Path(run_dir)
    manifest = verify_run(run_dir)
    logical_id = str(uuid.UUID(manifest["logical_run_id"]))
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("kaggle/mercedes-benz")
    if experiment is None or experiment.lifecycle_stage != "active":
        raise ValueError(
            "Experiment kaggle/mercedes-benz must be provisioned by an authorized MLflow operator "
            "using the client-owned namespace before import"
        )
    metrics = json.loads((run_dir / "metrics.json").read_text())
    digest = sha256(run_dir / "manifest.json")
    tags = {
        "project_id": PROJECT_ID,
        "repository": REPOSITORY,
        "run_kind": manifest["run_kind"],
        "source_sha256": manifest["source"]["python_source_sha256"],
        "source_commit": manifest["source"].get("git_commit") or "unavailable",
    }
    logged_metrics = {"oof." + key: value for key, value in metrics["pooled_oof"].items()}
    for family in metrics["model_results"]:
        logged_metrics.update(
            {f"model.{family['model']}.{key}": family[key] for key in ("r2", "rmse", "mae")}
        )
    return import_verified(
        client,
        experiment_id=experiment.experiment_id,
        run_dir=run_dir,
        logical_id=logical_id,
        manifest_digest=digest,
        artifacts={**manifest["artifacts"], "manifest.json": digest},
        tags=tags,
        run_name=manifest["config"]["name"],
        params={
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in manifest["config"].items()
        },
        metrics=logged_metrics,
        resume=resume,
    )
