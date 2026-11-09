import json

import pytest

mlflow = pytest.importorskip("mlflow")
pytest.importorskip("sqlalchemy")

from test_contracts import fixture_data  # noqa: E402

from mercedes.pipeline import DEFAULT, train  # noqa: E402
from mercedes.tracking import track  # noqa: E402


def test_partial_delivery_requires_explicit_recovery_and_keeps_remote_identity(
    tmp_path, monkeypatch
):
    data = fixture_data(tmp_path / "data")
    run = tmp_path / "run"
    train(
        data,
        run,
        {**DEFAULT, "folds": 2, "models": ["ridge"], "model_seeds": [170904], "threads": 1},
        fixture=True,
    )
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    client = mlflow.MlflowClient(tracking_uri=uri)
    with pytest.raises(ValueError, match="authorized MLflow operator"):
        track(run, uri)
    experiment = client.create_experiment(
        "kaggle/mercedes-benz", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    original = mlflow.MlflowClient.log_artifact
    attempts = 0

    def interrupted(self, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            raise ConnectionError("controlled fixture upload failure")
        return original(self, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(mlflow.MlflowClient, "log_artifact", interrupted)
        with pytest.raises(ConnectionError):
            track(run, uri)
    failed = client.search_runs([experiment])[0]
    assert failed.info.status == "FAILED"
    with pytest.raises(ValueError, match="Incomplete"):
        track(run, uri)
    resumed = track(run, uri, resume=True)
    assert resumed["mlflow_run_id"] == failed.info.run_id
    assert resumed["status"] == "resumed"
    assert client.get_run(failed.info.run_id).info.status == "FINISHED"
    assert track(run, uri)["status"] == "already_imported"
    assert len(client.search_runs([experiment])) == 1
    events = [json.loads(line) for line in (run / "tracking-events.jsonl").read_text().splitlines()]
    assert [event["status"] for event in events] == [
        "uploading",
        "failed",
        "uploading",
        "resumed",
        "already_imported",
    ]
