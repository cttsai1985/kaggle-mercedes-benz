import json
import shutil

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from test_contracts import fixture_data

from mercedes.pipeline import DEFAULT, features, train
from mercedes.serving import export_model


def test_export_survives_producer_removal_and_preserves_unseen_category_predictions(tmp_path):
    data = fixture_data(tmp_path / "data")
    run = tmp_path / "run"
    manifest = train(
        data,
        run,
        {**DEFAULT, "folds": 3, "trees": 5, "threads": 1, "model_seeds": [170904]},
        fixture=True,
    )
    example = pd.read_csv(data / "test.csv").drop(columns="ID")
    schema = json.loads((run / "schema.json").read_text())
    expected = np.mean(
        [
            joblib.load(run / r["file"]).predict(features(example, schema))
            for r in manifest["model_records"]
        ],
        axis=0,
    )
    output = tmp_path / "package"
    export_model(run, output, example)
    shutil.rmtree(run)
    loaded = mlflow.pyfunc.load_model(str(output))
    actual = loaded.predict(example)
    np.testing.assert_allclose(actual.y, expected, atol=1e-10, rtol=0)
    assert not any(p.name in {"oof.csv", "submission.csv", "folds.csv"} for p in output.rglob("*"))
    with pytest.raises(ValueError, match="already exists"):
        export_model(run, output, example)
    with pytest.raises(Exception, match="missing|Missing"):
        loaded.predict(example.drop(columns="Xcat"))
    artifact = next((output / "artifacts").rglob("schema.json"))
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        mlflow.pyfunc.load_model(str(output))
