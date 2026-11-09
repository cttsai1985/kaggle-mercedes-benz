import json

import joblib
import numpy as np
import pandas as pd
import pytest

from mercedes.pipeline import (
    DEFAULT,
    evaluate,
    features,
    make_pipeline,
    predict,
    read_data,
    resolve_config,
    schema_for,
    source_identity,
    train,
    verify_run,
)


def fixture_data(path):
    path.mkdir()
    frame = pd.DataFrame(
        {
            "ID": np.arange(30),
            "Xcat": ["a", "b", "c"] * 10,
            "Xnum": np.arange(30) / 10,
            "Xmissing": [np.nan] * 30,
            "y": np.linspace(10, 80, 30),
        }
    )
    frame.loc[29, "y"] = 265.32  # Extreme targets must remain eligible.
    test = pd.DataFrame(
        {
            "ID": [101, 102],
            "Xcat": ["new", "a"],
            "Xnum": [2.0, np.nan],
            "Xmissing": [np.nan, np.nan],
        }
    )
    frame.to_csv(path / "train.csv", index=False)
    test.to_csv(path / "test.csv", index=False)
    pd.DataFrame({"ID": test.ID, "y": 0.0}).to_csv(path / "sample_submission.csv", index=False)
    return path


@pytest.fixture
def completed(tmp_path):
    data = fixture_data(tmp_path / "data")
    run = tmp_path / "run"
    config = {**DEFAULT, "folds": 3, "trees": 5, "threads": 1, "model_seeds": [170904]}
    manifest = train(data, run, config, fixture=True)
    return data, run, manifest


def test_unknown_holdout_category_and_heldout_changes_do_not_fit_transform():
    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Xcat": ["a", "b", "a", "exclusive"],
            "Xnum": [1.0, 2, 3, 1000],
            "y": [1.0, 2, 3, 1e10],
        }
    )
    schema = schema_for(frame)
    config = resolve_config({"threads": 1})
    x = features(frame, schema)
    first = make_pipeline(schema, config, "ridge", 170904).fit(x.iloc[:3], frame.y.iloc[:3])
    mutated = frame.copy()
    mutated.loc[3, ["Xcat", "Xnum", "y"]] = ["different", -1e8, -1e15]
    second = make_pipeline(schema, config, "ridge", 170904).fit(
        features(mutated, schema).iloc[:3], mutated.y.iloc[:3]
    )
    encoder = first.named_steps["preprocess"].named_transformers_["categorical"]
    assert list(encoder.categories_[0]) == ["a", "b"]
    assert np.isfinite(first.predict(x.iloc[3:])).all()
    np.testing.assert_array_equal(first.predict(x), second.predict(x))
    assert "ID" not in first.feature_names_in_


def test_complete_oof_and_all_models_reload_with_fixed_equal_mean(completed, tmp_path):
    data, run, manifest = completed
    assert manifest["run_kind"] == "synthetic_fixture"
    assert len(manifest["model_records"]) == 9
    retained = pd.read_csv(run / "oof.csv")
    assert retained.ID.tolist() == list(range(30))
    assert retained.target.max() == 265.32
    prediction_columns = [name for name in retained if ".seed-" in name]
    np.testing.assert_allclose(retained.prediction, retained[prediction_columns].mean(axis=1))
    receipt = evaluate(run, data)
    assert receipt["models_reloaded"] == 9
    assert receipt["maximum_prediction_delta"] < 1e-10
    output = tmp_path / "predictions.csv"
    predicted = predict(run, data / "test.csv", output)
    assert predicted["rows"] == 2
    assert pd.read_csv(output).ID.tolist() == [101, 102]
    np.testing.assert_allclose(pd.read_csv(output).y, pd.read_csv(run / "submission.csv").y)
    with pytest.raises(FileExistsError):
        train(data, run, manifest["config"])


def test_schema_order_identity_and_tampering_fail_closed(completed, tmp_path):
    data, run, _ = completed
    test = pd.read_csv(data / "test.csv")
    test[["ID", "Xnum", "Xcat", "Xmissing"]].to_csv(tmp_path / "reordered.csv", index=False)
    with pytest.raises(ValueError, match="schema/order"):
        predict(run, tmp_path / "reordered.csv", tmp_path / "no.csv")
    test.loc[0, "ID"] = 1
    test.to_csv(tmp_path / "overlap.csv", index=False)
    with pytest.raises(ValueError, match="overlap"):
        predict(run, tmp_path / "overlap.csv", tmp_path / "no.csv")
    with (run / "oof.csv").open("a") as stream:
        stream.write("corrupt\n")
    with pytest.raises(ValueError, match="checksum"):
        verify_run(run)


def test_raw_data_requires_exact_test_id_and_column_contract(tmp_path):
    data = fixture_data(tmp_path / "data")
    sample = pd.read_csv(data / "sample_submission.csv").iloc[::-1]
    sample.to_csv(data / "sample_submission.csv", index=False)
    with pytest.raises(ValueError, match="Sample submission"):
        read_data(data)


def test_optional_projection_is_fitted_within_saved_fold(tmp_path):
    data = fixture_data(tmp_path / "data")
    frame, _, _, schema = read_data(data)
    config = resolve_config({"projection_components": 1, "threads": 1})
    model = make_pipeline(schema, config, "ridge", 170904)
    model.fit(features(frame.iloc[:20], schema), frame.y.iloc[:20])
    assert model.named_steps["projection"].projection_.components_.shape[0] == 1
    checkpoint = tmp_path / "model.joblib"
    joblib.dump(model, checkpoint)
    np.testing.assert_array_equal(
        model.predict(features(frame.iloc[20:], schema)),
        joblib.load(checkpoint).predict(features(frame.iloc[20:], schema)),
    )


def test_config_rejects_unbounded_or_ambiguous_requests():
    for value in (
        {"threads": 8},
        {"model_seeds": [1, 1]},
        {"models": ["xgboost"]},
        {"trees": True},
        {"projection_components": -1},
        {"unknown": 1},
    ):
        with pytest.raises(ValueError):
            resolve_config(value)


def test_installed_source_identity_does_not_require_git(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        subprocess, "check_output", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError())
    )
    identity = source_identity()
    assert identity["git_commit"] is None
    assert len(identity["python_source_sha256"]) == 64


def test_evaluation_checks_target_and_protocol_relationships_even_with_valid_hashes(completed):
    from mercedes.pipeline import sha256, write_json

    data, run, manifest = completed
    oof = pd.read_csv(run / "oof.csv")
    oof.loc[0, "target"] += 10
    oof.to_csv(run / "oof.csv", index=False)
    manifest["artifacts"]["oof.csv"] = sha256(run / "oof.csv")
    write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="OOF targets"):
        evaluate(run, data)


def test_failed_fit_keeps_failure_evidence_and_cannot_be_verified(tmp_path, monkeypatch):
    from mercedes import pipeline

    data = fixture_data(tmp_path / "data")
    run = tmp_path / "run"

    def fail(*args, **kwargs):
        raise RuntimeError("controlled model fit failure")

    monkeypatch.setattr(pipeline, "make_pipeline", fail)
    with pytest.raises(RuntimeError, match="controlled"):
        train(data, run, {**DEFAULT, "folds": 2, "threads": 1}, fixture=True)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "RuntimeError"
    with pytest.raises(ValueError, match="completed"):
        verify_run(run)
