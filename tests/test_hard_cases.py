"""Known residual arithmetic and identity boundaries for retained OOF diagnostics."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "mercedes_hard_cases", Path(__file__).resolve().parents[1] / "scripts/analyze_hard_cases.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_known_residual_concentration_direction_and_seed_overlap():
    result = analysis.summarize(
        [1, 2, 3, 4], [1, 1, 3, 2], [0, 0, 1, 1], {1: [1, 1, 3, 2], 2: [1, 1.1, 3, 2.1]}
    )
    assert result["pooled_oof"]["r2"] == 0
    assert result["pooled_oof"]["rmse"] == pytest.approx(math.sqrt(5 / 4))
    assert result["pooled_oof"]["mae"] == 0.75
    assert result["hard_cohort"]["rows"] == 1
    assert result["hard_cohort"]["share_of_total_squared_error"] == 0.8
    assert result["hard_cohort"]["underpredicted_rows"] == 1
    assert result["seed_error_overlap"]["jaccard"] == 1
    assert np.array_equal(analysis.hardest(np.ones(4)), [True, False, False, False])


def test_population_and_declared_mean_must_match():
    train = pd.DataFrame({"ID": ["a", "b"], "y": [1.0, 2.0]})
    assignments = pd.DataFrame({"ID": ["a", "b"], "fold": [0, 1]})
    saved = pd.DataFrame(
        {
            "ID": ["a", "b"],
            "fold": [0, 1],
            "target": [1.0, 2.0],
            "ridge.seed-10": [0.5, 1.5],
            "ridge.seed-20": [0.7, 1.7],
            "prediction": [0.6, 1.6],
        }
    )
    config = {"folds": 2, "models": ["ridge"], "model_seeds": [10, 20]}
    analysis.validate_population(train, saved, assignments, config)
    changed = saved.copy()
    changed.loc[0, "target"] = 9.0
    with pytest.raises(ValueError, match="targets differ"):
        analysis.validate_population(train, changed, assignments, config)
    changed = saved.copy()
    changed.loc[0, "ridge.seed-10"] = 9.0
    with pytest.raises(ValueError, match="equal mean"):
        analysis.validate_population(train, changed, assignments, config)
    with pytest.raises(ValueError, match="ID order differs"):
        analysis.validate_population(train, saved.iloc[::-1], assignments, config)


def test_invalid_metrics_and_artifact_tampering_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="nonconstant"):
        analysis.summarize([1, 1], [1, 1], [0, 1], {1: [1, 1], 2: [1, 1]})
    with pytest.raises(ValueError, match="Nonfinite"):
        analysis.summarize([1, 2], [1, float("nan")], [0, 1], {1: [1, 1], 2: [1, 1]})
    path = tmp_path / "oof.csv"
    path.write_text("original")
    expected = analysis.digest(path)
    path.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        analysis.verified_file(tmp_path, "oof.csv", expected)
    with pytest.raises(ValueError, match="basename"):
        analysis.verified_file(tmp_path, "../escape", expected)


def test_observed_feature_repetition_is_separate_from_error_cohort():
    frame = pd.DataFrame({"ID": ["a", "b", "c", "d"], "X": [1, 1, 2, 3], "y": [1, 3, 5, 6]})
    result = analysis.configuration_repetition(frame, np.array([True, False, False, False]))
    assert result["rows_with_repeated_features"] == 2
    assert result["hard_rows_with_repeated_features"] == 1
    assert result["maximum_observed_target_range_for_identical_features"] == 2
