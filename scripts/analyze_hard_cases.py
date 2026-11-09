"""Describe retained OOF hard cases; never refit, filter evaluation, or select a model."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

COHORT_FRACTION = 0.05


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verified_file(root, name, expected):
    require(Path(name).name == name, "Artifact name must be a basename")
    path = root / name
    require(path.is_file() and not path.is_symlink(), "Expected a regular artifact")
    require(digest(path) == expected, f"Artifact checksum differs: {name}")
    return path


def hardest(error):
    count = max(1, math.ceil(COHORT_FRACTION * len(error)))
    mask = np.zeros(len(error), dtype=bool)
    mask[np.argsort(-error, kind="stable")[:count]] = True
    return mask


def summarize(target, prediction, folds, seeds):
    target, prediction = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    folds = np.asarray(folds)
    require(
        target.ndim == 1 and target.shape == prediction.shape == folds.shape and len(target),
        "Invalid vectors",
    )
    require(np.isfinite(target).all() and np.isfinite(prediction).all(), "Nonfinite values")
    require(np.isfinite(folds).all() and np.equal(folds, np.floor(folds)).all(), "Invalid folds")
    residual = target - prediction  # Positive means duration was underestimated.
    squared = residual**2
    sse = float(squared.sum())
    sst = float(np.sum((target - target.mean()) ** 2))
    require(sst > 0, "R2 requires nonconstant targets")
    hard = hardest(squared)
    tail_threshold = float(np.quantile(target, 0.9))
    tail = target >= tail_threshold

    def cohort(mask):
        count = int(mask.sum())
        return {
            "rows": count,
            "share_of_total_squared_error": float(squared[mask].sum() / sse) if sse else 0.0,
            "mean_absolute_error": float(np.abs(residual[mask]).mean()) if count else None,
            "mean_signed_residual": float(residual[mask].mean()) if count else None,
            "underpredicted_rows": int((mask & (residual > 0)).sum()),
            "overpredicted_rows": int((mask & (residual < 0)).sum()),
            "hard_rows": int((mask & hard).sum()),
        }

    seed_masks, seed_predictions = {}, []
    for seed, values in seeds.items():
        values = np.asarray(values, dtype=float)
        require(
            values.shape == target.shape and np.isfinite(values).all(), "Invalid seed prediction"
        )
        seed_masks[str(seed)] = hardest((target - values) ** 2)
        seed_predictions.append(values)
    require(len(seed_masks) == 2, "This diagnostic requires exactly two declared seeds")
    left, right = seed_masks.values()
    disagreement = np.abs(seed_predictions[0] - seed_predictions[1])
    return {
        "rows": len(target),
        "pooled_oof": {
            "r2": 1 - sse / sst,
            "rmse": float(np.sqrt(sse / len(target))),
            "mae": float(np.abs(residual).mean()),
            "sse": sse,
            "sst": sst,
        },
        "cohort_rule": {
            "loss": "squared residual",
            "fraction": COHORT_FRACTION,
            "count_rule": "ceiling",
            "tie_break": "retained OOF order",
        },
        "hard_cohort": {
            **cohort(hard),
            "minimum_selected_absolute_error": float(np.sqrt(squared[hard].min())),
            "mean_target": float(target[hard].mean()),
            "mean_prediction": float(prediction[hard].mean()),
            "target_upper_decile_rows": int((hard & tail).sum()),
        },
        "target_upper_decile": {"threshold_inclusive": tail_threshold, **cohort(tail)},
        "folds": [{"fold": int(fold), **cohort(folds == fold)} for fold in np.unique(folds)],
        "seed_error_overlap": {
            "seed_ids": list(seed_masks),
            "hard_rows_per_seed": int(left.sum()),
            "intersection_rows": int((left & right).sum()),
            "union_rows": int((left | right).sum()),
            "jaccard": float((left & right).sum() / (left | right).sum()),
            "ensemble_hard_rows_in_both_seed_cohorts": int((hard & left & right).sum()),
            "mean_absolute_prediction_difference_all_rows": float(disagreement.mean()),
            "mean_absolute_prediction_difference_hard_rows": float(disagreement[hard].mean()),
        },
    }


def validate_population(train, saved, assignments, config):
    for frame in [train, saved, assignments]:
        require(frame.ID.notna().all() and frame.ID.is_unique, "IDs must be present and unique")
    require(
        train.ID.equals(saved.ID) and train.ID.equals(assignments.ID),
        "Exact OOF/source ID order differs",
    )
    require(
        np.array_equal(train.y.to_numpy(), saved.target.to_numpy()),
        "OOF targets differ from raw input",
    )
    require(
        np.array_equal(saved.fold.to_numpy(), assignments.fold.to_numpy()),
        "Saved fold assignment differs",
    )
    require(set(saved.fold) == set(range(config["folds"])), "Declared folds differ")
    seeds, models = config["model_seeds"], config["models"]
    require(len(seeds) == len(set(seeds)) == 2, "Require two unique model seeds")
    require(len(models) == len(set(models)) and len(models), "Invalid model families")
    columns = [f"{model}.seed-{seed}" for seed in seeds for model in models]
    require(
        list(saved) == ["ID", "fold", "target", *columns, "prediction"], "OOF member schema differs"
    )
    members = saved[columns].to_numpy(dtype=float)
    require(np.isfinite(members).all(), "Nonfinite member predictions")
    require(
        np.allclose(members.mean(axis=1), saved.prediction, atol=1e-10, rtol=0),
        "Ensemble differs from declared equal mean",
    )
    return {
        seed: saved[[f"{model}.seed-{seed}" for model in models]].to_numpy(dtype=float).mean(axis=1)
        for seed in seeds
    }


def configuration_repetition(train, hard):
    """Describe identical observed feature vectors, not asserted vehicle identities."""
    columns = [name for name in train if name not in {"ID", "y"}]
    require(bool(columns) and len(hard) == len(train), "Configuration population differs")
    grouped = train.groupby(columns, dropna=False, sort=False).y
    repeated = grouped.transform("size").to_numpy() > 1
    spread = (grouped.transform("max") - grouped.transform("min")).to_numpy()
    return {
        "rows_with_repeated_features": int(repeated.sum()),
        "population_fraction": float(repeated.mean()),
        "hard_rows_with_repeated_features": int((hard & repeated).sum()),
        "hard_cohort_fraction": float(repeated[hard].mean()),
        "repeated_rows_with_different_observed_targets": int((repeated & (spread > 0)).sum()),
        "maximum_observed_target_range_for_identical_features": float(spread.max()),
        "interpretation": (
            "Exact observed configuration equality does not establish vehicle identity or leakage."
        ),
    }


def analyze(run_dir, data_dir):
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("project_id") == "mercedes-benz" and manifest.get("status") == "completed",
        "Expected completed Mercedes-Benz run",
    )
    require(manifest.get("run_kind") == "real_data_model_run", "Expected real-data run")
    for name, expected in manifest["artifacts"].items():
        verified_file(run_dir, name, expected)
    for name, expected in manifest["input_sha256"].items():
        verified_file(data_dir, name, expected)
    config = manifest["config"]
    require(json.loads((run_dir / "config.json").read_text()) == config, "Configuration differs")
    require(digest(run_dir / "folds.csv") == manifest["split_sha256"], "Split checksum differs")
    train = pd.read_csv(data_dir / "train.csv", dtype={"ID": str})
    saved = pd.read_csv(run_dir / "oof.csv", dtype={"ID": str})
    assignments = pd.read_csv(run_dir / "folds.csv", dtype={"ID": str})
    require(len(train) == manifest["train_rows"], "Training population differs")
    seeds = validate_population(train, saved, assignments, config)
    result = summarize(
        train.y.to_numpy(), saved.prediction.to_numpy(), saved.fold.to_numpy(), seeds
    )
    result["configuration_repetition"] = configuration_repetition(
        train, hardest((train.y.to_numpy() - saved.prediction.to_numpy()) ** 2)
    )
    retained = json.loads((run_dir / "metrics.json").read_text())["pooled_oof"]
    require(
        all(
            np.isclose(result["pooled_oof"][name], retained[name], rtol=0, atol=1e-12)
            for name in ["r2", "rmse", "mae"]
        ),
        "Pooled metric reconciliation failed",
    )
    return {
        "schema_version": 1,
        "project_id": "mercedes-benz",
        "analysis_kind": "retained_oof_hard_cases",
        "producer_manifest_sha256": digest(manifest_path),
        "producer_source": manifest["source"],
        "input_sha256": manifest["input_sha256"],
        "split_sha256": manifest["split_sha256"],
        "oof_sha256": manifest["artifacts"]["oof.csv"],
        "analyzer_sha256": digest(Path(__file__)),
        "prediction_reconciliation": "equal family/seed mean verified",
        "results": result,
        "limitations": [
            "Error-selected diagnostic cohort, not a new test or an improvement.",
            "Two seeds share one fold plan; overlap does not establish generalization.",
            "Target tail membership is retrospective, not known from predictions alone.",
            "No outlier removal, relabelling, retraining, weight search or promotion performed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(analyze(args.run_dir, args.data_dir), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
