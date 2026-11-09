"""Reconcile saved OOF evidence without fitting models or selecting a new recipe."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(target, prediction):
    """Direct arithmetic, independent of the producer's sklearn metric helpers."""
    require(np.isfinite(target).all() and np.isfinite(prediction).all(), "Nonfinite values")
    residual = target - prediction
    sse = float(np.dot(residual, residual))
    centered = target - target.mean()
    sst = float(np.dot(centered, centered))
    require(sst > 0, "R2 needs a nonconstant target")
    return {
        "rows": len(target),
        "r2": 1.0 - sse / sst,
        "rmse": float(np.sqrt(sse / len(target))),
        "mae": float(np.abs(residual).mean()),
        "sse": sse,
        "sst": sst,
    }


def summarize(target, prediction, assignments):
    return {
        "pooled": metrics(target, prediction),
        "folds": [
            {"fold": int(fold), **metrics(target[mask], prediction[mask])}
            for fold in np.unique(assignments)
            for mask in [assignments == fold]
        ],
    }


class Reconciliation:
    def __init__(self):
        self.count = 0
        self.maximum_absolute_error = 0.0

    def compare(self, actual, retained, label):
        actual, retained = np.asarray(actual), np.asarray(retained)
        require(actual.shape == retained.shape, f"Shape mismatch: {label}")
        require(np.isfinite(actual).all() and np.isfinite(retained).all(), f"Nonfinite {label}")
        error = float(np.max(np.abs(actual - retained)))
        self.maximum_absolute_error = max(self.maximum_absolute_error, error)
        self.count += actual.size
        require(np.allclose(actual, retained, rtol=1e-12, atol=1e-10), f"Mismatch: {label}")

    def score(self, actual, retained, label):
        for key in ("r2", "rmse", "mae"):
            self.compare(actual[key], retained[key], f"{label}.{key}")


def analyze_run(directory, data_dir, checks):
    manifest = read_json(directory / "manifest.json")
    require(manifest["status"] == "completed", "Run is not completed")
    require(manifest["run_kind"] == "real_data_model_run", "Expected real-data run")
    require(manifest["project_id"] == "mercedes-benz", "Wrong project")
    config = manifest["config"]
    expected_records = {
        (seed, model, fold)
        for seed in config["model_seeds"]
        for model in config["models"]
        for fold in range(config["folds"])
    }
    records = manifest["model_records"]
    require(
        {(r["seed"], r["model"], r["fold"]) for r in records} == expected_records
        and len(records) == len(expected_records),
        "Incomplete or duplicate model identities",
    )
    expected_artifacts = {
        "config.json",
        "schema.json",
        "folds.csv",
        "metrics.json",
        "oof.csv",
        "submission.csv",
    } | {f"model-{seed}-{model}-fold-{fold:02d}.joblib" for seed, model, fold in expected_records}
    require(set(manifest["artifacts"]) == expected_artifacts, "Unexpected artifact set")
    for name, expected in manifest["artifacts"].items():
        path = directory / name
        require(not path.is_symlink() and digest(path) == expected, f"Artifact changed: {name}")
    require(read_json(directory / "config.json") == config, "Config differs from manifest")
    require(digest(directory / "folds.csv") == manifest["split_sha256"], "Split hash differs")
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        require(digest(data_dir / name) == manifest["input_sha256"][name], f"Input changed: {name}")

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    saved = pd.read_csv(directory / "oof.csv")
    folds = pd.read_csv(directory / "folds.csv")
    submission = pd.read_csv(directory / "submission.csv")
    require(len(train) == manifest["train_rows"], "Train row count differs")
    require(len(test) == manifest["test_rows"], "Test row count differs")
    require(train.ID.is_unique and test.ID.is_unique, "Duplicate source IDs")
    require(not set(train.ID) & set(test.ID), "Train/test IDs overlap")
    for frame in (saved, folds):
        require(train.ID.equals(frame.ID), "OOF/fold IDs are not in exact source order")
    require(list(submission) == ["ID", "y"], "Unexpected submission schema")
    require(test.ID.equals(submission.ID) and test.ID.equals(sample.ID), "Test ID order differs")
    require(np.isfinite(submission.y.to_numpy()).all(), "Nonfinite test predictions")
    schema = read_json(directory / "schema.json")
    require(
        schema["columns"] == [c for c in train if c not in {"ID", "y"}], "Feature order differs"
    )
    target = train.y.to_numpy(dtype=float)
    require(np.array_equal(target, saved.target.to_numpy()), "OOF targets differ from source")
    assignments = folds.fold.to_numpy()
    require(np.array_equal(assignments, saved.fold.to_numpy()), "OOF folds differ")
    # Reconstruct the shuffled KFold assignment without invoking producer split code.
    permutation = np.arange(len(train))
    np.random.RandomState(config["split_seed"]).shuffle(permutation)
    expected_assignments = np.full(len(train), -1)
    for fold, indices in enumerate(np.array_split(permutation, config["folds"])):
        expected_assignments[indices] = fold
    require(
        np.array_equal(assignments, expected_assignments), "Declared split cannot be reproduced"
    )

    columns = [
        f"{model}.seed-{seed}" for seed in config["model_seeds"] for model in config["models"]
    ]
    require(
        list(saved) == ["ID", "fold", "target", *columns, "prediction"], "Unexpected OOF schema"
    )
    predictions = {"equal_mean": saved[columns].to_numpy().mean(axis=1)}
    checks.compare(predictions["equal_mean"], saved.prediction.to_numpy(), "equal blend")
    for model in config["models"]:
        selected = [f"{model}.seed-{seed}" for seed in config["model_seeds"]]
        predictions[f"family:{model}"] = saved[selected].to_numpy().mean(axis=1)
    for seed in config["model_seeds"]:
        selected = [f"{model}.seed-{seed}" for model in config["models"]]
        predictions[f"seed:{seed}"] = saved[selected].to_numpy().mean(axis=1)
    for column in columns:
        predictions[f"member:{column}"] = saved[column].to_numpy()
    results = {name: summarize(target, pred, assignments) for name, pred in predictions.items()}
    retained = read_json(directory / "metrics.json")
    checks.score(results["equal_mean"]["pooled"], retained["pooled_oof"], "pooled")
    for item in retained["model_results"]:
        checks.score(results[f"family:{item['model']}"]["pooled"], item, "family")
    for item in retained["seed_results"]:
        checks.score(results[f"seed:{item['seed']}"]["pooled"], item, "seed")
    for record in records:
        name = f"member:{record['model']}.seed-{record['seed']}"
        item = results[name]["folds"][record["fold"]]
        require(item["rows"] == record["validation_rows"], "Validation row count differs")
        require(len(target) - item["rows"] == record["train_rows"], "Fit row count differs")
        checks.score(item, record, "saved fold record")
    for item in retained["analysis"]["folds"]:
        computed = results["equal_mean"]["folds"][item["fold"]]
        checks.score(computed, item, "fold diagnostic")
        checks.compare(computed["sse"], item["sum_squared_error"], "fold SSE")
        checks.compare(computed["sst"], item["target_sum_squared_deviation"], "fold SST")

    squared = np.square(target - predictions["equal_mean"])
    total = float(squared.sum())
    max_target = target == target.max()
    max_target_folds = sorted(int(v) for v in np.unique(assignments[max_target]))
    top_target_indices = np.argsort(target)[-int(np.ceil(len(target) * 0.01)) :]
    diagnostics = {
        "total_sse": total,
        "largest_residual_sse_share": float(squared.max() / total),
        "top_10_residual_sse_share": float(np.sort(squared)[-10:].sum() / total),
        "maximum_target": float(target.max()),
        "maximum_target_rows": int(max_target.sum()),
        "maximum_target_folds": max_target_folds,
        "maximum_target_sse_share": float(squared[max_target].sum() / total),
        "maximum_target_within_fold_sse_share": {
            str(fold): float(
                squared[max_target & (assignments == fold)].sum()
                / squared[assignments == fold].sum()
            )
            for fold in max_target_folds
        },
        "top_ceil_one_percent_target_rows": len(top_target_indices),
        "top_target_cohort_sse_share": float(squared[top_target_indices].sum() / total),
        "mean_fold_r2": float(np.mean([f["r2"] for f in results["equal_mean"]["folds"]])),
        "within_fold_sst_sum": float(sum(f["sst"] for f in results["equal_mean"]["folds"])),
        "global_sst": results["equal_mean"]["pooled"]["sst"],
        "between_fold_sst": float(
            sum(
                (assignments == fold).sum()
                * (target[assignments == fold].mean() - target.mean()) ** 2
                for fold in np.unique(assignments)
            )
        ),
        "ridge_seed_max_prediction_delta": float(
            np.max(
                np.abs(
                    saved[f"ridge.seed-{config['model_seeds'][0]}"]
                    - saved[f"ridge.seed-{config['model_seeds'][1]}"]
                )
            )
        ),
    }
    checks.compare(
        diagnostics["within_fold_sst_sum"] + diagnostics["between_fold_sst"],
        diagnostics["global_sst"],
        "SST decomposition",
    )
    checks.compare(total, retained["analysis"]["total_squared_error"], "total SSE")
    checks.compare(
        diagnostics["largest_residual_sse_share"],
        retained["analysis"]["largest_error_share"],
        "largest error",
    )
    checks.compare(
        diagnostics["top_10_residual_sse_share"],
        retained["analysis"]["top_10_error_share"],
        "top ten errors",
    )
    return manifest, {
        "manifest_sha256": digest(directory / "manifest.json"),
        "oof_sha256": manifest["artifacts"]["oof.csv"],
        "submission_sha256": manifest["artifacts"]["submission.csv"],
        "source": manifest["source"],
        "config": config,
        "environment": manifest["environment"],
        "elapsed_seconds": manifest["elapsed_seconds"],
        "saved_models": len(records),
        "results": results,
        "diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "projection", "data-dir", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    checks = Reconciliation()
    before, baseline = analyze_run(args.baseline, args.data_dir, checks)
    after, projection = analyze_run(args.projection, args.data_dir, checks)
    for key in ("input_sha256", "split_sha256", "source", "environment", "train_rows", "test_rows"):
        require(before[key] == after[key], f"Unmatched {key}")
    require(before["source"]["git_dirty"] is False, "Producer source was not clean")
    differences = [key for key in before["config"] if before["config"][key] != after["config"][key]]
    require(set(differences) == {"name", "projection_components"}, "Uncontrolled config difference")
    require(
        before["config"]["projection_components"] == 0
        and after["config"]["projection_components"] == 16,
        "Expected 0 versus 16 SVD coordinates",
    )
    deltas = {}
    for name, result in baseline["results"].items():
        other = projection["results"][name]
        deltas[name] = {
            "pooled": {
                metric: other["pooled"][metric] - result["pooled"][metric]
                for metric in ("r2", "rmse", "mae", "sse")
            },
            "folds": [
                {
                    "fold": x["fold"],
                    **{metric: y[metric] - x[metric] for metric in ("r2", "rmse", "mae", "sse")},
                }
                for x, y in zip(result["folds"], other["folds"], strict=True)
            ],
        }
    report = {
        "schema_version": 1,
        "project_id": "mercedes-benz",
        "study": "Fixed-recipe appended-SVD comparison; no new model fitting or selection",
        "input_sha256": before["input_sha256"],
        "split_sha256": before["split_sha256"],
        "config_differences": differences,
        "analysis_script_sha256": digest(__file__),
        "runs": {"baseline": baseline, "projection": projection},
        "projection_minus_baseline": deltas,
        "verification": {
            "status": "passed",
            "method": "Direct NumPy SSE/SST arithmetic from private OOF CSVs and raw labels",
            "numeric_reconciliations": int(checks.count),
            "maximum_absolute_error": checks.maximum_absolute_error,
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-12,
            "scope": "Hashes, exact IDs/targets/folds, equal mean and retained metrics; "
            "saved-model reconstruction is separately performed by mercedes evaluate",
        },
        "interpretation": {
            "projection": "Appended SVD retains all inputs; no new information or compression",
            "seeds": "Same folds; projection recipe changes SVD and tree seeds together. "
            "Ridge is deterministic conditional on representation, not across refitted SVDs.",
            "diagnostics": "Post-fit descriptions do not identify bad labels or justify deletion",
            "selection": "Predeclared equal mean; no OOF weight search or winner promotion",
            "limits": "Two seeds, one split plan, CPU models; no leaderboard or production claim",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["verification"], indent=2))


if __name__ == "__main__":
    main()
