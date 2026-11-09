"""Fold-local fitting, immutable evidence and independent model reconstruction."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from mercedes import PROJECT_ID, __version__

FILES = ("train.csv", "test.csv", "sample_submission.csv")
DEFAULT = {
    "name": "fold-local-equal-mean",
    "folds": 5,
    "split_seed": 904,
    "model_seeds": [170904, 170905],
    "models": ["ridge", "random_forest", "extra_trees"],
    "trees": 160,
    "max_depth": 12,
    "min_samples_leaf": 4,
    "ridge_alpha": 10.0,
    "projection_components": 0,
    "threads": 4,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity():
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    source = {"python_source_sha256": digest.hexdigest(), "package_version": __version__}
    try:
        root = package.parents[1]
        if not (root / ".git").exists() or package != root / "src" / "mercedes":
            raise FileNotFoundError("Installed package has no source-checkout identity")
        source["git_commit"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        source["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        source.update(git_commit=None, git_dirty=None)
    return source


def resolve_config(value):
    if set(value) - DEFAULT.keys():
        raise ValueError("Unknown configuration key")
    config = {**DEFAULT, **value}
    for key, lower, upper in [
        ("folds", 2, 20),
        ("trees", 1, 2000),
        ("max_depth", 1, 100),
        ("min_samples_leaf", 1, 1000),
        ("projection_components", 0, 100),
        ("threads", 1, 4),
    ]:
        if type(config[key]) is not int or not lower <= config[key] <= upper:
            raise ValueError(f"Invalid {key}")
    if type(config["split_seed"]) is not int or config["split_seed"] < 0:
        raise ValueError("Invalid split seed")
    seeds, models = config["model_seeds"], config["models"]
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("Use unique nonnegative model seeds")
    if (
        not models
        or len(models) != len(set(models))
        or not set(models) <= {"ridge", "random_forest", "extra_trees"}
    ):
        raise ValueError("Unsupported or duplicate models")
    if not np.isfinite(config["ridge_alpha"]) or config["ridge_alpha"] <= 0:
        raise ValueError("ridge_alpha must be finite and positive")
    return config


def schema_for(frame):
    columns = [name for name in frame.columns if name not in {"ID", "y"}]
    if not columns:
        raise ValueError("No model features")
    categorical = [name for name in columns if not pd.api.types.is_numeric_dtype(frame[name])]
    return {
        "columns": columns,
        "categorical": categorical,
        "numeric": [name for name in columns if name not in categorical],
    }


def features(frame, schema):
    actual = [name for name in frame.columns if name not in {"ID", "y"}]
    if actual != schema["columns"]:
        raise ValueError("Feature schema/order differs from the fitted model")
    result = frame[schema["columns"]].copy()
    for name in schema["categorical"]:
        result[name] = result[name].astype("string").fillna("__missing__").astype(str)
    for name in schema["numeric"]:
        result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
        if np.isinf(result[name].to_numpy()).any():
            raise ValueError("Numeric features contain infinity")
    return result


def read_data(data_dir):
    data_dir = Path(data_dir)
    train, test, sample = [pd.read_csv(data_dir / name) for name in FILES]
    for frame in (train, test, sample):
        if "ID" not in frame or frame.ID.isna().any() or not frame.ID.is_unique:
            raise ValueError("IDs must be present, nonmissing and unique")
    if "y" not in train or "y" in test or not np.isfinite(train.y.to_numpy()).all():
        raise ValueError("Training target must be finite and absent from test")
    if set(train.ID.astype(str)) & set(test.ID.astype(str)):
        raise ValueError("Train/test IDs overlap")
    if list(sample.columns) != ["ID", "y"] or not sample.ID.astype(str).equals(test.ID.astype(str)):
        raise ValueError("Sample submission IDs/order/schema differ from test")
    schema = schema_for(train)
    features(train, schema)
    features(test, schema)
    return train, test, sample, schema


class AppendedSVD(TransformerMixin, BaseEstimator):
    """Fit a projection on the current training fold, retaining original features."""

    def __init__(self, n_components=0, random_state=0):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, x, y=None):
        if self.n_components:
            if self.n_components >= min(x.shape):
                raise ValueError("Projection budget must be smaller than training dimensions")
            self.projection_ = TruncatedSVD(
                n_components=self.n_components, random_state=self.random_state
            ).fit(x)
        return self

    def transform(self, x):
        return np.column_stack([x, self.projection_.transform(x)]) if self.n_components else x


def make_pipeline(schema, config, model_name, seed):
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                schema["categorical"],
            ),
            ("numeric", numeric, schema["numeric"]),
        ],
        remainder="drop",
    )
    if model_name == "ridge":
        regressor = Ridge(alpha=config["ridge_alpha"], solver="lsqr")
    else:
        cls = RandomForestRegressor if model_name == "random_forest" else ExtraTreesRegressor
        regressor = cls(
            n_estimators=config["trees"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=1.0,
            random_state=seed,
            n_jobs=config["threads"],
        )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("projection", AppendedSVD(config["projection_components"], seed)),
            ("regressor", regressor),
        ]
    )


def scores(actual, predicted):
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def error_analysis(target, prediction, assignments):
    squared = np.square(target - prediction)
    total = float(squared.sum())
    order = np.argsort(squared)[::-1]
    return {
        "rows": len(target),
        "total_squared_error": total,
        "largest_error_share": float(squared[order[0]] / total) if total else 0.0,
        "top_10_error_share": float(squared[order[:10]].sum() / total) if total else 0.0,
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
        "folds": [
            {
                "fold": int(fold),
                "rows": int((assignments == fold).sum()),
                "sum_squared_error": float(squared[assignments == fold].sum()),
                "target_sum_squared_deviation": float(
                    np.square(
                        target[assignments == fold] - target[assignments == fold].mean()
                    ).sum()
                ),
                "target_mean": float(target[assignments == fold].mean()),
                "target_max": float(target[assignments == fold].max()),
                **scores(target[assignments == fold], prediction[assignments == fold]),
            }
            for fold in np.unique(assignments)
        ],
        "interpretation": (
            "Error concentration and fold spread are descriptive; all rows remain in evaluation."
        ),
    }


def train(data_dir, run_dir, config, *, fixture=False):
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    config = resolve_config(config)
    frame, test, _, schema = read_data(data_dir)
    if len(frame) < config["folds"] * 2:
        raise ValueError("Too few rows for meaningful fold metrics")
    x, xt, y = features(frame, schema), features(test, schema), frame.y.to_numpy(dtype=float)
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": 1,
        "project_id": PROJECT_ID,
        "logical_run_id": str(uuid.uuid4()),
        "status": "running",
        "run_kind": "synthetic_fixture" if fixture else "real_data_model_run",
        "source": source_identity(),
        "config": config,
        "input_sha256": {name: sha256(data_dir / name) for name in FILES},
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "joblib")
            },
        },
        "train_rows": len(frame),
        "test_rows": len(test),
        "model_records": [],
    }
    write_json(run_dir / "manifest.json", manifest)
    try:
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "schema.json", schema)
        splits = list(
            KFold(config["folds"], shuffle=True, random_state=config["split_seed"]).split(x)
        )
        assignments = np.full(len(x), -1, dtype=int)
        for fold, (_, valid) in enumerate(splits):
            assignments[valid] = fold
        pd.DataFrame({"ID": frame.ID, "fold": assignments}).to_csv(
            run_dir / "folds.csv", index=False
        )
        manifest["split_sha256"] = sha256(run_dir / "folds.csv")
        oof = np.full((len(y), len(config["model_seeds"]), len(config["models"])), np.nan)
        predictions = np.zeros((len(test), len(config["model_seeds"]), len(config["models"])))
        with threadpool_limits(limits=config["threads"]):
            for si, seed in enumerate(config["model_seeds"]):
                for mi, name in enumerate(config["models"]):
                    for fold, (fit, valid) in enumerate(splits):
                        estimator = make_pipeline(schema, config, name, seed)
                        estimator.fit(x.iloc[fit], y[fit])
                        validation = estimator.predict(x.iloc[valid])
                        oof[valid, si, mi] = validation
                        predictions[:, si, mi] += estimator.predict(xt) / config["folds"]
                        filename = f"model-{seed}-{name}-fold-{fold:02d}.joblib"
                        joblib.dump(estimator, run_dir / filename)
                        record = {
                            "seed": seed,
                            "model": name,
                            "fold": fold,
                            "file": filename,
                            "train_rows": len(fit),
                            "validation_rows": len(valid),
                            **scores(y[valid], validation),
                        }
                        manifest["model_records"].append(record)
                        print(json.dumps(record), flush=True)
        if not np.isfinite(oof).all() or not np.isfinite(predictions).all():
            raise ValueError("Incomplete or nonfinite fold predictions")
        columns = {
            f"{name}.seed-{seed}": oof[:, si, mi]
            for si, seed in enumerate(config["model_seeds"])
            for mi, name in enumerate(config["models"])
        }
        ensemble = oof.mean(axis=(1, 2))
        pd.DataFrame(
            {"ID": frame.ID, "fold": assignments, "target": y, **columns, "prediction": ensemble}
        ).to_csv(run_dir / "oof.csv", index=False)
        pd.DataFrame({"ID": test.ID, "y": predictions.mean(axis=(1, 2))}).to_csv(
            run_dir / "submission.csv", index=False
        )
        seed_metrics = [
            {"seed": seed, **scores(y, oof[:, si].mean(axis=1))}
            for si, seed in enumerate(config["model_seeds"])
        ]
        metrics = {
            "pooled_oof": scores(y, ensemble),
            "seed_results": seed_metrics,
            "model_results": [
                {"model": name, **scores(y, oof[:, :, mi].mean(axis=1))}
                for mi, name in enumerate(config["models"])
            ],
            "analysis": error_analysis(y, ensemble, assignments),
            "ensemble": "predeclared equal mean across families and training seeds",
            "selection": "No OOF blend-weight search or target-row removal",
            "seed_interpretation": (
                "Two training seeds on identical folds give descriptive sensitivity, not a CI. "
                "Ridge is deterministic conditional on its fitted feature representation; "
                "duplicate Ridge predictions do not provide stochastic-replication evidence."
            ),
        }
        write_json(run_dir / "metrics.json", metrics)
        artifacts = [p for p in run_dir.iterdir() if p.name != "manifest.json"]
        manifest.update(
            status="completed",
            elapsed_seconds=time.monotonic() - started,
            artifacts={p.name: sha256(p) for p in sorted(artifacts)},
        )
        write_json(run_dir / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(run_dir / "manifest.json", manifest)
        raise


def verify_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("project_id") != PROJECT_ID or manifest.get("status") != "completed":
        raise ValueError("Expected a completed Mercedes-Benz run")
    config = resolve_config(manifest["config"])
    records = manifest["model_records"]
    if any(
        r["file"] != f"model-{r['seed']}-{r['model']}-fold-{r['fold']:02d}.joblib" for r in records
    ):
        raise ValueError("Fold model filename differs from its identity")
    expected = {
        (seed, model, fold)
        for seed in config["model_seeds"]
        for model in config["models"]
        for fold in range(config["folds"])
    }
    if {(r["seed"], r["model"], r["fold"]) for r in records} != expected or len(records) != len(
        expected
    ):
        raise ValueError("Incomplete or duplicate saved fold models")
    allowed = {
        "config.json",
        "schema.json",
        "folds.csv",
        "metrics.json",
        "oof.csv",
        "submission.csv",
    }
    allowed |= {r["file"] for r in records}
    if set(manifest["artifacts"]) != allowed:
        raise ValueError("Unexpected artifact set")
    for name, digest in manifest["artifacts"].items():
        path = run_dir / name
        if (
            Path(name).name != name
            or path.is_symlink()
            or not path.is_file()
            or sha256(path) != digest
        ):
            raise ValueError(f"Artifact checksum mismatch: {name}")
    if json.loads((run_dir / "config.json").read_text()) != config:
        raise ValueError("Saved config differs from manifest")
    if sha256(run_dir / "folds.csv") != manifest["split_sha256"]:
        raise ValueError("Fold assignment checksum differs")
    return manifest


def predict(run_dir, data, output):
    run_dir, data, output = Path(run_dir), Path(data), Path(output)
    manifest = verify_run(run_dir)
    frame = pd.read_csv(data)
    if "y" in frame or "ID" not in frame or frame.ID.isna().any() or not frame.ID.is_unique:
        raise ValueError("Predict expects unique test IDs and no target")
    training_ids = set(pd.read_csv(run_dir / "folds.csv").ID.astype(str))
    if training_ids & set(frame.ID.astype(str)):
        raise ValueError("Prediction IDs overlap training")
    schema = json.loads((run_dir / "schema.json").read_text())
    x = features(frame, schema)
    values = []
    with threadpool_limits(limits=manifest["config"]["threads"]):
        for record in manifest["model_records"]:
            values.append(joblib.load(run_dir / record["file"]).predict(x))
    prediction = np.mean(values, axis=0)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predictions")
    pd.DataFrame({"ID": frame.ID, "y": prediction}).to_csv(output, index=False)
    receipt = {
        "status": "model_predictions",
        "rows": len(frame),
        "input_sha256": sha256(data),
        "manifest_sha256": sha256(run_dir / "manifest.json"),
        "output_sha256": sha256(output),
        "model_count": len(values),
        "aggregation": "equal mean of all fold/family/seed predictions",
    }
    write_json(output.with_suffix(".receipt.json"), receipt)
    return receipt


def evaluate(run_dir, data_dir):
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    manifest = verify_run(run_dir)
    if {name: sha256(data_dir / name) for name in FILES} != manifest["input_sha256"]:
        raise ValueError("Evaluation input differs from the recorded snapshot")
    frame, test, _, schema = read_data(data_dir)
    if schema != json.loads((run_dir / "schema.json").read_text()):
        raise ValueError("Schema differs")
    config = manifest["config"]
    x, xt, y = features(frame, schema), features(test, schema), frame.y.to_numpy(dtype=float)
    saved = pd.read_csv(run_dir / "oof.csv")
    folds = pd.read_csv(run_dir / "folds.csv")
    if not frame.ID.astype(str).equals(folds.ID.astype(str)) or not frame.ID.astype(str).equals(
        saved.ID.astype(str)
    ):
        raise ValueError("OOF ID coverage/order differs")
    expected_columns = [
        "ID",
        "fold",
        "target",
        *[f"{name}.seed-{seed}" for seed in config["model_seeds"] for name in config["models"]],
        "prediction",
    ]
    if list(saved.columns) != expected_columns or list(folds.columns) != ["ID", "fold"]:
        raise ValueError("OOF/fold schema differs from the declared protocol")
    if not np.array_equal(saved.target.to_numpy(), y):
        raise ValueError("OOF targets differ from the training snapshot")
    if not np.array_equal(saved.fold.to_numpy(), folds.fold.to_numpy()):
        raise ValueError("OOF fold values differ from retained assignments")
    if not np.isfinite(saved.drop(columns=["ID"]).to_numpy()).all():
        raise ValueError("OOF contains nonfinite numeric evidence")
    splits = list(KFold(config["folds"], shuffle=True, random_state=config["split_seed"]).split(x))
    reconstructed = np.zeros(len(y))
    reconstructed_test = np.zeros(len(test))
    maximum = 0.0
    with threadpool_limits(limits=config["threads"]):
        for record in manifest["model_records"]:
            fit, valid = splits[record["fold"]]
            if record["train_rows"] != len(fit) or record["validation_rows"] != len(valid):
                raise ValueError("Fold row counts differ from the declared protocol")
            if not np.all(folds.fold.iloc[valid] == record["fold"]):
                raise ValueError("Saved folds differ from the declared protocol")
            estimator = joblib.load(run_dir / record["file"])
            values = estimator.predict(x.iloc[valid])
            column = f"{record['model']}.seed-{record['seed']}"
            maximum = max(maximum, float(np.max(np.abs(values - saved[column].iloc[valid]))))
            reconstructed[valid] += values / (len(config["models"]) * len(config["model_seeds"]))
            reconstructed_test += estimator.predict(xt) / len(manifest["model_records"])
    maximum = max(maximum, float(np.max(np.abs(reconstructed - saved.prediction))))
    submission = pd.read_csv(run_dir / "submission.csv")
    if list(submission.columns) != ["ID", "y"] or not np.isfinite(submission.y.to_numpy()).all():
        raise ValueError("Submission schema or finite-value contract differs")
    if not test.ID.astype(str).equals(submission.ID.astype(str)):
        raise ValueError("Submission ID order differs")
    maximum = max(maximum, float(np.max(np.abs(reconstructed_test - submission.y))))
    if maximum > 1e-10:
        raise ValueError("Saved predictions differ from reloaded models")
    metrics = scores(y, reconstructed)
    retained_metrics = json.loads((run_dir / "metrics.json").read_text())["pooled_oof"]
    if any(abs(metrics[name] - retained_metrics[name]) > 1e-10 for name in metrics):
        raise ValueError("Retained metrics differ from reloaded predictions")
    return {
        "status": "verified_model_reload",
        "models_reloaded": len(manifest["model_records"]),
        "maximum_prediction_delta": maximum,
        "tolerance": 1e-10,
        "metrics": metrics,
        "test_rows": len(test),
        "train_rows": len(frame),
    }
