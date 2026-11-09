"""Export the evaluated fold ensemble as a self-contained MLflow prediction model."""

import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from threadpoolctl import threadpool_limits

from mercedes.pipeline import features, sha256, source_identity, verify_run


class TestDurationModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        root = Path(context.artifacts["ensemble"])
        contract = json.loads((root / "serving-contract.json").read_text())
        names = contract["models"]
        if (
            not names
            or len(names) != len(set(names))
            or any(Path(name).name != name for name in names)
            or set(contract["files"]) != {"schema.json", *names}
        ):
            raise ValueError("Serving ensemble model set is incomplete or invalid")
        for name, expected in contract["files"].items():
            if Path(name).name != name or sha256(root / name) != expected:
                raise ValueError("Serving ensemble checksum mismatch")
        self.schema = json.loads((root / "schema.json").read_text())
        self.models = [joblib.load(root / name) for name in contract["models"]]

    def predict(self, context, model_input, params=None):
        if not isinstance(model_input, pd.DataFrame) or model_input.empty:
            raise ValueError("Provide a nonempty feature table")
        if list(model_input.columns) != self.schema["columns"]:
            raise ValueError("Feature schema/order differs from the fitted ensemble")
        x = features(model_input, self.schema)
        with threadpool_limits(limits=2):
            values = np.mean([model.predict(x) for model in self.models], axis=0)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite duration prediction")
        return pd.DataFrame({"y": values})


def export_model(run_dir, output, example):
    """Copy only fitted models/schema; keep row-level experiment artifacts private."""
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Serving export already exists; select a fresh output directory")
    manifest = verify_run(run_dir)
    export_source = source_identity()
    schema = json.loads((run_dir / "schema.json").read_text())
    example = example[schema["columns"]].copy()
    if example.empty:
        raise ValueError("Provide a representative input example")
    # Explicit float inputs avoid integer inference that rejects future missing values.
    for name in schema["numeric"]:
        example[name] = example[name].astype(float)
    for name in schema["categorical"]:
        example[name] = example[name].astype(str)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mercedes-export-") as directory:
        bundle = Path(directory) / "ensemble"
        bundle.mkdir()
        names = [record["file"] for record in manifest["model_records"]]
        for name in ["schema.json", *names]:
            shutil.copy2(run_dir / name, bundle / name)
        contract = {
            "models": names,
            "files": {name: sha256(bundle / name) for name in ["schema.json", *names]},
            "producer_manifest_sha256": sha256(run_dir / "manifest.json"),
            "producer_source": manifest["source"],
            "export_source": export_source,
            "aggregation": "equal mean of every saved fold/family/seed",
        }
        (bundle / "serving-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        code = Path(directory) / "code" / "mercedes"
        shutil.copytree(
            Path(__file__).parent, code, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        model = TestDurationModel()
        context = type("Context", (), {"artifacts": {"ensemble": str(bundle)}})()
        model.load_context(context)
        expected = model.predict(context, example)
        mlflow.pyfunc.save_model(
            path=str(output),
            python_model=TestDurationModel(),
            artifacts={"ensemble": str(bundle)},
            code_paths=[str(code)],
            input_example=example,
            signature=infer_signature(example, expected),
            pip_requirements=[
                f"{name}=={importlib.metadata.version(name)}"
                for name in ["mlflow", "numpy", "pandas", "scikit-learn", "joblib", "threadpoolctl"]
            ],
            metadata={
                "project_id": "mercedes-benz",
                "producer_manifest_sha256": contract["producer_manifest_sha256"],
                "model_count": len(names),
                "export_source": export_source,
                "input_boundary": "ordered configuration features; no ID or target",
            },
        )
    return {
        "status": "exported",
        "export_source": export_source,
        "models": len(names),
        "example_rows": len(example),
        "producer_manifest_sha256": contract["producer_manifest_sha256"],
    }
