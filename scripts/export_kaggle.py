"""Stage a code-only wheel and offline Kaggle notebook; never upload or submit."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from mercedes import PROJECT_ID, __version__
from mercedes.pipeline import sha256, source_identity, write_json


def export(output, owner="cttsai"):
    project = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(
            "Choose a fresh staging directory; existing code versions are not overwritten"
        )
    output.mkdir(parents=True)
    assets, kernel = output / "code-dataset", output / "kernel"
    assets.mkdir()
    kernel.mkdir()
    with tempfile.TemporaryDirectory(prefix="mercedes-wheel-") as staging:
        subprocess.run(["uv", "build", "--wheel", "--out-dir", staging], cwd=project, check=True)
        wheels = list(Path(staging).glob("*.whl"))
        expected = f"mercedes_benz_pipeline-{__version__}-py3-none-any.whl"
        if len(wheels) != 1 or wheels[0].name != expected:
            raise ValueError("Expected one newly built wheel with the current project version")
        wheel = assets / expected
        shutil.copy2(wheels[0], wheel)
    shutil.copy2(project / "LICENSE", assets / "LICENSE")
    configs = {}
    for path in sorted((project / "configs").glob("*.json")):
        destination = assets / ("config-" + path.name)
        shutil.copy2(path, destination)
        configs[destination.name] = sha256(destination)
    source = {
        "schema_version": 1,
        "project_id": PROJECT_ID,
        "repository": "https://github.com/cttsai1985/kaggle-mercedes-benz",
        "wheel": wheel.name,
        "wheel_sha256": sha256(wheel),
        "source": source_identity(),
        "lock_sha256": sha256(project / "uv.lock"),
        "config_sha256": configs,
        "contains_competition_data": False,
    }
    write_json(assets / "source.json", source)
    write_json(
        assets / "dataset-metadata.json",
        {
            "id": f"{owner}/kaggle-mercedes-benz-code",
            "title": "Kaggle Mercedes Benz code",
            "licenses": [{"name": "MIT"}],
        },
    )
    shutil.copy2(project / "notebooks/kaggle/rerun.ipynb", kernel / "rerun.ipynb")
    metadata = json.loads((project / "notebooks/kaggle/kernel-metadata.json").read_text())
    metadata["id"] = f"{owner}/kaggle-mercedes-benz-solution-mlops"
    metadata["title"] = "Kaggle Mercedes Benz Solution MLOps"
    metadata["dataset_sources"] = [f"{owner}/kaggle-mercedes-benz-code"]
    write_json(kernel / "kernel-metadata.json", metadata)
    print(f"Prepared code-only assets at {output}; no remote resource changed.")
    print("Dataset upload must retain JSON/config bytes: use --keep-tabular.")
    print("Create the dataset privately (CLI default); keep kernel metadata is_private=true.")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".local/kaggle-bundle"))
    parser.add_argument("--owner", default="cttsai", help="Kaggle username, not GitHub username")
    args = parser.parse_args()
    export(args.output, args.owner)


if __name__ == "__main__":
    main()
