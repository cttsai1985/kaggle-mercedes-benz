"""Download the three authorized inputs with Kaggle CLI; retain exact byte identities."""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

from mercedes.pipeline import FILES, sha256, write_json


def download(directory):
    directory = Path(directory).resolve()
    project = Path(__file__).resolve().parents[1]
    if directory.is_relative_to(project) and not directory.is_relative_to(project / ".local"):
        raise ValueError("Store raw data outside the checkout or under ignored .local")
    directory.mkdir(parents=True, exist_ok=True)
    executable = str(Path(sys.executable).with_name("kaggle"))
    environment = None
    receipt = {"competition": "mercedes-benz-greener-manufacturing", "files": {}}
    for name in FILES:
        archive = directory / (name + ".zip")
        if not archive.exists():
            if environment is None:
                environment = os.environ.copy()
                if not environment.get("KAGGLE_API_TOKEN"):
                    auth = subprocess.run(
                        [executable, "auth", "print-access-token"], capture_output=True, text=True
                    )
                    if auth.returncode or not auth.stdout.strip() or "\n" in auth.stdout.strip():
                        raise RuntimeError("Authenticate with kaggle auth login")
                    environment["KAGGLE_API_TOKEN"] = auth.stdout.strip()
            result = subprocess.run(
                [
                    executable,
                    "competitions",
                    "download",
                    receipt["competition"],
                    "-f",
                    archive.name,
                    "-p",
                    str(directory),
                    "-q",
                ],
                env=environment,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise RuntimeError(f"Kaggle download failed for {name}; check competition access")
        with zipfile.ZipFile(archive) as bundle:
            members = [m for m in bundle.infolist() if PurePosixPath(m.filename).name == name]
            if len(members) != 1:
                raise ValueError("Expected one named competition CSV")
            member = members[0]
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe archive path")
            target = directory / name
            if target.exists():
                import hashlib

                if sha256(target) != hashlib.sha256(bundle.read(member)).hexdigest():
                    raise ValueError("Existing raw input differs from its archive")
            else:
                with bundle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        receipt["files"][name] = {
            "sha256": sha256(target),
            "archive_sha256": sha256(archive),
            "bytes": target.stat().st_size,
        }
    write_json(directory / "download-manifest.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    print(download(parser.parse_args().data_dir))
