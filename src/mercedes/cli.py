"""Explicit modeling commands; downloading, tracking and publishing remain opt-in."""

import argparse
import json
import os
from pathlib import Path

from mercedes.pipeline import evaluate, predict, read_data, train, verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("validate")
    command.add_argument("--data-dir", type=Path, required=True)
    command = commands.add_parser("train")
    command.add_argument("--data-dir", type=Path, required=True)
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--fixture", action="store_true")
    for name in ("evaluate", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--data-dir", type=Path, required=True)
        command.add_argument("--run-dir", type=Path, required=True)
    command = commands.add_parser("verify")
    command.add_argument("--run-dir", type=Path, required=True)
    command = commands.add_parser("predict")
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--data", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("track")
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    command.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "validate":
        frame, test, _, schema = read_data(args.data_dir)
        result = {"train_rows": len(frame), "test_rows": len(test), "schema": schema}
    elif args.command == "train":
        result = train(
            args.data_dir, args.run_dir, json.loads(args.config.read_text()), fixture=args.fixture
        )
    elif args.command == "track":
        from mercedes.tracking import track

        result = track(args.run_dir, args.tracking_uri, resume=args.resume)
    elif args.command == "verify":
        manifest = verify_run(args.run_dir)
        result = {"status": "verified_artifacts", "logical_run_id": manifest["logical_run_id"]}
    elif args.command == "predict":
        result = predict(args.run_dir, args.data, args.output)
    else:
        result = evaluate(args.run_dir, args.data_dir)
        if args.command == "analyze":
            result["analysis"] = json.loads((args.run_dir / "metrics.json").read_text())["analysis"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
