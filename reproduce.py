"""Repeat the recorded three-segment training protocol and selected-policy holdout."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for the 40 + 40 + 80 minute protocol")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    previous = None
    for segment in range(1, 4):
        directory = output / f"segment_{segment}"
        command = [sys.executable, "-m", "error_defined_q.train", "--execute",
                   "--config", str(root / "configs" / f"segment_{segment}.json"),
                   "--output", str(directory)]
        if previous is not None:
            command.extend(["--continue-from", str(previous)])
        subprocess.run(command, check=True)
        completion = json.loads((directory / "completion.json").read_text())
        if completion["status"] not in ("completed", "time_budget"):
            raise RuntimeError(f"Segment {segment} stopped: {completion['status']}")
        previous = directory / "final_checkpoint.pt"

    # Selection uses monitoring alone; the holdout is evaluated after selection.
    monitors = [json.loads(line) for line in (directory / "monitoring.jsonl").read_text().splitlines()]
    eligible = [record for record in monitors if record["complete"]
                and (directory / f"epoch_{record['completed_epochs']:06d}.pt").is_file()]
    if not eligible:
        raise RuntimeError("No completed monitoring checkpoint available for evaluation")
    best = max(eligible, key=lambda record: (record["summary"]["success_fraction"],
                                             -record["completed_epochs"]))
    source = directory / f"epoch_{best['completed_epochs']:06d}.pt"
    selected = output / "selected_checkpoint.pt"
    shutil.copyfile(source, selected)
    selection = {
        "rule": "Highest completed greedy monitoring landing rate in segment 3; earliest epoch breaks ties",
        "source_checkpoint": str(source.relative_to(output)),
        "checkpoint_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        "monitor": best,
        "holdout_seeds": [3000001, 3000002, 3000003, 3000004, 3000005],
        "holdout_episodes_per_seed": 100,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    subprocess.run([
        sys.executable, "-m", "error_defined_q.evaluate", "--checkpoint", str(selected),
        "--seeds", "3000001", "3000002", "3000003", "3000004", "3000005",
        "--episodes", "100", "--output", str(output / "evaluation_holdout.json"),
    ], check=True)


if __name__ == "__main__":
    main()
