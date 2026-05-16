#!/usr/bin/env python3
"""Regenerate output files for all testcases under testcases/.

Run from the repo root:
    uv run python scripts/regen_testcase_outputs.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import app.main as app_module


def regen_case(case_dir: Path) -> None:
    os.environ["CONFIG_PATH"] = str(case_dir / "config.yaml")
    step_inputs = sorted(case_dir.glob("input_*.json"))

    with TestClient(app_module.app) as client:
        for step_input in step_inputs:
            step_num = step_input.stem.split("_", 1)[1]  # e.g. "001"
            input_raw = json.loads(step_input.read_text())

            ingest_outputs = [
                client.post("/ingest?include_features=true", json={"payload": doc}).json()
                for doc in input_raw["ingest"]
            ]
            score_outputs = [
                client.post("/score?explain=true&include_features=true", json={"payload": doc}).json()
                for doc in input_raw["scores"]
            ]

            (case_dir / f"output_{step_num}_ingest.json").write_text(json.dumps(ingest_outputs, indent=2) + "\n")
            (case_dir / f"output_{step_num}_score.json").write_text(json.dumps(score_outputs, indent=2) + "\n")

    (case_dir / "output.json").unlink(missing_ok=True)


def main() -> None:
    testcases_dir = ROOT / "testcases"

    for suite_dir in sorted(testcases_dir.iterdir()):
        if not suite_dir.is_dir():
            continue
        print(f"{suite_dir.name}/")
        for case_dir in sorted(suite_dir.iterdir()):
            if case_dir.is_dir() and (case_dir / "config.yaml").exists():
                regen_case(case_dir)
                print(f"  {case_dir.name}")

    print("Done.")


if __name__ == "__main__":
    main()
