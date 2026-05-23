#!/usr/bin/env python3
"""Regenerate output files for all testcases under testcases/.

Run from the repo root:
    uv run python scripts/regen_testcase_outputs.py                            # all cases
    uv run python scripts/regen_testcase_outputs.py signin-logs-scenario-001  # one case by name
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

logging.getLogger("httpx").setLevel(logging.WARNING)

from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import app.main as app_module

from utils import load_testcase_config, progress, should_generate




def regen_case(
    case_dir: Path,
    label: str,
    input_overrides: dict[str, Path] | None = None,
    ingest_output_dir: Path | None = None,
) -> int:
    """Regenerate outputs for one case. Returns total number of events processed.

    If ingest_output_dir is set, ingest responses are always captured and written
    there (even when gen_ingest is False), so callers can use them for metrics.
    """
    cfg = load_testcase_config(case_dir)
    os.environ["CONFIG_PATH"] = str(case_dir / "config.yaml")

    step_map: dict[str, Path] = {}
    for p in sorted(case_dir.glob("input_*.json")):
        step_map[p.stem.split("_", 1)[1]] = p
    if input_overrides:
        step_map.update(input_overrides)
    step_inputs = [step_map[k] for k in sorted(step_map)]

    total_events = 0

    with TestClient(app_module.app) as client:
        for step_input in step_inputs:
            step_num = step_input.stem.split("_", 1)[1]  # e.g. "001"
            input_raw = json.loads(step_input.read_text())

            ingest_docs = input_raw["ingest"]
            score_docs  = input_raw["scores"]
            total       = len(ingest_docs) + len(score_docs)
            gen_ingest  = should_generate(step_num, cfg["output-ingest"])
            gen_score   = should_generate(step_num, cfg["output-score"])

            t_step_start = time.perf_counter()

            capture_ingest = gen_ingest or ingest_output_dir is not None
            ingest_outputs = []
            for i, doc in enumerate(ingest_docs):
                elapsed = time.perf_counter() - t_step_start
                ms_per_event = elapsed / (i + 1) * 1000 if i > 0 else None
                progress(label, i + 1, total, ms_per_event)
                if capture_ingest:
                    ingest_outputs.append(
                        client.post("/ingest?include_features=true", json={"payload": doc}).json()
                    )
                else:
                    client.post("/ingest", json={"payload": doc})

            score_outputs = []
            for i, doc in enumerate(score_docs):
                elapsed = time.perf_counter() - t_step_start
                ms_per_event = elapsed / (len(ingest_docs) + i + 1) * 1000
                progress(label, len(ingest_docs) + i + 1, total, ms_per_event)
                if gen_score:
                    score_outputs.append(
                        client.post("/score?explain=true&include_features=true", json={"payload": doc}).json()
                    )
                else:
                    client.post("/score", json={"payload": doc})

            print()  # end progress line
            total_events += total

            if gen_ingest:
                (case_dir / f"output_{step_num}_ingest.json").write_text(json.dumps(ingest_outputs, indent=2) + "\n")
            elif ingest_output_dir is not None:
                (ingest_output_dir / f"output_{step_num}_ingest.json").write_text(json.dumps(ingest_outputs, indent=2) + "\n")
            if gen_score:
                (case_dir / f"output_{step_num}_score.json").write_text(json.dumps(score_outputs, indent=2) + "\n")

    (case_dir / "output.json").unlink(missing_ok=True)
    return total_events


def main(root_dir: Path | None = None) -> None:
    filter_name = sys.argv[1] if len(sys.argv) > 1 else None
    testcases_dir = root_dir or (ROOT / "testcases")

    cases = [
        (suite_dir, case_dir)
        for suite_dir in sorted(testcases_dir.iterdir()) if suite_dir.is_dir()
        for case_dir in sorted(suite_dir.iterdir())
        if case_dir.is_dir() and (case_dir / "config.yaml").exists()
        and (not filter_name or case_dir.name == filter_name)
    ]

    if filter_name and not cases:
        print(f"No testcase named '{filter_name}' found.", file=sys.stderr)
        sys.exit(1)

    total_cases = len(cases)
    total_events = 0
    t_start = time.perf_counter()

    for i, (suite_dir, case_dir) in enumerate(cases, start=1):
        label = f"[{i}/{total_cases}] {suite_dir.name}/{case_dir.name}"
        total_events += regen_case(case_dir, label)

    elapsed = time.perf_counter() - t_start
    avg_ms = (elapsed / total_events * 1000) if total_events else 0
    print(
        f"Done. ({total_cases} case{'s' if total_cases != 1 else ''} regenerated"
        f", {total_events} events"
        f", {elapsed:.1f}s total"
        f", {avg_ms:.2f}ms/event)"
    )


if __name__ == "__main__":
    main()
