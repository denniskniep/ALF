"""Shared logic for running detector testcases and benchcases."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest
from fastapi.testclient import TestClient

import app.main as app_module

from scripts.utils import load_testcase_config, progress, should_generate

should_check = should_generate


def list_cases(root: Path) -> list[str]:
    if not root.exists():
        return []
    ids = []
    for suite_dir in sorted(root.iterdir()):
        if not suite_dir.is_dir():
            continue
        for case_dir in sorted(suite_dir.iterdir()):
            if case_dir.is_dir() and (case_dir / "config.yaml").exists():
                ids.append(f"{suite_dir.name}/{case_dir.name}")
    return ids


def _list_sort_key(item: Any) -> str:
    if isinstance(item, dict):
        for k in ("field", "name"):
            if k in item:
                return str(item[k])
    return json.dumps(item, sort_keys=True, default=str)


def assert_equal(actual: Any, expected: Any) -> None:
    """Recursively compare two JSON-parsed objects.

    Dicts are compared key-set-first (order-independent).
    Lists of dicts are sorted by a natural key before comparison so the check
    does not depend on insertion or serialisation order.
    Floats are compared with abs=1.0 tolerance.
    Everything else (str, int, bool, None) must match exactly.
    """
    if isinstance(expected, dict):
        assert set(actual.keys()) == set(expected.keys())
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        if expected and isinstance(expected[0], dict):
            actual_s = sorted(actual, key=_list_sort_key)
            expected_s = sorted(expected, key=_list_sort_key)
        else:
            actual_s, expected_s = actual, expected
        for a, e in zip(actual_s, expected_s):
            assert_equal(a, e)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1.0)
    else:
        assert actual == expected


def run_case(root: Path, case_path: str, monkeypatch: Any) -> None:
    suite, case = case_path.split("/", 1)
    case_dir = root / suite / case
    monkeypatch.setenv("CONFIG_PATH", str(case_dir / "config.yaml"))

    cfg   = load_testcase_config(case_dir)
    steps = sorted(case_dir.glob("input_*.json"))
    assert steps, f"no input_*.json files found in {case_dir}"

    with TestClient(app_module.app) as client:
        for step_input in steps:
            step_num     = step_input.stem.split("_", 1)[1]  # e.g. "001"
            raw          = json.loads(step_input.read_text())
            check_ingest = should_check(step_num, cfg["output-ingest"])
            check_score  = should_check(step_num, cfg["output-score"])

            ingest_docs  = raw["ingest"]
            score_docs   = raw["scores"]
            total        = len(ingest_docs) + len(score_docs)
            label        = f"{case_path} step {step_num}"
            t_step_start = time.perf_counter()

            def _ms(done: int) -> float | None:
                elapsed = time.perf_counter() - t_step_start
                return elapsed / done * 1000 if done > 1 else None

            if check_ingest:
                expected_ingest = json.loads((case_dir / f"output_{step_num}_ingest.json").read_text())
                assert len(ingest_docs) == len(expected_ingest)
                for i, (doc, expected) in enumerate(zip(ingest_docs, expected_ingest)):
                    progress(label, i + 1, total, _ms(i + 1), file=sys.stderr)
                    resp = client.post("/ingest?include_features=true", json={"payload": doc})
                    assert resp.status_code == 200
                    assert_equal(resp.json(), expected)
            else:
                for i, doc in enumerate(ingest_docs):
                    progress(label, i + 1, total, _ms(i + 1), file=sys.stderr)
                    assert client.post("/ingest", json={"payload": doc}).status_code == 200

            if check_score:
                expected_scores = json.loads((case_dir / f"output_{step_num}_score.json").read_text())
                assert len(score_docs) == len(expected_scores)
                for i, (doc, expected) in enumerate(zip(score_docs, expected_scores)):
                    done = len(ingest_docs) + i + 1
                    progress(label, done, total, _ms(done), file=sys.stderr)
                    resp = client.post("/score?explain=true&include_features=true", json={"payload": doc})
                    assert resp.status_code == 200
                    assert_equal(resp.json(), expected)
            else:
                for i, doc in enumerate(score_docs):
                    done = len(ingest_docs) + i + 1
                    progress(label, done, total, _ms(done), file=sys.stderr)
                    assert client.post("/score", json={"payload": doc}).status_code == 200

            sys.stderr.write("\n")
            sys.stderr.flush()
