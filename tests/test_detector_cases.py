from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as app_module

_TESTCASES_DIR = Path(__file__).parent.parent / "testcases"


def _list_sort_key(item: Any) -> str:
    if isinstance(item, dict):
        for k in ("field", "name"):
            if k in item:
                return str(item[k])
    return json.dumps(item, sort_keys=True, default=str)


def _assert_equal(actual: Any, expected: Any) -> None:
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
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        if expected and isinstance(expected[0], dict):
            actual_s = sorted(actual, key=_list_sort_key)
            expected_s = sorted(expected, key=_list_sort_key)
        else:
            actual_s, expected_s = actual, expected
        for a, e in zip(actual_s, expected_s):
            _assert_equal(a, e)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1.0)
    else:
        assert actual == expected


def _case_ids() -> list[str]:
    """Return 'suite/case' identifiers for every testcase under testcases/."""
    if not _TESTCASES_DIR.exists():
        return []
    ids = []
    for suite_dir in sorted(_TESTCASES_DIR.iterdir()):
        if not suite_dir.is_dir():
            continue
        for case_dir in sorted(suite_dir.iterdir()):
            if case_dir.is_dir() and (case_dir / "config.yaml").exists():
                ids.append(f"{suite_dir.name}/{case_dir.name}")
    return ids


@pytest.mark.parametrize("case_path", _case_ids())
def test_case(case_path: str, monkeypatch) -> None:
    suite, case = case_path.split("/", 1)
    case_dir = _TESTCASES_DIR / suite / case
    monkeypatch.setenv("CONFIG_PATH", str(case_dir / "config.yaml"))

    steps = sorted(case_dir.glob("input_*.json"))
    assert steps, f"no input_*.json files found in {case_dir}"

    with TestClient(app_module.app) as client:
        for step_input in steps:
            step_num = step_input.stem.split("_", 1)[1]  # e.g. "001"
            raw = json.loads(step_input.read_text())
            expected_ingest = json.loads((case_dir / f"output_{step_num}_ingest.json").read_text())
            expected_scores = json.loads((case_dir / f"output_{step_num}_score.json").read_text())

            assert len(raw["ingest"]) == len(expected_ingest)
            for doc, expected in zip(raw["ingest"], expected_ingest):
                resp = client.post("/ingest?include_features=true", json={"payload": doc})
                assert resp.status_code == 200
                _assert_equal(resp.json(), expected)

            assert len(raw["scores"]) == len(expected_scores)
            for doc, expected in zip(raw["scores"], expected_scores):
                resp = client.post("/score?explain=true&include_features=true", json={"payload": doc})
                assert resp.status_code == 200
                _assert_equal(resp.json(), expected)
