from __future__ import annotations

from pathlib import Path

import pytest

from tests.case_runner import list_cases, run_case

_ROOT = Path(__file__).parent.parent / "benchcases"


@pytest.mark.parametrize("case_path", list_cases(_ROOT))
def test_bench_case(case_path: str, monkeypatch) -> None:
    run_case(_ROOT, case_path, monkeypatch)
