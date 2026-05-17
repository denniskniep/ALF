from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

def load_testcase_config(case_dir: Path) -> dict:
    """Load config.testcase.yaml if present.

    Returns a dict with keys:
      output-ingest: None (all steps) | [] (none) | ["001", ...]
      output-score:  None (all steps) | [] (none) | ["001", ...]
    """
    path = case_dir / "config.testcase.yaml"
    if not path.exists():
        return {"output-ingest": None, "output-score": None}
    cfg = yaml.safe_load(path.read_text()) or {}
    return {
        "output-ingest": cfg.get("output-ingest"),
        "output-score":  cfg.get("output-score"),
    }


def should_generate(step_num: str, steps_cfg: Any) -> bool:
    """True when output for this step should be generated."""
    if steps_cfg is None:
        return True
    return step_num in steps_cfg


_BAR_WIDTH = 36


def progress(
    label: str,
    current: int,
    total: int,
    ms_per_event: float | None = None,
    file: Any = None,
) -> None:
    import sys
    out = file or sys.stdout
    filled = int(_BAR_WIDTH * current / total) if total else _BAR_WIDTH
    bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
    timing = f" {ms_per_event:.0f}ms/event" if ms_per_event is not None else ""
    suffix = f"  [{bar}] {current}/{total}{timing} "
    term_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    max_label = term_width - len(suffix) - 2  # 2 for leading "  "
    if 0 < max_label < len(label):
        label = label[:max_label - 1] + "…"
    out.write(f"\r  {label}{suffix}")
    out.flush()
