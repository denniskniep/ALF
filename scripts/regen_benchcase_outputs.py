#!/usr/bin/env python3
"""Regenerate output files for all benchcases under benchcases/.

Run from the repo root:
    uv run python scripts/regen_benchcase_outputs.py                     # all bench cases
    uv run python scripts/regen_benchcase_outputs.py signin-logs-scenario-001     # one bench case by name
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from regen_testcase_outputs import ROOT, main as _main
from utils import load_testcase_config, should_generate


def _write_metrics(case_dir: Path) -> None:
    cfg = load_testcase_config(case_dir)

    tp_total = fp_total = tn_total = fn_total = 0
    fp_scores: list[float] = []
    fn_scores: list[float] = []
    has_any = False

    for step_input in sorted(case_dir.glob("input_*.json")):
        step_num = step_input.stem.split("_", 1)[1]
        if not should_generate(step_num, cfg["output-score"]):
            continue
        output_score_path = case_dir / f"output_{step_num}_score.json"
        if not output_score_path.exists():
            continue

        score_docs = json.loads(step_input.read_text()).get("scores", [])
        score_outputs = json.loads(output_score_path.read_text())

        for doc, resp in zip(score_docs, score_outputs):
            gt_positive = doc.get("description", "NORMAL").upper().startswith("ANOMALY")
            pred_positive = resp.get("status", "") in ("ANOMALOUS", "HIGHLY_ANOMALOUS")
            score = resp.get("composite_score")
            if gt_positive and pred_positive:
                tp_total += 1
            elif not gt_positive and pred_positive:
                fp_total += 1
                if score is not None:
                    fp_scores.append(score)
            elif not gt_positive and not pred_positive:
                tn_total += 1
            else:
                fn_total += 1
                if score is not None:
                    fn_scores.append(score)
        has_any = True

    if not has_any:
        return

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def _fmt(v: float | None) -> str:
        return "null" if v is None else str(round(v, 6))

    def _fmt_scores(scores: list[float]) -> str:
        if not scores:
            return "[]"
        items = ", ".join(str(s) for s in scores)
        return f"[{items}]"

    results_block = (
        "results:\n"
        f"  f1: {_fmt(f1)}\n"
        f"  recall: {_fmt(recall)}\n"
        f"  precision: {_fmt(precision)}\n"
        f"  tp: {tp_total}\n"
        f"  fp: {fp_total}\n"
        f"  tn: {tn_total}\n"
        f"  fn: {fn_total}\n"
        f"  fp_scores: {_fmt_scores(fp_scores)}\n"
        f"  fn_scores: {_fmt_scores(fn_scores)}\n"
    )

    config_path = case_dir / "config.yaml"
    text = config_path.read_text()
    lines = text.splitlines(keepends=True)

    # Find existing results block (top-level key at column 0)
    start = next((i for i, l in enumerate(lines) if l.startswith("results:")), None)
    if start is not None:
        end = start + 1
        while end < len(lines) and (lines[end][:1] in (" ", "\t") or lines[end].strip() == ""):
            end += 1
        lines[start:end] = [results_block]
        text = "".join(lines)
    else:
        text = text.rstrip("\n") + "\n" + results_block

    config_path.write_text(text)


def main() -> None:
    filter_name = sys.argv[1] if len(sys.argv) > 1 else None
    benchcases_dir = ROOT / "benchcases"

    cases = [
        case_dir
        for suite_dir in sorted(benchcases_dir.iterdir()) if suite_dir.is_dir()
        for case_dir in sorted(suite_dir.iterdir())
        if case_dir.is_dir() and (case_dir / "config.yaml").exists()
        and (not filter_name or case_dir.name == filter_name)
    ]

    if filter_name and not cases:
        print(f"No benchcase named '{filter_name}' found.", file=sys.stderr)
        sys.exit(1)

    _main(root_dir=benchcases_dir)

    for case_dir in cases:
        _write_metrics(case_dir)


if __name__ == "__main__":
    main()
