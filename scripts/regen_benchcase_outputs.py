#!/usr/bin/env python3
"""Regenerate output files for all benchcases under benchcases/.

Run from the repo root:
    uv run python scripts/regen_benchcase_outputs.py                     # all bench cases
    uv run python scripts/regen_benchcase_outputs.py signin-logs-scenario-001     # one bench case by name
"""
from __future__ import annotations

import csv as _csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from regen_testcase_outputs import ROOT, regen_case
from utils import load_testcase_config, should_generate


# ------------------------------------------------------------------ #
# CSV input                                                            #
# ------------------------------------------------------------------ #

def _eval_anomaly_expr(expr: str, row: dict[str, str]) -> bool:
    rewritten = re.sub(r'\.([A-Za-z_]\w*)', lambda m: f"row.get('{m.group(1)}')", expr)
    return bool(eval(rewritten, {"__builtins__": {}}, {"row": row}))  # noqa: S307


def _transform_csv_to_input(csv_path: Path, cfg: dict) -> dict[str, Any]:
    csv_opts = cfg.get("csv", {})
    delimiter = csv_opts.get("delimiter", ",")
    encoding = csv_opts.get("encoding", "utf-8")

    score_fraction = float(cfg.get("score_split_by", 0.2))
    anomaly_expr: str | None = cfg.get("anomalyWhenMatch")

    rows: list[dict[str, str]] = []
    with open(csv_path, encoding=encoding, newline="") as f:
        reader = _csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rows.append(dict(row))

    n_total = len(rows)
    n_scores = int(n_total * score_fraction)
    ingest_rows = rows[: n_total - n_scores]
    score_rows = rows[n_total - n_scores :]

    ingest = [dict(r) for r in ingest_rows]
    scores = []
    for r in score_rows:
        payload = dict(r)
        if anomaly_expr:
            payload["description"] = "ANOMALY" if _eval_anomaly_expr(anomaly_expr, r) else "NORMAL"
        scores.append(payload)

    return {"ingest": ingest, "scores": scores}


# ------------------------------------------------------------------ #
# ARFF input                                                           #
# ------------------------------------------------------------------ #

def _parse_arff_attributes(path: Path) -> list[tuple[str, str]]:
    """Parse @attribute lines from an ARFF file.

    Returns a list of (name, kind) where kind is 'numeric' or 'nominal'.
    """
    attributes: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            upper = stripped.upper()
            if upper.startswith("@DATA"):
                break
            if not upper.startswith("@ATTRIBUTE"):
                continue
            rest = stripped[len("@attribute"):].strip()
            if rest.startswith("'"):
                end = rest.index("'", 1)
                name = rest[1:end]
                type_str = rest[end + 1:].strip()
            else:
                parts = rest.split(None, 1)
                name = parts[0]
                type_str = parts[1].strip() if len(parts) > 1 else ""
            kind = "numeric" if type_str.upper() in ("REAL", "INTEGER", "NUMERIC") else "nominal"
            attributes.append((name, kind))
    return attributes


def _transform_arff_to_input(arff_path: Path, cfg: dict) -> dict[str, Any]:
    attributes = _parse_arff_attributes(arff_path)
    attr_names = [name for name, _ in attributes]
    attr_kinds = {name: kind for name, kind in attributes}

    limit: int | None = cfg.get("limit")
    score_fraction = float(cfg.get("score_split_by", 0.2))
    anomaly_expr: str | None = cfg.get("anomalyWhenMatch")

    rows: list[dict[str, Any]] = []
    in_data = False
    with open(arff_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            if stripped.upper().startswith("@DATA"):
                in_data = True
                continue
            if not in_data:
                continue
            values = next(_csv.reader([stripped]))
            if len(values) != len(attr_names):
                continue
            row: dict[str, Any] = {}
            for name, raw_val in zip(attr_names, values):
                raw = raw_val.strip().strip("'")
                if raw == "?":
                    row[name] = None
                elif attr_kinds[name] == "numeric":
                    try:
                        row[name] = float(raw)
                    except ValueError:
                        row[name] = None
                else:
                    row[name] = raw
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break

    n_total = len(rows)
    n_scores = int(n_total * score_fraction)
    ingest_rows = rows[: n_total - n_scores]
    score_rows = rows[n_total - n_scores :]

    ingest = [dict(r) for r in ingest_rows]
    scores = []
    for r in score_rows:
        payload = dict(r)
        if anomaly_expr:
            row_str = {k: (str(v) if v is not None else "") for k, v in r.items()}
            payload["description"] = "ANOMALY" if _eval_anomaly_expr(anomaly_expr, row_str) else "NORMAL"
        scores.append(payload)

    return {"ingest": ingest, "scores": scores}


# ------------------------------------------------------------------ #
# NPZ input                                                            #
# ------------------------------------------------------------------ #

def _transform_npz_to_input(npz_path: Path, cfg: dict) -> dict[str, Any]:
    import numpy as np

    data = np.load(str(npz_path), allow_pickle=True)
    X: Any = data["X"]
    y: Any = data["y"]

    limit: int | None = cfg.get("limit")
    if limit is not None:
        X, y = X[:limit], y[:limit]

    unique_labels = set(int(v) for v in y)
    if unique_labels - {0, 1}:
        raise ValueError(f"npz y contains unexpected values: {unique_labels}. Expected only 0 and 1.")

    feature_names: list[str] = cfg.get("feature_names") or [f"feature_{i}" for i in range(X.shape[1])]
    score_fraction = float(cfg.get("score_split_by", 0.2))
    anomaly_expr: str | None = cfg.get("anomalyWhenMatch")

    n = len(X)
    n_scores = int(n * score_fraction)

    def _row(arr: Any) -> dict[str, Any]:
        return {name: float(val) for name, val in zip(feature_names, arr)}

    def _is_anomaly(i: int) -> bool:
        if anomaly_expr:
            row_str = {k: str(v) for k, v in _row(X[i]).items()}
            row_str["y"] = str(int(y[i]))
            return _eval_anomaly_expr(anomaly_expr, row_str)
        return int(y[i]) == 1

    ingest = [{**_row(X[i]), "description": "NORMAL"} for i in range(n - n_scores) if not _is_anomaly(i)]
    scores = []
    for i in range(n - n_scores, n):
        payload = _row(X[i])
        payload["description"] = "ANOMALY" if _is_anomaly(i) else "NORMAL"
        scores.append(payload)

    return {"ingest": ingest, "scores": scores}


# ------------------------------------------------------------------ #
# Unified input loader                                                 #
# ------------------------------------------------------------------ #

_TRANSFORMS = {
    "csv": (_transform_csv_to_input, lambda cfg: cfg.get("url", "").rsplit("/", 1)[-1]),
    "npz": (_transform_npz_to_input, lambda cfg: cfg.get("url", "").rsplit("/", 1)[-1]),
    "arff": (_transform_arff_to_input, lambda cfg: cfg.get("url", "").rsplit("/", 1)[-1]),
}


def _ensure_generated_inputs(case_dir: Path) -> dict[str, Path]:
    """Process the ``input:`` array from config.testcase.yaml.

    Each item must have exactly one key (``csv:`` or ``npz:``) whose value is
    the source config for that step.  Downloads source files to ``.local/`` and
    generates ``.local/input_<step>.json`` (both cached — delete ``.local/`` to
    force regeneration).

    Returns a dict mapping step_num → Path of generated input file.
    """
    testcase_cfg_path = case_dir / "config.testcase.yaml"
    if not testcase_cfg_path.exists():
        return {}
    raw = yaml.safe_load(testcase_cfg_path.read_text()) or {}
    input_items: list[dict] = raw.get("input") or []
    if not input_items:
        return {}

    local_dir = case_dir / ".local"
    local_dir.mkdir(exist_ok=True)

    overrides: dict[str, Path] = {}
    for item in input_items:
        kind = next((k for k in _TRANSFORMS if k in item), None)
        if kind is None:
            known = ", ".join(_TRANSFORMS)
            raise ValueError(f"input item must have one of: {known}. Got keys: {list(item)}")

        transform_fn, filename_fn = _TRANSFORMS[kind]
        cfg: dict = item[kind]
        url: str = cfg["url"]
        step: str = str(cfg.get("step", "001"))

        src_path = local_dir / filename_fn(cfg)
        if not src_path.exists():
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, str(src_path))

        input_path = local_dir / f"input_{step}.json"
        if not input_path.exists():
            data = transform_fn(src_path, cfg)
            input_path.write_text(json.dumps(data, indent=2) + "\n")
            print(
                f"  Generated {input_path.relative_to(case_dir)}"
                f" ({len(data['ingest'])} ingest, {len(data['scores'])} scores)"
            )

        overrides[step] = input_path

    return overrides


# ------------------------------------------------------------------ #
# Metrics                                                              #
# ------------------------------------------------------------------ #

def _write_metrics(case_dir: Path, input_overrides: dict[str, Path] | None = None) -> None:
    cfg = load_testcase_config(case_dir)

    step_map: dict[str, Path] = {}
    for p in sorted(case_dir.glob("input_*.json")):
        step_map[p.stem.split("_", 1)[1]] = p
    if input_overrides:
        step_map.update(input_overrides)

    tp_total = fp_total = tn_total = fn_total = 0
    train_count = 0
    score_count = 0
    accepted_count = 0
    unaccepted_count = 0
    fp_scores: list[float] = []
    fn_scores: list[float] = []
    has_any = False

    for step_num, step_input in sorted(step_map.items()):
        if should_generate(step_num, cfg["output-score"]):
            output_score_path = case_dir / f"output_{step_num}_score.json"
            if output_score_path.exists():
                input_data = json.loads(step_input.read_text())
                train_count += len(input_data.get("ingest", []))
                score_docs = input_data.get("scores", [])
                score_count += len(score_docs)
                score_outputs = json.loads(output_score_path.read_text())
                for doc, resp in zip(score_docs, score_outputs):
                    if resp.get("accepted"):
                        accepted_count += 1
                    else:
                        unaccepted_count += 1
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

        ingest_out = case_dir / f"output_{step_num}_ingest.json"
        if not ingest_out.exists():
            ingest_out = case_dir / ".local" / f"output_{step_num}_ingest.json"
        if ingest_out.exists():
            for resp in json.loads(ingest_out.read_text()):
                if resp.get("accepted"):
                    accepted_count += 1
                else:
                    unaccepted_count += 1

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

    normal_count = tn_total + fp_total
    anomaly_count = tp_total + fn_total

    results_block = (
        "results:\n"
        f"  train_count: {train_count}\n"
        f"  score_count: {score_count}\n"
        f"  accepted_count: {accepted_count}\n"
        f"  unaccepted_count: {unaccepted_count}\n"
        f"  normal_count: {normal_count}\n"
        f"  anomaly_count: {anomaly_count}\n"
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


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

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

    total_cases = len(cases)
    total_events = 0
    t_start = time.perf_counter()

    for i, case_dir in enumerate(cases, start=1):
        suite_name = case_dir.parent.name
        label = f"[{i}/{total_cases}] {suite_name}/{case_dir.name}"
        overrides = _ensure_generated_inputs(case_dir) or None
        local_dir = case_dir / ".local"
        local_dir.mkdir(exist_ok=True)
        total_events += regen_case(case_dir, label, input_overrides=overrides, ingest_output_dir=local_dir)
        _write_metrics(case_dir, input_overrides=overrides)

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
