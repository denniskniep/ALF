#!/usr/bin/env python3
"""
Fetch auth events from Elasticsearch, train anomaly models on a 3-week baseline,
then score the most-recent week and report results.

Training window:  now-4w → now-1w
Detection window: now-1w → now

Config via CLI args or env vars

Usage:
  uv run python scripts/train_and_detect_from_elastic.py \\
      --es-url https://es.example.com:9200 \\
      --es-api-key YOUR_API_KEY \\
      --index auth-events \\
      [--filter 'department.id: EXAMPLE-DEPT'] \\
      [--api-url http://localhost:8000] \\
      [--page-size 500] \\
      [--anomalous-only] \\
      [--explain]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

logging.getLogger("httpx").setLevel(logging.WARNING)

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from app.orchestration import labels  # noqa: E402


# ---------------------------------------------------------------------------
# Elasticsearch helpers
# ---------------------------------------------------------------------------

TIMESTAMP_FIELD = "@timestamp"


def _es_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"ApiKey {api_key}",
        "Content-Type": "application/json",
    }


def _build_query(lucene_filter: str | None, gte: str, lt: str) -> dict[str, Any]:
    clauses: list[dict] = [{"range": {TIMESTAMP_FIELD: {"gte": gte, "lt": lt}}}]
    if lucene_filter:
        clauses.append({"query_string": {"query": lucene_filter}})
    return {"bool": {"filter": clauses}}


def _count_window(
    es_url: str,
    api_key: str,
    index: str,
    lucene_filter: str | None,
    gte: str,
    lt: str,
) -> int:
    """Return the total number of documents in [gte, lt) without fetching them."""
    url = f"{es_url.rstrip('/')}/{index}/_count"
    with httpx.Client(headers=_es_headers(api_key), timeout=60.0) as client:
        resp = client.post(url, json={"query": _build_query(lucene_filter, gte, lt)})
        if resp.status_code != 200:
            print(f"  ERROR: _count returned {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
            resp.raise_for_status()
        return resp.json()["count"]


def _fetch_window(
    es_url: str,
    api_key: str,
    index: str,
    lucene_filter: str | None,
    gte: str,
    lt: str,
    page_size: int,
    debug_es: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield documents in [gte, lt) page by page using search_after."""
    url = f"{es_url.rstrip('/')}/{index}/_search"
    body: dict[str, Any] = {
        "query": _build_query(lucene_filter, gte, lt),
        "sort": [{TIMESTAMP_FIELD: "asc"}],
        "size": page_size,
        "_source": True,
    }
    if debug_es:
        import json
        print(f"\n  [debug] POST {url}")
        print(f"  [debug] body: {json.dumps(body, indent=2)}")

    after: list | None = None
    with httpx.Client(headers=_es_headers(api_key), timeout=60.0) as client:
        while True:
            if after is not None:
                body["search_after"] = after
            resp = client.post(url, json=body)
            if resp.status_code != 200:
                print(
                    f"  ERROR: Elasticsearch returned {resp.status_code}: {resp.text[:300]}",
                    file=sys.stderr,
                )
                resp.raise_for_status()
            hits = resp.json()["hits"]["hits"]
            if not hits:
                break
            for hit in hits:
                yield hit["_source"]
            after = hits[-1]["sort"]
            if len(hits) < page_size:
                break


# ---------------------------------------------------------------------------
# Anomaly API helpers
# ---------------------------------------------------------------------------

def _ingest(api_url: str, client: Any, payload: dict) -> None:
    resp = client.post(f"{api_url}/ingest", json={"payload": payload})
    if resp.status_code != 200:
        print(
            f"  WARN: /ingest returned {resp.status_code}: {resp.text[:200]}",
            file=sys.stderr,
        )


def _score(
    api_url: str, client: Any, payload: dict, explain: bool
) -> dict | None:
    params = {"explain": "true"} if explain else {}
    resp = client.post(f"{api_url}/score", json={"payload": payload}, params=params)
    if resp.status_code == 200:
        return resp.json()
    print(
        f"  WARN: /score returned {resp.status_code}: {resp.text[:200]}",
        file=sys.stderr,
    )
    return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and detect anomalies using Elasticsearch data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--es-url",
        default=os.environ.get("ES_URL"),
        help="Elasticsearch base URL (env: ES_URL)",
    )
    p.add_argument(
        "--es-api-key",
        default=os.environ.get("ES_API_KEY"),
        help="Elasticsearch API key (env: ES_API_KEY)",
    )
    p.add_argument(
        "--index",
        default=os.environ.get("ES_INDEX"),
        help="Elasticsearch index or data-view name (env: ES_INDEX)",
    )
    p.add_argument(
        "--api-url",
        default=os.environ.get("ANOMALY_API_URL", None),
        help="Anomaly detection API base URL (env: ANOMALY_API_URL)",
    )
    p.add_argument(
        "--filter",
        default=None,
        metavar="LUCENE_EXPR",
        help="Lucene filter expression applied to all ES queries (e.g. 'field:value AND other:x')",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=500,
        help="Number of documents per Elasticsearch page (default: 500)",
    )
    p.add_argument(
        "--anomalous-only",
        action="store_true",
        help="Only display anomalous results in the output table",
    )
    p.add_argument(
        "--explain",
        action="store_true",
        help="Request explanations for detection results",
    )
    p.add_argument(
        "--debug-es",
        action="store_true",
        help="Print the Elasticsearch query body before executing it",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help=(
            "Run the anomaly API in-process via TestClient instead of hitting an "
            "external server. Both training and detection share one in-memory app "
            "instance so model state is preserved between phases."
        ),
    )
    p.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config.yml"),
        help="Path to config.yml; relative paths are resolved from the repo root (default: <repo-root>/config.yml, used with --local)",
    )
    p.add_argument(
        "--train-from",
        default="now-4w/d",
        help="Training window start — Elasticsearch date math (default: now-4w/d)",
    )
    p.add_argument(
        "--train-to",
        default="now-1w/d",
        help="Training window end — Elasticsearch date math (default: now-1w/d)",
    )
    p.add_argument(
        "--detect-from",
        default="now-1w/d",
        help="Detection window start — Elasticsearch date math (default: now-1w/d)",
    )
    p.add_argument(
        "--detect-to",
        default="now/d",
        help="Detection window end — Elasticsearch date math (default: now/d)",
    )
    return p.parse_args()


def _require(value: str | None, name: str) -> str:
    if not value:
        print(f"ERROR: {name} is required (set via CLI arg or env var)", file=sys.stderr)
        sys.exit(1)
    return value


def _make_client(local: bool, config: str | None = None) -> tuple[Any, str]:
    """Return (client_context_manager, api_url) for either local or remote mode.

    Local mode: TestClient runs the FastAPI app in-process. Both training and
    detection must share the same client instance so in-memory model state is
    preserved between phases — a second TestClient would start a fresh app.

    Remote mode: a plain httpx.Client connecting to the configured server URL.
    """
    if local:
        if config:
            os.environ["CONFIG_PATH"] = config
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import app.main as app_module
        from fastapi.testclient import TestClient
        return TestClient(app_module.app), "http://testserver"
    return httpx.Client(timeout=30.0), ""


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path
    args.config = str(config_path)

    es_url = _require(args.es_url, "--es-url / ES_URL")
    es_api_key = _require(args.es_api_key, "--es-api-key / ES_API_KEY")
    index = _require(args.index, "--index / ES_INDEX")
    lucene_filter: str | None = args.filter

    client_ctx, api_url = _make_client(args.local, args.config)
    if not api_url:
        api_url = args.api_url.rstrip("/")

    # Both phases share one client so in-memory model state survives between them.
    with client_ctx as api_client:
        # ------------------------------------------------------------------
        # Phase 1: training — ingest now-4w → now-1w
        # ------------------------------------------------------------------
        filter_label = f"filter={lucene_filter}" if lucene_filter else "no filter"
        print(f"\n[1/2] TRAINING  from:{args.train_from}  to:{args.train_to}  {filter_label}")
        print(f"      ES: {es_url}/{index}")
        print(f"      API: {api_url}/ingest")

        train_total = _count_window(es_url, es_api_key, index, lucene_filter, args.train_from, args.train_to)
        print(f"      {train_total} documents")

        train_count = 0
        train_errors = 0
        for doc in _fetch_window(
            es_url, es_api_key, index,
            lucene_filter, args.train_from, args.train_to,
            args.page_size, args.debug_es,
        ):
            try:
                _ingest(api_url, api_client, doc)
                train_count += 1
            except Exception as exc:
                train_errors += 1
                if train_errors <= 3:
                    print(f"  WARN: ingest error: {exc}", file=sys.stderr)
            processed = train_count + train_errors
            print(f"\r  ingested {processed} / {train_total}", end="", flush=True)

        print(f"\r  ingested {train_count} / {train_total}  ({train_errors} errors)")

        # ------------------------------------------------------------------
        # Phase 2: detection — score now-1w → now
        # ------------------------------------------------------------------
        print(f"\n[2/2] DETECTION from:{args.detect_from}  to:{args.detect_to}  {filter_label}")
        print(f"      ES: {es_url}/{index}")
        print(f"      API: {api_url}/score")

        detect_total = _count_window(es_url, es_api_key, index, lucene_filter, args.detect_from, args.detect_to)
        print(f"      {detect_total} documents")

        results: list[dict] = []
        score_count = 0
        score_errors = 0
        for doc in _fetch_window(
            es_url, es_api_key, index,
            lucene_filter, args.detect_from, args.detect_to,
            args.page_size, args.debug_es,
        ):
            result = _score(api_url, api_client, doc, args.explain)
            score_count += 1
            if result is None:
                score_errors += 1
            else:
                result["_timestamp"] = doc.get(TIMESTAMP_FIELD, "")
                if not args.anomalous_only or result["status"] in (labels.ANOMALOUS, labels.HIGHLY_ANOMALOUS):
                    results.append(result)
            print(f"\r  scored {score_count} / {detect_total}", end="", flush=True)

        print(f"\r  scored {score_count} / {detect_total}  ({score_errors} errors)")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    anomalous = [r for r in results if r["status"] in (labels.ANOMALOUS, labels.HIGHLY_ANOMALOUS)]

    if detect_total == 0:
        print("\n[Results] 0 events in the detection window.")
        if train_total == 0:
            print("  Training window also returned 0 events — filter likely matches nothing.")
            if lucene_filter:
                print(f"  Filter used: {lucene_filter}")
                print("  Tip: quote hyphenated values: field:\"EXAMPLE-VALUE\"")
        else:
            print(f"  Training window had {train_total} events; detection window had none.")
            print("  This usually means no data exists for that time range with this filter.")
        return

    print(f"\n[Results] {score_count} scored  |  {len(anomalous)} anomalous", end="")
    if args.anomalous_only and score_count > len(results):
        print(f"  ({score_count - len(results)} hidden by --anomalous-only)", end="")
    print()

    if not results:
        print("  No anomalous events found. Remove --anomalous-only to see all scored events.")
        return

    # Derive identifier column keys from the first result that carries them.
    id_keys: list[str] = []
    for r in results:
        if r.get("identifiers"):
            id_keys = list(r["identifiers"].keys())
            break

    ID_COL_W = 30
    col_w = {"ts": 26, "score": 7, "status": 18}
    header = (
        f"{'timestamp':<{col_w['ts']}}"
        f"{'score':>{col_w['score']}}"
        f"  {'status':<{col_w['status']}}"
    )
    for key in id_keys:
        header += f"  {key.split('.')[-1]:<{ID_COL_W}}"

    total_w = sum(col_w.values()) + max(0, len(id_keys) - 1) * 2 + len(id_keys) * (ID_COL_W + 2)
    print(f"\n{header}")
    print("-" * total_w)

    for r in results:
        ts = r.get("_timestamp", "")[:26]
        score = r.get("composite_score")
        score_str = f"{score:.1f}" if score is not None else "N/A"
        status = r.get("status", "")

        row = (
            f"{ts:<{col_w['ts']}}"
            f"{score_str:>{col_w['score']}}"
            f"  {status:<{col_w['status']}}"
        )
        ids = r.get("identifiers") or {}
        for key in id_keys:
            val = str(ids.get(key) or "-")[:ID_COL_W]
            row += f"  {val:<{ID_COL_W}}"
        print(row)
        print()

        if args.explain:
            # Collect raw feature values from every cohort (explanation always present,
            # delta=null for untrained cohorts).
            all_features: dict[str, Any] = {}
            for cohort in r.get("scores", []):
                for feat in (cohort.get("explanation") or {}).get("features") or []:
                    if feat["field"] not in all_features:
                        all_features[feat["field"]] = feat["value"]
            if all_features:
                field_w = max(len(f) + 1 for f in all_features)
                print()
                for field, value in sorted(all_features.items()):
                    print(f"    {field + ':':<{field_w}}  {value}")
                print()

            trained_weight = sum(
                c.get("weight", 0) for c in r.get("scores", [])
                if c.get("anomaly_score") is not None
            )
            all_feats = [
                feat
                for cohort in r.get("scores", [])
                for feat in ((cohort.get("explanation") or {}).get("features") or [])
                if feat.get("delta") is not None
            ][:5 * len(r.get("scores", []))]
            field_w = max((len(f["field"]) + 1 for f in all_feats), default=0)
            delta_w = max((len(f"{f['delta']:+.1f}") for f in all_feats), default=0)

            for cohort in r.get("scores", []):
                key_vals = ", ".join(
                    v for v in cohort.get("key", {}).values()
                    if v and v != "__missing__"
                )
                cohort_score = cohort.get("anomaly_score")
                key_part = f" value={key_vals}" if key_vals else ""
                if cohort_score is None:
                    print(f"    [{cohort['name']}]{key_part} insufficient_data")
                    continue
                weight = cohort.get("weight", 0)
                pct = (weight / trained_weight * 100) if trained_weight > 0 else 0
                print(f"    [{cohort['name']}]{key_part} score={cohort_score:.1f} ({pct:.0f}%)")
                expl = cohort.get("explanation")
                for feat in [f for f in (expl.get("features") or []) if f.get("delta") is not None][:5] if expl else []:
                    field_str = f"{feat['field']}:"
                    delta_str = f"{feat['delta']:+.1f}"
                    print(f"      {field_str:<{field_w}}  delta={delta_str:<{delta_w}}  value={feat['value']}")
            print("\n")


if __name__ == "__main__":
    main()
