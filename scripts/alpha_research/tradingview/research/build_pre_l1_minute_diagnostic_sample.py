"""Build a matched PRE-L1 positive/negative minute diagnostic sample.

The purpose is candidate-feature diagnosis, not a trading strategy backtest.
Positive samples are all PRE-L1 rows where ``tomorrow_l1`` is true.  Negative
samples are matched by entry date, PRE-L1 type, and score bucket, so minute
features can be compared against a local control group instead of the full
class-imbalanced PRE-L1 pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.alpha_research.tradingview.research.fetch_pre_l1_baostock_minutes import (
    DEFAULT_EVENTS_PATH,
    DEFAULT_OUTPUT_DIR,
    build_requests,
    expand_required_dates,
    repo_path,
)


DEFAULT_REPORT_DIR = "scripts/alpha_research/tradingview/reports/pre_l1_minute_diagnostic_sample"
DEFAULT_NEGATIVE_OUTPUT_DIR = "lab/a_share_research/minute/baostock_5m_pre_l1_negative_sample"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build PRE-L1 minute diagnostic sample")
    parser.add_argument("--events-path", default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--negative-output-dir", default=DEFAULT_NEGATIVE_OUTPUT_DIR)
    parser.add_argument("--positive-output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=20260612)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--hold-trading-days", type=int, default=1)
    parser.add_argument("--require-open-executable", action="store_true")
    return parser.parse_args()


def normalize_date(values: Any) -> pd.Series:
    """Normalize date-like values."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def load_candidate_events(path: Path, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Load PRE-L1 candidate events."""
    events = pd.read_parquet(path)
    events["signal_date"] = normalize_date(events["signal_date"])
    events["entry_date"] = normalize_date(events["entry_date"])
    events = events[events["has_entry"].fillna(False).astype(bool)].copy()
    events = events[events["vt_symbol"].astype(str).str.endswith((".SSE", ".SZSE"))].copy()
    if start_date:
        events = events[events["entry_date"] >= pd.Timestamp(start_date)]
    if end_date:
        events = events[events["entry_date"] <= pd.Timestamp(end_date)]
    events = events.sort_values(["entry_date", "vt_symbol", "pre_l1_type"]).reset_index(drop=True)
    events.insert(0, "pre_l1_event_id", range(len(events)))
    return events


def add_match_buckets(events: pd.DataFrame) -> pd.DataFrame:
    """Add strata columns for matched sampling."""
    out = events.copy()
    out["score_bucket"] = pd.cut(
        out["pre_l1_score"],
        bins=[-1, 20, 40, 60, 80, 100],
        labels=["0_20", "21_40", "41_60", "61_80", "81_100"],
    )
    out["entry_date_key"] = out["entry_date"].dt.strftime("%Y-%m-%d")
    out["match_key"] = (
        out["entry_date_key"].astype(str)
        + "|"
        + out["pre_l1_type"].astype(str)
        + "|"
        + out["score_bucket"].astype(str)
    )
    return out


def sample_negatives(events: pd.DataFrame, negative_ratio: int, random_state: int) -> pd.DataFrame:
    """Sample matched negatives from strata that contain positives."""
    positives = events[events["tomorrow_l1"].fillna(False).astype(bool)].copy()
    negatives = events[~events["tomorrow_l1"].fillna(False).astype(bool)].copy()
    if positives.empty:
        raise ValueError("no positive tomorrow_l1 rows found")
    sampled: list[pd.DataFrame] = []
    rng = random_state
    positive_counts = positives.groupby("match_key", observed=False).size()
    for match_key, positive_count in positive_counts.items():
        pool = negatives[negatives["match_key"] == match_key]
        if pool.empty:
            continue
        n = min(len(pool), int(positive_count) * negative_ratio)
        sampled.append(pool.sample(n=n, random_state=rng))
        rng += 1
    if not sampled:
        raise ValueError("no matched negative rows sampled")
    return pd.concat(sampled, ignore_index=True)


def build_cohort(events: pd.DataFrame, negative_ratio: int, random_state: int, require_open_executable: bool) -> pd.DataFrame:
    """Build positive + matched-negative cohort."""
    data = add_match_buckets(events)
    if require_open_executable:
        data = data[data["open_executable"].fillna(False).astype(bool)].copy()
    positives = data[data["tomorrow_l1"].fillna(False).astype(bool)].copy()
    negatives = sample_negatives(data, negative_ratio, random_state)
    positives["sample_label"] = "positive_tomorrow_l1"
    negatives["sample_label"] = "negative_matched"
    cohort = pd.concat([positives, negatives], ignore_index=True)
    cohort = cohort.sort_values(["entry_date", "sample_label", "vt_symbol"]).reset_index(drop=True)
    cohort.insert(0, "minute_sample_id", range(len(cohort)))
    return cohort


def concrete_csv_path(path: Path) -> Path:
    """Return the concrete CSV path for a request path."""
    return path.with_suffix(".csv")


def write_negative_fetch_plan(cohort: pd.DataFrame, output_dir: Path, hold_trading_days: int) -> tuple[Path, pd.DataFrame]:
    """Write Baostock fetch plan for matched negative rows only."""
    negatives = cohort[cohort["sample_label"].eq("negative_matched")].copy()
    required_dates = expand_required_dates(negatives, hold_trading_days=hold_trading_days)
    requests = build_requests(required_dates, output_dir)
    rows = []
    for request in requests:
        csv_path = concrete_csv_path(request.output_path)
        rows.append(
            {
                "vt_symbol": request.vt_symbol,
                "bs_code": request.bs_code,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "output_path": str(request.output_path),
                "csv_path": str(csv_path),
                "exists": csv_path.exists(),
            }
        )
    plan = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "fetch_plan_pre_l1_negative_sample.csv"
    plan.to_csv(plan_path, index=False)
    return plan_path, plan


def write_outputs(cohort: pd.DataFrame, plan: pd.DataFrame, plan_path: Path, report_dir: Path, args: argparse.Namespace) -> None:
    """Write cohort and metadata outputs."""
    report_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(report_dir / "pre_l1_minute_diagnostic_cohort.parquet", index=False)
    cohort.to_csv(report_dir / "pre_l1_minute_diagnostic_cohort.csv", index=False)
    summary = (
        cohort.groupby(["sample_label", "score_bucket", "pre_l1_type"], dropna=False, observed=False)
        .size()
        .reset_index(name="events")
    )
    summary.to_csv(report_dir / "pre_l1_minute_diagnostic_cohort_summary.csv", index=False)
    metadata = {
        "events": len(cohort),
        "positive_events": int(cohort["sample_label"].eq("positive_tomorrow_l1").sum()),
        "negative_events": int(cohort["sample_label"].eq("negative_matched").sum()),
        "negative_ratio": args.negative_ratio,
        "random_state": args.random_state,
        "hold_trading_days": args.hold_trading_days,
        "require_open_executable": bool(args.require_open_executable),
        "negative_fetch_plan": str(plan_path),
        "negative_requests": len(plan),
        "negative_pending_requests": int((~plan["exists"]).sum()) if not plan.empty else 0,
    }
    (report_dir / "pre_l1_minute_diagnostic_cohort.metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Run the cohort builder."""
    args = parse_args()
    events = load_candidate_events(repo_path(args.events_path), args.start_date, args.end_date)
    cohort = build_cohort(events, args.negative_ratio, args.random_state, args.require_open_executable)
    plan_path, plan = write_negative_fetch_plan(
        cohort,
        repo_path(args.negative_output_dir),
        args.hold_trading_days,
    )
    write_outputs(cohort, plan, plan_path, repo_path(args.report_dir), args)
    print(
        "pre_l1_minute_diagnostic_sample "
        f"events={len(cohort)} positives={cohort['sample_label'].eq('positive_tomorrow_l1').sum()} "
        f"negatives={cohort['sample_label'].eq('negative_matched').sum()} "
        f"negative_requests={len(plan)} pending={int((~plan['exists']).sum()) if not plan.empty else 0} "
        f"plan={plan_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
