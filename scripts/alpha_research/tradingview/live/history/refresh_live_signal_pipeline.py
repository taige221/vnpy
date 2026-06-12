"""Refresh the live-safe L1 signal snapshot directly from local daily data.

This is the operational refresh entrypoint. It rebuilds the upstream TrendRSI
event tables in an ignored live work directory, then publishes only the
live-safe snapshot consumed by daily candidate generation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CONFIG_PATH = "scripts/alpha_research/tradingview/configs/l1_execution_v1.json"
DEFAULT_WORK_DIR = "scripts/alpha_research/tradingview/data/live_refresh_work"
DEFAULT_START_DATE = "2020-01-01"

V4_START_GRADE_SCRIPT = (
    "examples/alpha_research/tradingview/vnpy_alpha/"
    "run_trend_rsi_v4_start_grade_study.py"
)
V5_PLAYBOOK_SCRIPT = (
    "examples/alpha_research/tradingview/vnpy_alpha/"
    "run_trend_rsi_v5_market_playbook_study.py"
)
A5_SNAPSHOT_SCRIPT = (
    "examples/alpha_research/tradingview/vnpy_alpha/"
    "build_a5_signal_snapshot.py"
)
LIVE_SNAPSHOT_SCRIPT = (
    "scripts/alpha_research/tradingview/live/history/"
    "build_live_signal_snapshot.py"
)
CANDIDATE_SCRIPT = (
    "scripts/alpha_research/tradingview/live/"
    "run_l1_daily_candidates.py"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Refresh live-safe TrendRSI/L1 snapshot from local daily bars"
    )
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--lab-path", default="lab/a_share_research")
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--live-output-path",
        default=None,
        help="Published live snapshot path. Defaults to config live_signal_snapshot.",
    )
    parser.add_argument(
        "--live-metadata-path",
        default=None,
        help="Published live metadata path. Defaults to live-output-path with .metadata.json.",
    )
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument(
        "--end",
        default=None,
        help="Signal refresh end date. Defaults to the latest panel trade date.",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="Comma-separated years for A5 snapshot build. Defaults to start..end years.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit symbols for smoke tests.")
    parser.add_argument("--skip-pvcorr", action="store_true", help="Skip pvcorr proxy in v4 refresh.")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--run-candidates",
        action="store_true",
        help="Run daily candidate generation after publishing the live snapshot.",
    )
    parser.add_argument(
        "--candidate-signal-date",
        default=None,
        help="Candidate signal date. Defaults to --end/latest panel date.",
    )
    parser.add_argument(
        "--candidate-include-below-threshold",
        action="store_true",
        help="Allow candidate report output when the v1 day filter is below threshold.",
    )
    parser.add_argument(
        "--live-open-source",
        choices=["none", "tencent"],
        default="none",
        help="Optional runtime open quote overlay for candidate generation.",
    )
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    """Resolve a path from the repository root unless already absolute."""
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def repo_arg(path: str | Path) -> str:
    """Return a path string relative to the repo when possible."""
    resolved = repo_path(path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def load_config(path: str) -> dict[str, Any]:
    """Load the v1 execution config."""
    with repo_path(path).open(encoding="utf-8") as file:
        return json.load(file)


def latest_panel_trade_date(lab_path: str) -> pd.Timestamp:
    """Return the latest trade date available in the local research panel."""
    panel_path = repo_path(lab_path) / "panel" / "research_panel_daily"
    files = sorted(panel_path.glob("year=*/data_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no panel parquet files under {panel_path}")

    latest: pd.Timestamp | None = None
    for file_path in files:
        frame = pd.read_parquet(file_path, columns=["trade_date"])
        if frame.empty:
            continue
        current = pd.to_datetime(frame["trade_date"], errors="coerce").max()
        if pd.isna(current):
            continue
        current = pd.Timestamp(current).normalize()
        latest = current if latest is None or current > latest else latest

    if latest is None:
        raise ValueError(f"no valid trade_date values under {panel_path}")
    return latest


def derive_years(start: str, end: str) -> str:
    """Build a comma-separated inclusive year list."""
    start_year = pd.Timestamp(start).year
    end_year = pd.Timestamp(end).year
    return ",".join(str(year) for year in range(start_year, end_year + 1))


def command_text(command: list[str]) -> str:
    """Render a command for logs."""
    return " ".join(command)


def run_command(command: list[str], *, dry_run: bool) -> None:
    """Run one subprocess from the repository root."""
    print(f"$ {command_text(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def summarize_live_snapshot(path: Path) -> None:
    """Print max signal-date diagnostics for the live snapshot."""
    frame = pd.read_parquet(path, columns=["signal_date", "buy_point_type", "vt_symbol"])
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce").dt.normalize()
    max_date = frame["signal_date"].max()
    print(
        "live_snapshot_summary "
        f"rows={len(frame)} max_signal_date={max_date.date()}",
        flush=True,
    )
    max_by_type = frame.groupby("buy_point_type")["signal_date"].max().sort_index()
    for buy_point_type, signal_date in max_by_type.items():
        rows = frame[
            frame["buy_point_type"].eq(buy_point_type)
            & frame["signal_date"].eq(signal_date)
        ]
        print(
            f"live_snapshot_type_max {buy_point_type} "
            f"date={signal_date.date()} rows={len(rows)}",
            flush=True,
        )


def main() -> None:
    """Run the live refresh pipeline."""
    args = parse_args()
    config = load_config(args.config_path)
    end_date = args.end or str(latest_panel_trade_date(args.lab_path).date())
    years = args.years or derive_years(args.start, end_date)

    work_dir = repo_path(args.work_dir)
    v4_dir = work_dir / "v4_start_grade"
    v5_dir = work_dir / "v5_playbook"
    a5_dir = work_dir / "a5_signal_snapshot"
    live_snapshot_path = repo_path(
        args.live_output_path
        or config.get("live_signal_snapshot")
        or config.get("source_snapshot")
        or "scripts/alpha_research/tradingview/data/live_signal_snapshot.parquet"
    )
    live_metadata_path = (
        repo_path(args.live_metadata_path)
        if args.live_metadata_path
        else live_snapshot_path.with_suffix(".metadata.json")
    )

    if not args.dry_run:
        v4_dir.mkdir(parents=True, exist_ok=True)
        v5_dir.mkdir(parents=True, exist_ok=True)
        a5_dir.mkdir(parents=True, exist_ok=True)
        live_snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    v4_command = [
        sys.executable,
        repo_arg(V4_START_GRADE_SCRIPT),
        "--lab-path",
        args.lab_path,
        "--start",
        args.start,
        "--end",
        end_date,
        "--output-dir",
        repo_arg(v4_dir),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.limit is not None:
        v4_command.extend(["--limit", str(args.limit)])
    if args.skip_pvcorr:
        v4_command.append("--skip-pvcorr")

    v5_command = [
        sys.executable,
        repo_arg(V5_PLAYBOOK_SCRIPT),
        "--events-path",
        repo_arg(v4_dir / "trendRsiv4_start_grade_events.csv"),
        "--output-dir",
        repo_arg(v5_dir),
    ]

    a5_command = [
        sys.executable,
        repo_arg(A5_SNAPSHOT_SCRIPT),
        "--lab-path",
        args.lab_path,
        "--event-path",
        repo_arg(v5_dir / "trendRsiv5_market_playbook_events.csv"),
        "--output-dir",
        repo_arg(a5_dir),
        "--years",
        years,
        "--progress-every",
        str(args.progress_every),
    ]

    live_command = [
        sys.executable,
        repo_arg(LIVE_SNAPSHOT_SCRIPT),
        "--config-path",
        args.config_path,
        "--research-snapshot-path",
        repo_arg(a5_dir / "a5_signal_snapshot.parquet"),
        "--output-path",
        repo_arg(live_snapshot_path),
        "--metadata-path",
        repo_arg(live_metadata_path),
    ]

    print(
        "refresh_context "
        f"start={args.start} end={end_date} years={years} "
        f"work_dir={repo_arg(work_dir)}",
        flush=True,
    )
    for command in (v4_command, v5_command, a5_command, live_command):
        run_command(command, dry_run=args.dry_run)

    if not args.dry_run:
        summarize_live_snapshot(live_snapshot_path)

    if args.run_candidates:
        candidate_date = args.candidate_signal_date or end_date
        candidate_command = [
            sys.executable,
            repo_arg(CANDIDATE_SCRIPT),
            "--config-path",
            args.config_path,
            "--snapshot-kind",
            "live",
            "--signal-date",
            candidate_date,
            "--live-open-source",
            args.live_open_source,
        ]
        if args.candidate_include_below_threshold:
            candidate_command.append("--include-below-threshold")
        run_command(candidate_command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
