"""Refresh the live-safe L1 snapshot for one signal date only.

This is the operational fast path for the frozen L1 v1 workflow. It computes
only the target-date L1 reversal-start rows from local daily bars and the local
research panel, then replaces that date's L1 rows in the live snapshot.

It intentionally does not read the canonical research snapshot and does not
write future, next-open, or risk labels.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
TV_ALPHA = REPO_ROOT / "examples" / "alpha_research" / "tradingview" / "vnpy_alpha"
sys.path.insert(0, str(TV_ALPHA))
sys.path.insert(0, str(REPO_ROOT))

from build_a5_signal_snapshot import (  # noqa: E402
    add_factor_and_rank_fields,
    build_event_row,
    event_id,
    load_combo_factors,
)
from run_trend_rsi_event_study import TrendRsiConfig, compute_base_columns, load_daily  # noqa: E402
from run_trend_rsi_v3_event_study import V3FilterConfig, load_panel_frames  # noqa: E402
from run_trend_rsi_v3_start_bucket_study import (  # noqa: E402
    align_context,
    build_industry_bucket_contexts,
    build_market_bucket_context,
    compute_tag_arrays,
)
from run_trend_rsi_v4_start_grade_study import (  # noqa: E402
    build_pvcorr60_context,
    collect_v4_start_events,
)
from run_trend_rsi_v5_market_playbook_study import add_playbook_columns  # noqa: E402
from scripts.alpha_research.tradingview.signal_core.buy_points import (  # noqa: E402
    BUY_POINT_TYPE_L1,
    add_buy_point_columns,
)


DEFAULT_CONFIG_PATH = "scripts/alpha_research/tradingview/configs/l1_execution_v1.json"
DEFAULT_COMBO_FACTOR_PATH = (
    "examples/alpha_research/tradingview/reports/a5_tp_hit_diagnostic_combo_candidates/"
    "combo_factor_candidates.parquet"
)
DEFAULT_LIVE_OUTPUT_PATH = "scripts/alpha_research/tradingview/data/live_signal_snapshot.parquet"
DEFAULT_CANDIDATE_SCRIPT = "scripts/alpha_research/tradingview/live/run_l1_daily_candidates.py"
SNAPSHOT_VERSION = "a5_signal_snapshot_v1"
SIGNAL_SOURCE = "trendRsiv5_A5_live_daily"
FORBIDDEN_PREFIXES = ("future_", "next_", "risk_")
FORBIDDEN_COLUMNS = {
    "suggested_entry_date",
    "suggested_entry_price",
    "tradable_flag",
    "tradable_reason",
    "execution_bucket",
    "open_executable",
    "next_open_executable",
    "next_open_selected",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Refresh live-safe L1 rows for one signal date")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--lab-path", default="lab/a_share_research")
    parser.add_argument("--target-date", "--end", dest="target_date", default=None)
    parser.add_argument("--lookback-bars", type=int, default=500)
    parser.add_argument(
        "--factor-warmup-days",
        type=int,
        default=520,
        help="Calendar-day warmup before target date for live core4/pvcorr factor ranks.",
    )
    parser.add_argument("--min-history-bars", type=int, default=400)
    parser.add_argument("--new-stock-days", type=int, default=120)
    parser.add_argument("--min-bars-panel", type=int, default=1)
    parser.add_argument("--component-column", default="in_a_share_active_ge1000")
    parser.add_argument("--combo-factor-path", default=DEFAULT_COMBO_FACTOR_PATH)
    parser.add_argument("--skip-combo-factor-file", action="store_true")
    parser.add_argument("--live-output-path", default=None)
    parser.add_argument("--live-metadata-path", default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-candidates", action="store_true")
    parser.add_argument("--candidate-include-below-threshold", action="store_true")
    parser.add_argument(
        "--live-open-source",
        choices=["none", "tencent"],
        default="none",
        help="Optional runtime open quote overlay for candidate generation.",
    )
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    """Resolve paths from the repository root unless already absolute."""
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def repo_arg(path: str | Path) -> str:
    """Return a repository-relative path string when possible."""
    resolved = repo_path(path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def normalize_date(values: Any) -> pd.Series:
    """Normalize date-like values."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def load_config(path: str) -> dict[str, Any]:
    """Load the v1 execution config."""
    with repo_path(path).open(encoding="utf-8") as file:
        return json.load(file)


def latest_panel_trade_date(lab_path: str) -> pd.Timestamp:
    """Return the latest trade date in the local research panel."""
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


def resolve_window_start(lab_path: str, target_date: pd.Timestamp, lookback_bars: int) -> pd.Timestamp:
    """Resolve the first panel trade date for the rolling live window."""
    panel_path = repo_path(lab_path) / "panel" / "research_panel_daily"
    dates: list[pd.Timestamp] = []
    for file_path in sorted(panel_path.glob("year=*/data_*.parquet")):
        frame = pd.read_parquet(file_path, columns=["trade_date"])
        if frame.empty:
            continue
        values = normalize_date(frame["trade_date"])
        values = values[(values <= target_date) & values.notna()]
        if not values.empty:
            dates.extend(pd.Timestamp(value).normalize() for value in values.unique())
    if not dates:
        raise ValueError(f"no panel trade dates <= {target_date.date()}")
    index = pd.DatetimeIndex(sorted(set(dates)))
    if len(index) <= lookback_bars:
        return pd.Timestamp(index[0]).normalize()
    return pd.Timestamp(index[-lookback_bars]).normalize()


def load_signal_day_panel(lab_path: str, target_date: pd.Timestamp) -> pd.DataFrame:
    """Load event-date panel metadata used by the live-safe snapshot."""
    panel_path = repo_path(lab_path) / "panel" / "research_panel_daily"
    file_path = panel_path / f"year={target_date.year}" / "data_0.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"missing panel parquet for {target_date.year}: {file_path}")
    columns = [
        "trade_date",
        "vt_symbol",
        "name",
        "list_date",
        "turnover_rate",
        "circ_mv",
        "total_mv",
        "raw_close",
        "up_limit",
        "down_limit",
        "is_limit_up",
        "is_limit_down",
        "has_valid_bar",
        "is_list_life_valid",
        "in_a_share_active_ge1000",
        "sw_l1_name",
    ]
    frame = pd.read_parquet(file_path, columns=columns)
    frame["signal_date"] = normalize_date(frame["trade_date"])
    frame = frame[frame["signal_date"].eq(target_date)].copy()
    if frame.empty:
        raise ValueError(f"no panel rows for {target_date.date()}")
    frame["list_date"] = pd.to_datetime(frame["list_date"], errors="coerce")
    frame["signal_age_days"] = (target_date - frame["list_date"]).dt.days
    frame["is_st"] = frame["name"].fillna("").astype(str).str.contains("ST|退")
    frame = frame.drop(columns=["trade_date"])
    return frame.drop_duplicates(["signal_date", "vt_symbol"])


def bool_from_row(row: pd.Series, column: str, default: bool = False) -> bool:
    """Read a bool-like value from a row."""
    value = row.get(column, default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return bool(value)


def join_signal_day_panel(rows: pd.DataFrame, panel: pd.DataFrame, *, new_stock_days: int) -> pd.DataFrame:
    """Attach signal-date panel fields without next-open or future labels."""
    if rows.empty:
        return rows.copy()
    out = rows.merge(panel, on=["signal_date", "vt_symbol"], how="left")
    out["signal_age_days"] = pd.to_numeric(out["signal_age_days"], errors="coerce")
    out["is_new_stock"] = out["signal_age_days"] < new_stock_days
    for column in (
        "is_st",
        "has_valid_bar",
        "is_list_life_valid",
        "is_limit_up",
        "is_limit_down",
        "in_a_share_active_ge1000",
    ):
        if column in out.columns:
            out[column] = out[column].astype("boolean").fillna(False).astype(bool)
        else:
            out[column] = False
    return out


def collect_target_l1_rows(
    lab_path: str,
    *,
    target_date: pd.Timestamp,
    window_start: pd.Timestamp,
    factor_start: pd.Timestamp,
    min_history_bars: int,
    component_column: str,
    progress_every: int,
) -> pd.DataFrame:
    """Compute live-safe L1 rows for a single target date."""
    lab = repo_path(lab_path)
    daily_path = lab / "daily"
    panel_path = lab / "panel" / "research_panel_daily"
    config = TrendRsiConfig()
    filter_config = V3FilterConfig(use_vol_quality_filter=False)

    print("loading live market contexts", flush=True)
    panel = load_panel_frames(panel_path, window_start, target_date)
    if panel.empty:
        raise ValueError(f"no panel context rows for {window_start.date()}..{target_date.date()}")
    market_context = build_market_bucket_context(panel, config)
    industry_contexts, symbol_industry = build_industry_bucket_contexts(
        panel,
        market_context,
        config,
        filter_config,
    )
    print(
        f"loaded context panel_rows={len(panel)} market_days={len(market_context)} "
        f"industries={len(industry_contexts)}",
        flush=True,
    )

    pvcorr_context = build_pvcorr60_context(
        panel_path,
        factor_start,
        target_date,
        component_column,
    )
    pvcorr_context = pvcorr_context.rename(columns={"signal_date_key": "signal_date"})
    pvcorr_context = pvcorr_context[pvcorr_context["signal_date"].eq(target_date)].copy()

    rows: list[dict[str, Any]] = []
    considered = 0
    skipped_short_history = 0
    files = sorted(daily_path.glob("*.parquet"))
    for file_path in files:
        vt_symbol = file_path.stem
        daily = load_daily(file_path, window_start, target_date)
        if len(daily) < min_history_bars:
            skipped_short_history += 1
            continue
        if daily.empty or pd.Timestamp(daily["datetime"].max()).normalize() < target_date:
            continue

        data = compute_base_columns(daily, config)
        industry_name = symbol_industry.get(vt_symbol)
        tags = compute_tag_arrays(
            data,
            align_context(data, market_context),
            align_context(data, industry_contexts.get(industry_name, pd.DataFrame())),
            filter_config,
        )
        start_events = collect_v4_start_events(vt_symbol, data, config, filter_config, tags)
        if not start_events:
            considered += 1
            if progress_every and considered % progress_every == 0:
                print(f"progress symbols={considered}/{len(files)} l1_rows={len(rows)}", flush=True)
            continue

        target_events = [
            event
            for event in start_events
            if pd.Timestamp(event["signal_date"]).normalize() == target_date
        ]
        if target_events:
            parents = add_playbook_columns(pd.DataFrame(target_events), "v5_no_pvcorr", use_pvcorr=False)
            parents = parents[
                parents["passes_v4_default"].astype(bool)
                & parents["v5_action_bucket"].astype(str).eq("A5_normal")
            ].copy()
            for parent_row in parents.to_dict("records"):
                signal_idx = int(parent_row["signal_idx"])
                row = build_event_row(
                    vt_symbol,
                    data,
                    tags,
                    signal_idx=signal_idx,
                    buy_point_type=BUY_POINT_TYPE_L1,
                    parent=parent_row,
                    parent_start_date=target_date,
                    confirmed=False,
                )
                row["snapshot_version"] = SNAPSHOT_VERSION
                row["signal_source"] = SIGNAL_SOURCE
                row["history_bars"] = int(len(data))
                row["history_window_start"] = window_start
                row["history_min_required"] = int(min_history_bars)
                row["history_bucket"] = "normal_history" if len(data) >= min_history_bars else "short_history"
                row["live_refresh_mode"] = "l1_target_date"
                row["live_refresh_target_date"] = target_date
                row["event_id"] = event_id(vt_symbol, target_date, BUY_POINT_TYPE_L1, target_date)
                rows.append(row)

        considered += 1
        if progress_every and considered % progress_every == 0:
            print(
                f"progress symbols={considered}/{len(files)} l1_rows={len(rows)} "
                f"short_history_skips={skipped_short_history}",
                flush=True,
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["signal_date"] = normalize_date(out["signal_date"])
    out["parent_start_date"] = normalize_date(out["parent_start_date"])
    out = add_buy_point_columns(out)
    out = out.merge(
        pvcorr_context[
            [
                "signal_date",
                "vt_symbol",
                "combo_core4_control",
                "combo_core4_pvcorr60_15pct",
                "pvcorr60_bucket",
                "pvcorr60_bucket_score",
            ]
        ],
        on=["signal_date", "vt_symbol"],
        how="left",
    )
    out["parent_combo_core4_control"] = out["combo_core4_control"]
    out["parent_combo_core4_pvcorr60_15pct"] = out["combo_core4_pvcorr60_15pct"]
    out["parent_pvcorr60_bucket"] = out["pvcorr60_bucket"].fillna("Q0_no_combo")
    out["parent_pvcorr60_bucket_score"] = out["pvcorr60_bucket_score"]
    return out


def forbidden_columns(columns: list[str]) -> list[str]:
    """Return columns forbidden in the live-safe snapshot."""
    return sorted(
        column
        for column in columns
        if column in FORBIDDEN_COLUMNS or column.startswith(FORBIDDEN_PREFIXES)
    )


def validate_live_safe(frame: pd.DataFrame) -> None:
    """Validate live-safe snapshot fields."""
    blocked = forbidden_columns(list(frame.columns))
    if blocked:
        raise ValueError(f"live snapshot contains forbidden columns: {blocked}")
    if "event_id" in frame.columns and int(frame.duplicated(["event_id"]).sum()):
        raise ValueError("live snapshot contains duplicate event_id rows")


def align_columns(existing: pd.DataFrame, new_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align existing and new row columns before concatenation."""
    columns = list(existing.columns)
    for column in new_rows.columns:
        if column not in columns:
            columns.append(column)
    return existing.reindex(columns=columns), new_rows.reindex(columns=columns)


def load_existing_live(path: Path) -> pd.DataFrame:
    """Load the existing live snapshot or an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["signal_date"] = normalize_date(frame["signal_date"])
    return add_buy_point_columns(frame)


def replace_target_l1_rows(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    *,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Replace target-date L1 rows while preserving other dates and signal types."""
    if existing.empty:
        out = new_rows.copy()
    else:
        mask = existing["signal_date"].eq(target_date) & existing["buy_point_type"].eq(BUY_POINT_TYPE_L1)
        kept = existing[~mask].copy()
        kept, aligned_new = align_columns(kept, new_rows)
        out = pd.concat([kept, aligned_new], ignore_index=True)
    if out.empty:
        return out
    out["signal_date"] = normalize_date(out["signal_date"])
    out = add_buy_point_columns(out)
    out = out.sort_values(["signal_date", "vt_symbol", "buy_point_type", "event_id"]).reset_index(drop=True)
    validate_live_safe(out)
    return out


def write_metadata(
    metadata_path: Path,
    snapshot: pd.DataFrame,
    *,
    source: str,
    target_date: pd.Timestamp,
    lookback_bars: int,
    factor_warmup_days: int,
    min_history_bars: int,
    target_l1_rows: int,
) -> dict[str, Any]:
    """Write live snapshot metadata."""
    metadata = {
        "kind": "live_signal_snapshot",
        "source": source,
        "refresh_mode": "l1_target_date_live_only",
        "target_date": str(target_date.date()),
        "lookback_bars": int(lookback_bars),
        "factor_warmup_days": int(factor_warmup_days),
        "min_history_bars": int(min_history_bars),
        "target_l1_rows": int(target_l1_rows),
        "rows": int(len(snapshot)),
        "columns": int(snapshot.shape[1]),
        "signal_date_min": str(snapshot["signal_date"].min().date()) if not snapshot.empty else None,
        "signal_date_max": str(snapshot["signal_date"].max().date()) if not snapshot.empty else None,
        "buy_point_counts": {
            str(key): int(value)
            for key, value in snapshot.groupby("buy_point_type", sort=True).size().to_dict().items()
        }
        if not snapshot.empty
        else {},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def run_candidate_script(
    *,
    config_path: str,
    snapshot_path: Path,
    target_date: pd.Timestamp,
    include_below_threshold: bool,
    live_open_source: str,
    dry_run: bool,
) -> None:
    """Run the daily candidate output script."""
    command = [
        sys.executable,
        repo_arg(DEFAULT_CANDIDATE_SCRIPT),
        "--config-path",
        config_path,
        "--snapshot-kind",
        "live",
        "--snapshot-path",
        repo_arg(snapshot_path),
        "--signal-date",
        str(target_date.date()),
        "--live-open-source",
        live_open_source,
    ]
    if include_below_threshold:
        command.append("--include-below-threshold")
    print("$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    """Refresh one target date's live L1 rows."""
    args = parse_args()
    config = load_config(args.config_path)
    target_date = (
        pd.Timestamp(args.target_date).normalize()
        if args.target_date
        else latest_panel_trade_date(args.lab_path)
    )
    live_output_path = repo_path(
        args.live_output_path
        or config.get("live_signal_snapshot")
        or config.get("source_snapshot")
        or DEFAULT_LIVE_OUTPUT_PATH
    )
    live_metadata_path = (
        repo_path(args.live_metadata_path)
        if args.live_metadata_path
        else live_output_path.with_suffix(".metadata.json")
    )
    window_start = resolve_window_start(args.lab_path, target_date, args.lookback_bars)
    factor_start = target_date - pd.Timedelta(days=args.factor_warmup_days)
    print(
        "l1_live_refresh_context "
        f"target_date={target_date.date()} window_start={window_start.date()} "
        f"factor_start={factor_start.date()} lookback_bars={args.lookback_bars} "
        f"min_history_bars={args.min_history_bars}",
        flush=True,
    )

    if args.dry_run:
        print(f"would_write live_snapshot={repo_arg(live_output_path)} metadata={repo_arg(live_metadata_path)}")
        if args.run_candidates:
            run_candidate_script(
                config_path=args.config_path,
                snapshot_path=live_output_path,
                target_date=target_date,
                include_below_threshold=args.candidate_include_below_threshold,
                live_open_source=args.live_open_source,
                dry_run=True,
            )
        return

    l1_rows = collect_target_l1_rows(
        args.lab_path,
        target_date=target_date,
        window_start=window_start,
        factor_start=factor_start,
        min_history_bars=args.min_history_bars,
        component_column=args.component_column,
        progress_every=args.progress_every,
    )
    if not l1_rows.empty:
        combo_factors = (
            pd.DataFrame(columns=["datetime", "vt_symbol"])
            if args.skip_combo_factor_file
            else load_combo_factors(repo_path(args.combo_factor_path))
        )
        l1_rows = add_factor_and_rank_fields(l1_rows, combo_factors)
        panel = load_signal_day_panel(args.lab_path, target_date)
        l1_rows = join_signal_day_panel(l1_rows, panel, new_stock_days=args.new_stock_days)

    existing = load_existing_live(live_output_path)
    live = replace_target_l1_rows(existing, l1_rows, target_date=target_date)
    live_output_path.parent.mkdir(parents=True, exist_ok=True)
    live.to_parquet(live_output_path, index=False)
    metadata = write_metadata(
        live_metadata_path,
        live,
        source="local_daily_and_research_panel",
        target_date=target_date,
        lookback_bars=args.lookback_bars,
        factor_warmup_days=args.factor_warmup_days,
        min_history_bars=args.min_history_bars,
        target_l1_rows=len(l1_rows),
    )

    print(
        "wrote "
        f"{repo_arg(live_output_path)} rows={metadata['rows']} "
        f"target_l1_rows={len(l1_rows)} "
        f"date_range={metadata['signal_date_min']}..{metadata['signal_date_max']}",
        flush=True,
    )
    print(f"wrote {repo_arg(live_metadata_path)}", flush=True)

    if args.run_candidates:
        if l1_rows.empty:
            print(f"no_l1_candidates target_date={target_date.date()} candidates_skipped=true", flush=True)
        else:
            run_candidate_script(
                config_path=args.config_path,
                snapshot_path=live_output_path,
                target_date=target_date,
                include_below_threshold=args.candidate_include_below_threshold,
                live_open_source=args.live_open_source,
                dry_run=False,
            )


if __name__ == "__main__":
    main()
