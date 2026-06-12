"""Backtest delayed 5-minute confirmation entries for PRE-L1 tomorrow-L1 events.

The companion feature study shows whether early intraday bars contain useful
information.  This script checks the tradable version: if a condition is only
known after a 5/15/30-minute bar closes, entry happens at the next 5-minute
bar's open, and future MFE/MAE/TP labels only use bars after that entry.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from scripts.alpha_research.tradingview.research.run_pre_l1_minute_feature_study import (
    DEFAULT_EVENTS_PATH,
    DEFAULT_PLAN_PATH,
    load_events,
    repo_path,
    safe_return,
)


DEFAULT_OUTPUT_DIR = "scripts/alpha_research/tradingview/reports/pre_l1_minute_delayed_entry_study"
DEFAULT_MINUTE_PANEL_PATH = "lab/a_share_research/minute/baostock_5m/tomorrow_l1_5m_panel.parquet"
TRADING_DAY_BARS = 48


@dataclass(frozen=True)
class CandidateSpec:
    """One delayed-entry candidate definition."""

    name: str
    confirm_bars: int
    min_close_return: float | None = None
    max_close_return: float | None = None
    min_mae: float | None = None
    min_mfe: float | None = None
    gap_min: float | None = -0.03
    gap_max: float | None = 0.03
    require_open_executable: bool = True
    entry_mode: str = "next_open"


CANDIDATES = [
    CandidateSpec(
        name="open_all",
        confirm_bars=0,
        gap_min=None,
        gap_max=None,
        require_open_executable=False,
        entry_mode="open",
    ),
    CandidateSpec(
        name="open_exec_gap_-3_+3",
        confirm_bars=0,
        entry_mode="open",
    ),
    CandidateSpec(
        name="5m_green_next_open",
        confirm_bars=1,
        min_close_return=0.0,
    ),
    CandidateSpec(
        name="5m_strong_1pct_next_open",
        confirm_bars=1,
        min_close_return=0.01,
    ),
    CandidateSpec(
        name="15m_green_next_open",
        confirm_bars=3,
        min_close_return=0.0,
    ),
    CandidateSpec(
        name="30m_green_next_open",
        confirm_bars=6,
        min_close_return=0.0,
    ),
    CandidateSpec(
        name="30m_avoid_weak_next_open",
        confirm_bars=6,
        min_close_return=-0.005,
        min_mae=-0.02,
    ),
    CandidateSpec(
        name="30m_strong_1pct_next_open",
        confirm_bars=6,
        min_close_return=0.01,
    ),
    CandidateSpec(
        name="30m_mfe_3pct_next_open",
        confirm_bars=6,
        min_mfe=0.03,
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Study PRE-L1 delayed 5m confirmation entries")
    parser.add_argument("--events-path", default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--plan-path", default=DEFAULT_PLAN_PATH)
    parser.add_argument("--minute-panel-path", default=DEFAULT_MINUTE_PANEL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--storage-format", choices=["csv"], default="csv")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--tp-levels", default="0.03,0.05")
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--rebuild-minute-panel", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def concrete_csv_path(output_path: str) -> Path:
    """Return the concrete CSV cache path from a plan output path."""
    return repo_path(output_path).with_suffix(".csv")


def build_minute_panel_from_plan(plan_path: Path, panel_path: Path, progress_every: int) -> pd.DataFrame:
    """Build a consolidated minute panel from the full fetch plan."""
    plan = pd.read_csv(plan_path)
    paths = [concrete_csv_path(str(value)) for value in plan["output_path"]]
    paths = [path for path in paths if path.exists() and path.stat().st_size > 80]
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    batches: list[pl.DataFrame] = []
    progress_every = max(1, progress_every)
    for start in range(0, len(paths), progress_every):
        chunk = paths[start : start + progress_every]
        frames = [pl.read_csv(path) for path in chunk]
        if frames:
            batches.append(pl.concat(frames, how="vertical_relaxed"))
        print(
            f"minute_panel_build progress files={min(start + len(chunk), len(paths))}/{len(paths)}",
            flush=True,
        )
    if not batches:
        raise RuntimeError(f"no minute CSV files found from {plan_path}")
    panel = pl.concat(batches, how="vertical_relaxed")
    panel.write_parquet(panel_path)
    metadata = {
        "plan_path": str(plan_path),
        "panel_path": str(panel_path),
        "csv_files": len(paths),
        "rows": panel.height,
    }
    panel_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"minute_panel_build wrote {panel_path} rows={panel.height}", flush=True)
    return panel.to_pandas()


def load_minute_panel(plan_path: Path, panel_path: Path, rebuild: bool, progress_every: int) -> pd.DataFrame:
    """Load or build the consolidated minute panel."""
    metadata_path = panel_path.with_suffix(".metadata.json")
    if rebuild or not panel_path.exists() or not metadata_path.exists():
        panel = build_minute_panel_from_plan(plan_path, panel_path, progress_every)
    else:
        panel = pd.read_parquet(panel_path)
    if panel.empty:
        return panel
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in panel.columns:
            panel[column] = pd.to_numeric(panel[column], errors="coerce")
    time_text = panel["time"].astype(str).str.zfill(17)
    panel["minute_hhmm"] = time_text.str.slice(8, 12)
    panel["minute_dt"] = pd.to_datetime(
        panel["date"].dt.strftime("%Y-%m-%d") + " " + panel["minute_hhmm"].astype(str),
        format="%Y-%m-%d %H%M",
        errors="coerce",
    )
    return panel.sort_values(["vt_symbol", "date", "time"]).reset_index(drop=True)


def event_price_ratio(row: pd.Series, minute_open: float) -> float | None:
    """Return the daily-to-minute price ratio used for scaled comparisons."""
    daily_open = row.get("entry_open")
    if pd.isna(daily_open) or float(daily_open) <= 0:
        return None
    return minute_open / float(daily_open)


def candidate_passes(spec: CandidateSpec, row: pd.Series, entry_day: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
    """Check whether a candidate passes using only bars up to confirmation."""
    if entry_day.empty:
        return False, {}
    open_executable = row.get("open_executable")
    if spec.require_open_executable and not bool(open_executable):
        return False, {}
    gap = row.get("entry_gap_pct")
    if spec.gap_min is not None and (pd.isna(gap) or float(gap) < spec.gap_min):
        return False, {}
    if spec.gap_max is not None and (pd.isna(gap) or float(gap) > spec.gap_max):
        return False, {}

    if spec.confirm_bars <= 0:
        window = entry_day.head(1)
    else:
        if len(entry_day) <= spec.confirm_bars:
            return False, {}
        window = entry_day.head(spec.confirm_bars)

    minute_open = float(entry_day.iloc[0]["open"])
    close_return = safe_return(window.iloc[-1]["close"], minute_open)
    mfe = safe_return(window["high"].max(), minute_open)
    mae = safe_return(window["low"].min(), minute_open)
    if spec.min_close_return is not None and (close_return is None or close_return <= spec.min_close_return):
        return False, {}
    if spec.max_close_return is not None and (close_return is None or close_return > spec.max_close_return):
        return False, {}
    if spec.min_mae is not None and (mae is None or mae <= spec.min_mae):
        return False, {}
    if spec.min_mfe is not None and (mfe is None or mfe < spec.min_mfe):
        return False, {}

    ratio = event_price_ratio(row, minute_open)
    signal_high = row.get("signal_high")
    signal_low = row.get("signal_low")
    if ratio is not None:
        if pd.notna(signal_high):
            signal_high = float(signal_high) * ratio
        if pd.notna(signal_low):
            signal_low = float(signal_low) * ratio
    metadata = {
        "confirm_time": window.iloc[-1].get("minute_hhmm"),
        "confirm_close_return": close_return,
        "confirm_mfe": mfe,
        "confirm_mae": mae,
        "confirm_break_signal_high": bool(pd.notna(signal_high) and window["high"].max() >= signal_high),
        "confirm_hold_signal_low": bool(pd.notna(signal_low) and window["low"].min() >= signal_low),
    }
    return True, metadata


def entry_index_for_spec(spec: CandidateSpec) -> int | None:
    """Return the bar index used for execution."""
    if spec.entry_mode == "open":
        return 0
    if spec.entry_mode == "next_open":
        return spec.confirm_bars
    raise ValueError(f"unsupported entry_mode: {spec.entry_mode}")


def select_horizon_path(after_entry: pd.DataFrame, entry_date: pd.Timestamp, horizon_days: int) -> pd.DataFrame:
    """Select rows from entry through the requested trading-day horizon."""
    if after_entry.empty:
        return after_entry
    dates = [pd.Timestamp(value).normalize() for value in after_entry["date"].dropna().unique()]
    dates = sorted(date for date in dates if date >= entry_date)
    if not dates:
        return after_entry.iloc[0:0]
    keep_dates = set(dates[: horizon_days + 1])
    return after_entry[after_entry["date"].isin(keep_dates)].copy()


def first_tp_hit(path: pd.DataFrame, entry_price: float, tp_level: float) -> tuple[bool, int | None, float | None]:
    """Return whether target was hit, plus bar and trading-day offsets."""
    if path.empty or entry_price <= 0:
        return False, None, None
    target = entry_price * (1.0 + tp_level)
    hit = path[path["high"] >= target]
    if hit.empty:
        return False, None, None
    first_index = int(hit.index[0])
    bar_offset = first_index - int(path.index[0])
    day_offset = bar_offset / TRADING_DAY_BARS
    return True, bar_offset, day_offset


def path_metrics(path: pd.DataFrame, entry_price: float, tp_levels: list[float], fee: float, prefix: str) -> dict[str, Any]:
    """Calculate return, MFE/MAE, and TP labels for one horizon path."""
    out: dict[str, Any] = {
        f"{prefix}_has_path": False,
    }
    if path.empty or entry_price <= 0:
        return out
    fixed_return = safe_return(path.iloc[-1]["close"], entry_price)
    mfe = safe_return(path["high"].max(), entry_price)
    mae = safe_return(path["low"].min(), entry_price)
    out.update(
        {
            f"{prefix}_has_path": True,
            f"{prefix}_bars": len(path),
            f"{prefix}_fixed_return": fixed_return - fee if fixed_return is not None else None,
            f"{prefix}_mfe": mfe,
            f"{prefix}_mae": mae,
        }
    )
    for tp_level in tp_levels:
        label = f"tp{int(round(tp_level * 100))}"
        hit, bars_to_hit, day_offset = first_tp_hit(path, entry_price, tp_level)
        plan_return = tp_level - fee if hit else out[f"{prefix}_fixed_return"]
        out.update(
            {
                f"{prefix}_{label}_hit": hit,
                f"{prefix}_{label}_bars_to_hit": bars_to_hit,
                f"{prefix}_{label}_day_offset": day_offset,
                f"{prefix}_{label}_plan_return": plan_return,
            }
        )
    return out


def delayed_entry_rows(
    row: pd.Series,
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    tp_levels: list[float],
    fee: float,
) -> list[dict[str, Any]]:
    """Build candidate delayed-entry rows for one event."""
    entry_date = pd.Timestamp(row["entry_date"]).normalize()
    if frame.empty or "date" not in frame.columns:
        return []
    entry_day = frame[frame["date"] == entry_date].copy().reset_index(drop=True)
    if entry_day.empty:
        return []
    full_path = frame[frame["date"] >= entry_date].copy().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    minute_open = float(entry_day.iloc[0]["open"])
    for spec in candidates:
        passed, confirm = candidate_passes(spec, row, entry_day)
        if not passed:
            continue
        entry_index = entry_index_for_spec(spec)
        if entry_index is None or entry_index >= len(entry_day):
            continue
        entry_bar = entry_day.iloc[entry_index]
        entry_price = float(entry_bar["open"])
        entry_dt = entry_bar.get("minute_dt")
        if pd.isna(entry_dt) or entry_price <= 0:
            continue
        after_entry = full_path[full_path["minute_dt"] >= entry_dt].copy().reset_index(drop=True)
        event_row: dict[str, Any] = {
            "event_id": int(row["event_id"]),
            "candidate": spec.name,
            "vt_symbol": row["vt_symbol"],
            "signal_date": row["signal_date"],
            "entry_date": row["entry_date"],
            "pre_l1_type": row.get("pre_l1_type"),
            "pre_l1_score": row.get("pre_l1_score"),
            "component_count": row.get("component_count"),
            "entry_gap_pct": row.get("entry_gap_pct"),
            "open_executable": row.get("open_executable"),
            "entry_limit_up": row.get("entry_limit_up"),
            "minute_open": minute_open,
            "delayed_entry_time": entry_bar.get("minute_hhmm"),
            "delayed_entry_price": entry_price,
            "entry_price_vs_open": safe_return(entry_price, minute_open),
            "entry_delay_bars": entry_index,
            "entry_delay_minutes": entry_index * 5,
            **confirm,
        }
        for horizon in (0, 1, 3):
            path = select_horizon_path(after_entry, entry_date, horizon)
            event_row.update(path_metrics(path, entry_price, tp_levels, fee, f"h{horizon}d"))
        rows.append(event_row)
    return rows


def build_delayed_entries_from_panel(
    events: pd.DataFrame,
    minute_panel: pd.DataFrame,
    candidates: list[CandidateSpec],
    tp_levels: list[float],
    fee: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build event-level delayed-entry rows."""
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    panel_by_symbol = {
        str(vt_symbol): group.reset_index(drop=True)
        for vt_symbol, group in minute_panel.groupby("vt_symbol", sort=False)
    }
    for event in events.itertuples(index=False):
        row = pd.Series(event._asdict())
        frame = panel_by_symbol.get(str(row["vt_symbol"]))
        if frame is None or frame.empty:
            missing_rows.append(
                {
                    "event_id": int(row["event_id"]),
                    "vt_symbol": row["vt_symbol"],
                    "entry_date": row["entry_date"],
                    "reason": "missing_symbol_minutes",
                }
            )
            continue
        event_rows = delayed_entry_rows(row, frame, candidates, tp_levels, fee)
        if event_rows:
            rows.extend(event_rows)
        else:
            missing_rows.append(
                {
                    "event_id": int(row["event_id"]),
                    "vt_symbol": row["vt_symbol"],
                    "entry_date": row["entry_date"],
                    "reason": "no_minute_entry_day_or_no_candidate",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(missing_rows)


def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    """Summarize a delayed-entry candidate group."""
    out: dict[str, Any] = {
        "events": len(group),
        "entry_price_vs_open_mean": group["entry_price_vs_open"].mean(),
        "entry_price_vs_open_median": group["entry_price_vs_open"].median(),
        "entry_delay_minutes_mean": group["entry_delay_minutes"].mean(),
    }
    for horizon in (0, 1, 3):
        out.update(
            {
                f"h{horizon}d_fixed_mean": group[f"h{horizon}d_fixed_return"].mean(),
                f"h{horizon}d_fixed_median": group[f"h{horizon}d_fixed_return"].median(),
                f"h{horizon}d_mfe_median": group[f"h{horizon}d_mfe"].median(),
                f"h{horizon}d_mae_median": group[f"h{horizon}d_mae"].median(),
                f"h{horizon}d_tp3_rate": group[f"h{horizon}d_tp3_hit"].mean(),
                f"h{horizon}d_tp3_plan_mean": group[f"h{horizon}d_tp3_plan_return"].mean(),
                f"h{horizon}d_tp5_rate": group[f"h{horizon}d_tp5_hit"].mean(),
                f"h{horizon}d_tp5_plan_mean": group[f"h{horizon}d_tp5_plan_return"].mean(),
            }
        )
    return out


def candidate_summary(entries: pd.DataFrame) -> pd.DataFrame:
    """Summarize delayed-entry results by candidate."""
    rows: list[dict[str, Any]] = []
    for candidate, group in entries.groupby("candidate", sort=False):
        row = summarize_group(group)
        row["candidate"] = candidate
        rows.append(row)
    return pd.DataFrame(rows)


def year_summary(entries: pd.DataFrame) -> pd.DataFrame:
    """Summarize delayed-entry results by candidate and year."""
    rows: list[dict[str, Any]] = []
    data = entries.copy()
    data["entry_year"] = pd.to_datetime(data["entry_date"]).dt.year
    for (candidate, year), group in data.groupby(["candidate", "entry_year"], sort=False):
        row = summarize_group(group)
        row.update({"candidate": candidate, "entry_year": int(year)})
        rows.append(row)
    return pd.DataFrame(rows)


def parse_tp_levels(text: str) -> list[float]:
    """Parse comma-separated TP levels."""
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("--tp-levels cannot be empty")
    return values


def write_outputs(entries: pd.DataFrame, missing: pd.DataFrame, output_dir: Path) -> None:
    """Write delayed-entry event and summary outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    entries.to_csv(output_dir / "pre_l1_minute_delayed_entry_events.csv", index=False)
    entries.to_parquet(output_dir / "pre_l1_minute_delayed_entry_events.parquet", index=False)
    missing.to_csv(output_dir / "pre_l1_minute_delayed_entry_missing.csv", index=False)
    candidate_summary(entries).to_csv(output_dir / "minute_delayed_entry_candidate_summary.csv", index=False)
    year_summary(entries).to_csv(output_dir / "minute_delayed_entry_candidate_by_year.csv", index=False)


def main() -> None:
    """Run the delayed-entry study."""
    args = parse_args()
    events = load_events(repo_path(args.events_path), args.start_date, args.end_date, args.max_events)
    minute_panel = load_minute_panel(
        repo_path(args.plan_path),
        repo_path(args.minute_panel_path),
        args.rebuild_minute_panel,
        args.progress_every,
    )
    tp_levels = parse_tp_levels(args.tp_levels)
    fee = args.fee_bps / 10000.0
    entries, missing = build_delayed_entries_from_panel(events, minute_panel, CANDIDATES, tp_levels, fee)
    write_outputs(entries, missing, repo_path(args.output_dir))
    print(
        "pre_l1_minute_delayed_entry_study "
        f"events={len(events)} entries={len(entries)} missing_or_no_candidate={len(missing)} "
        f"output_dir={repo_path(args.output_dir)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
