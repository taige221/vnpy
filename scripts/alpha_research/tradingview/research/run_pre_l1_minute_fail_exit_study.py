"""Study intraday failure exits after open entry for PRE-L1 tomorrow-L1 events.

This script models the A-share T+1 constraint explicitly:

* entry happens at the L1 day first 5-minute open;
* 5/15/30-minute failure signals can be observed on the entry day;
* no same-day exit is allowed;
* if a failure signal fires, the earliest exit is the next trading day's open.

The output compares full-exit and half-reduce variants against a no-intraday
baseline that also disallows same-day TP exits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.alpha_research.tradingview.research.run_pre_l1_minute_delayed_entry_study import (
    DEFAULT_MINUTE_PANEL_PATH,
    load_minute_panel,
    parse_tp_levels,
)
from scripts.alpha_research.tradingview.research.run_pre_l1_minute_feature_study import (
    DEFAULT_EVENTS_PATH,
    DEFAULT_PLAN_PATH,
    load_events,
    repo_path,
    safe_return,
)


DEFAULT_OUTPUT_DIR = "scripts/alpha_research/tradingview/reports/pre_l1_minute_fail_exit_study"


@dataclass(frozen=True)
class FailureSpec:
    """One entry-day failure rule."""

    name: str
    check_bars: int
    max_close_return: float | None = None
    max_mae: float | None = None
    min_mfe: float | None = None
    mode: str = "any"


@dataclass(frozen=True)
class ExitSpec:
    """One failure-exit policy."""

    name: str
    failure: FailureSpec | None
    exit_fraction: float


FAILURE_SPECS = [
    FailureSpec(name="5m_red", check_bars=1, max_close_return=0.0),
    FailureSpec(name="15m_red", check_bars=3, max_close_return=0.0),
    FailureSpec(name="30m_red", check_bars=6, max_close_return=0.0),
    FailureSpec(name="30m_weak_any", check_bars=6, max_close_return=-0.005, max_mae=-0.02, mode="any"),
    FailureSpec(name="30m_no_push", check_bars=6, max_close_return=0.005, min_mfe=0.01, mode="all"),
]

EXIT_SPECS = [
    ExitSpec(name="baseline_no_intraday_exit", failure=None, exit_fraction=0.0),
    *[
        ExitSpec(name=f"{failure.name}_exit100_next_open", failure=failure, exit_fraction=1.0)
        for failure in FAILURE_SPECS
    ],
    *[
        ExitSpec(name=f"{failure.name}_reduce50_next_open", failure=failure, exit_fraction=0.5)
        for failure in FAILURE_SPECS
    ],
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Study PRE-L1 minute failure exits under T+1")
    parser.add_argument("--events-path", default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--plan-path", default=DEFAULT_PLAN_PATH)
    parser.add_argument("--minute-panel-path", default=DEFAULT_MINUTE_PANEL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--tp-levels", default="0.03,0.05")
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--gap-min", type=float, default=-0.03)
    parser.add_argument("--gap-max", type=float, default=0.03)
    parser.add_argument("--horizon-days", type=int, default=3)
    parser.add_argument("--rebuild-minute-panel", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def eligible_events(events: pd.DataFrame, gap_min: float, gap_max: float) -> pd.DataFrame:
    """Filter to the open-executable gap window used as the current execution pool."""
    data = events.copy()
    return data[
        data["open_executable"].fillna(False).astype(bool)
        & data["entry_gap_pct"].between(gap_min, gap_max, inclusive="both")
    ].reset_index(drop=True)


def minute_window_stats(entry_day: pd.DataFrame, bars: int, entry_price: float) -> dict[str, Any]:
    """Calculate early-entry-day window stats."""
    if len(entry_day) < bars or entry_price <= 0:
        return {"available": False}
    window = entry_day.head(bars)
    return {
        "available": True,
        "confirm_time": window.iloc[-1].get("minute_hhmm"),
        "close_return": safe_return(window.iloc[-1]["close"], entry_price),
        "mfe": safe_return(window["high"].max(), entry_price),
        "mae": safe_return(window["low"].min(), entry_price),
    }


def failure_fired(spec: FailureSpec, stats: dict[str, Any]) -> bool:
    """Return whether a failure rule fires from observed entry-day stats."""
    if not stats.get("available", False):
        return False
    checks: list[bool] = []
    close_return = stats.get("close_return")
    mae = stats.get("mae")
    mfe = stats.get("mfe")
    if spec.max_close_return is not None:
        checks.append(close_return is not None and close_return <= spec.max_close_return)
    if spec.max_mae is not None:
        checks.append(mae is not None and mae <= spec.max_mae)
    if spec.min_mfe is not None:
        checks.append(mfe is not None and mfe < spec.min_mfe)
    if not checks:
        return False
    if spec.mode == "any":
        return any(checks)
    if spec.mode == "all":
        return all(checks)
    raise ValueError(f"unsupported failure mode: {spec.mode}")


def future_dates_after_entry(frame: pd.DataFrame, entry_date: pd.Timestamp, horizon_days: int) -> list[pd.Timestamp]:
    """Return future trading dates after the entry date."""
    dates = sorted(pd.Timestamp(value).normalize() for value in frame["date"].dropna().unique())
    dates = [date for date in dates if date > entry_date]
    return dates[:horizon_days]


def future_path(frame: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Return future rows for selected trading dates."""
    if not dates:
        return frame.iloc[0:0]
    return frame[frame["date"].isin(set(dates))].copy().reset_index(drop=True)


def first_open_for_date(frame: pd.DataFrame, date: pd.Timestamp) -> float | None:
    """Return the first 5-minute open for a date."""
    day = frame[frame["date"] == date].sort_values("time")
    if day.empty:
        return None
    value = day.iloc[0]["open"]
    return float(value) if pd.notna(value) else None


def continuation_plan(path: pd.DataFrame, entry_price: float, tp_level: float, fee: float) -> dict[str, Any]:
    """Plan return with no same-day exits; path begins after the entry date."""
    out: dict[str, Any] = {"tp_hit": False, "plan_return": None, "timeout_return": None}
    if path.empty or entry_price <= 0:
        return out
    target = entry_price * (1.0 + tp_level)
    hit = path[path["high"] >= target]
    timeout_return = safe_return(path.iloc[-1]["close"], entry_price)
    timeout_return = timeout_return - fee if timeout_return is not None else None
    if not hit.empty:
        out.update(
            {
                "tp_hit": True,
                "plan_return": tp_level - fee,
                "tp_hit_date": hit.iloc[0]["date"],
                "tp_hit_time": hit.iloc[0].get("minute_hhmm"),
                "timeout_return": timeout_return,
            }
        )
    else:
        out.update({"plan_return": timeout_return, "timeout_return": timeout_return})
    return out


def path_mfe_mae(path: pd.DataFrame, entry_price: float) -> tuple[float | None, float | None]:
    """Calculate path MFE and MAE."""
    if path.empty or entry_price <= 0:
        return None, None
    return safe_return(path["high"].max(), entry_price), safe_return(path["low"].min(), entry_price)


def event_fail_exit_rows(
    row: pd.Series,
    frame: pd.DataFrame,
    exit_specs: list[ExitSpec],
    tp_levels: list[float],
    fee: float,
    horizon_days: int,
) -> list[dict[str, Any]]:
    """Build failure-exit rows for one open-entry event."""
    entry_date = pd.Timestamp(row["entry_date"]).normalize()
    entry_day = frame[frame["date"] == entry_date].copy().sort_values("time").reset_index(drop=True)
    if entry_day.empty:
        return []
    entry_price = float(entry_day.iloc[0]["open"])
    if entry_price <= 0:
        return []
    future_dates = future_dates_after_entry(frame, entry_date, horizon_days)
    if not future_dates:
        return []
    next_open = first_open_for_date(frame, future_dates[0])
    if next_open is None:
        return []
    future = future_path(frame, future_dates)
    full_path = pd.concat([entry_day, future], ignore_index=True)
    day0_mfe, day0_mae = path_mfe_mae(entry_day, entry_price)
    full_mfe, full_mae = path_mfe_mae(full_path, entry_price)
    next_open_return = safe_return(next_open, entry_price)
    next_open_return = next_open_return - fee if next_open_return is not None else None
    fail_path_high = max(float(entry_day["high"].max()), float(next_open))
    fail_path_low = min(float(entry_day["low"].min()), float(next_open))
    fail_path_mfe = safe_return(fail_path_high, entry_price)
    fail_path_mae = safe_return(fail_path_low, entry_price)

    failure_stats = {
        spec.name: minute_window_stats(entry_day, spec.check_bars, entry_price)
        for spec in FAILURE_SPECS
    }
    continuation_by_tp = {
        tp_level: continuation_plan(future, entry_price, tp_level, fee)
        for tp_level in tp_levels
    }

    rows: list[dict[str, Any]] = []
    for exit_spec in exit_specs:
        if exit_spec.failure is None:
            fired = False
            stats = {"available": True}
        else:
            stats = failure_stats[exit_spec.failure.name]
            fired = failure_fired(exit_spec.failure, stats)
        event_row: dict[str, Any] = {
            "event_id": int(row["event_id"]),
            "exit_policy": exit_spec.name,
            "vt_symbol": row["vt_symbol"],
            "signal_date": row["signal_date"],
            "entry_date": row["entry_date"],
            "pre_l1_type": row.get("pre_l1_type"),
            "pre_l1_score": row.get("pre_l1_score"),
            "component_count": row.get("component_count"),
            "entry_gap_pct": row.get("entry_gap_pct"),
            "entry_price": entry_price,
            "next_open_date": future_dates[0],
            "next_open_price": next_open,
            "next_open_return": next_open_return,
            "failure_fired": fired,
            "failure_available": stats.get("available", False),
            "failure_confirm_time": stats.get("confirm_time"),
            "failure_close_return": stats.get("close_return"),
            "failure_mfe": stats.get("mfe"),
            "failure_mae": stats.get("mae"),
            "exit_fraction": exit_spec.exit_fraction if fired else 0.0,
            "day0_mfe": day0_mfe,
            "day0_mae": day0_mae,
            "horizon_mfe": full_mfe,
            "horizon_mae": full_mae,
        }
        if fired and exit_spec.exit_fraction > 0:
            if exit_spec.exit_fraction >= 1:
                policy_mfe = fail_path_mfe
                policy_mae = fail_path_mae
            else:
                policy_mfe = (
                    exit_spec.exit_fraction * fail_path_mfe + (1.0 - exit_spec.exit_fraction) * full_mfe
                    if fail_path_mfe is not None and full_mfe is not None
                    else None
                )
                policy_mae = (
                    exit_spec.exit_fraction * fail_path_mae + (1.0 - exit_spec.exit_fraction) * full_mae
                    if fail_path_mae is not None and full_mae is not None
                    else None
                )
        else:
            policy_mfe = full_mfe
            policy_mae = full_mae
        event_row.update(
            {
                "fail_path_mfe": fail_path_mfe,
                "fail_path_mae": fail_path_mae,
                "policy_mfe": policy_mfe,
                "policy_mae": policy_mae,
            }
        )
        for tp_level in tp_levels:
            label = f"tp{int(round(tp_level * 100))}"
            continuation = continuation_by_tp[tp_level]
            continuation_return = continuation.get("plan_return")
            if fired and exit_spec.exit_fraction >= 1.0:
                plan_return = next_open_return
                tp_hit = False
            elif fired and exit_spec.exit_fraction > 0:
                if next_open_return is None or continuation_return is None:
                    plan_return = None
                else:
                    plan_return = (
                        exit_spec.exit_fraction * next_open_return
                        + (1.0 - exit_spec.exit_fraction) * float(continuation_return)
                    )
                tp_hit = bool(continuation.get("tp_hit", False))
            else:
                plan_return = continuation_return
                tp_hit = bool(continuation.get("tp_hit", False))
            event_row.update(
                {
                    f"{label}_plan_return": plan_return,
                    f"{label}_hit": tp_hit,
                    f"{label}_continuation_return": continuation_return,
                    f"{label}_timeout_return": continuation.get("timeout_return"),
                }
            )
        rows.append(event_row)
    return rows


def build_fail_exit_rows(
    events: pd.DataFrame,
    minute_panel: pd.DataFrame,
    exit_specs: list[ExitSpec],
    tp_levels: list[float],
    fee: float,
    horizon_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build event-policy rows and missing rows."""
    panel_by_symbol = {
        str(vt_symbol): group.reset_index(drop=True)
        for vt_symbol, group in minute_panel.groupby("vt_symbol", sort=False)
    }
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        row = pd.Series(event._asdict())
        frame = panel_by_symbol.get(str(row["vt_symbol"]))
        if frame is None or frame.empty:
            missing.append(
                {
                    "event_id": int(row["event_id"]),
                    "vt_symbol": row["vt_symbol"],
                    "entry_date": row["entry_date"],
                    "reason": "missing_symbol_minutes",
                }
            )
            continue
        event_rows = event_fail_exit_rows(row, frame, exit_specs, tp_levels, fee, horizon_days)
        if not event_rows:
            missing.append(
                {
                    "event_id": int(row["event_id"]),
                    "vt_symbol": row["vt_symbol"],
                    "entry_date": row["entry_date"],
                    "reason": "missing_entry_or_future_path",
                }
            )
            continue
        rows.extend(event_rows)
    return pd.DataFrame(rows), pd.DataFrame(missing)


def summarize_group(group: pd.DataFrame, tp_levels: list[float]) -> dict[str, Any]:
    """Summarize one policy group."""
    out: dict[str, Any] = {
        "events": len(group),
        "fail_rate": group["failure_fired"].mean(),
        "exit_fraction_mean": group["exit_fraction"].mean(),
        "next_open_return_failed_mean": group.loc[group["failure_fired"], "next_open_return"].mean(),
        "day0_mfe_median": group["day0_mfe"].median(),
        "day0_mae_median": group["day0_mae"].median(),
        "horizon_mfe_median": group["horizon_mfe"].median(),
        "horizon_mae_median": group["horizon_mae"].median(),
        "policy_mfe_median": group["policy_mfe"].median(),
        "policy_mae_median": group["policy_mae"].median(),
        "policy_mae_p10": group["policy_mae"].quantile(0.10),
    }
    for tp_level in tp_levels:
        label = f"tp{int(round(tp_level * 100))}"
        out.update(
            {
                f"{label}_plan_mean": group[f"{label}_plan_return"].mean(),
                f"{label}_plan_median": group[f"{label}_plan_return"].median(),
                f"{label}_plan_p10": group[f"{label}_plan_return"].quantile(0.10),
                f"{label}_hit_rate": group[f"{label}_hit"].mean(),
                f"{label}_timeout_mean": group[f"{label}_timeout_return"].mean(),
            }
        )
    return out


def policy_summary(rows: pd.DataFrame, tp_levels: list[float]) -> pd.DataFrame:
    """Summarize by exit policy."""
    out: list[dict[str, Any]] = []
    for policy, group in rows.groupby("exit_policy", sort=False):
        item = summarize_group(group, tp_levels)
        item["exit_policy"] = policy
        out.append(item)
    return pd.DataFrame(out)


def year_summary(rows: pd.DataFrame, tp_levels: list[float]) -> pd.DataFrame:
    """Summarize by exit policy and entry year."""
    data = rows.copy()
    data["entry_year"] = pd.to_datetime(data["entry_date"]).dt.year
    out: list[dict[str, Any]] = []
    for (policy, year), group in data.groupby(["exit_policy", "entry_year"], sort=False):
        item = summarize_group(group, tp_levels)
        item.update({"exit_policy": policy, "entry_year": int(year)})
        out.append(item)
    return pd.DataFrame(out)


def write_outputs(rows: pd.DataFrame, missing: pd.DataFrame, output_dir: Path, tp_levels: list[float]) -> None:
    """Write event and summary outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "pre_l1_minute_fail_exit_events.csv", index=False)
    rows.to_parquet(output_dir / "pre_l1_minute_fail_exit_events.parquet", index=False)
    missing.to_csv(output_dir / "pre_l1_minute_fail_exit_missing.csv", index=False)
    policy_summary(rows, tp_levels).to_csv(output_dir / "minute_fail_exit_policy_summary.csv", index=False)
    year_summary(rows, tp_levels).to_csv(output_dir / "minute_fail_exit_policy_by_year.csv", index=False)


def main() -> None:
    """Run the failure-exit study."""
    args = parse_args()
    events = load_events(repo_path(args.events_path), args.start_date, args.end_date, args.max_events)
    events = eligible_events(events, args.gap_min, args.gap_max)
    minute_panel = load_minute_panel(
        repo_path(args.plan_path),
        repo_path(args.minute_panel_path),
        args.rebuild_minute_panel,
        args.progress_every,
    )
    tp_levels = parse_tp_levels(args.tp_levels)
    fee = args.fee_bps / 10000.0
    rows, missing = build_fail_exit_rows(events, minute_panel, EXIT_SPECS, tp_levels, fee, args.horizon_days)
    write_outputs(rows, missing, repo_path(args.output_dir), tp_levels)
    print(
        "pre_l1_minute_fail_exit_study "
        f"events={len(events)} rows={len(rows)} missing={len(missing)} "
        f"output_dir={repo_path(args.output_dir)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
