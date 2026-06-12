"""Study 5-minute intraday features for PRE-L1 tomorrow-L1 events.

This script joins PRE-L1 event rows with the Baostock 5-minute cache fetched by
``fetch_pre_l1_baostock_minutes.py``.  It is a research-only script: all
``future`` or realized-return fields are labels, not live inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EVENTS_PATH = (
    "scripts/alpha_research/tradingview/reports/pre_l1_near_start_study/"
    "pre_l1_near_start_events.parquet"
)
DEFAULT_PLAN_PATH = "lab/a_share_research/minute/baostock_5m/fetch_plan_tomorrow_l1.csv"
DEFAULT_OUTPUT_DIR = "scripts/alpha_research/tradingview/reports/pre_l1_minute_feature_study"

WINDOW_BARS = {
    "5m": 1,
    "15m": 3,
    "30m": 6,
    "60m": 12,
}

LABEL_COLUMNS = [
    "fixed_1d_net_return",
    "fixed_3d_net_return",
    "fixed_5d_net_return",
    "tp3_hit_1d",
    "tp3_hit_3d",
    "tp3_hit_5d",
    "tp5_hit_1d",
    "tp5_hit_3d",
    "tp5_hit_5d",
    "mfe_1d",
    "mfe_3d",
    "mfe_5d",
    "mae_1d",
    "mae_3d",
    "mae_5d",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Study PRE-L1 5-minute features")
    parser.add_argument("--events-path", default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--plan-path", default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--storage-format", choices=["csv"], default="csv")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-events", type=int, default=None)
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    """Resolve paths relative to the repository root."""
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def normalize_date(values: Any) -> pd.Series:
    """Normalize date-like values to midnight timestamps."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def concrete_cache_path(output_path: str, storage_format: str) -> Path:
    """Return the actual minute-cache path for the requested storage format."""
    if storage_format != "csv":
        raise ValueError(f"unsupported storage format: {storage_format}")
    path = repo_path(output_path)
    return path.with_suffix(".csv")


def load_events(path: Path, start_date: str | None, end_date: str | None, max_events: int | None) -> pd.DataFrame:
    """Load PRE-L1 rows and keep the tomorrow-L1 set."""
    events = pd.read_parquet(path)
    events["signal_date"] = normalize_date(events["signal_date"])
    events["entry_date"] = normalize_date(events["entry_date"])
    events = events[events["tomorrow_l1"].fillna(False).astype(bool)].copy()
    if start_date:
        events = events[events["entry_date"] >= pd.Timestamp(start_date)]
    if end_date:
        events = events[events["entry_date"] <= pd.Timestamp(end_date)]
    events = events.sort_values(["entry_date", "vt_symbol"]).reset_index(drop=True)
    if max_events is not None:
        events = events.head(max_events).copy()
    events.insert(0, "event_id", range(len(events)))
    return events


def load_plan(path: Path, storage_format: str) -> dict[str, list[dict[str, Any]]]:
    """Load fetch plan rows grouped by vt_symbol."""
    plan = pd.read_csv(path)
    plan["start_date"] = normalize_date(plan["start_date"])
    plan["end_date"] = normalize_date(plan["end_date"])
    plan["cache_path"] = plan["output_path"].map(lambda value: concrete_cache_path(str(value), storage_format))
    out: dict[str, list[dict[str, Any]]] = {}
    for vt_symbol, group in plan.groupby("vt_symbol", sort=False):
        records = group.sort_values("start_date").to_dict("records")
        out[str(vt_symbol)] = records
    return out


def find_cache_path(plan_by_symbol: dict[str, list[dict[str, Any]]], vt_symbol: str, entry_date: pd.Timestamp) -> Path | None:
    """Find the cache file that covers an event's entry date."""
    records = plan_by_symbol.get(vt_symbol, [])
    for record in records:
        if record["start_date"] <= entry_date <= record["end_date"]:
            return Path(record["cache_path"])
    return None


def read_minute_cache(path: Path, cache: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    """Read one cached minute CSV once."""
    if path in cache:
        return cache[path]
    if not path.exists():
        frame = pd.DataFrame()
    else:
        frame = pd.read_csv(path)
    if frame.empty:
        cache[path] = frame
        return frame
    if "date" in frame.columns:
        frame["date"] = normalize_date(frame["date"])
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "time" in frame.columns:
        time_text = frame["time"].astype(str).str.zfill(17)
        frame["minute_hhmm"] = time_text.str.slice(8, 12)
    cache[path] = frame
    return frame


def safe_return(numerator: float | int | None, denominator: float | int | None) -> float | None:
    """Calculate a simple return and guard invalid prices."""
    if numerator is None or denominator is None:
        return None
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) <= 0:
        return None
    return float(numerator) / float(denominator) - 1.0


def first_time_hit(day: pd.DataFrame, level: float) -> str | None:
    """Return the first 5-minute timestamp where high reaches a level."""
    if day.empty or pd.isna(level):
        return None
    hit = day[day["high"] >= level]
    if hit.empty:
        return None
    return str(hit.iloc[0].get("minute_hhmm", ""))


def event_minute_features(row: pd.Series, day: pd.DataFrame) -> dict[str, Any]:
    """Calculate one event's intraday features."""
    out: dict[str, Any] = {
        "event_id": int(row["event_id"]),
        "vt_symbol": row["vt_symbol"],
        "signal_date": row["signal_date"],
        "entry_date": row["entry_date"],
        "pre_l1_type": row.get("pre_l1_type"),
        "pre_l1_score": row.get("pre_l1_score"),
        "component_count": row.get("component_count"),
        "entry_gap_pct": row.get("entry_gap_pct"),
        "open_gap_bucket": row.get("open_gap_bucket"),
        "open_confirm_bucket": row.get("open_confirm_bucket"),
        "open_executable": row.get("open_executable"),
        "entry_limit_up": row.get("entry_limit_up"),
        "entry_limit_down": row.get("entry_limit_down"),
        "liquidity_ok": row.get("liquidity_ok"),
        "has_minute_entry_day": False,
    }
    for column in LABEL_COLUMNS:
        if column in row.index:
            out[column] = row.get(column)

    if day.empty:
        return out

    day = day.sort_values("time").reset_index(drop=True)
    minute_open = float(day.iloc[0]["open"])
    daily_open = row.get("entry_open")
    daily_to_minute_ratio = None
    if pd.notna(daily_open) and float(daily_open) > 0:
        daily_to_minute_ratio = minute_open / float(daily_open)
    price_open = minute_open
    day_high = day["high"].max()
    day_low = day["low"].min()
    day_close = day.iloc[-1]["close"]
    day_volume = day["volume"].sum()
    day_amount = day["amount"].sum()

    signal_high = row.get("signal_high")
    signal_low = row.get("signal_low")
    if daily_to_minute_ratio is not None:
        if pd.notna(signal_high):
            signal_high = float(signal_high) * daily_to_minute_ratio
        if pd.notna(signal_low):
            signal_low = float(signal_low) * daily_to_minute_ratio
    tp3_level = price_open * 1.03
    tp5_level = price_open * 1.05

    out.update(
        {
            "has_minute_entry_day": True,
            "minute_bars_0d": len(day),
            "minute_first_time": day.iloc[0].get("minute_hhmm"),
            "minute_last_time": day.iloc[-1].get("minute_hhmm"),
            "minute_first_open": minute_open,
            "minute_first_close": day.iloc[0]["close"],
            "minute_open_diff_vs_daily_open": safe_return(minute_open, daily_open),
            "daily_to_minute_price_ratio": daily_to_minute_ratio,
            "minute_day_close_return": safe_return(day_close, price_open),
            "minute_day_mfe": safe_return(day_high, price_open),
            "minute_day_mae": safe_return(day_low, price_open),
            "minute_tp3_hit_0d": bool(day_high >= tp3_level),
            "minute_tp5_hit_0d": bool(day_high >= tp5_level),
            "minute_tp3_first_time_0d": first_time_hit(day, tp3_level),
            "minute_tp5_first_time_0d": first_time_hit(day, tp5_level),
        }
    )

    for label, bars in WINDOW_BARS.items():
        window = day.head(bars)
        high = window["high"].max()
        low = window["low"].min()
        close = window.iloc[-1]["close"]
        volume = window["volume"].sum()
        amount = window["amount"].sum()
        out.update(
            {
                f"{label}_close_return": safe_return(close, price_open),
                f"{label}_mfe": safe_return(high, price_open),
                f"{label}_mae": safe_return(low, price_open),
                f"{label}_range_pct": safe_return(high, low),
                f"{label}_volume_share_0d": float(volume / day_volume) if day_volume > 0 else None,
                f"{label}_amount_share_0d": float(amount / day_amount) if day_amount > 0 else None,
                f"{label}_break_signal_high": bool(pd.notna(signal_high) and high >= signal_high),
                f"{label}_hold_signal_low": bool(pd.notna(signal_low) and low >= signal_low),
                f"{label}_close_above_signal_high": bool(pd.notna(signal_high) and close >= signal_high),
            }
        )
    return out


def build_features(events: pd.DataFrame, plan_by_symbol: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """Build event-level minute feature rows."""
    cache: dict[Path, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        row = pd.Series(event._asdict())
        entry_date = pd.Timestamp(row["entry_date"]).normalize()
        path = find_cache_path(plan_by_symbol, str(row["vt_symbol"]), entry_date)
        if path is None:
            rows.append(event_minute_features(row, pd.DataFrame()))
            rows[-1]["minute_cache_path"] = None
            rows[-1]["minute_cache_missing"] = True
            continue
        frame = read_minute_cache(path, cache)
        if frame.empty or "date" not in frame.columns:
            day = pd.DataFrame()
        else:
            day = frame[frame["date"] == entry_date]
        features = event_minute_features(row, day)
        features["minute_cache_path"] = str(path)
        features["minute_cache_missing"] = not path.exists()
        rows.append(features)
    return pd.DataFrame(rows)


def add_buckets(features: pd.DataFrame) -> pd.DataFrame:
    """Add coarse buckets used in the summary tables."""
    out = features.copy()
    out["entry_gap_bucket_calc"] = pd.cut(
        out["entry_gap_pct"],
        bins=[-10, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 10],
        labels=["<=-5%", "-5~-3%", "-3~-1%", "-1~+1%", "+1~+3%", "+3~+5%", ">+5%"],
    )
    for window in ("5m", "15m", "30m", "60m"):
        out[f"{window}_close_return_bucket"] = pd.cut(
            out[f"{window}_close_return"],
            bins=[-10, -0.03, -0.01, 0, 0.01, 0.03, 10],
            labels=["<=-3%", "-3~-1%", "-1~0%", "0~+1%", "+1~+3%", ">+3%"],
        )
        out[f"{window}_mfe_bucket"] = pd.cut(
            out[f"{window}_mfe"],
            bins=[-10, 0, 0.01, 0.03, 0.05, 10],
            labels=["<=0%", "0~+1%", "+1~+3%", "+3~+5%", ">+5%"],
        )
        out[f"{window}_mae_bucket"] = pd.cut(
            out[f"{window}_mae"],
            bins=[-10, -0.05, -0.03, -0.01, 0, 10],
            labels=["<=-5%", "-5~-3%", "-3~-1%", "-1~0%", ">=0%"],
        )
    for column in ("30m_volume_share_0d", "30m_amount_share_0d"):
        bucket = f"{column}_quintile"
        valid = out[column].dropna()
        out[bucket] = pd.NA
        if valid.nunique() >= 5:
            out.loc[valid.index, bucket] = pd.qcut(valid, 5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"])
    return out


def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    """Summarize one feature bucket."""
    out: dict[str, Any] = {
        "events": len(group),
        "fixed_1d_mean": group["fixed_1d_net_return"].mean(),
        "fixed_1d_median": group["fixed_1d_net_return"].median(),
        "fixed_3d_mean": group["fixed_3d_net_return"].mean(),
        "fixed_3d_median": group["fixed_3d_net_return"].median(),
        "tp3_1d_rate": group["tp3_hit_1d"].mean(),
        "tp3_3d_rate": group["tp3_hit_3d"].mean(),
        "tp5_3d_rate": group["tp5_hit_3d"].mean(),
        "mfe_3d_median": group["mfe_3d"].median(),
        "mae_3d_median": group["mae_3d"].median(),
        "minute_day_mfe_median": group["minute_day_mfe"].median(),
        "minute_day_mae_median": group["minute_day_mae"].median(),
    }
    if "fixed_5d_net_return" in group.columns:
        out["fixed_5d_mean"] = group["fixed_5d_net_return"].mean()
    return out


def bool_mask(series: pd.Series) -> pd.Series:
    """Convert a nullable/object boolean-like series into a plain bool mask."""
    return series.map(lambda value: bool(value) if pd.notna(value) else False)


def bucket_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Build bucket-level summaries for selected intraday feature buckets."""
    bucket_columns = [
        "entry_gap_bucket_calc",
        "5m_close_return_bucket",
        "15m_close_return_bucket",
        "30m_close_return_bucket",
        "60m_close_return_bucket",
        "30m_mfe_bucket",
        "30m_mae_bucket",
        "30m_volume_share_0d_quintile",
        "30m_amount_share_0d_quintile",
        "30m_break_signal_high",
        "30m_hold_signal_low",
        "30m_close_above_signal_high",
    ]
    rows: list[dict[str, Any]] = []
    available = features[features["has_minute_entry_day"].fillna(False)].copy()
    for column in bucket_columns:
        if column not in available.columns:
            continue
        for value, group in available.groupby(column, dropna=False, observed=False):
            if len(group) == 0:
                continue
            summary = summarize_group(group)
            summary.update({"bucket_field": column, "bucket": str(value)})
            rows.append(summary)
    return pd.DataFrame(rows)


def candidate_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Compare simple open-confirm candidate filters."""
    available = features[features["has_minute_entry_day"].fillna(False)].copy()
    gap = available["entry_gap_pct"]
    break_high_30m = bool_mask(available["30m_break_signal_high"])
    hold_low_30m = bool_mask(available["30m_hold_signal_low"])
    candidates: dict[str, pd.Series] = {
        "all_minute_available": pd.Series(True, index=available.index),
        "gap_-3_to_+3": gap.between(-0.03, 0.03, inclusive="both"),
        "gap_-1_to_+3": gap.between(-0.01, 0.03, inclusive="both"),
        "gap_-3_to_+3_and_5m_green": gap.between(-0.03, 0.03, inclusive="both")
        & (available["5m_close_return"] > 0),
        "gap_-3_to_+3_and_15m_green": gap.between(-0.03, 0.03, inclusive="both")
        & (available["15m_close_return"] > 0),
        "gap_-3_to_+3_and_30m_green": gap.between(-0.03, 0.03, inclusive="both")
        & (available["30m_close_return"] > 0),
        "gap_-3_to_+3_30m_break_signal_high": gap.between(-0.03, 0.03, inclusive="both")
        & break_high_30m,
        "gap_-3_to_+3_30m_hold_signal_low": gap.between(-0.03, 0.03, inclusive="both")
        & hold_low_30m,
        "gap_-3_to_+3_30m_green_hold_low": gap.between(-0.03, 0.03, inclusive="both")
        & (available["30m_close_return"] > 0)
        & hold_low_30m,
        "gap_-3_to_+3_30m_green_break_high": gap.between(-0.03, 0.03, inclusive="both")
        & (available["30m_close_return"] > 0)
        & break_high_30m,
        "gap_-1_to_+3_30m_green_break_high": gap.between(-0.01, 0.03, inclusive="both")
        & (available["30m_close_return"] > 0)
        & break_high_30m,
        "avoid_early_weakness_30m": gap.between(-0.03, 0.03, inclusive="both")
        & (available["30m_close_return"] > -0.005)
        & (available["30m_mae"] > -0.02),
    }
    rows: list[dict[str, Any]] = []
    for name, mask in candidates.items():
        group = available[mask.fillna(False)]
        if group.empty:
            continue
        summary = summarize_group(group)
        summary.update({"candidate": name})
        rows.append(summary)
    return pd.DataFrame(rows)


def year_candidate_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Build candidate summaries by entry year."""
    rows: list[dict[str, Any]] = []
    available = features[features["has_minute_entry_day"].fillna(False)].copy()
    available["entry_year"] = pd.to_datetime(available["entry_date"]).dt.year
    break_high_30m = bool_mask(available["30m_break_signal_high"])
    candidates = candidate_summary(available)
    for candidate in candidates["candidate"]:
        if candidate == "all_minute_available":
            mask = pd.Series(True, index=available.index)
        elif candidate == "gap_-3_to_+3":
            mask = available["entry_gap_pct"].between(-0.03, 0.03, inclusive="both")
        elif candidate == "gap_-1_to_+3":
            mask = available["entry_gap_pct"].between(-0.01, 0.03, inclusive="both")
        elif candidate == "gap_-3_to_+3_and_30m_green":
            mask = available["entry_gap_pct"].between(-0.03, 0.03, inclusive="both") & (
                available["30m_close_return"] > 0
            )
        elif candidate == "gap_-3_to_+3_30m_green_break_high":
            mask = (
                available["entry_gap_pct"].between(-0.03, 0.03, inclusive="both")
                & (available["30m_close_return"] > 0)
                & break_high_30m
            )
        elif candidate == "avoid_early_weakness_30m":
            mask = (
                available["entry_gap_pct"].between(-0.03, 0.03, inclusive="both")
                & (available["30m_close_return"] > -0.005)
                & (available["30m_mae"] > -0.02)
            )
        else:
            continue
        for year, group in available[mask.fillna(False)].groupby("entry_year"):
            summary = summarize_group(group)
            summary.update({"candidate": candidate, "entry_year": int(year)})
            rows.append(summary)
    return pd.DataFrame(rows)


def write_outputs(features: pd.DataFrame, output_dir: Path) -> None:
    """Write event-level and summary outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    features = add_buckets(features)
    features.to_csv(output_dir / "pre_l1_minute_features_tomorrow_l1.csv", index=False)
    features.to_parquet(output_dir / "pre_l1_minute_features_tomorrow_l1.parquet", index=False)
    bucket_summary(features).to_csv(output_dir / "minute_feature_bucket_summary.csv", index=False)
    candidate_summary(features).to_csv(output_dir / "minute_open_confirm_candidate_summary.csv", index=False)
    year_candidate_summary(features).to_csv(output_dir / "minute_open_confirm_candidate_by_year.csv", index=False)


def main() -> None:
    """Run the study."""
    args = parse_args()
    events = load_events(repo_path(args.events_path), args.start_date, args.end_date, args.max_events)
    plan_by_symbol = load_plan(repo_path(args.plan_path), args.storage_format)
    features = build_features(events, plan_by_symbol)
    write_outputs(features, repo_path(args.output_dir))
    available = int(features["has_minute_entry_day"].fillna(False).sum())
    print(
        "pre_l1_minute_feature_study "
        f"events={len(features)} minute_available={available} "
        f"missing={len(features) - available} output_dir={repo_path(args.output_dir)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
