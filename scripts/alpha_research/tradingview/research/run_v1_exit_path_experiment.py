"""Run path-level exit experiments on the frozen L1 execution v1 selections.

The input event file is expected to already contain the clean timeline
selection: T-close day filter, T-close top-N ranking, then T+1 executable top-K.
This script does not re-rank or re-filter events. It only reloads daily bars and
compares exit rules over the exact selected event set.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EVENT_PATH = (
    "scripts/alpha_research/tradingview/reports/v1_core4_top2_event_detail/"
    "l1_rank_score_event_selection.csv"
)
DEFAULT_PANEL_PATH = "lab/a_share_research/panel/research_panel_daily"
DEFAULT_OUTPUT_DIR = "scripts/alpha_research/tradingview/reports/v1_exit_path_experiment"
OPEN_RATE = 0.0005
CLOSE_RATE = 0.0015
ANNUAL_DAYS = 252.0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run frozen L1 v1 path exit experiments")
    parser.add_argument("--event-path", default=DEFAULT_EVENT_PATH)
    parser.add_argument("--panel-path", default=DEFAULT_PANEL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--open-rate", type=float, default=OPEN_RATE)
    parser.add_argument("--close-rate", type=float, default=CLOSE_RATE)
    parser.add_argument("--target-pcts", default="0.03,0.05,0.08,0.10")
    parser.add_argument("--timeout-bars", default="3,5")
    parser.add_argument("--min-exit-bars", type=int, default=1)
    parser.add_argument("--warmup-days", type=int, default=420)
    parser.add_argument("--exit-buffer-days", type=int, default=30)
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated float list."""
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"empty float list: {value}")
    return sorted(set(values))


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated int list."""
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"invalid int list: {value}")
    return sorted(set(values))


def normalize_date(values: Any) -> pd.Series:
    """Normalize date-like values."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def numeric(frame: pd.DataFrame, column: str, default: float = math.nan) -> pd.Series:
    """Return a numeric series, or a default-valued series if missing."""
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def net_return(entry_price: float, exit_price: float, open_rate: float, close_rate: float) -> float:
    """Return net return after proportional buy/sell costs."""
    if entry_price <= 0 or exit_price <= 0:
        return math.nan
    return exit_price * (1.0 - close_rate) / (entry_price * (1.0 + open_rate)) - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Return max drawdown of a compounded return series."""
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return math.nan
    equity = (1.0 + values).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def annual_return(returns: pd.Series, *, calendar_days: int) -> float:
    """Return full-calendar compounded annual return."""
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if values.empty or calendar_days <= 0:
        return math.nan
    total_return = float((1.0 + values).prod() - 1.0)
    if total_return <= -1.0:
        return -1.0
    return (1.0 + total_return) ** (ANNUAL_DAYS / calendar_days) - 1.0


def sharpe_ratio(returns: pd.Series) -> float:
    """Return annualized Sharpe with zero risk-free rate."""
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    std = float(values.std(ddof=1))
    if values.empty or std <= 0:
        return math.nan
    return float(values.mean() / std * math.sqrt(ANNUAL_DAYS))


def pct(value: float) -> str:
    """Format a float as a percent."""
    if pd.isna(value):
        return ""
    return f"{value * 100:.2f}%"


def load_events(path: Path) -> pd.DataFrame:
    """Load and normalize frozen event selections."""
    events = pd.read_csv(path)
    required = {
        "event_id",
        "vt_symbol",
        "signal_date",
        "suggested_entry_date",
        "suggested_entry_price",
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"event file missing columns: {missing}")

    events = events.copy()
    events["signal_date"] = normalize_date(events["signal_date"])
    events["suggested_entry_date"] = normalize_date(events["suggested_entry_date"])
    events["signal_year"] = events["signal_date"].dt.year.astype("Int64")
    if "rank_score_column" not in events.columns:
        events["rank_score_column"] = "unknown"
    if "top_k" not in events.columns:
        events["top_k"] = events.groupby("signal_date")["event_id"].transform("count")
    return events.drop_duplicates(["rank_score_column", "top_k", "event_id"]).reset_index(drop=True)


def wilder(series: pd.Series, window: int) -> pd.Series:
    """Wilder-style exponential smoothing."""
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def add_path_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add EMA, RSI, and ATR columns needed by adaptive exits."""
    out = frame.sort_values(["vt_symbol", "datetime"]).copy()

    def enrich_symbol(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        close = group["close"].astype(float)
        high = group["high"].astype(float)
        low = group["low"].astype(float)
        prev_close = close.shift(1)

        group["ema_fast"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
        group["ema_mid"] = close.ewm(span=60, adjust=False, min_periods=60).mean()

        up = close.diff().clip(lower=0.0)
        down = (-close.diff()).clip(lower=0.0)
        rs = wilder(up, 14) / wilder(down, 14)
        group["rsi"] = 100.0 - 100.0 / (1.0 + rs)

        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        group["atr"] = wilder(true_range, 14)
        return group

    parts = [enrich_symbol(group) for _, group in out.groupby("vt_symbol", sort=False)]
    if not parts:
        return out
    return pd.concat(parts, ignore_index=True)


def load_panel_data(
    panel_path: Path,
    *,
    events: pd.DataFrame,
    warmup_days: int,
    exit_buffer_days: int,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Load local panel rows for selected symbols and build a trade calendar."""
    symbols = sorted(events["vt_symbol"].dropna().astype(str).unique())
    start = min(events["signal_date"].min(), events["suggested_entry_date"].min()) - pd.Timedelta(
        days=warmup_days
    )
    end = max(events["signal_date"].max(), events["suggested_entry_date"].max()) + pd.Timedelta(
        days=exit_buffer_days
    )

    columns = [
        "trade_date",
        "vt_symbol",
        "open",
        "high",
        "low",
        "close",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "volume",
        "up_limit",
        "down_limit",
        "is_limit_up",
        "is_limit_down",
        "has_valid_bar",
    ]
    filters = [
        ("vt_symbol", "in", symbols),
        ("trade_date", ">=", start.strftime("%Y-%m-%d")),
        ("trade_date", "<=", end.strftime("%Y-%m-%d")),
    ]
    try:
        panel = pd.read_parquet(panel_path, columns=columns, filters=filters)
    except Exception:
        panel = pd.read_parquet(panel_path, columns=columns)
        panel["datetime"] = normalize_date(panel["trade_date"])
        panel = panel[
            panel["vt_symbol"].astype(str).isin(symbols)
            & (panel["datetime"] >= start)
            & (panel["datetime"] <= end)
        ].copy()
    else:
        panel["datetime"] = normalize_date(panel["trade_date"])

    panel = panel.dropna(subset=["datetime", "vt_symbol", "open", "high", "low", "close"]).copy()
    if "has_valid_bar" in panel.columns:
        panel = panel[panel["has_valid_bar"].astype("boolean").fillna(False).astype(bool)].copy()
    panel["vt_symbol"] = panel["vt_symbol"].astype(str)
    for column in ("is_limit_up", "is_limit_down"):
        if column not in panel.columns:
            panel[column] = False
        panel[column] = panel[column].astype("boolean").fillna(False).astype(bool)

    panel = add_path_indicators(panel)
    by_symbol = {
        symbol: group.sort_values("datetime").reset_index(drop=True)
        for symbol, group in panel.groupby("vt_symbol", sort=False)
    }
    eval_start = events["signal_date"].min()
    eval_end = events["signal_date"].max()
    calendar_values = sorted(
        panel.loc[
            (panel["datetime"] >= eval_start) & (panel["datetime"] <= eval_end),
            "datetime",
        ]
        .dropna()
        .unique()
    )
    calendar = pd.DatetimeIndex(calendar_values)
    return by_symbol, calendar


def index_by_date(data: pd.DataFrame) -> dict[pd.Timestamp, int]:
    """Map normalized trade dates to row offsets."""
    return {
        pd.Timestamp(value).normalize(): int(index)
        for index, value in data["datetime"].items()
    }


def value_at(data: pd.DataFrame, index: int, column: str, default: float = math.nan) -> float:
    """Return a float value from a row."""
    if column not in data.columns:
        return default
    value = data.at[index, column]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bool_at(data: pd.DataFrame, index: int, column: str) -> bool:
    """Return a bool value from a row."""
    if column not in data.columns:
        return False
    return bool(data.at[index, column])


def base_exit_row(
    data: pd.DataFrame,
    event: pd.Series,
    *,
    entry_idx: int,
    exit_idx: int,
    entry_price: float,
    exit_price: float,
    exit_rule: str,
    exit_reason: str,
    open_rate: float,
    close_rate: float,
    target_hit: bool,
    stop_hit: bool,
    profit_protect_hit: bool,
    confirmed: bool,
) -> dict[str, Any]:
    """Build a normalized exit result row."""
    window = data.iloc[entry_idx : exit_idx + 1]
    return {
        "event_id": event["event_id"],
        "vt_symbol": event["vt_symbol"],
        "signal_date": pd.Timestamp(event["signal_date"]).normalize(),
        "signal_year": int(event["signal_year"]),
        "rank_score_column": event.get("rank_score_column", "unknown"),
        "top_k": int(event.get("top_k", 0)),
        "signal_close_rank": int(event.get("signal_close_rank", 0))
        if pd.notna(event.get("signal_close_rank", math.nan))
        else math.nan,
        "next_open_execution_rank": int(event.get("next_open_execution_rank", 0))
        if pd.notna(event.get("next_open_execution_rank", math.nan))
        else math.nan,
        "start_type": event.get("start_type", ""),
        "start_grade": event.get("start_grade", ""),
        "start_score": float(event.get("start_score", math.nan)),
        "entry_date": pd.Timestamp(data.at[entry_idx, "datetime"]).normalize(),
        "exit_date": pd.Timestamp(data.at[exit_idx, "datetime"]).normalize(),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": exit_price / entry_price - 1.0 if entry_price > 0 else math.nan,
        "net_return": net_return(entry_price, exit_price, open_rate, close_rate),
        "holding_bars": int(exit_idx - entry_idx),
        "exit_rule": exit_rule,
        "exit_reason": exit_reason,
        "target_hit": bool(target_hit),
        "stop_hit": bool(stop_hit),
        "profit_protect_hit": bool(profit_protect_hit),
        "confirmed": bool(confirmed),
        "limit_up_seen": bool(window["is_limit_up"].any()) if "is_limit_up" in window else False,
        "limit_down_seen": bool(window["is_limit_down"].any()) if "is_limit_down" in window else False,
        "mfe": float(window["high"].max() / entry_price - 1.0) if entry_price > 0 else math.nan,
        "mae": float(window["low"].min() / entry_price - 1.0) if entry_price > 0 else math.nan,
    }


def fixed_exit(
    data: pd.DataFrame,
    event: pd.Series,
    *,
    entry_idx: int,
    horizon: int,
    open_rate: float,
    close_rate: float,
) -> dict[str, Any] | None:
    """Exit at the close after a fixed number of bars."""
    exit_idx = entry_idx + horizon
    if exit_idx >= len(data):
        return None
    entry_price = float(event.get("suggested_entry_price", math.nan))
    if not math.isfinite(entry_price) or entry_price <= 0:
        entry_price = value_at(data, entry_idx, "open")
    exit_price = value_at(data, exit_idx, "close")
    return base_exit_row(
        data,
        event,
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        entry_price=entry_price,
        exit_price=exit_price,
        exit_rule=f"fixed_{horizon}d_path",
        exit_reason=f"fixed_{horizon}d_close",
        open_rate=open_rate,
        close_rate=close_rate,
        target_hit=False,
        stop_hit=False,
        profit_protect_hit=False,
        confirmed=False,
    )


def target_timeout_exit(
    data: pd.DataFrame,
    event: pd.Series,
    *,
    entry_idx: int,
    target_pct: float,
    max_bars: int,
    min_exit_bars: int,
    open_rate: float,
    close_rate: float,
    stop_price: float | None = None,
    exit_rule: str,
) -> dict[str, Any] | None:
    """Exit on target touch, optional stop touch, or timeout close."""
    max_exit_idx = min(entry_idx + max_bars, len(data) - 1)
    if max_exit_idx <= entry_idx:
        return None
    entry_price = float(event.get("suggested_entry_price", math.nan))
    if not math.isfinite(entry_price) or entry_price <= 0:
        entry_price = value_at(data, entry_idx, "open")
    target_price = entry_price * (1.0 + target_pct)
    exit_idx = max_exit_idx
    exit_price = value_at(data, max_exit_idx, "close")
    exit_reason = "timeout"
    target_hit = False
    stop_hit = False

    first_exit_idx = min(entry_idx + min_exit_bars, max_exit_idx)
    for i in range(first_exit_idx, max_exit_idx + 1):
        low = value_at(data, i, "low")
        high = value_at(data, i, "high")
        if stop_price is not None and math.isfinite(stop_price) and low <= stop_price:
            exit_idx = i
            exit_price = stop_price
            exit_reason = "stop_priority"
            stop_hit = True
            break
        if high >= target_price:
            exit_idx = i
            exit_price = target_price
            exit_reason = "target_hit"
            target_hit = True
            break

    return base_exit_row(
        data,
        event,
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        entry_price=entry_price,
        exit_price=exit_price,
        exit_rule=exit_rule,
        exit_reason=exit_reason,
        open_rate=open_rate,
        close_rate=close_rate,
        target_hit=target_hit,
        stop_hit=stop_hit,
        profit_protect_hit=False,
        confirmed=False,
    )


def limit_up_next_open_exit(
    data: pd.DataFrame,
    event: pd.Series,
    *,
    entry_idx: int,
    max_bars: int,
    open_rate: float,
    close_rate: float,
    target_pct: float | None = None,
    min_exit_bars: int = 1,
    exit_rule: str,
) -> dict[str, Any] | None:
    """Exit next open after the first limit-up, else optional target or timeout."""
    max_exit_idx = min(entry_idx + max_bars, len(data) - 1)
    if max_exit_idx <= entry_idx:
        return None
    entry_price = float(event.get("suggested_entry_price", math.nan))
    if not math.isfinite(entry_price) or entry_price <= 0:
        entry_price = value_at(data, entry_idx, "open")
    target_price = entry_price * (1.0 + target_pct) if target_pct is not None else math.nan

    exit_idx = max_exit_idx
    exit_price = value_at(data, max_exit_idx, "close")
    exit_reason = "timeout"
    target_hit = False

    for i in range(entry_idx, max_exit_idx + 1):
        can_sell = i >= entry_idx + min_exit_bars
        if can_sell and target_pct is not None and value_at(data, i, "high") >= target_price:
            exit_idx = i
            exit_price = target_price
            exit_reason = "target_hit"
            target_hit = True
            break
        if bool_at(data, i, "is_limit_up") and i + 1 < len(data):
            exit_idx = min(i + 1, max_exit_idx)
            exit_price = value_at(data, exit_idx, "open")
            exit_reason = "limit_up_next_open"
            break

    return base_exit_row(
        data,
        event,
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        entry_price=entry_price,
        exit_price=exit_price,
        exit_rule=exit_rule,
        exit_reason=exit_reason,
        open_rate=open_rate,
        close_rate=close_rate,
        target_hit=target_hit,
        stop_hit=False,
        profit_protect_hit=False,
        confirmed=False,
    )


def profit_protect_exit(
    data: pd.DataFrame,
    event: pd.Series,
    *,
    entry_idx: int,
    target_pct: float,
    trigger_pct: float,
    max_bars: int,
    open_rate: float,
    close_rate: float,
    disaster_stop_price: float,
    trail_atr_mult: float,
    protect_floor_pct: float,
    exit_rule: str,
) -> dict[str, Any] | None:
    """Exit on target, disaster stop, or next-open profit protection."""
    max_exit_idx = min(entry_idx + max_bars, len(data) - 1)
    if max_exit_idx <= entry_idx:
        return None
    entry_price = float(event.get("suggested_entry_price", math.nan))
    if not math.isfinite(entry_price) or entry_price <= 0:
        entry_price = value_at(data, entry_idx, "open")

    target_price = entry_price * (1.0 + target_pct)
    trigger_price = entry_price * (1.0 + trigger_pct)
    protect_floor = entry_price * (1.0 + protect_floor_pct)
    signal_high = float(event.get("event_high", math.nan))
    highest_high = entry_price
    profit_protect_active = False
    confirmed = False

    exit_idx = max_exit_idx
    exit_price = value_at(data, max_exit_idx, "close")
    exit_reason = "timeout"
    target_hit = False
    stop_hit = False
    profit_protect_hit = False

    for i in range(entry_idx + 1, max_exit_idx + 1):
        high = value_at(data, i, "high")
        low = value_at(data, i, "low")
        close = value_at(data, i, "close")
        ema_fast = value_at(data, i, "ema_fast")
        ema_mid = value_at(data, i, "ema_mid")
        rsi_value = value_at(data, i, "rsi")
        atr_value = value_at(data, i, "atr")
        highest_high = max(highest_high, high)

        if math.isfinite(disaster_stop_price) and low <= disaster_stop_price:
            exit_idx = i
            exit_price = disaster_stop_price
            exit_reason = "disaster_stop"
            stop_hit = True
            break

        if high >= target_price:
            exit_idx = i
            exit_price = target_price
            exit_reason = "target_hit"
            target_hit = True
            break

        if close > signal_high or (close > ema_fast and rsi_value >= 50.0) or close > ema_mid:
            confirmed = True

        if high >= trigger_price:
            profit_protect_active = True

        if profit_protect_active:
            trail_stop = highest_high - atr_value * trail_atr_mult if math.isfinite(atr_value) else math.nan
            protect_stop = max(protect_floor, trail_stop if math.isfinite(trail_stop) else protect_floor)
            if close < protect_stop:
                exit_idx = min(i + 1, len(data) - 1)
                exit_price = value_at(data, exit_idx, "open")
                exit_reason = "profit_protect"
                profit_protect_hit = True
                break

    return base_exit_row(
        data,
        event,
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        entry_price=entry_price,
        exit_price=exit_price,
        exit_rule=exit_rule,
        exit_reason=exit_reason,
        open_rate=open_rate,
        close_rate=close_rate,
        target_hit=target_hit,
        stop_hit=stop_hit,
        profit_protect_hit=profit_protect_hit,
        confirmed=confirmed,
    )


def confirm3_wide_profit_exit(
    data: pd.DataFrame,
    event: pd.Series,
    *,
    entry_idx: int,
    open_rate: float,
    close_rate: float,
    wide_stop_price: float,
) -> dict[str, Any] | None:
    """TP5 with wide stop, profit protection, and 3-day unconfirmed exit."""
    max_exit_idx = min(entry_idx + 5, len(data) - 1)
    if max_exit_idx <= entry_idx:
        return None
    entry_price = float(event.get("suggested_entry_price", math.nan))
    if not math.isfinite(entry_price) or entry_price <= 0:
        entry_price = value_at(data, entry_idx, "open")

    target_price = entry_price * 1.05
    trigger_price = entry_price * 1.03
    protect_floor = entry_price * 1.002
    signal_high = float(event.get("event_high", math.nan))
    highest_high = entry_price
    profit_protect_active = False
    confirmed = False

    exit_idx = max_exit_idx
    exit_price = value_at(data, max_exit_idx, "close")
    exit_reason = "timeout"
    target_hit = False
    stop_hit = False
    profit_protect_hit = False

    for i in range(entry_idx + 1, max_exit_idx + 1):
        bars_held = i - entry_idx
        high = value_at(data, i, "high")
        low = value_at(data, i, "low")
        close = value_at(data, i, "close")
        ema_fast = value_at(data, i, "ema_fast")
        ema_mid = value_at(data, i, "ema_mid")
        rsi_value = value_at(data, i, "rsi")
        atr_value = value_at(data, i, "atr")
        highest_high = max(highest_high, high)

        if math.isfinite(wide_stop_price) and low <= wide_stop_price:
            exit_idx = i
            exit_price = wide_stop_price
            exit_reason = "wide_stop"
            stop_hit = True
            break

        if high >= target_price:
            exit_idx = i
            exit_price = target_price
            exit_reason = "target_hit"
            target_hit = True
            break

        if close > signal_high or (close > ema_fast and rsi_value >= 50.0) or close > ema_mid:
            confirmed = True

        if high >= trigger_price:
            profit_protect_active = True

        if profit_protect_active:
            trail_stop = highest_high - atr_value * 2.5 if math.isfinite(atr_value) else math.nan
            protect_stop = max(protect_floor, trail_stop if math.isfinite(trail_stop) else protect_floor)
            if close < protect_stop:
                exit_idx = min(i + 1, len(data) - 1)
                exit_price = value_at(data, exit_idx, "open")
                exit_reason = "profit_protect"
                profit_protect_hit = True
                break

        if bars_held >= 3 and not confirmed:
            exit_idx = min(i + 1, len(data) - 1)
            exit_price = value_at(data, exit_idx, "open")
            exit_reason = "unconfirmed_3d_exit"
            break

    return base_exit_row(
        data,
        event,
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        entry_price=entry_price,
        exit_price=exit_price,
        exit_rule="tp5_confirm3_wide_profit_5d_path",
        exit_reason=exit_reason,
        open_rate=open_rate,
        close_rate=close_rate,
        target_hit=target_hit,
        stop_hit=stop_hit,
        profit_protect_hit=profit_protect_hit,
        confirmed=confirmed,
    )


def simulate_exits(
    events: pd.DataFrame,
    symbol_data: dict[str, pd.DataFrame],
    *,
    target_pcts: list[float],
    timeout_bars: list[int],
    min_exit_bars: int,
    open_rate: float,
    close_rate: float,
) -> pd.DataFrame:
    """Simulate all configured exit rules for each frozen event."""
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        data = symbol_data.get(str(event["vt_symbol"]))
        if data is None or data.empty:
            continue
        date_index = index_by_date(data)
        entry_idx = date_index.get(pd.Timestamp(event["suggested_entry_date"]).normalize())
        if entry_idx is None:
            continue

        disaster_stop_price = float(event.get("risk_disaster_stop_price", math.nan))
        wide_stop_price = float(event.get("risk_wide_stop_price", math.nan))

        for horizon in sorted(set([1, 3, 5, *timeout_bars])):
            row = fixed_exit(
                data,
                event,
                entry_idx=entry_idx,
                horizon=horizon,
                open_rate=open_rate,
                close_rate=close_rate,
            )
            if row is not None:
                rows.append(row)

        for max_bars in timeout_bars:
            for target_pct in target_pcts:
                target_label = int(round(target_pct * 100))
                row = target_timeout_exit(
                    data,
                    event,
                    entry_idx=entry_idx,
                    target_pct=target_pct,
                    max_bars=max_bars,
                    min_exit_bars=min_exit_bars,
                    open_rate=open_rate,
                    close_rate=close_rate,
                    exit_rule=f"tp{target_label}_timeout_{max_bars}d_path",
                )
                if row is not None:
                    rows.append(row)

        for target_pct in (0.05, 0.08):
            target_label = int(round(target_pct * 100))
            row = target_timeout_exit(
                data,
                event,
                entry_idx=entry_idx,
                target_pct=target_pct,
                max_bars=5,
                min_exit_bars=min_exit_bars,
                open_rate=open_rate,
                close_rate=close_rate,
                stop_price=wide_stop_price,
                exit_rule=f"tp{target_label}_wide_stop_5d_path",
            )
            if row is not None:
                rows.append(row)

        row = target_timeout_exit(
            data,
            event,
            entry_idx=entry_idx,
            target_pct=0.05,
            max_bars=5,
            min_exit_bars=min_exit_bars,
            open_rate=open_rate,
            close_rate=close_rate,
            stop_price=disaster_stop_price,
            exit_rule="tp5_disaster_stop_5d_path",
        )
        if row is not None:
            rows.append(row)

        for target_pct, trigger_pct in [(0.05, 0.03), (0.08, 0.05)]:
            target_label = int(round(target_pct * 100))
            row = profit_protect_exit(
                data,
                event,
                entry_idx=entry_idx,
                target_pct=target_pct,
                trigger_pct=trigger_pct,
                max_bars=5,
                open_rate=open_rate,
                close_rate=close_rate,
                disaster_stop_price=disaster_stop_price,
                trail_atr_mult=2.5,
                protect_floor_pct=0.002,
                exit_rule=f"tp{target_label}_profit_protect_5d_path",
            )
            if row is not None:
                rows.append(row)

        for target_pct in [None, 0.05, 0.08]:
            label = "limitup" if target_pct is None else f"tp{int(round(target_pct * 100))}_limitup"
            row = limit_up_next_open_exit(
                data,
                event,
                entry_idx=entry_idx,
                max_bars=5,
                open_rate=open_rate,
                close_rate=close_rate,
                target_pct=target_pct,
                min_exit_bars=min_exit_bars,
                exit_rule=f"{label}_next_open_5d_path",
            )
            if row is not None:
                rows.append(row)

        row = confirm3_wide_profit_exit(
            data,
            event,
            entry_idx=entry_idx,
            open_rate=open_rate,
            close_rate=close_rate,
            wide_stop_price=wide_stop_price,
        )
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


def build_summaries(
    detail: pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build event, daily, overall, yearly, and reason summaries."""
    daily = (
        detail.groupby(["rank_score_column", "top_k", "exit_rule", "signal_date", "signal_year"], sort=True)
        .agg(
            selected_events=("event_id", "count"),
            daily_return=("net_return", "mean"),
            target_hit_rate=("target_hit", "mean"),
            stop_hit_rate=("stop_hit", "mean"),
            profit_protect_hit_rate=("profit_protect_hit", "mean"),
            confirmed_rate=("confirmed", "mean"),
            limit_up_seen_rate=("limit_up_seen", "mean"),
            avg_holding_bars=("holding_bars", "mean"),
            mfe_median=("mfe", "median"),
            mae_median=("mae", "median"),
        )
        .reset_index()
    )

    daily_calendar_rows: list[dict[str, Any]] = []
    for (rank_score, top_k, exit_rule), group in daily.groupby(
        ["rank_score_column", "top_k", "exit_rule"], sort=True
    ):
        returns_by_date = group.set_index("signal_date")["daily_return"]
        curve = pd.Series(0.0, index=calendar)
        curve.loc[curve.index.intersection(returns_by_date.index)] = returns_by_date
        signal_returns = pd.to_numeric(group["daily_return"], errors="coerce").dropna()
        daily_calendar_rows.append(
            {
                "rank_score_column": rank_score,
                "top_k": int(top_k),
                "exit_rule": exit_rule,
                "calendar_days": int(len(curve)),
                "signal_days": int(signal_returns.shape[0]),
                "events": int(group["selected_events"].sum()),
                "avg_signal_day_return": float(signal_returns.mean()),
                "median_signal_day_return": float(signal_returns.median()),
                "p10_signal_day_return": float(signal_returns.quantile(0.10)),
                "positive_signal_day_rate": float((signal_returns > 0).mean()),
                "avg_selected_events": float(group["selected_events"].mean()),
                "target_hit_rate": float(group["target_hit_rate"].mean()),
                "stop_hit_rate": float(group["stop_hit_rate"].mean()),
                "profit_protect_hit_rate": float(group["profit_protect_hit_rate"].mean()),
                "confirmed_rate": float(group["confirmed_rate"].mean()),
                "limit_up_seen_rate": float(group["limit_up_seen_rate"].mean()),
                "avg_holding_bars": float(group["avg_holding_bars"].mean()),
                "total_compounded_return": float((1.0 + curve).prod() - 1.0),
                "annual_return": annual_return(curve, calendar_days=len(curve)),
                "sharpe": sharpe_ratio(curve),
                "max_drawdown": max_drawdown(curve),
            }
        )
    overall = pd.DataFrame(daily_calendar_rows)

    by_year_rows: list[dict[str, Any]] = []
    for (rank_score, top_k, exit_rule, year), group in daily.groupby(
        ["rank_score_column", "top_k", "exit_rule", "signal_year"], sort=True
    ):
        returns = pd.to_numeric(group["daily_return"], errors="coerce").dropna()
        by_year_rows.append(
            {
                "rank_score_column": rank_score,
                "top_k": int(top_k),
                "exit_rule": exit_rule,
                "signal_year": int(year),
                "signal_days": int(returns.shape[0]),
                "events": int(group["selected_events"].sum()),
                "avg_signal_day_return": float(returns.mean()),
                "median_signal_day_return": float(returns.median()),
                "p10_signal_day_return": float(returns.quantile(0.10)),
                "positive_signal_day_rate": float((returns > 0).mean()),
                "target_hit_rate": float(group["target_hit_rate"].mean()),
                "stop_hit_rate": float(group["stop_hit_rate"].mean()),
                "profit_protect_hit_rate": float(group["profit_protect_hit_rate"].mean()),
                "limit_up_seen_rate": float(group["limit_up_seen_rate"].mean()),
                "avg_holding_bars": float(group["avg_holding_bars"].mean()),
                "compounded_signal_return": float((1.0 + returns).prod() - 1.0),
                "max_drawdown_signal_days": max_drawdown(returns),
            }
        )
    by_year = pd.DataFrame(by_year_rows)

    reason = (
        detail.groupby(["rank_score_column", "top_k", "exit_rule", "exit_reason"], sort=True)
        .agg(
            events=("event_id", "count"),
            avg_net_return=("net_return", "mean"),
            median_net_return=("net_return", "median"),
            avg_holding_bars=("holding_bars", "mean"),
        )
        .reset_index()
    )
    reason["reason_share"] = reason["events"] / reason.groupby(
        ["rank_score_column", "top_k", "exit_rule"]
    )["events"].transform("sum")
    return daily, overall, by_year, reason


def write_report(
    output_dir: Path,
    *,
    events: pd.DataFrame,
    overall: pd.DataFrame,
    by_year: pd.DataFrame,
    reason: pd.DataFrame,
) -> None:
    """Write a compact Markdown report."""
    preferred_rules = [
        "fixed_1d_path",
        "fixed_3d_path",
        "fixed_5d_path",
        "tp3_timeout_3d_path",
        "tp5_timeout_3d_path",
        "tp8_timeout_3d_path",
        "tp10_timeout_3d_path",
        "tp3_timeout_5d_path",
        "tp5_timeout_5d_path",
        "tp8_timeout_5d_path",
        "tp10_timeout_5d_path",
        "tp5_wide_stop_5d_path",
        "tp8_wide_stop_5d_path",
        "tp5_profit_protect_5d_path",
        "tp8_profit_protect_5d_path",
        "limitup_next_open_5d_path",
        "tp5_limitup_next_open_5d_path",
        "tp8_limitup_next_open_5d_path",
        "tp5_confirm3_wide_profit_5d_path",
    ]
    compact = overall[overall["exit_rule"].isin(preferred_rules)].copy()
    compact["rule_order"] = compact["exit_rule"].map(
        {rule: index for index, rule in enumerate(preferred_rules)}
    )
    compact = compact.sort_values(["rank_score_column", "top_k", "rule_order"])

    display = compact[
        [
            "exit_rule",
            "signal_days",
            "events",
            "avg_signal_day_return",
            "p10_signal_day_return",
            "positive_signal_day_rate",
            "target_hit_rate",
            "stop_hit_rate",
            "profit_protect_hit_rate",
            "avg_holding_bars",
            "annual_return",
            "sharpe",
            "max_drawdown",
        ]
    ].copy()
    for column in [
        "avg_signal_day_return",
        "p10_signal_day_return",
        "positive_signal_day_rate",
        "target_hit_rate",
        "stop_hit_rate",
        "profit_protect_hit_rate",
        "annual_return",
        "max_drawdown",
    ]:
        display[column] = display[column].map(pct)
    display["sharpe"] = display["sharpe"].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    display["avg_holding_bars"] = display["avg_holding_bars"].map(
        lambda value: "" if pd.isna(value) else f"{value:.2f}"
    )

    top_by_ann = overall.sort_values("annual_return", ascending=False).head(8)
    top_display = top_by_ann[
        ["exit_rule", "annual_return", "sharpe", "max_drawdown", "avg_signal_day_return"]
    ].copy()
    for column in ["annual_return", "max_drawdown", "avg_signal_day_return"]:
        top_display[column] = top_display[column].map(pct)
    top_display["sharpe"] = top_display["sharpe"].map(
        lambda value: "" if pd.isna(value) else f"{value:.2f}"
    )

    reason_focus = reason[reason["exit_rule"].isin(preferred_rules)].copy()
    reason_focus = reason_focus.sort_values(["exit_rule", "events"], ascending=[True, False])
    reason_focus["reason_share"] = reason_focus["reason_share"].map(pct)
    reason_focus["avg_net_return"] = reason_focus["avg_net_return"].map(pct)
    reason_focus["median_net_return"] = reason_focus["median_net_return"].map(pct)

    lines = [
        "# L1 Execution v1 Exit Path Experiment",
        "",
        "Input is the frozen clean-timeline event selection; this script does not re-rank or re-filter.",
        "",
        f"- Events: {len(events):,}",
        f"- Signal days: {events['signal_date'].nunique():,}",
        f"- Symbols: {events['vt_symbol'].nunique():,}",
        f"- Date range: {events['signal_date'].min().date()} to {events['signal_date'].max().date()}",
        "",
        "## Main Rule Comparison",
        "",
        display.to_markdown(index=False),
        "",
        "## Top Rules by Full-Calendar Annual Return",
        "",
        top_display.to_markdown(index=False),
        "",
        "## Exit Reason Summary",
        "",
        reason_focus[
            [
                "exit_rule",
                "exit_reason",
                "events",
                "reason_share",
                "avg_net_return",
                "median_net_return",
                "avg_holding_bars",
            ]
        ].to_markdown(index=False),
        "",
        "## Year Output",
        "",
        "See `v1_exit_path_by_year.csv` for annual stability by exit rule.",
        "",
    ]
    (output_dir / "v1_exit_path_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(display.to_string(index=False))


def main() -> None:
    """Run the experiment."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(Path(args.event_path))
    target_pcts = parse_float_list(args.target_pcts)
    timeout_bars = parse_int_list(args.timeout_bars)
    symbol_data, calendar = load_panel_data(
        Path(args.panel_path),
        events=events,
        warmup_days=args.warmup_days,
        exit_buffer_days=args.exit_buffer_days,
    )
    if not symbol_data:
        raise RuntimeError("no symbol data loaded")
    if calendar.empty:
        raise RuntimeError("empty trade calendar")

    detail = simulate_exits(
        events,
        symbol_data,
        target_pcts=target_pcts,
        timeout_bars=timeout_bars,
        min_exit_bars=args.min_exit_bars,
        open_rate=args.open_rate,
        close_rate=args.close_rate,
    )
    if detail.empty:
        raise RuntimeError("no exit rows generated")

    daily, overall, by_year, reason = build_summaries(detail, calendar=calendar)
    detail.to_csv(output_dir / "v1_exit_path_event_results.csv", index=False)
    daily.to_csv(output_dir / "v1_exit_path_daily.csv", index=False)
    overall.to_csv(output_dir / "v1_exit_path_summary.csv", index=False)
    by_year.to_csv(output_dir / "v1_exit_path_by_year.csv", index=False)
    reason.to_csv(output_dir / "v1_exit_path_reason_summary.csv", index=False)
    write_report(output_dir, events=events, overall=overall, by_year=by_year, reason=reason)
    print(f"wrote={output_dir}")


if __name__ == "__main__":
    main()
