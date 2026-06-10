"""run_l1_honest_portfolio.py — the realizable number behind the clean-timeline +1598%.

It REUSES the exact selection from ``run_l1_clean_timeline_backtest.py`` (same snapshot,
same ``core4_mean >= 0.52`` day filter, same top20 rank pool, same next-open
executability, same top2 pick) so selection is identical and not re-argued. ONLY the
P&L accounting changes.

The clean-timeline P&L is a *cohort* number: per signal-day it takes the equal-weight
mean of that day's <=2 picks' **3-day** trade return, then compounds those across the
154 signal-days (``prod(1+daily_return)``). That compounds a 3-day trade as if it were
realized in one day, and ignores that consecutive entry-days' holds OVERLAP — so the
same capital is counted ~3x and there is no cash drag. The resulting +1598% (top2,
fixed_3d) is not a realizable equity curve.

Here every trade instead occupies its 3 trading days on a shared calendar, and capital
is split across ``CAPACITY = top_k * hold = 6`` slots (idle slots earn 0 = cash). We
compound the real daily portfolio return to get honest CAGR / annualised Sharpe / true
max drawdown, then add a drop-top-names kill-test, a per-year table, and a random-2 vs
ranked-2 concentration check. fixed_3d net returns already include costs (the snapshot
``future_open_to_close_net_return_3d`` field).

Run: rtk python3 scripts/alpha_research/demo/run_l1_honest_portfolio.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# scripts now live in scripts/alpha_research/demo/; the selection pipeline is in ../tradingview/research/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tradingview", "research"))
from run_l1_clean_timeline_backtest import (  # noqa: E402
    DAY_FILTER_THRESHOLD, DEFAULT_SNAPSHOT_PATH, GAP_MAX, GAP_MIN, MIN_CIRC_MV,
    MIN_TURNOVER_RATE, NEXT_OPEN_TOP_K, SIGNAL_CLOSE_TOP_N, add_next_open_execution_fields,
    add_offline_returns, add_signal_close_day_filter, add_signal_close_rank_pool,
    normalize_date, select_next_open_topk, signal_close_l1_pool,
)

HOLD = 3                                  # fixed_3d hold, in trading days
CAPACITY = NEXT_OPEN_TOP_K * HOLD         # 2 picks/day * 3-day hold = 6 concurrent slots
TDAYS = 244.0                             # A-share trading days / year
RET = "fixed_3d_return"
DEFAULT_EVENT_PATH = (
    "scripts/alpha_research/tradingview/reports/v1_core4_top2_event_detail/"
    "l1_rank_score_event_selection.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run honest L1 portfolio accounting")
    parser.add_argument(
        "--event-path",
        default=DEFAULT_EVENT_PATH,
        help="Frozen selected-event CSV. Use empty/off/missing to recompute clean-timeline selection.",
    )
    parser.add_argument("--random-sims", type=int, default=200)
    return parser.parse_args()


# ------------------------------------------------------------------- portfolio engine
def calendar_of(snapshot: pd.DataFrame) -> np.ndarray:
    """Trading-day grid: every date referenced as a signal / entry / next day."""
    dates = pd.concat([
        normalize_date(snapshot["signal_date"]),
        normalize_date(snapshot["suggested_entry_date"]),
        normalize_date(snapshot["next_trade_date"]),
    ]).dropna().unique()
    return np.sort(dates)


def daily_series(trades: pd.DataFrame, n_days: int, *, full_invest: bool) -> np.ndarray:
    """Per-calendar-day portfolio return. Each trade spreads its (1+R)^(1/H)-1 daily
    mark across its H active days; capital is /CAPACITY (idle=cash) or /active (ceiling)."""
    acc = np.zeros(n_days)
    cnt = np.zeros(n_days)
    amort = (1.0 + trades[RET].to_numpy()) ** (1.0 / HOLD) - 1.0
    eidx = trades["entry_idx"].to_numpy()
    for a, j in zip(amort, eidx, strict=False):
        hi = min(j + HOLD, n_days)
        acc[j:hi] += a
        cnt[j:hi] += 1
    if full_invest:
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    return acc / CAPACITY


def curve_stats(r: np.ndarray, lo: int, hi: int) -> dict:
    """CAGR / annualised Sharpe / maxDD / total return over calendar span [lo, hi]."""
    rr = r[lo:hi + 1]
    if rr.size == 0:
        return dict(cagr=np.nan, sharpe=np.nan, maxdd=np.nan, total=np.nan)
    eq = np.cumprod(1.0 + rr)
    span = hi - lo + 1
    cagr = float(eq[-1] ** (TDAYS / span) - 1.0)
    sharpe = float(rr.mean() / rr.std() * np.sqrt(TDAYS)) if rr.std() > 0 else np.nan
    maxdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return dict(cagr=cagr, sharpe=sharpe, maxdd=maxdd, total=float(eq[-1] - 1.0))


def span_of(trades: pd.DataFrame, n_days: int) -> tuple[int, int]:
    lo = int(trades["entry_idx"].min())
    hi = int(min(trades["entry_idx"].max() + HOLD - 1, n_days - 1))
    return lo, hi


def pct(x: float) -> str:
    return "n/a" if x != x else f"{x * 100:+.1f}%"


def build_clean_timeline_selection(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Recompute the clean-timeline top2 selection from the snapshot."""
    pool = signal_close_l1_pool(snapshot)
    pool, _ = add_signal_close_day_filter(pool, threshold=DAY_FILTER_THRESHOLD)
    pool = add_signal_close_rank_pool(pool, top_n=SIGNAL_CLOSE_TOP_N)
    execp = add_next_open_execution_fields(
        pool,
        gap_min=GAP_MIN,
        gap_max=GAP_MAX,
        min_circ_mv=MIN_CIRC_MV,
        min_turnover_rate=MIN_TURNOVER_RATE,
    )
    execp["next_open_selected"] = False
    return add_offline_returns(select_next_open_topk(execp, top_k=NEXT_OPEN_TOP_K))


def load_selection(event_path: str, snapshot: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Load the frozen event selection when available, else recompute it."""
    path = Path(event_path)
    if event_path.lower() not in {"", "none", "off", "false"} and path.exists():
        selected = pd.read_csv(path)
        selected["signal_date"] = normalize_date(selected["signal_date"])
        selected["suggested_entry_date"] = normalize_date(selected["suggested_entry_date"])
        if RET not in selected.columns:
            selected = add_offline_returns(selected)
        selected = selected.drop_duplicates(["event_id"]).copy()
        return selected, f"frozen event file: {path}"
    return build_clean_timeline_selection(snapshot), "recomputed clean-timeline selection"


# ------------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    snap = pd.read_parquet(DEFAULT_SNAPSHOT_PATH)
    snap["signal_date"] = normalize_date(snap["signal_date"])

    selected, selection_source = load_selection(args.event_path, snap)

    pool = signal_close_l1_pool(snap)
    pool, _ = add_signal_close_day_filter(pool, threshold=DAY_FILTER_THRESHOLD)
    pool = add_signal_close_rank_pool(pool, top_n=SIGNAL_CLOSE_TOP_N)
    execp = add_next_open_execution_fields(
        pool, gap_min=GAP_MIN, gap_max=GAP_MAX,
        min_circ_mv=MIN_CIRC_MV, min_turnover_rate=MIN_TURNOVER_RATE)

    cal = calendar_of(snap)
    n = len(cal)
    idx = pd.Series(np.arange(n), index=pd.DatetimeIndex(cal))

    sel = selected.copy()
    sel["entry_dt"] = normalize_date(sel["suggested_entry_date"])
    sel = sel[sel["entry_dt"].notna() & sel[RET].notna()].copy()
    sel["entry_idx"] = sel["entry_dt"].map(idx).astype("Int64")
    sel = sel[sel["entry_idx"].notna()].copy()
    sel["entry_idx"] = sel["entry_idx"].astype(int)
    sel["year"] = pd.DatetimeIndex(sel["entry_dt"]).year

    lo, hi = span_of(sel, n)
    span_days = hi - lo + 1

    # --- their cohort number, reproduced (the +1598%) -----------------------------
    cohort = sel.groupby("signal_date")[RET].mean()
    cohort_total = float((1.0 + cohort).prod() - 1.0)
    cohort_sharpe = float(cohort.mean() / cohort.std() * np.sqrt(TDAYS)) if cohort.std() > 0 else np.nan

    # --- honest realizable books --------------------------------------------------
    r_cap = daily_series(sel, n, full_invest=False)
    r_full = daily_series(sel, n, full_invest=True)
    cnt = np.zeros(n)
    for j in sel["entry_idx"].to_numpy():
        cnt[min(j, n - 1):min(j + HOLD, n)] += 1
    exposure = float((cnt[lo:hi + 1] > 0).mean())
    avg_conc = float(cnt[lo:hi + 1][cnt[lo:hi + 1] > 0].mean())
    max_conc = int(cnt.max())

    s_cap = curve_stats(r_cap, lo, hi)
    s_full = curve_stats(r_full, lo, hi)
    # in-market: only days holding >=1 position (signal quality, no cash drag)
    inmkt = r_full[lo:hi + 1][cnt[lo:hi + 1] > 0]
    inmkt_sharpe = float(inmkt.mean() / inmkt.std() * np.sqrt(TDAYS)) if inmkt.std() > 0 else np.nan

    print("=" * 90)
    print("L1 v1 — HONEST REALIZABLE PORTFOLIO")
    print(f"  selection source={selection_source}")
    print(f"  trades={len(sel)}  entry-days={sel['signal_date'].nunique()}  "
          f"calendar span={span_days} trading days  capacity={CAPACITY} slots (top2 x 3d)")
    print(f"  market exposure={exposure * 100:.0f}% of span   avg concurrent={avg_conc:.1f}  max concurrent={max_conc}")
    print("=" * 90)
    print(f"  clean-timeline COHORT (their +X%):   total {pct(cohort_total):>10}   "
          f"naive ann-Sharpe {cohort_sharpe:+.2f}   <- compounds a 3d trade as 1 day, ignores overlap")
    print("  " + "-" * 86)
    print(f"  HONEST fixed-capacity (cash drag):   CAGR {pct(s_cap['cagr']):>9}   Sharpe {s_cap['sharpe']:+.2f}   "
          f"maxDD {pct(s_cap['maxdd']):>8}   total {pct(s_cap['total']):>9}")
    print(f"  HONEST fully-invested (ceiling):     CAGR {pct(s_full['cagr']):>9}   Sharpe {s_full['sharpe']:+.2f}   "
          f"maxDD {pct(s_full['maxdd']):>8}   total {pct(s_full['total']):>9}")
    print(f"  in-market days only (signal qual):   Sharpe {inmkt_sharpe:+.2f}   (no cash drag, days with >=1 position)")

    # --- concentration / kill-test ------------------------------------------------
    by_sym = sel.groupby("vt_symbol")[RET].sum().sort_values(ascending=False)
    by_trade = sel[RET].sort_values(ascending=False)
    tot = float(sel[RET].sum())
    top10_share = float(by_trade.head(10).sum() / tot) if tot != 0 else np.nan
    print("\n[concentration]")
    print(f"  sum of trade returns={tot:+.2f};  top-10 winning trades = {top10_share * 100:.0f}% of it;  "
          f"largest single name = {by_sym.iloc[0] / tot * 100:.0f}% ({by_sym.index[0]})")
    for k in (3, 5, 10):
        drop = set(by_sym.head(k).index)
        sub = sel[~sel["vt_symbol"].isin(drop)]
        s = curve_stats(daily_series(sub, n, full_invest=False), *span_of(sub, n))
        print(f"  [kill-test] drop top-{k:>2} P&L names -> fixed-capacity CAGR {pct(s['cagr']):>9}  Sharpe {s['sharpe']:+.2f}")

    # --- per-year -----------------------------------------------------------------
    print("\n[per-year]  fixed-capacity book")
    print(f"  {'year':6}{'trades':>8}{'CAGR-in-yr':>12}{'Sharpe':>9}{'maxDD':>9}")
    yrday = pd.DatetimeIndex(cal).year
    for y in sorted(sel["year"].unique()):
        mask = (yrday == y)
        ylo, yhi = int(np.argmax(mask)), int(n - 1 - np.argmax(mask[::-1]))
        s = curve_stats(r_cap, ylo, yhi)
        nt = int((sel["year"] == y).sum())
        print(f"  {y:<6}{nt:>8}{pct(s['total']):>12}{s['sharpe']:>+9.2f}{pct(s['maxdd']):>9}")

    # --- random-2 vs ranked-2 (concentration at the ACTUAL k) ---------------------
    elig = execp[execp["signal_close_day_filter_pass"] & execp["next_open_executable"]].copy()
    elig = add_offline_returns(elig)
    elig["entry_dt"] = normalize_date(elig["suggested_entry_date"])
    elig = elig[elig["entry_dt"].notna() & elig[RET].notna()].copy()
    elig["entry_idx"] = elig["entry_dt"].map(idx)
    elig = elig[elig["entry_idx"].notna()].copy()
    elig["entry_idx"] = elig["entry_idx"].astype(int)
    rng = np.random.default_rng(7)
    sims = []
    selected_counts = sel.groupby("signal_date").size()
    elig = elig[elig["signal_date"].isin(selected_counts.index)].copy()
    grp = list(elig.groupby("signal_date"))
    for _ in range(args.random_sims):
        parts = []
        for signal_date, g in grp:
            target_n = int(selected_counts.get(signal_date, min(NEXT_OPEN_TOP_K, len(g))))
            parts.append(
                g.sample(n=min(target_n, len(g)), random_state=int(rng.integers(1 << 31)))
            )
        rnd = pd.concat(parts)
        s = curve_stats(daily_series(rnd, n, full_invest=False), *span_of(rnd, n))
        sims.append((s["cagr"], s["sharpe"]))
    sca = np.array([x[0] for x in sims])
    ssh = np.array([x[1] for x in sims])
    print("\n[random matched-count vs ranked]  (same signal days and same selected count/day)")
    print(f"  ranked-2 (real):   fixed-capacity CAGR {pct(s_cap['cagr']):>9}   Sharpe {s_cap['sharpe']:+.2f}")
    print(f"  random ({args.random_sims} sims) CAGR  p10 {pct(np.nanpercentile(sca, 10))}  p50 {pct(np.nanpercentile(sca, 50))}  "
          f"p90 {pct(np.nanpercentile(sca, 90))}")
    print(f"  random ({args.random_sims} sims) Sharpe p10 {np.nanpercentile(ssh, 10):+.2f}  p50 {np.nanpercentile(ssh, 50):+.2f}  "
          f"p90 {np.nanpercentile(ssh, 90):+.2f}   ranked percentile = {(ssh < s_cap['sharpe']).mean() * 100:.0f}%")
    print("\nRead: the cohort total is a horizon+overlap artifact; the realizable book is the fixed-")
    print("capacity row. Trust Sharpe (they never computed it) + the kill-test + the per-year 2023 row.")


if __name__ == "__main__":
    main()
