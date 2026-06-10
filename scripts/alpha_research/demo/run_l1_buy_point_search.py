"""run_l1_buy_point_search.py — find NEW buy points on top of the frozen L1 book.

The snapshot already labels three entry types (L1_reversal_start, L2_reversal_confirm,
L3_pullback_relay) but v1 trades only L1. This runs L2 and L3 — and the stacked
combination — through the *same* honest portfolio engine and the *same* unre-tuned
pipeline (core4_mean>=0.52 day gate, top20 rank pool, next-open top2, fixed_3d), so a
new buy point only counts if it clears the identical bar L1 cleared. No per-type
threshold mining.

Why this is the highest-value first step: the honest L1 book is only ~23% market
exposure. If L2/L3 fire on DIFFERENT days with a positive edge, stacking them fills the
idle days — lifting CAGR at the same Sharpe. We therefore report each type's standalone
honest stats, the day-level orthogonality between types, and the stacked book.

Run: rtk python3 scripts/alpha_research/demo/run_l1_buy_point_search.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import os
import sys

# this script lives in scripts/alpha_research/demo/ alongside run_l1_honest_portfolio;
# the selection pipeline (run_l1_clean_timeline_backtest) is in ../tradingview/research/
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tradingview", "research"))
from run_l1_clean_timeline_backtest import (  # noqa: E402
    DAY_FILTER_THRESHOLD, DEFAULT_SNAPSHOT_PATH, GAP_MAX, GAP_MIN, MIN_CIRC_MV,
    MIN_TURNOVER_RATE, NEXT_OPEN_TOP_K, SIGNAL_CLOSE_TOP_N, add_next_open_execution_fields,
    add_offline_returns, add_signal_close_day_filter, add_signal_close_rank_pool,
    bool_series, normalize_date, select_next_open_topk,
)
from run_l1_honest_portfolio import RET, calendar_of  # noqa: E402

HOLD = 3
TDAYS = 244.0
TYPES = ["L1_reversal_start", "L2_reversal_confirm", "L3_pullback_relay"]
SHORT = {"L1_reversal_start": "L1", "L2_reversal_confirm": "L2", "L3_pullback_relay": "L3"}


# ---------------------------------------------------------------- selection (parameterized type)
def pool_of_type(snap: pd.DataFrame, bpt: str) -> pd.DataFrame:
    """Same gates as signal_close_l1_pool but for an arbitrary buy_point_type."""
    mask = (
        snap["buy_point_type"].eq(bpt)
        & bool_series(snap, "has_valid_bar", default=True)
        & bool_series(snap, "is_list_life_valid", default=True)
        & ~bool_series(snap, "is_st")
        & ~bool_series(snap, "is_new_stock")
    )
    return snap[mask].copy()


def select_type(snap: pd.DataFrame, bpt: str) -> pd.DataFrame:
    pool = pool_of_type(snap, bpt)
    pool, _ = add_signal_close_day_filter(pool, threshold=DAY_FILTER_THRESHOLD)
    pool = add_signal_close_rank_pool(pool, top_n=SIGNAL_CLOSE_TOP_N)
    execp = add_next_open_execution_fields(
        pool, gap_min=GAP_MIN, gap_max=GAP_MAX,
        min_circ_mv=MIN_CIRC_MV, min_turnover_rate=MIN_TURNOVER_RATE)
    execp["next_open_selected"] = False
    return add_offline_returns(select_next_open_topk(execp, top_k=NEXT_OPEN_TOP_K))


def to_trades(sel: pd.DataFrame, idx: pd.Series) -> pd.DataFrame:
    t = sel.copy()
    t["entry_dt"] = normalize_date(t["suggested_entry_date"])
    t = t[t["entry_dt"].notna() & t[RET].notna()].copy()
    t["entry_idx"] = t["entry_dt"].map(idx)
    t = t[t["entry_idx"].notna()].copy()
    t["entry_idx"] = t["entry_idx"].astype(int)
    t["year"] = pd.DatetimeIndex(t["entry_dt"]).year
    return t


# ---------------------------------------------------------------- portfolio (capacity-parameterized)
def daily_series(trades: pd.DataFrame, n_days: int, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    acc = np.zeros(n_days)
    cnt = np.zeros(n_days)
    amort = (1.0 + trades[RET].to_numpy()) ** (1.0 / HOLD) - 1.0
    for a, j in zip(amort, trades["entry_idx"].to_numpy(), strict=False):
        hi = min(j + HOLD, n_days)
        acc[j:hi] += a
        cnt[j:hi] += 1
    return acc / capacity, cnt


def stats(trades: pd.DataFrame, n_days: int, capacity: int) -> dict:
    r, cnt = daily_series(trades, n_days, capacity)
    lo = int(trades["entry_idx"].min())
    hi = int(min(trades["entry_idx"].max() + HOLD - 1, n_days - 1))
    rr, cc = r[lo:hi + 1], cnt[lo:hi + 1]
    eq = np.cumprod(1.0 + rr)
    span = hi - lo + 1
    out = dict(
        trades=len(trades),
        expo=float((cc > 0).mean()),
        cagr=float(eq[-1] ** (TDAYS / span) - 1.0),
        sharpe=float(rr.mean() / rr.std() * np.sqrt(TDAYS)) if rr.std() > 0 else np.nan,
        maxdd=float((eq / np.maximum.accumulate(eq) - 1.0).min()),
        r=r, lo=lo, hi=hi,
    )
    return out


def drop_topk_sharpe(trades: pd.DataFrame, n_days: int, capacity: int, k: int) -> float:
    by = trades.groupby("vt_symbol")[RET].sum().sort_values(ascending=False)
    sub = trades[~trades["vt_symbol"].isin(set(by.head(k).index))]
    return stats(sub, n_days, capacity)["sharpe"]


def worst_year(trades: pd.DataFrame, cal: np.ndarray, capacity: int) -> tuple[int, float]:
    n = len(cal)
    r, _ = daily_series(trades, n, capacity)
    yrday = pd.DatetimeIndex(cal).year
    worst = (None, np.inf)
    for y in sorted(trades["year"].unique()):
        m = (yrday == y)
        ylo, yhi = int(np.argmax(m)), int(n - 1 - np.argmax(m[::-1]))
        rr = r[ylo:yhi + 1]
        sh = rr.mean() / rr.std() * np.sqrt(TDAYS) if rr.std() > 0 else np.nan
        if sh == sh and sh < worst[1]:
            worst = (y, float(sh))
    return worst


def pct(x: float) -> str:
    return "n/a" if x != x else f"{x * 100:+.1f}%"


def main() -> None:
    snap = pd.read_parquet(DEFAULT_SNAPSHOT_PATH)
    snap["signal_date"] = normalize_date(snap["signal_date"])
    cal = calendar_of(snap)
    n = len(cal)
    idx = pd.Series(np.arange(n), index=pd.DatetimeIndex(cal))

    cap = NEXT_OPEN_TOP_K * HOLD  # 6 per type
    trades = {bpt: to_trades(select_type(snap, bpt), idx) for bpt in TYPES}

    print("=" * 96)
    print("BUY-POINT SEARCH — each type through the SAME honest pipeline (no per-type re-tuning)")
    print(f"  fixed-capacity book, capacity={cap} slots/type;  honest Sharpe / CAGR / kill-test / worst year")
    print("=" * 96)
    print(f"  {'type':5}{'trades':>8}{'expo':>7}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>8}{'drop5-Sh':>10}{'worst-year':>16}")
    print("  " + "-" * 92)
    seldays = {}
    for bpt in TYPES:
        t = trades[bpt]
        s = stats(t, n, cap)
        d5 = drop_topk_sharpe(t, n, cap, 5)
        wy, wsh = worst_year(t, cal, cap)
        seldays[bpt] = set(t["entry_idx"].tolist())
        print(f"  {SHORT[bpt]:5}{s['trades']:>8}{s['expo'] * 100:>6.0f}%{pct(s['cagr']):>9}{s['sharpe']:>+8.2f}"
              f"{pct(s['maxdd']):>8}{d5:>+10.2f}{f'{wsh:+.2f}({wy})':>16}")

    # day-level orthogonality (do types fire on different entry days?)
    print("\n[time orthogonality]  entry-day Jaccard overlap (low = fills different days = additive exposure)")
    for i, a in enumerate(TYPES):
        for b in TYPES[i + 1:]:
            j = len(seldays[a] & seldays[b]) / max(len(seldays[a] | seldays[b]), 1)
            print(f"  {SHORT[a]} ∩ {SHORT[b]}:  {j * 100:4.0f}%")

    # stacked book: hold all three types' top2 each day (2 per type), equal capital
    print("\n[stacked]  run L1+L2+L3 simultaneously (2 each/day), equal capital across 3x6=18 slots")
    allt = pd.concat([trades[b] for b in TYPES], ignore_index=True)
    ss = stats(allt, n, cap * len(TYPES))
    l1 = stats(trades["L1_reversal_start"], n, cap)
    print(f"  {'L1 alone':16}  expo {l1['expo'] * 100:3.0f}%   CAGR {pct(l1['cagr']):>8}   Sharpe {l1['sharpe']:+.2f}   maxDD {pct(l1['maxdd'])}")
    print(f"  {'L1+L2+L3 stacked':16}  expo {ss['expo'] * 100:3.0f}%   CAGR {pct(ss['cagr']):>8}   Sharpe {ss['sharpe']:+.2f}   maxDD {pct(ss['maxdd'])}")
    for pair in (["L1_reversal_start", "L3_pullback_relay"], ["L1_reversal_start", "L2_reversal_confirm"]):
        p = pd.concat([trades[b] for b in pair], ignore_index=True)
        s = stats(p, n, cap * len(pair))
        lbl = "+".join(SHORT[b] for b in pair)
        print(f"  {lbl:16}  expo {s['expo'] * 100:3.0f}%   CAGR {pct(s['cagr']):>8}   Sharpe {s['sharpe']:+.2f}   maxDD {pct(s['maxdd'])}")

    print("\nRead: a new buy point earns its place only if it (a) clears the same gate standalone")
    print("(positive Sharpe, kill-test-robust, no dead year) AND (b) fires on DIFFERENT days so the")
    print("stacked book lifts exposure/CAGR without crushing Sharpe. Dilution => leave it out (the")
    print("crypto 'weak+orthogonal != additive' lesson).")


if __name__ == "__main__":
    main()
