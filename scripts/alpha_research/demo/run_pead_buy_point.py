"""run_pead_buy_point.py — a GENUINELY NEW, orthogonal buy point: post-earnings drift (PEAD).

The technical buy-point space is mined to saturation (the alpha-highlights Combo Expansion shows
every technical add — low-vol/turnover/box/double-bottom/PB — DILUTES core4; L2/L3 dilute too). The
only factor orthogonal to the reversal/turnover/pvcorr complex in the candidate corr-test is
fundamental growth. So this tests the A-share analogue of the crypto "new data axis" move: an
EARNINGS-EVENT buy point, orthogonal to price by construction.

Signal (all from research_panel_daily, look-ahead-safe — fi_ann_date<=trade_date is 99.7%, enforced):
  * EVENT  = a fresh report drops: fi_ann_date changes vs the prior trading row (and is already public).
  * SURPRISE = fi_netprofit_yoy (net-profit YoY) of that just-released report; rank events each day,
    take the top-K highest-growth (PEAD: strong earnings -> upward drift).
  * ENTRY  = next-day open, net of costs (5bp buy / 15bp sell, same as the L1 book); exclude opens
    locked at up-limit (unfillable), ST/new/invalid-life, BSE, circ_mv<20e8, turnover<0.5%.

Scored through the SAME honest position-level portfolio (capacity = K*hold slots, idle=cash) as the
L1 audit, then: hold x K sweep, kill-test, per-year, and the decisive test — is PEAD time-orthogonal
to L1 and does STACKING it on the L1 book add exposure/return without crushing Sharpe (vs the L2/L3
dilution). Prior is GUARDED: A-share reversal is strong and may front-run earnings; honest test decides.

Run: rtk python3 scripts/alpha_research/demo/run_pead_buy_point.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "tradingview", "research"))
from run_l1_clean_timeline_backtest import DEFAULT_SNAPSHOT_PATH, normalize_date  # noqa: E402
from run_l1_honest_portfolio import build_clean_timeline_selection  # noqa: E402

PANEL_GLOB = "lab/a_share_research/panel/research_panel_daily/**/*.parquet"
OPEN_RATE, CLOSE_RATE = 0.0005, 0.0015
MIN_CIRC_MV, MIN_TURN, TDAYS = 200000.0, 0.5, 244.0
HOLDS = [3, 5, 10]
KS = [2, 5, 10]


def load_panel() -> pd.DataFrame:
    want = ["trade_date", "vt_symbol", "open", "close", "up_limit", "name", "list_date",
            "is_list_life_valid", "has_valid_bar", "circ_mv", "turnover_rate", "fi_ann_date",
            "fi_netprofit_yoy"]
    fs = sorted(glob.glob(PANEL_GLOB, recursive=True))
    avail = set(pq.read_schema(fs[0]).names)
    cols = [c for c in want if c in avail]
    if set(want) - avail:
        print(f"  [warn] panel missing {sorted(set(want) - avail)} (skipped)", file=sys.stderr)
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["fi_ann_date"] = pd.to_datetime(df["fi_ann_date"], errors="coerce")
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    return df.sort_values(["vt_symbol", "trade_date"]).reset_index(drop=True)


def _b(s: pd.Series, default: bool) -> pd.Series:
    return s.astype("boolean").fillna(default).astype(bool)


def build_events(df: pd.DataFrame) -> pd.DataFrame:
    """Mark earnings-event rows + carry next-day (entry) tradability fields. Hold-independent."""
    g = df.groupby("vt_symbol", sort=False)
    df["event"] = (df["fi_ann_date"].notna()
                   & (df["fi_ann_date"] != g["fi_ann_date"].shift(1))
                   & (df["fi_ann_date"] <= df["trade_date"]))
    for c in ["trade_date", "open", "up_limit", "name", "list_date", "is_list_life_valid",
              "has_valid_bar", "circ_mv", "turnover_rate"]:
        df[f"n_{c}"] = g[c].shift(-1)
    return df


def trades_for_hold(df: pd.DataFrame, hold: int) -> pd.DataFrame:
    g = df.groupby("vt_symbol", sort=False)
    gross = g["close"].shift(-hold) / df["n_open"] - 1.0
    net = (1.0 + gross) * (1.0 - CLOSE_RATE) / (1.0 + OPEN_RATE) - 1.0
    age = (pd.to_datetime(df["n_trade_date"]) - df["n_list_date"]).dt.days
    is_st = df["n_name"].astype(str).str.contains("ST", case=False, na=False)
    ok = (df["event"]
          & _b(df["n_has_valid_bar"], True) & _b(df["n_is_list_life_valid"], True)
          & ~is_st & (age >= 60)
          & (df["n_open"] < df["n_up_limit"] * 0.999)
          & (pd.to_numeric(df["n_circ_mv"], errors="coerce") >= MIN_CIRC_MV)
          & (pd.to_numeric(df["n_turnover_rate"], errors="coerce") >= MIN_TURN)
          & ~df["vt_symbol"].astype(str).str.endswith(".BSE")
          & net.notna() & df["fi_netprofit_yoy"].notna())
    t = df.loc[ok, ["n_trade_date", "vt_symbol", "fi_netprofit_yoy"]].copy()
    t["ret"] = net[ok].to_numpy()
    t["entry_date"] = pd.to_datetime(t["n_trade_date"])
    t["growth"] = pd.to_numeric(t["fi_netprofit_yoy"], errors="coerce")
    t["year"] = t["entry_date"].dt.year
    return t


def topk_per_day(t: pd.DataFrame, k: int) -> pd.DataFrame:
    return t.sort_values(["entry_date", "growth"], ascending=[True, False]).groupby("entry_date").head(k)


# ----------------------------------------------------------------- honest portfolio (capacity model)
def daily_series(trades: pd.DataFrame, n: int, hold: int, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    acc = np.zeros(n)
    cnt = np.zeros(n)
    amort = (1.0 + trades["ret"].to_numpy()) ** (1.0 / hold) - 1.0
    for a, j in zip(amort, trades["eidx"].to_numpy(), strict=False):
        hi = min(j + hold, n)
        acc[j:hi] += a
        cnt[j:hi] += 1
    return acc / capacity, cnt


def stats(trades: pd.DataFrame, n: int, hold: int, capacity: int) -> dict:
    if trades.empty:
        return dict(trades=0, expo=np.nan, cagr=np.nan, sharpe=np.nan, maxdd=np.nan)
    r, cnt = daily_series(trades, n, hold, capacity)
    lo = int(trades["eidx"].min())
    hi = int(min(trades["eidx"].max() + hold - 1, n - 1))
    rr, cc = r[lo:hi + 1], cnt[lo:hi + 1]
    eq = np.cumprod(1.0 + rr)
    span = hi - lo + 1
    return dict(trades=len(trades), expo=float((cc > 0).mean()),
                cagr=float(eq[-1] ** (TDAYS / span) - 1.0),
                sharpe=float(rr.mean() / rr.std() * np.sqrt(TDAYS)) if rr.std() > 0 else np.nan,
                maxdd=float((eq / np.maximum.accumulate(eq) - 1.0).min()))


def pct(x: float) -> str:
    return "n/a" if x != x else f"{x * 100:+.1f}%"


def attach_idx(trades: pd.DataFrame, idx: pd.Series) -> pd.DataFrame:
    t = trades.copy()
    t["eidx"] = t["entry_date"].map(idx)
    t = t[t["eidx"].notna()].copy()
    t["eidx"] = t["eidx"].astype(int)
    return t


def main() -> None:
    df = build_events(load_panel())
    cal = np.sort(df["trade_date"].unique())
    n = len(cal)
    idx = pd.Series(np.arange(n), index=pd.DatetimeIndex(cal))
    n_events = int(df["event"].sum())
    print("=" * 96)
    print("PEAD BUY POINT — earnings-event entry, ranked by net-profit YoY, through the honest engine")
    print(f"  panel {len(df):,} rows, {df['vt_symbol'].nunique()} symbols, {cal[0].astype('datetime64[D]')}"
          f"..{cal[-1].astype('datetime64[D]')};  raw earnings events: {n_events:,}")
    print("=" * 96)
    print(f"  {'hold':>5}{'K':>4}{'trades':>8}{'expo':>7}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>8}{'medGrowth':>11}")
    print("  " + "-" * 92)
    best = None
    full = {h: attach_idx(trades_for_hold(df, h), idx) for h in HOLDS}
    for h in HOLDS:
        for k in KS:
            t = attach_idx(topk_per_day(full[h].drop(columns="eidx"), k), idx)
            s = stats(t, n, h, k * h)
            mg = t["growth"].median()
            print(f"  {h:>5}{k:>4}{s['trades']:>8}{s['expo'] * 100:>6.0f}%{pct(s['cagr']):>9}"
                  f"{s['sharpe']:>+8.2f}{pct(s['maxdd']):>8}{mg:>+10.0f}%")
            if best is None or (s["sharpe"] == s["sharpe"] and s["sharpe"] > best[0]):
                best = (s["sharpe"], h, k, t)

    _, bh, bk, bt = best
    cap = bk * bh
    print(f"\n[best config] hold={bh} K={bk}  (Sharpe {best[0]:+.2f})")
    by = bt.groupby("vt_symbol")["ret"].sum().sort_values(ascending=False)
    for kk in (3, 5, 10):
        sub = bt[~bt["vt_symbol"].isin(set(by.head(kk).index))]
        print(f"  [kill-test] drop top-{kk:>2} names -> Sharpe {stats(sub, n, bh, cap)['sharpe']:+.2f}")
    print("  [per-year] ", end="")
    yrday = pd.DatetimeIndex(cal).year
    for y in sorted(bt["year"].unique()):
        m = (yrday == y)
        r, _ = daily_series(bt, n, bh, cap)
        ylo, yhi = int(np.argmax(m)), int(n - 1 - np.argmax(m[::-1]))
        rr = r[ylo:yhi + 1]
        sh = rr.mean() / rr.std() * np.sqrt(TDAYS) if rr.std() > 0 else np.nan
        print(f"{y}:{sh:+.1f} ", end="")
    print()

    # decisive: orthogonality + stacking vs the L1 book (does PEAD ADD, unlike L2/L3?)
    l1 = build_clean_timeline_selection(pd.read_parquet(DEFAULT_SNAPSHOT_PATH).assign(
        signal_date=lambda d: normalize_date(d["signal_date"])))
    l1 = l1[l1["suggested_entry_date"].notna() & l1["fixed_3d_return"].notna()].copy()
    l1t = pd.DataFrame({"entry_date": normalize_date(l1["suggested_entry_date"]),
                        "vt_symbol": l1["vt_symbol"], "ret": l1["fixed_3d_return"].to_numpy()})
    l1t = attach_idx(l1t[l1t["entry_date"].notna()], idx)
    l1s = stats(l1t, n, 3, 2 * 3)
    pday = set(bt["eidx"])
    lday = set(l1t["eidx"])
    jacc = len(pday & lday) / max(len(pday | lday), 1)
    print(f"\n[vs L1]  L1(fixed3,K2): expo {l1s['expo'] * 100:.0f}%  CAGR {pct(l1s['cagr'])}  Sharpe {l1s['sharpe']:+.2f}")
    print(f"         PEAD entry-day overlap with L1 (Jaccard): {jacc * 100:.0f}%  (low = orthogonal in time)")
    # stack: hold both books, equal capital across the union of slots
    stack = pd.concat([bt.assign(_h=bh), l1t.assign(_h=3)], ignore_index=True)
    acc = np.zeros(n)
    for _, row in stack.iterrows():
        h = int(row["_h"])
        j = int(row["eidx"])
        hi = min(j + h, n)
        acc[j:hi] += (1.0 + row["ret"]) ** (1.0 / h) - 1.0
    capstack = cap + 6
    r = acc / capstack
    lo, hi = min(bt["eidx"].min(), l1t["eidx"].min()), max(bt["eidx"].max() + bh, l1t["eidx"].max() + 3)
    rr = r[lo:hi]
    eq = np.cumprod(1 + rr)
    sh = rr.mean() / rr.std() * np.sqrt(TDAYS) if rr.std() > 0 else np.nan
    expo = (np.array([1 if x != 0 else 0 for x in rr]).mean())
    print(f"[stacked L1+PEAD]  expo {expo * 100:.0f}%  CAGR {pct(eq[-1] ** (TDAYS / len(rr)) - 1):>8}  "
          f"Sharpe {sh:+.2f}  maxDD {pct((eq / np.maximum.accumulate(eq) - 1).min())}")
    print("\nRead: PEAD earns its place only if standalone-positive (kill-test-robust, no dead year) AND")
    print("stacking it LIFTS the L1 book (exposure/CAGR up, Sharpe not crushed) — unlike L2/L3 which diluted.")


if __name__ == "__main__":
    main()
