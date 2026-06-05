"""
LLM-driven alpha factor search loop (Paradigm A, manual-generator variant).

Idea
----
vnpy.alpha already abstracts a factor into a *generatable, auto-scorable string*:
``calculate_by_expression(df, expr)`` evals the expression against the operator
vocabulary (ts_/cs_/ta_/math + OHLCV terminals). So any generator that emits a
legal DSL string + an automatic IC score = a closed search loop. Here the
generator is YOU, the LLM, talking in the conversation.

Each round:
  1. You read OPERATOR_VOCAB + the hypothesis + the scored ``history`` below.
  2. You propose a batch of expressions as a Python list.
  3. ``AlphaLoop.score_batch([...])`` computes an out-of-sample rank-IC for each,
     ranks them, and prints a report.
  4. You read the report and propose the next, better batch.

Why not ``AlphaDataset.show_feature_performance``?
  That draws an interactive alphalens tear sheet and returns ``None`` — useless
  for a loop. Scoring needs a *scalar*, so we compute cross-sectional rank-IC
  directly with polars (cheap, no per-round process pool, no alphalens).

Requires the ``alpha`` extra::

    pip install -e ".[alpha]"     # polars, alphalens-reloaded, ta-lib, ...

Anti-overfitting reminder
-------------------------
When you auto-generate many factors, some look great by luck (multiple testing).
The honest number is the TEST-segment IC. Watch the train/test gap; prefer
factors stable across segments with an economic story, not single-segment spikes.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import polars as pl

from vnpy.alpha.dataset.utility import calculate_by_expression, to_datetime


# Operator vocabulary — the alphabet you compose expressions from.
# Terminals (data columns): open, high, low, close, volume, vwap, turnover, open_interest
# Note: prices are normalized by the first close; vwap = turnover / volume.
# Quirk: in numeric ops put the variable before the constant, e.g. `close * 2`.
OPERATOR_VOCAB: str = """
TERMINALS : open high low close volume vwap turnover open_interest  (+ float/int constants)

ts_ (per-symbol, time-series; window is an int unless noted)
  ts_delay(x, w)            value w bars ago
  ts_delta(x, w)            x - ts_delay(x, w)
  ts_min/ts_max(x, w)       rolling min / max
  ts_argmax/ts_argmin(x, w) 1-based position of rolling max / min
  ts_rank(x, w)             percentile rank of current value in window (0..1)
  ts_sum/ts_mean/ts_std(x, w)
  ts_quantile(x, w, q)      rolling q-quantile, q in 0..1
  ts_slope/ts_rsquare/ts_resi(x, w)   rolling linear-regression vs time
  ts_corr(x, y, w)/ts_cov(x, y, w)    rolling corr / cov of two series
  ts_decay_linear(x, w)     linearly-decaying weighted mean
  ts_product(x, w)          rolling product
  ts_delay-friendly helpers : ts_less(x, y), ts_greater(x, y), ts_log(x), ts_abs(x)

cs_ (per-datetime, cross-section over the universe)
  cs_rank(x)                rank within the day
  cs_mean/cs_std/cs_sum(x)
  cs_scale(x)               x / sum(|x|) within the day

ta_ (TA-Lib, per-symbol)
  ta_rsi(close, w)          RSI
  ta_atr(high, low, close, w)  ATR

math / elementwise
  log(x) abs(x) sign(x)
  less(x, y) greater(x, y)          min / max of two
  pow1(x, c)  pow2(x, y)            safe powers (c float / y series)
  quesval(thr, x, a, b)            a if thr < x else b   (thr is a float)
  quesval2(thr, x, a, b)           same but thr is a series
  operators: + - * / // % **  abs  -x   and comparisons > >= < <= == !=
""".strip()


# The framework's default label: forward return from T+1 to T+3 (see alpha_101/158).
DEFAULT_LABEL_EXPR: str = "ts_delay(close, -3) / ts_delay(close, -1) - 1"


def _float_or_nan(value: Any) -> float:
    """Convert a scalar value from Polars to float."""
    if value is None:
        return float("nan")
    return float(value)


class AlphaLoop:
    """Score factor expressions by out-of-sample cross-sectional rank-IC.

    Parameters
    ----------
    bar_df:
        Long-format bars with columns ``datetime``, ``vt_symbol`` and the
        terminal feature columns (open/high/low/close/volume/vwap/...).
        Exactly what ``AlphaLab.load_bar_df`` returns.
    train_period / test_period:
        ``(start, end)`` date strings/datetimes. Train IC guides you; only the
        test IC is the honest score. (A valid split is omitted here on purpose —
        add one once you start tuning to avoid leaking the test set.)
    label_expr:
        Forward-return target. Defaults to the framework's T+1->T+3 return.
    min_universe:
        Skip any day whose cross-section has fewer than this many names (an IC
        from 2-3 stocks is noise).
    """

    def __init__(
        self,
        bar_df: pl.DataFrame,
        train_period: tuple[str, str],
        test_period: tuple[str, str],
        label_expr: str = DEFAULT_LABEL_EXPR,
        min_universe: int = 5,
    ) -> None:
        self.bar_df: pl.DataFrame = bar_df
        self.train_period: tuple[datetime, datetime] = (to_datetime(train_period[0]), to_datetime(train_period[1]))
        self.test_period: tuple[datetime, datetime] = (to_datetime(test_period[0]), to_datetime(test_period[1]))
        self.min_universe: int = min_universe

        # Precompute the label once; every candidate is scored against it.
        label_df: pl.DataFrame = calculate_by_expression(bar_df, label_expr)
        self.label_df: pl.DataFrame = label_df.rename({"data": "label"})

        self.history: list[dict] = []
        self._seen: dict[str, dict] = {}

        # Factor-panel cache so scoring, correlation and compositing reuse one computation.
        self._factors: dict[str, pl.DataFrame | None] = {}
        self._factor_err: dict[str, str] = {}

    # ------------------------------------------------------------------ scoring
    def factor_frame(self, expr: str) -> pl.DataFrame | None:
        """Compute (and cache) the [datetime, vt_symbol, factor] panel for an expression.

        Returns None on a bad expression (error stashed in ``self._factor_err``).
        """
        expr = expr.strip()
        if expr in self._factors:
            return self._factors[expr]

        try:
            f = calculate_by_expression(self.bar_df, expr).rename({"data": "factor"})
            self._factors[expr] = f
        except Exception as exc:                                    # bad DSL -> None, don't crash the loop
            self._factors[expr] = None
            self._factor_err[expr] = f"{type(exc).__name__}: {exc}"
        return self._factors[expr]

    def score(self, expr: str) -> dict:
        """Score a single expression; cached so re-proposing is free."""
        expr = expr.strip()
        if expr in self._seen:
            return self._seen[expr]

        rec: dict = {"expr": expr}
        f = self.factor_frame(expr)

        if f is None:
            rec.update(
                train_ic=float("nan"), train_icir=float("nan"), train_days=0,
                test_ic=float("nan"), test_icir=float("nan"), test_days=0,
                error=self._factor_err.get(expr, "unknown error"),
            )
        else:
            merged = f.join(self.label_df, on=["datetime", "vt_symbol"], how="inner")
            tr = self._segment_ic(merged, *self.train_period)
            te = self._segment_ic(merged, *self.test_period)
            rec.update(
                train_ic=tr["ic"], train_icir=tr["icir"], train_days=tr["days"],
                test_ic=te["ic"], test_icir=te["icir"], test_days=te["days"],
                error=None,
            )

        self._seen[expr] = rec
        self.history.append(rec)
        return rec

    def score_batch(self, exprs: list[str]) -> list[dict]:
        """Score a batch and print the round report. This is your per-round call."""
        recs = [self.score(e) for e in exprs]
        self.print_report()
        return recs

    def _segment_ic(self, merged: pl.DataFrame, start: datetime, end: datetime) -> dict:
        """Mean daily rank-IC and ICIR over a date range."""
        seg = (
            merged
            .filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))
            .with_columns(pl.col("factor").fill_nan(None), pl.col("label").fill_nan(None))
            .drop_nulls(["factor", "label"])
        )
        if seg.is_empty():
            return {"ic": float("nan"), "icir": float("nan"), "days": 0}

        # Rank-IC = Pearson corr of within-day ranks (== Spearman).
        seg = seg.with_columns(
            pl.col("factor").rank().over("datetime").alias("rf"),
            pl.col("label").rank().over("datetime").alias("rl"),
        )
        daily = (
            seg.group_by("datetime")
            .agg(pl.corr("rf", "rl").alias("ic"), pl.len().alias("n"))
            .filter(pl.col("n") >= self.min_universe)
        )
        ic = daily["ic"].drop_nulls().drop_nans()
        if ic.len() == 0:
            return {"ic": float("nan"), "icir": float("nan"), "days": 0}

        mean_ic = _float_or_nan(ic.mean())
        std_ic = _float_or_nan(ic.std()) if ic.len() > 1 else float("nan")
        icir = mean_ic / std_ic if std_ic and not math.isnan(std_ic) and std_ic != 0 else float("nan")
        return {"ic": mean_ic, "icir": icir, "days": int(ic.len())}

    # ----------------------------------------------------- orthogonality / combine
    def _period(self, segment: str) -> tuple[datetime, datetime]:
        return self.train_period if segment == "train" else self.test_period

    def _percentile_frame(self, expr: str, start: datetime | None = None, end: datetime | None = None) -> pl.DataFrame | None:
        """Per-day cross-sectional percentile rank (0..1) of a factor; the unit for combining."""
        f = self.factor_frame(expr)
        if f is None:
            return None
        if start is not None:
            f = f.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))
        return (
            f.with_columns(pl.col("factor").fill_nan(None))
            .drop_nulls("factor")
            .with_columns((pl.col("factor").rank().over("datetime") / pl.len().over("datetime")).alias("pr"))
            .select("datetime", "vt_symbol", "pr")
        )

    def corr_matrix(self, exprs: list[str], segment: str = "test") -> pl.DataFrame:
        """Pairwise rank correlation between factors (redundancy check).

        Uses per-day percentile ranks pooled over the segment (~average daily Spearman).
        Two factors with corr ~1 are the same bet; near 0 means orthogonal / additive.
        """
        start, end = self._period(segment)
        ranks: dict[str, pl.DataFrame] = {}
        for i, e in enumerate(exprs):
            r = self._percentile_frame(e, start, end)
            if r is not None:
                ranks[e] = r.rename({"pr": f"f{i}"})

        base: pl.DataFrame | None = None
        for r in ranks.values():
            base = r if base is None else base.join(r, on=["datetime", "vt_symbol"], how="inner")

        print(f"\nrank-corr matrix ({segment}, n={0 if base is None else base.height:,} pairs)")
        for i, e in enumerate(exprs):
            tag = f"f{i}" if e in ranks else f"f{i} (ERR)"
            print(f"  {tag:>3}  {e}")
        cols = [f"f{i}" for i, e in enumerate(exprs) if e in ranks]
        print("       " + " ".join(f"{c:>6}" for c in cols))
        for ci in cols:
            row = []
            for cj in cols:
                v = base.select(pl.corr(ci, cj)).item() if base is not None else float("nan")
                row.append(f"{v:>6.2f}" if v is not None else "   nan")
            print(f"  {ci:>4} " + " ".join(row))
        return base if base is not None else pl.DataFrame()

    def composite_frame(self, exprs: list[str], weights: list[float] | None = None) -> pl.DataFrame:
        """Combine factors into one panel: per-day percentile-rank each, weighted sum.

        Only coins present in *all* component factors on a given day survive (inner join).
        """
        weights = weights or [1.0] * len(exprs)

        base: pl.DataFrame | None = None
        for i, (e, w) in enumerate(zip(exprs, weights, strict=False)):
            pr = self._percentile_frame(e)
            if pr is None:
                raise ValueError(f"composite component failed: {e} -> {self._factor_err.get(e)}")
            pr = pr.select("datetime", "vt_symbol", (pl.col("pr") * w).alias(f"c{i}"))
            base = pr if base is None else base.join(pr, on=["datetime", "vt_symbol"], how="inner")

        if base is None:
            raise ValueError("No valid composite components")

        return (
            base.with_columns(pl.sum_horizontal([f"c{i}" for i in range(len(exprs))]).alias("factor"))
            .select("datetime", "vt_symbol", "factor")
        )

    def composite(self, exprs: list[str], weights: list[float] | None = None, name: str = "COMPOSITE") -> dict:
        """Build a composite factor and score its IC vs the best single component."""
        weights = weights or [1.0] * len(exprs)
        comp = self.composite_frame(exprs, weights)
        merged = comp.join(self.label_df, on=["datetime", "vt_symbol"], how="inner")

        tr = self._segment_ic(merged, *self.train_period)
        te = self._segment_ic(merged, *self.test_period)
        rec = {"expr": name, "train_ic": tr["ic"], "train_icir": tr["icir"],
               "test_ic": te["ic"], "test_icir": te["icir"], "error": None}

        best_single = max(
            (self.score(e)["test_ic"] for e in exprs if self.score(e)["error"] is None),
            key=abs, default=float("nan"),
        )
        print(f"\n{name}: testIC={te['ic']:+.4f} (IR {te['icir']:+.2f})  trainIC={tr['ic']:+.4f}  "
              f"| best single testIC={best_single:+.4f}  | {len(exprs)} factors, weights={weights}")
        return rec

    # ----------------------------------------------------- tradability backtest
    def backtest(
        self,
        factor_df: pl.DataFrame,
        segment: str = "test",
        quantile: float = 0.2,
        hold: int = 1,
        fee: float = 0.0005,
        winsor: float | None = None,
        long_only: bool = False,
        name: str = "portfolio",
    ) -> dict:
        """Vectorized dollar-neutral long-short decile backtest with turnover costs.

        Long the top `quantile`, short the bottom `quantile`, equal-weight each leg
        (gross exposure 2x, dollar-neutral). Rebalance every `hold` bars, charging
        `fee` per unit of weight traded. This — not IC — tells you if a factor survives
        costs. Low-turnover factors (low-vol) should net positive; fast signals may not.

        `winsor`: clip per-bar forward returns to +/-winsor before P&L (tames crypto's fat
        right tail, which otherwise blows up the short leg even when rank-IC is positive).
        `long_only`: hold the top quantile only (vs equal-weight universe) — sidesteps the
        short-side lottery blowup entirely.
        """
        bars_per_year = 2190.0 / hold                      # 4h bars: 6/day * 365

        # K-bar forward return per symbol
        px = (
            self.bar_df.select("datetime", "vt_symbol", "close")
            .sort(["vt_symbol", "datetime"])
            .with_columns((pl.col("close").shift(-hold).over("vt_symbol") / pl.col("close") - 1.0).alias("fwd"))
        )
        start, end = self._period(segment)
        df = (
            factor_df.join(px.select("datetime", "vt_symbol", "fwd"), on=["datetime", "vt_symbol"], how="inner")
            .with_columns(pl.col("factor").fill_nan(None))
            .drop_nulls(["factor", "fwd"])
            .filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))
        )

        if winsor:
            df = df.with_columns(pl.col("fwd").clip(-winsor, winsor))

        # Keep only every `hold`-th timestamp as a rebalance date
        dts = df.select("datetime").unique().sort("datetime")["datetime"]
        rebal = dts.gather(list(range(0, dts.len(), hold)))
        df = df.filter(pl.col("datetime").is_in(rebal))

        # Long/short weights, equal-weight per leg, dollar-neutral
        df = df.with_columns((pl.col("factor").rank().over("datetime") / pl.len().over("datetime")).alias("pr"))
        df = df.with_columns(
            pl.when(pl.col("pr") >= 1 - quantile).then(1.0)
            .when(pl.col("pr") <= quantile).then(-1.0)
            .otherwise(0.0).alias("side")
        )
        df = df.with_columns(
            (pl.col("side") == 1).sum().over("datetime").alias("nL"),
            (pl.col("side") == -1).sum().over("datetime").alias("nS"),
            pl.len().over("datetime").alias("N"),
        )
        if long_only:
            # Long top quantile vs a diffuse short of the whole universe (no concentrated lottery short)
            df = df.with_columns(
                (pl.when(pl.col("side") == 1).then(1.0 / pl.col("nL")).otherwise(0.0) - 1.0 / pl.col("N")).alias("w")
            )
        else:
            df = df.with_columns(
                pl.when(pl.col("side") == 1).then(1.0 / pl.col("nL"))
                .when(pl.col("side") == -1).then(-1.0 / pl.col("nS"))
                .otherwise(0.0).alias("w")
            )

        # Gross return per rebalance
        port = df.group_by("datetime").agg((pl.col("w") * pl.col("fwd")).sum().alias("gross")).sort("datetime")

        # Turnover = sum |w_t - w_{t-1}| (prev weight per symbol across rebalances)
        wdf = (
            df.select("datetime", "vt_symbol", "w").sort(["vt_symbol", "datetime"])
            .with_columns(pl.col("w").shift(1).over("vt_symbol").fill_null(0.0).alias("w_prev"))
        )
        turn = wdf.group_by("datetime").agg((pl.col("w") - pl.col("w_prev")).abs().sum().alias("turnover"))
        port = port.join(turn, on="datetime", how="left").sort("datetime").with_columns(
            (pl.col("gross") - fee * pl.col("turnover")).alias("net")
        )

        def sharpe(s: pl.Series) -> float:
            mean: float = _float_or_nan(s.mean())
            std: float = _float_or_nan(s.std())
            return mean / std * math.sqrt(bars_per_year) if std else float("nan")

        gross_shp, net_shp = sharpe(port["gross"]), sharpe(port["net"])
        net_total = _float_or_nan((1.0 + port["net"]).cum_prod()[-1]) - 1.0
        avg_turn = _float_or_nan(port["turnover"].mean())
        ann_net = _float_or_nan(port["net"].mean()) * bars_per_year

        print(
            f"\n[{name}] {segment}  hold={hold}bar  q={quantile}  fee={fee*1e4:.0f}bp\n"
            f"   gross Sharpe={gross_shp:5.2f}   NET Sharpe={net_shp:5.2f}\n"
            f"   ann net return={ann_net*100:6.1f}%   total net={net_total*100:6.1f}%\n"
            f"   avg turnover/rebal={avg_turn:.2f}   fee drag/yr={fee*avg_turn*bars_per_year*100:5.1f}%   rebals={port.height}"
        )
        return {"name": name, "gross_sharpe": gross_shp, "net_sharpe": net_shp,
                "ann_net": ann_net, "net_total": net_total, "avg_turnover": avg_turn}

    # ------------------------------------------------------------------ reports
    def print_report(self, top: int | None = None) -> None:
        """Print all scored factors ranked by |test IC| (the honest score)."""
        rows = sorted(
            self.history,
            key=lambda r: (-abs(r["test_ic"]) if not math.isnan(r["test_ic"]) else float("inf")),
        )
        if top:
            rows = rows[:top]

        print(f"\n{'#':>2}  {'testIC':>8} {'testIR':>7}  {'trainIC':>8} {'trainIR':>7}  expr")
        print("-" * 96)
        for i, r in enumerate(rows, 1):
            if r["error"]:
                print(f"{i:>2}  {'ERR':>8} {'':>7}  {'':>8} {'':>7}  {r['expr'][:48]}   <- {r['error'][:40]}")
                continue
            print(
                f"{i:>2}  {r['test_ic']:>8.4f} {r['test_icir']:>7.2f}  "
                f"{r['train_ic']:>8.4f} {r['train_icir']:>7.2f}  {r['expr']}"
            )
        print("-" * 96)
        print("Honest score = testIC. Big trainIC but small testIC = overfit. testIR = IC mean/std.\n")

    def report_text(self, top: int | None = 20) -> str:
        """Compact history string to paste back to the LLM as feedback."""
        rows = sorted(
            self.history,
            key=lambda r: (-abs(r["test_ic"]) if not math.isnan(r["test_ic"]) else float("inf")),
        )[: top or len(self.history)]
        lines = [f"testIC={r['test_ic']:+.4f} trainIC={r['train_ic']:+.4f} | {r['expr']}"
                 if not r["error"] else f"ERROR({r['error'][:30]}) | {r['expr']}"
                 for r in rows]
        return "\n".join(lines)


# ------------------------------------------------------------------- data loaders
def load_panel(parquet_path: str, start: str | None = None, min_close: float = 0.0) -> pl.DataFrame:
    """Load a long OHLCV panel exported from the claude-trader candle_cache.

    Columns: datetime, vt_symbol, open/high/low/close/volume/turnover/vwap/open_interest.
    Casts numerics to Float64 (the export is Float32 / Decimal, which can lose precision
    in the rolling-regression operators) and optionally trims early thin-cross-section days.

    NOTE — scale-free factor design: prices here are NOT normalized per coin. Cross-sectional
    factors must therefore use returns / ratios / ranks, never raw price *levels* or *deltas*
    (BTC moving $1000 vs a meme coin moving $0.001 would otherwise dominate every ranking).
    """
    df = pl.read_parquet(parquet_path)

    numeric = ["open", "high", "low", "close", "volume", "turnover", "vwap", "open_interest"]
    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in numeric if c in df.columns])

    if start:
        df = df.filter(pl.col("datetime") >= to_datetime(start))
    if min_close:
        df = df.filter(pl.col("close") >= min_close)

    return df.sort(["datetime", "vt_symbol"])


# ---------------------------------------------------------------------------- demo
def make_demo_bars(n_symbols: int = 30, n_days: int = 600, seed: int = 0) -> pl.DataFrame:
    """Synthetic OHLCV panel for smoke-testing the loop mechanics (no real data needed)."""
    import numpy as np

    rng = np.random.default_rng(seed)

    # Contiguous daily datetime index of length n_days.
    base = pl.date_range(datetime(2020, 1, 1), datetime(2030, 1, 1), "1d", eager=True).cast(pl.Datetime).head(n_days)

    frames = []
    for s in range(n_symbols):
        ret = rng.normal(0.0003, 0.02, n_days)
        close = 100 * np.exp(np.cumsum(ret))
        high = close * (1 + rng.uniform(0, 0.02, n_days))
        low = close * (1 - rng.uniform(0, 0.02, n_days))
        open_ = low + rng.uniform(0, 1, n_days) * (high - low)
        volume = rng.uniform(1e6, 5e6, n_days)
        turnover = volume * close
        c0 = float(close[0])
        frames.append(pl.DataFrame({
            "datetime": base,
            "vt_symbol": [f"S{s:03d}.DEMO"] * n_days,
            "open": open_ / c0, "high": high / c0, "low": low / c0, "close": close / c0,
            "volume": volume, "turnover": turnover, "open_interest": [0.0] * n_days,
            "vwap": (turnover / volume) / c0,
        }))
    return pl.concat(frames).sort(["datetime", "vt_symbol"])


if __name__ == "__main__":
    # Smoke test on synthetic data. With real data, build bar_df via:
    #   from vnpy.alpha import AlphaLab
    #   lab = AlphaLab("./lab/csi300")
    #   symbols = lab.load_component_symbols("000300.SSE", "2020-01-01", "2023-01-01")
    #   bar_df = lab.load_bar_df(symbols, "d", "2020-01-01", "2023-01-01", extended_days=60)
    print(OPERATOR_VOCAB)

    bars = make_demo_bars()
    loop = AlphaLoop(bars, train_period=("2020-01-01", "2021-03-31"), test_period=("2021-04-01", "2021-08-31"))

    # ---- One round. Replace this list each round with your fresh proposals. ----
    loop.score_batch([
        "-1 * ts_corr(open, volume, 10)",                       # alpha6: price-volume divergence
        "cs_rank(ts_delta(close, 5))",                          # 5-day momentum, cross-sectional
        "-1 * cs_rank(ts_std(close, 20))",                      # low-vol preference
        "ts_corr(vwap, volume, 15)",                            # vwap-volume co-movement
        "cs_rank(-1 * ts_delta(close, 1) * sign(ts_delta(volume, 1)))",  # reversal gated by volume
    ])
