"""
Statistical alpha factor evaluation loop.

This module separates factor *proposal* from factor *validation*. A candidate
factor can come from a hand-written expression, a parameter grid, genetic search,
or an LLM; the loop always scores it with the same train/valid/test protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from vnpy.alpha.dataset.utility import calculate_by_expression, to_datetime


DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 20)
SEGMENTS: tuple[str, ...] = ("train", "valid", "test")


@dataclass(frozen=True)
class SegmentMetrics:
    """Scalar metrics for one factor on one segment and horizon."""

    ic: float
    icir: float
    days: int
    positive_rate: float
    samples_mean: float
    spread: float
    spread_ir: float


def forward_return_expr(horizon: int, entry_lag: int = 1) -> str:
    """Build a delayed-entry forward-return label expression.

    ``entry_lag=1, horizon=5`` means the signal is observed on T, the simulated
    entry price is T+1 close, and the exit price is T+6 close.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if entry_lag < 0:
        raise ValueError("entry_lag must be non-negative")

    exit_lag: int = entry_lag + horizon
    if entry_lag == 0:
        return f"ts_delay(close, -{exit_lag}) / close - 1"
    return f"ts_delay(close, -{exit_lag}) / ts_delay(close, -{entry_lag}) - 1"


def dedupe_expressions(exprs: list[str]) -> list[str]:
    """Deduplicate expressions while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for expr in exprs:
        text = expr.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def classic_price_expressions(windows: tuple[int, ...] = (5, 10, 20, 60, 120)) -> list[str]:
    """Generate a compact benchmark grid of non-fundamental OHLCV factors."""
    exprs: list[str] = []
    for window in windows:
        exprs.extend(
            [
                f"cs_rank(close / ts_delay(close, {window}) - 1)",
                f"-1 * cs_rank(close / ts_delay(close, {window}) - 1)",
                f"cs_rank((close - ts_min(low, {window})) / (ts_max(high, {window}) - ts_min(low, {window}) + 1e-12))",
                f"-1 * cs_rank(ts_std(close / ts_delay(close, 1) - 1, {window}))",
                f"cs_rank(volume / (ts_mean(volume, {window}) + 1e-12))",
                f"cs_rank(turnover / (ts_mean(turnover, {window}) + 1e-12))",
                f"-1 * cs_rank(ts_corr(close, volume, {window}))",
            ]
        )
    return dedupe_expressions(exprs)


def _empty_metrics() -> SegmentMetrics:
    """Return empty metrics for a segment with no valid observations."""
    return SegmentMetrics(
        ic=float("nan"),
        icir=float("nan"),
        days=0,
        positive_rate=float("nan"),
        samples_mean=float("nan"),
        spread=float("nan"),
        spread_ir=float("nan"),
    )


def _float_or_nan(value: Any) -> float:
    """Convert a scalar value from Polars to float."""
    if value is None:
        return float("nan")
    return float(value)


def _mean_std_ir(series: pl.Series) -> tuple[float, float, float]:
    """Return mean, standard deviation, and mean/std."""
    clean: pl.Series = series.drop_nulls().drop_nans()
    if clean.is_empty():
        return float("nan"), float("nan"), float("nan")

    mean: float = _float_or_nan(clean.mean())
    std: float = _float_or_nan(clean.std()) if clean.len() > 1 else float("nan")
    ir: float = mean / std if std and not math.isnan(std) else float("nan")
    return mean, std, ir


class StatAlphaLoop:
    """Score factor expressions with a train/valid/test statistical protocol."""

    def __init__(
        self,
        bar_df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        entry_lag: int = 1,
        min_universe: int = 300,
        quantile: float = 0.2,
        eligibility_df: pl.DataFrame | None = None,
    ) -> None:
        """Create the evaluation loop."""
        self.bar_df: pl.DataFrame = bar_df.sort(["datetime", "vt_symbol"])
        self.periods: dict[str, tuple[datetime, datetime]] = {
            "train": (to_datetime(train_period[0]), to_datetime(train_period[1])),
            "valid": (to_datetime(valid_period[0]), to_datetime(valid_period[1])),
            "test": (to_datetime(test_period[0]), to_datetime(test_period[1])),
        }
        self.horizons: tuple[int, ...] = tuple(sorted(set(horizons)))
        self.entry_lag: int = entry_lag
        self.min_universe: int = min_universe
        self.quantile: float = quantile
        self.eligibility_df: pl.DataFrame | None = self._normalize_eligibility(eligibility_df)

        self.label_dfs: dict[int, pl.DataFrame] = {
            horizon: calculate_by_expression(
                self.bar_df,
                forward_return_expr(horizon, entry_lag),
            ).rename({"data": "label"})
            for horizon in self.horizons
        }

        self.history: list[dict[str, Any]] = []
        self._seen: dict[str, list[dict[str, Any]]] = {}
        self._factors: dict[str, pl.DataFrame | None] = {}
        self._factor_errors: dict[str, str] = {}

    def factor_frame(self, expr: str) -> pl.DataFrame | None:
        """Calculate and cache a factor expression panel."""
        expr = expr.strip()
        if expr in self._factors:
            return self._factors[expr]

        try:
            factor_df: pl.DataFrame = calculate_by_expression(self.bar_df, expr).rename({"data": "factor"})
            self._factors[expr] = factor_df
        except Exception as exc:
            self._factors[expr] = None
            self._factor_errors[expr] = f"{type(exc).__name__}: {exc}"

        return self._factors[expr]

    def score(self, expr: str) -> list[dict[str, Any]]:
        """Score one expression for every configured horizon."""
        expr = expr.strip()
        if expr in self._seen:
            return self._seen[expr]

        factor_df: pl.DataFrame | None = self.factor_frame(expr)
        records: list[dict[str, Any]] = []

        for horizon in self.horizons:
            record: dict[str, Any] = {"expr": expr, "horizon": horizon}
            if factor_df is None:
                record["error"] = self._factor_errors.get(expr, "unknown factor error")
                for segment in SEGMENTS:
                    self._write_metrics(record, segment, _empty_metrics())
                records.append(record)
                continue

            merged: pl.DataFrame = factor_df.join(
                self.label_dfs[horizon],
                on=["datetime", "vt_symbol"],
                how="inner",
            )
            if self.eligibility_df is not None:
                merged = merged.join(self.eligibility_df, on=["datetime", "vt_symbol"], how="left")
                merged = merged.filter(pl.col("eligible").fill_null(False))

            record["error"] = None
            for segment in SEGMENTS:
                start, end = self.periods[segment]
                self._write_metrics(record, segment, self._segment_metrics(merged, start, end))
            records.append(record)

        self._seen[expr] = records
        self.history.extend(records)
        return records

    def score_batch(self, exprs: list[str], top: int | None = 20) -> list[dict[str, Any]]:
        """Score a candidate batch and print a compact report."""
        records: list[dict[str, Any]] = []
        for expr in exprs:
            records.extend(self.score(expr))
        self.print_report(top)
        return records

    def report_text(self, top: int | None = 20) -> str:
        """Return compact feedback text for the next proposal round."""
        rows: list[dict[str, Any]] = self._ranked_history(top)
        lines: list[str] = []
        for record in rows:
            if record["error"]:
                lines.append(f"ERROR({record['error']}) h={record['horizon']} | {record['expr']}")
            else:
                lines.append(
                    f"h={record['horizon']} testIC={record['test_ic']:+.4f} "
                    f"validIC={record['valid_ic']:+.4f} trainIC={record['train_ic']:+.4f} | {record['expr']}"
                )
        return "\n".join(lines)

    def history_frame(self) -> pl.DataFrame:
        """Return all scored records as a DataFrame."""
        if not self.history:
            return pl.DataFrame()
        return pl.DataFrame(self.history)

    def save_history(self, path: str | Path) -> None:
        """Persist scored history to CSV for later proposal rounds."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_frame().write_csv(output_path)

    def print_report(self, top: int | None = 20) -> None:
        """Print all scored candidates ranked by absolute test IC."""
        rows: list[dict[str, Any]] = self._ranked_history(top)

        print(f"\n{'#':>2} {'h':>3} {'testIC':>8} {'testIR':>7} {'validIC':>8} {'trainIC':>8} {'spread':>8} expr")
        print("-" * 112)
        for index, record in enumerate(rows, 1):
            if record["error"]:
                print(f"{index:>2} {record['horizon']:>3} {'ERR':>8} {'':>7} {'':>8} {'':>8} {'':>8} {record['expr']} <- {record['error']}")
                continue
            print(
                f"{index:>2} {record['horizon']:>3} "
                f"{record['test_ic']:>8.4f} {record['test_icir']:>7.2f} "
                f"{record['valid_ic']:>8.4f} {record['train_ic']:>8.4f} "
                f"{record['test_spread']:>8.4f} {record['expr']}"
            )
        print("-" * 112)
        print("Rank by |testIC| for review only; use validIC while iterating and keep test for final confirmation.\n")

    def rank_corr_matrix(self, exprs: list[str], segment: str = "valid") -> pl.DataFrame:
        """Return pooled per-day rank-correlation data for factor redundancy checks."""
        start, end = self.periods[segment]
        base: pl.DataFrame | None = None
        names: list[str] = []

        for index, expr in enumerate(exprs):
            frame: pl.DataFrame | None = self.factor_frame(expr)
            if frame is None:
                continue

            column: str = f"f{index}"
            ranked: pl.DataFrame = (
                frame.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))
                .with_columns(pl.col("factor").fill_nan(None))
                .drop_nulls("factor")
                .with_columns((pl.col("factor").rank().over("datetime") / pl.len().over("datetime")).alias(column))
                .select("datetime", "vt_symbol", column)
            )
            base = ranked if base is None else base.join(ranked, on=["datetime", "vt_symbol"], how="inner")
            names.append(column)

        if base is None:
            return pl.DataFrame()

        rows: list[dict[str, float | str]] = []
        for left in names:
            row: dict[str, float | str] = {"factor": left}
            for right in names:
                value: float | None = base.select(pl.corr(left, right)).item()
                row[right] = float(value) if value is not None else float("nan")
            rows.append(row)
        return pl.DataFrame(rows)

    @staticmethod
    def _normalize_eligibility(eligibility_df: pl.DataFrame | None) -> pl.DataFrame | None:
        """Normalize an optional eligibility panel."""
        if eligibility_df is None:
            return None
        required: set[str] = {"datetime", "vt_symbol", "eligible"}
        missing: set[str] = required.difference(eligibility_df.columns)
        if missing:
            raise ValueError(f"eligibility_df missing columns: {sorted(missing)}")
        return eligibility_df.select("datetime", "vt_symbol", pl.col("eligible").cast(pl.Boolean))

    @staticmethod
    def _write_metrics(record: dict[str, Any], segment: str, metrics: SegmentMetrics) -> None:
        """Flatten metrics into a score record."""
        record[f"{segment}_ic"] = metrics.ic
        record[f"{segment}_icir"] = metrics.icir
        record[f"{segment}_days"] = metrics.days
        record[f"{segment}_positive_rate"] = metrics.positive_rate
        record[f"{segment}_samples_mean"] = metrics.samples_mean
        record[f"{segment}_spread"] = metrics.spread
        record[f"{segment}_spread_ir"] = metrics.spread_ir

    def _segment_metrics(self, merged: pl.DataFrame, start: datetime, end: datetime) -> SegmentMetrics:
        """Calculate daily RankIC and top-bottom spread metrics for one segment."""
        segment: pl.DataFrame = (
            merged.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))
            .with_columns(pl.col("factor").fill_nan(None), pl.col("label").fill_nan(None))
            .drop_nulls(["factor", "label"])
        )
        if segment.is_empty():
            return _empty_metrics()

        ranked: pl.DataFrame = (
            segment.with_columns(
                pl.col("factor").rank().over("datetime").alias("factor_rank"),
                pl.col("label").rank().over("datetime").alias("label_rank"),
                pl.len().over("datetime").alias("sample_count"),
            )
            .filter(pl.col("sample_count") >= self.min_universe)
            .with_columns((pl.col("factor_rank") / pl.col("sample_count")).alias("factor_pct"))
        )
        if ranked.is_empty():
            return _empty_metrics()

        daily_ic: pl.DataFrame = (
            ranked.group_by("datetime")
            .agg(pl.corr("factor_rank", "label_rank").alias("ic"), pl.first("sample_count").alias("sample_count"))
            .sort("datetime")
        )
        ic_mean, _, icir = _mean_std_ir(daily_ic["ic"])
        ic_clean: pl.Series = daily_ic["ic"].drop_nulls().drop_nans()
        positive_rate: float = _float_or_nan((ic_clean > 0).mean()) if not ic_clean.is_empty() else float("nan")
        samples_mean: float = _float_or_nan(daily_ic["sample_count"].mean())

        daily_spread: pl.DataFrame = (
            ranked.group_by("datetime")
            .agg(
                pl.col("label").filter(pl.col("factor_pct") >= 1.0 - self.quantile).mean().alias("top_return"),
                pl.col("label").filter(pl.col("factor_pct") <= self.quantile).mean().alias("bottom_return"),
            )
            .with_columns((pl.col("top_return") - pl.col("bottom_return")).alias("spread"))
            .sort("datetime")
        )
        spread_mean, _, spread_ir = _mean_std_ir(daily_spread["spread"])

        return SegmentMetrics(
            ic=ic_mean,
            icir=icir,
            days=int(ic_clean.len()),
            positive_rate=positive_rate,
            samples_mean=samples_mean,
            spread=spread_mean,
            spread_ir=spread_ir,
        )

    def _ranked_history(self, top: int | None) -> list[dict[str, Any]]:
        """Sort history by absolute test IC, keeping failed records last."""
        rows: list[dict[str, Any]] = sorted(
            self.history,
            key=lambda record: (
                record["error"] is not None,
                -abs(record["test_ic"]) if not math.isnan(record["test_ic"]) else float("inf"),
            ),
        )
        return rows[:top] if top else rows
