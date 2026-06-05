from __future__ import annotations

from dataclasses import dataclass


DEFAULT_COMMON_TECHNICAL_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)
DEFAULT_MACD_PERIODS: tuple[int, int, int] = (12, 26, 9)
DEFAULT_COMMON_TRADE_MA_WINDOW: int = 10
DEFAULT_COMMON_TRADE_MIN_MA_BIAS: float = 0.05
DEFAULT_COMMON_TRADE_MIN_TURNOVER_RATE: float = 3.0
DEFAULT_COMMON_TRADE_MAX_TURNOVER_RATE: float = 6.5


@dataclass(frozen=True)
class CommonFactorSpec:
    """Named public factor expression."""

    name: str
    expression: str
    category: str
    description: str


def _dedupe_expressions(exprs: list[str]) -> list[str]:
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
    return _dedupe_expressions(exprs)


def common_technical_factor_specs(
    windows: tuple[int, ...] = DEFAULT_COMMON_TECHNICAL_WINDOWS,
    macd_periods: tuple[int, int, int] = DEFAULT_MACD_PERIODS,
) -> list[CommonFactorSpec]:
    """Generate named public OHLCV/technical factor expressions."""
    fast_period, slow_period, signal_period = macd_periods
    specs: list[CommonFactorSpec] = []

    for window in windows:
        specs.extend(
            [
                CommonFactorSpec(
                    name=f"common_momentum_{window}",
                    expression=f"close / (ts_delay(close, {window}) + 1e-12) - 1",
                    category="price",
                    description=f"{window}-bar close momentum",
                ),
                CommonFactorSpec(
                    name=f"common_reversal_{window}",
                    expression=f"-1 * (close / (ts_delay(close, {window}) + 1e-12) - 1)",
                    category="price",
                    description=f"Negative {window}-bar close momentum",
                ),
                CommonFactorSpec(
                    name=f"common_ma_bias_{window}",
                    expression=f"close / (ts_mean(close, {window}) + 1e-12) - 1",
                    category="trend",
                    description=f"Close distance from {window}-bar moving average",
                ),
                CommonFactorSpec(
                    name=f"common_ma_slope_{window}",
                    expression=f"ts_slope(ts_mean(close, {window}), {window}) / (ts_mean(close, {window}) + 1e-12)",
                    category="trend",
                    description=f"{window}-bar moving-average slope normalized by moving average",
                ),
                CommonFactorSpec(
                    name=f"common_range_position_{window}",
                    expression=f"(close - ts_min(low, {window})) / (ts_max(high, {window}) - ts_min(low, {window}) + 1e-12)",
                    category="price",
                    description=f"Close position inside the {window}-bar high-low range",
                ),
                CommonFactorSpec(
                    name=f"common_return_volatility_{window}",
                    expression=f"ts_std(close / (ts_delay(close, 1) + 1e-12) - 1, {window})",
                    category="risk",
                    description=f"{window}-bar close-return volatility",
                ),
                CommonFactorSpec(
                    name=f"common_volume_ratio_{window}",
                    expression=f"volume / (ts_mean(ts_delay(volume, 1), {window}) + 1e-12)",
                    category="volume",
                    description=f"Volume relative to trailing {window}-bar average volume",
                ),
                CommonFactorSpec(
                    name=f"common_turnover_ratio_{window}",
                    expression=f"turnover / (ts_mean(ts_delay(turnover, 1), {window}) + 1e-12)",
                    category="volume",
                    description=f"Turnover relative to trailing {window}-bar average turnover",
                ),
                CommonFactorSpec(
                    name=f"common_price_volume_corr_{window}",
                    expression=f"ts_corr(close, volume, {window})",
                    category="volume",
                    description=f"{window}-bar close-volume correlation",
                ),
                CommonFactorSpec(
                    name=f"common_rsi_{window}",
                    expression=f"ta_rsi(close, {window}) / 100.0",
                    category="oscillator",
                    description=f"{window}-bar RSI scaled to 0-1",
                ),
                CommonFactorSpec(
                    name=f"common_atr_pct_{window}",
                    expression=f"ta_atr(high, low, close, {window}) / (close + 1e-12)",
                    category="risk",
                    description=f"{window}-bar ATR normalized by close",
                ),
            ]
        )

    dif = f"ta_macd_dif(close, {fast_period}, {slow_period}, {signal_period})"
    dea = f"ta_macd_dea(close, {fast_period}, {slow_period}, {signal_period})"
    hist = f"ta_macd_hist(close, {fast_period}, {slow_period}, {signal_period})"
    specs.extend(
        [
            CommonFactorSpec(
                name=f"common_macd_dif_{fast_period}_{slow_period}_{signal_period}",
                expression=f"{dif} / (close + 1e-12)",
                category="macd",
                description="MACD DIF line normalized by close",
            ),
            CommonFactorSpec(
                name=f"common_macd_dea_{fast_period}_{slow_period}_{signal_period}",
                expression=f"{dea} / (close + 1e-12)",
                category="macd",
                description="MACD DEA/signal line normalized by close",
            ),
            CommonFactorSpec(
                name=f"common_macd_hist_{fast_period}_{slow_period}_{signal_period}",
                expression=f"{hist} / (close + 1e-12)",
                category="macd",
                description="MACD histogram normalized by close",
            ),
            CommonFactorSpec(
                name=f"common_macd_hist_slope_3_{fast_period}_{slow_period}_{signal_period}",
                expression=f"({hist} - ts_delay({hist}, 3)) / (close + 1e-12)",
                category="macd",
                description="Three-bar MACD histogram change normalized by close",
            ),
            CommonFactorSpec(
                name=f"common_macd_bullish_state_{fast_period}_{slow_period}_{signal_period}",
                expression=f"({dif} > {dea}) * ({hist} > 0)",
                category="macd",
                description="Binary MACD bullish state",
            ),
            CommonFactorSpec(
                name=f"common_macd_bearish_state_{fast_period}_{slow_period}_{signal_period}",
                expression=f"({dif} <= {dea}) * ({hist} <= 0)",
                category="macd",
                description="Binary MACD bearish state",
            ),
        ]
    )

    return specs


def common_technical_expressions(
    windows: tuple[int, ...] = DEFAULT_COMMON_TECHNICAL_WINDOWS,
    macd_periods: tuple[int, int, int] = DEFAULT_MACD_PERIODS,
) -> list[str]:
    """Return public technical factor expressions."""
    return _dedupe_expressions(
        [spec.expression for spec in common_technical_factor_specs(windows=windows, macd_periods=macd_periods)]
    )


def common_technical_factor_names(
    windows: tuple[int, ...] = DEFAULT_COMMON_TECHNICAL_WINDOWS,
    macd_periods: tuple[int, int, int] = DEFAULT_MACD_PERIODS,
) -> list[str]:
    """Return public technical factor names."""
    return [spec.name for spec in common_technical_factor_specs(windows=windows, macd_periods=macd_periods)]


def common_trade_filter_factor_specs(
    ma_window: int = DEFAULT_COMMON_TRADE_MA_WINDOW,
    min_ma_bias: float = DEFAULT_COMMON_TRADE_MIN_MA_BIAS,
    min_turnover_rate: float = DEFAULT_COMMON_TRADE_MIN_TURNOVER_RATE,
    max_turnover_rate: float = DEFAULT_COMMON_TRADE_MAX_TURNOVER_RATE,
) -> list[CommonFactorSpec]:
    """Return reusable trade-filter factors that may require enriched panel columns.

    These are intentionally separate from ``common_technical_factor_specs`` because
    ``turnover_rate`` is present in A-share panels but not every generic OHLCV feed.
    """
    ma_bias = f"close / (ts_mean(close, {ma_window}) + 1e-12) - 1"
    return [
        CommonFactorSpec(
            name=f"common_ma_bias_pass_{ma_window}",
            expression=f"({ma_bias}) >= {min_ma_bias:.8f}",
            category="trade_filter",
            description=f"Binary pass flag for close at least {min_ma_bias:.2%} above MA{ma_window}",
        ),
        CommonFactorSpec(
            name="common_turnover_rate",
            expression="turnover_rate",
            category="trade_filter",
            description="Raw turnover-rate panel column, typically percentage-point units in A-share data",
        ),
        CommonFactorSpec(
            name="common_turnover_rate_band",
            expression=f"(turnover_rate >= {min_turnover_rate:.8f}) * (turnover_rate <= {max_turnover_rate:.8f})",
            category="trade_filter",
            description="Binary pass flag for the preferred turnover-rate band",
        ),
    ]


def common_trade_filter_expressions(
    ma_window: int = DEFAULT_COMMON_TRADE_MA_WINDOW,
    min_ma_bias: float = DEFAULT_COMMON_TRADE_MIN_MA_BIAS,
    min_turnover_rate: float = DEFAULT_COMMON_TRADE_MIN_TURNOVER_RATE,
    max_turnover_rate: float = DEFAULT_COMMON_TRADE_MAX_TURNOVER_RATE,
) -> list[str]:
    """Return reusable trade-filter factor expressions."""
    return _dedupe_expressions(
        [
            spec.expression
            for spec in common_trade_filter_factor_specs(
                ma_window=ma_window,
                min_ma_bias=min_ma_bias,
                min_turnover_rate=min_turnover_rate,
                max_turnover_rate=max_turnover_rate,
            )
        ]
    )


def common_trade_filter_factor_names(
    ma_window: int = DEFAULT_COMMON_TRADE_MA_WINDOW,
    min_ma_bias: float = DEFAULT_COMMON_TRADE_MIN_MA_BIAS,
    min_turnover_rate: float = DEFAULT_COMMON_TRADE_MIN_TURNOVER_RATE,
    max_turnover_rate: float = DEFAULT_COMMON_TRADE_MAX_TURNOVER_RATE,
) -> list[str]:
    """Return reusable trade-filter factor names."""
    return [
        spec.name
        for spec in common_trade_filter_factor_specs(
            ma_window=ma_window,
            min_ma_bias=min_ma_bias,
            min_turnover_rate=min_turnover_rate,
            max_turnover_rate=max_turnover_rate,
        )
    ]
