from __future__ import annotations

from dataclasses import dataclass


DEFAULT_COMMON_TECHNICAL_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)
DEFAULT_MACD_PERIODS: tuple[int, int, int] = (12, 26, 9)
DEFAULT_COMMON_TRADE_MA_WINDOW: int = 10
DEFAULT_COMMON_TRADE_MIN_MA_BIAS: float = 0.05
DEFAULT_COMMON_TRADE_MIN_TURNOVER_RATE: float = 3.0
DEFAULT_COMMON_TRADE_MAX_TURNOVER_RATE: float = 6.5
DEFAULT_COMMON_POSITION_WINDOWS: tuple[int, ...] = (20, 60, 120)
DEFAULT_COMMON_OSCILLATOR_WINDOWS: tuple[int, ...] = (14, 20)
DEFAULT_COMMON_RELATIVE_WINDOWS: tuple[int, ...] = (20, 60)
DEFAULT_COMMON_LIMIT_STATE_WINDOWS: tuple[int, ...] = (5, 20)
DEFAULT_COMMON_FUNDAMENTAL_WINDOWS: tuple[int, ...] = (20, 60)
DEFAULT_GMMA_SHORT_PERIODS: tuple[int, ...] = (3, 5, 8, 10, 12, 15)
DEFAULT_GMMA_LONG_PERIODS: tuple[int, ...] = (30, 35, 40, 45, 50, 60)
DEFAULT_GMMA_SLOPE_WINDOW: int = 3
DEFAULT_GMMA_MIN_GAP: float = 0.002
DEFAULT_VEGAS_EMA_FILTER_PERIOD: int = 12
DEFAULT_VEGAS_FAST_CHANNEL: tuple[int, int] = (144, 169)
DEFAULT_VEGAS_SLOW_CHANNEL: tuple[int, int] = (576, 676)
DEFAULT_VEGAS_SLOPE_WINDOW: int = 8
DEFAULT_VEGAS_EMA_SLOPE_WINDOW: int = 3
DEFAULT_VEGAS_PRIOR_HIGH_WINDOW: int = 20
DEFAULT_VOLATILITY_COMPRESSION_WINDOWS: tuple[int, ...] = (20, 60)
DEFAULT_BOLLINGER_STD_MULTIPLIER: float = 2.0
DEFAULT_KELTNER_ATR_MULTIPLIER: float = 1.5


@dataclass(frozen=True)
class CommonFactorSpec:
    """Named public factor expression."""

    name: str
    expression: str
    category: str
    description: str
    required_columns: tuple[str, ...] = ()


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


def _ema_expr(period: int) -> str:
    return f"ta_ema(close, {period})"


def _average_expr(expressions: tuple[str, ...]) -> str:
    joined = " + ".join(f"({expression})" for expression in expressions)
    return f"({joined}) / {float(len(expressions)):.1f}"


def _fold_function_expr(function_name: str, expressions: tuple[str, ...]) -> str:
    if not expressions:
        raise ValueError("expressions must not be empty")
    result = expressions[0]
    for expression in expressions[1:]:
        result = f"{function_name}({result}, {expression})"
    return result


def _period_range_label(periods: tuple[int, ...]) -> str:
    if not periods:
        raise ValueError("periods must not be empty")
    ordered = tuple(sorted(periods))
    return f"{ordered[0]}_{ordered[-1]}"


def _channel_expr(periods: tuple[int, int]) -> tuple[str, str, str, str]:
    first = _ema_expr(periods[0])
    second = _ema_expr(periods[1])
    top = f"greater({first}, {second})"
    bottom = f"less({first}, {second})"
    mid = f"(({first}) + ({second})) / 2"
    width = f"(({top}) - ({bottom})) / (({mid}) + 1e-12)"
    return top, bottom, mid, width


def _donchian_width(window: int) -> str:
    return f"(ts_max(high, {window}) - ts_min(low, {window})) / (ts_mean(close, {window}) + 1e-12)"


def _bollinger_width(window: int, std_multiplier: float) -> str:
    return f"({2.0 * std_multiplier:.8f} * ts_std(close, {window})) / (ts_mean(close, {window}) + 1e-12)"


def _keltner_width(window: int, atr_multiplier: float) -> str:
    return f"({2.0 * atr_multiplier:.8f} * ta_atr(high, low, close, {window})) / (ts_mean(close, {window}) + 1e-12)"


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


def common_candlestick_factor_specs() -> list[CommonFactorSpec]:
    """Return single-bar OHLC candlestick structure factors."""
    true_range = "high - low"
    body = "close - open"
    body_abs = "abs(close - open)"
    upper_shadow = "high - greater(open, close)"
    lower_shadow = "less(open, close) - low"
    close_position = f"(close - low) / (({true_range}) + 1e-12)"
    open_position = f"(open - low) / (({true_range}) + 1e-12)"
    prev_close = "ts_delay(close, 1)"
    gap = f"open / (({prev_close}) + 1e-12) - 1"

    return [
        CommonFactorSpec(
            name="common_open_gap",
            expression=gap,
            category="candlestick",
            description="Open gap versus prior close",
        ),
        CommonFactorSpec(
            name="common_intraday_return",
            expression="close / (open + 1e-12) - 1",
            category="candlestick",
            description="Close-to-open intraday return",
        ),
        CommonFactorSpec(
            name="common_high_low_range_pct",
            expression=f"({true_range}) / (open + 1e-12)",
            category="candlestick",
            description="Intraday high-low range normalized by open",
        ),
        CommonFactorSpec(
            name="common_real_body_pct",
            expression=f"({body_abs}) / (open + 1e-12)",
            category="candlestick",
            description="Absolute candle body normalized by open",
        ),
        CommonFactorSpec(
            name="common_body_to_range",
            expression=f"({body_abs}) / (({true_range}) + 1e-12)",
            category="candlestick",
            description="Absolute candle body share of intraday range",
        ),
        CommonFactorSpec(
            name="common_close_position",
            expression=close_position,
            category="candlestick",
            description="Close position within intraday high-low range",
        ),
        CommonFactorSpec(
            name="common_open_position",
            expression=open_position,
            category="candlestick",
            description="Open position within intraday high-low range",
        ),
        CommonFactorSpec(
            name="common_upper_shadow_pct",
            expression=f"({upper_shadow}) / (open + 1e-12)",
            category="candlestick",
            description="Upper shadow normalized by open",
        ),
        CommonFactorSpec(
            name="common_lower_shadow_pct",
            expression=f"({lower_shadow}) / (open + 1e-12)",
            category="candlestick",
            description="Lower shadow normalized by open",
        ),
        CommonFactorSpec(
            name="common_upper_shadow_ratio",
            expression=f"({upper_shadow}) / (({true_range}) + 1e-12)",
            category="candlestick",
            description="Upper shadow share of intraday range",
        ),
        CommonFactorSpec(
            name="common_lower_shadow_ratio",
            expression=f"({lower_shadow}) / (({true_range}) + 1e-12)",
            category="candlestick",
            description="Lower shadow share of intraday range",
        ),
        CommonFactorSpec(
            name="common_candle_shift",
            expression="(close * 2 - high - low) / (open + 1e-12)",
            category="candlestick",
            description="Close skew versus high-low midpoint normalized by open",
        ),
        CommonFactorSpec(
            name="common_bullish_body_state",
            expression=f"({body}) > 0",
            category="candlestick",
            description="Binary bullish candle body state",
        ),
        CommonFactorSpec(
            name="common_bearish_body_state",
            expression=f"({body}) < 0",
            category="candlestick",
            description="Binary bearish candle body state",
        ),
        CommonFactorSpec(
            name="common_gap_fill_state",
            expression=f"((open > ({prev_close})) * (low <= ({prev_close}))) + ((open < ({prev_close})) * (high >= ({prev_close})))",
            category="candlestick",
            description="Binary state for same-day fill of an up or down open gap",
        ),
    ]


def common_candlestick_expressions() -> list[str]:
    """Return candlestick factor expressions."""
    return _dedupe_expressions([spec.expression for spec in common_candlestick_factor_specs()])


def common_candlestick_factor_names() -> list[str]:
    """Return candlestick factor names."""
    return [spec.name for spec in common_candlestick_factor_specs()]


def common_price_volume_accumulation_factor_specs(
    windows: tuple[int, ...] = DEFAULT_COMMON_TECHNICAL_WINDOWS,
) -> list[CommonFactorSpec]:
    """Return volume-confirmation and accumulation/distribution factors."""
    specs: list[CommonFactorSpec] = []
    close_return = "close / (ts_delay(close, 1) + 1e-12) - 1"
    typical_price = "(high + low + close) / 3"
    typical_delta = f"({typical_price}) - ts_delay(({typical_price}), 1)"
    clv = "((close - low) - (high - close)) / (high - low + 1e-12)"

    for window in windows:
        volume_sum = f"ts_sum(volume, {window}) + 1e-12"
        turnover_sum = f"ts_sum(turnover, {window}) + 1e-12"
        signed_volume = "sign(close - ts_delay(close, 1)) * volume"
        signed_turnover = f"sign({typical_delta}) * turnover"
        volume_weighted_return = f"ts_sum(({close_return}) * volume, {window}) / ({volume_sum})"
        cmf = f"ts_sum(({clv}) * volume, {window}) / ({volume_sum})"

        specs.extend(
            [
                CommonFactorSpec(
                    name=f"common_obv_pressure_{window}",
                    expression=f"ts_sum({signed_volume}, {window}) / ({volume_sum})",
                    category="price_volume_accumulation",
                    description=f"{window}-bar signed-volume pressure similar to normalized OBV",
                ),
                CommonFactorSpec(
                    name=f"common_pvt_pressure_{window}",
                    expression=f"ts_sum(({close_return}) * volume, {window}) / ({volume_sum})",
                    category="price_volume_accumulation",
                    description=f"{window}-bar price-volume-trend pressure normalized by volume",
                ),
                CommonFactorSpec(
                    name=f"common_cmf_{window}",
                    expression=cmf,
                    category="price_volume_accumulation",
                    description=f"{window}-bar Chaikin money-flow style accumulation pressure",
                ),
                CommonFactorSpec(
                    name=f"common_money_flow_pressure_{window}",
                    expression=f"ts_sum({signed_turnover}, {window}) / ({turnover_sum})",
                    category="price_volume_accumulation",
                    description=f"{window}-bar signed turnover pressure by typical-price direction",
                ),
                CommonFactorSpec(
                    name=f"common_volume_weighted_return_{window}",
                    expression=volume_weighted_return,
                    category="price_volume_accumulation",
                    description=f"{window}-bar return weighted by volume share",
                ),
                CommonFactorSpec(
                    name=f"common_up_volume_ratio_{window}",
                    expression=f"ts_sum((close > ts_delay(close, 1)) * volume, {window}) / ({volume_sum})",
                    category="price_volume_accumulation",
                    description=f"{window}-bar share of volume traded on up-close bars",
                ),
                CommonFactorSpec(
                    name=f"common_down_volume_ratio_{window}",
                    expression=f"ts_sum((close < ts_delay(close, 1)) * volume, {window}) / ({volume_sum})",
                    category="price_volume_accumulation",
                    description=f"{window}-bar share of volume traded on down-close bars",
                ),
                CommonFactorSpec(
                    name=f"common_accumulation_slope_{window}",
                    expression=f"ts_slope(ts_sum(({clv}) * volume, {window}), {window}) / ({volume_sum})",
                    category="price_volume_accumulation",
                    description=f"{window}-bar slope of accumulation/distribution pressure",
                ),
            ]
        )

    return specs


def common_price_volume_accumulation_expressions(
    windows: tuple[int, ...] = DEFAULT_COMMON_TECHNICAL_WINDOWS,
) -> list[str]:
    """Return price-volume accumulation factor expressions."""
    return _dedupe_expressions(
        [spec.expression for spec in common_price_volume_accumulation_factor_specs(windows=windows)]
    )


def common_price_volume_accumulation_factor_names(
    windows: tuple[int, ...] = DEFAULT_COMMON_TECHNICAL_WINDOWS,
) -> list[str]:
    """Return price-volume accumulation factor names."""
    return [spec.name for spec in common_price_volume_accumulation_factor_specs(windows=windows)]


def common_liquidity_crowding_factor_specs(
    windows: tuple[int, ...] = DEFAULT_COMMON_TECHNICAL_WINDOWS,
) -> list[CommonFactorSpec]:
    """Return liquidity, capacity, and crowding factor expressions."""
    specs: list[CommonFactorSpec] = []
    close_return = "close / (ts_delay(close, 1) + 1e-12) - 1"

    for window in windows:
        amount_ratio = f"turnover / (ts_mean(ts_delay(turnover, 1), {window}) + 1e-12)"
        turnover_zscore = f"(turnover - ts_mean(turnover, {window})) / (ts_std(turnover, {window}) + 1e-12)"
        volume_zscore = f"(volume - ts_mean(volume, {window})) / (ts_std(volume, {window}) + 1e-12)"
        amihud = f"ts_mean(abs({close_return}) / (turnover + 1e-12), {window})"
        turnover_rate_zscore = (
            f"(turnover_rate - ts_mean(turnover_rate, {window})) / "
            f"(ts_std(turnover_rate, {window}) + 1e-12)"
        )

        specs.extend(
            [
                CommonFactorSpec(
                    name=f"common_log_turnover_mean_{window}",
                    expression=f"log(ts_mean(turnover, {window}) + 1)",
                    category="liquidity_crowding",
                    description=f"Log {window}-bar average turnover amount",
                ),
                CommonFactorSpec(
                    name=f"common_amount_ratio_{window}",
                    expression=amount_ratio,
                    category="liquidity_crowding",
                    description=f"Turnover amount relative to prior {window}-bar average",
                ),
                CommonFactorSpec(
                    name=f"common_turnover_zscore_{window}",
                    expression=turnover_zscore,
                    category="liquidity_crowding",
                    description=f"{window}-bar turnover amount z-score",
                ),
                CommonFactorSpec(
                    name=f"common_volume_zscore_{window}",
                    expression=volume_zscore,
                    category="liquidity_crowding",
                    description=f"{window}-bar volume z-score",
                ),
                CommonFactorSpec(
                    name=f"common_amihud_illiquidity_{window}",
                    expression=amihud,
                    category="liquidity_crowding",
                    description=f"{window}-bar Amihud-style return impact per turnover amount",
                ),
                CommonFactorSpec(
                    name=f"common_anti_amihud_illiquidity_{window}",
                    expression=f"-1 * ({amihud})",
                    category="liquidity_crowding",
                    description=f"Higher values indicate lower {window}-bar Amihud-style illiquidity",
                ),
                CommonFactorSpec(
                    name=f"common_liquidity_score_{window}",
                    expression=f"(cs_rank(log(ts_mean(turnover, {window}) + 1)) + cs_rank(-1 * ({amihud}))) / 2",
                    category="liquidity_crowding",
                    description=f"Composite {window}-bar liquidity score from turnover depth and low price impact",
                ),
                CommonFactorSpec(
                    name=f"common_turnover_rate_zscore_{window}",
                    expression=turnover_rate_zscore,
                    category="liquidity_crowding",
                    description=f"{window}-bar A-share turnover-rate z-score",
                    required_columns=("turnover_rate",),
                ),
                CommonFactorSpec(
                    name=f"common_turnover_rate_percentile_{window}",
                    expression=f"ts_rank(turnover_rate, {window})",
                    category="liquidity_crowding",
                    description=f"Time-series percentile rank of A-share turnover rate over {window} bars",
                    required_columns=("turnover_rate",),
                ),
                CommonFactorSpec(
                    name=f"common_crowding_score_{window}",
                    expression=(
                        f"cs_rank({amount_ratio})"
                        f" + cs_rank({turnover_zscore})"
                        f" + cs_rank({volume_zscore})"
                        f" + cs_rank({turnover_rate_zscore})"
                    ) + " / 4",
                    category="liquidity_crowding",
                    description=f"Composite {window}-bar crowding score from amount, volume, and turnover-rate surges",
                    required_columns=("turnover_rate",),
                ),
            ]
        )

    return specs


def common_liquidity_crowding_expressions(
    windows: tuple[int, ...] = DEFAULT_COMMON_TECHNICAL_WINDOWS,
) -> list[str]:
    """Return liquidity/crowding factor expressions."""
    return _dedupe_expressions([spec.expression for spec in common_liquidity_crowding_factor_specs(windows=windows)])


def common_liquidity_crowding_factor_names(
    windows: tuple[int, ...] = DEFAULT_COMMON_TECHNICAL_WINDOWS,
) -> list[str]:
    """Return liquidity/crowding factor names."""
    return [spec.name for spec in common_liquidity_crowding_factor_specs(windows=windows)]


def common_position_breakout_factor_specs(
    windows: tuple[int, ...] = DEFAULT_COMMON_POSITION_WINDOWS,
) -> list[CommonFactorSpec]:
    """Return range position, high/low distance, and breakout-state factors."""
    specs: list[CommonFactorSpec] = []

    for window in windows:
        prior_high = f"ts_max(ts_delay(high, 1), {window})"
        prior_low = f"ts_min(ts_delay(low, 1), {window})"
        current_range = f"ts_max(high, {window}) - ts_min(low, {window})"
        drawdown = f"close / (ts_max(high, {window}) + 1e-12) - 1"
        rebound = f"close / (ts_min(low, {window}) + 1e-12) - 1"

        specs.extend(
            [
                CommonFactorSpec(
                    name=f"common_close_to_high_{window}",
                    expression=f"(ts_max(high, {window}) - close) / (close + 1e-12)",
                    category="position_breakout",
                    description=f"Close distance from {window}-bar high",
                ),
                CommonFactorSpec(
                    name=f"common_close_to_low_{window}",
                    expression=f"(close - ts_min(low, {window})) / (close + 1e-12)",
                    category="position_breakout",
                    description=f"Close distance from {window}-bar low",
                ),
                CommonFactorSpec(
                    name=f"common_drawdown_from_high_{window}",
                    expression=drawdown,
                    category="position_breakout",
                    description=f"Current drawdown from {window}-bar high",
                ),
                CommonFactorSpec(
                    name=f"common_rebound_from_low_{window}",
                    expression=rebound,
                    category="position_breakout",
                    description=f"Current rebound from {window}-bar low",
                ),
                CommonFactorSpec(
                    name=f"common_days_since_high_{window}",
                    expression=f"({window} - ts_argmax(high, {window})) / {float(window):.1f}",
                    category="position_breakout",
                    description=f"Normalized bars since the {window}-bar high",
                ),
                CommonFactorSpec(
                    name=f"common_days_since_low_{window}",
                    expression=f"({window} - ts_argmin(low, {window})) / {float(window):.1f}",
                    category="position_breakout",
                    description=f"Normalized bars since the {window}-bar low",
                ),
                CommonFactorSpec(
                    name=f"common_prior_high_breakout_{window}",
                    expression=f"close / (({prior_high}) + 1e-12) - 1",
                    category="position_breakout",
                    description=f"Close breakout distance above the prior {window}-bar high",
                ),
                CommonFactorSpec(
                    name=f"common_prior_low_breakdown_{window}",
                    expression=f"close / (({prior_low}) + 1e-12) - 1",
                    category="position_breakout",
                    description=f"Close distance from the prior {window}-bar low",
                ),
                CommonFactorSpec(
                    name=f"common_new_high_state_{window}",
                    expression=f"close >= ({prior_high})",
                    category="position_breakout",
                    description=f"Binary close breakout above the prior {window}-bar high",
                ),
                CommonFactorSpec(
                    name=f"common_new_low_state_{window}",
                    expression=f"close <= ({prior_low})",
                    category="position_breakout",
                    description=f"Binary close breakdown below the prior {window}-bar low",
                ),
                CommonFactorSpec(
                    name=f"common_pullback_from_high_rank_{window}",
                    expression=f"ts_rank({drawdown}, {window})",
                    category="position_breakout",
                    description=f"Time-series rank of current drawdown from {window}-bar high",
                ),
                CommonFactorSpec(
                    name=f"common_low_reclaim_strength_{window}",
                    expression=f"(close - low) / (({current_range}) + 1e-12)",
                    category="position_breakout",
                    description=f"Close reclaim strength after intraday low relative to {window}-bar range",
                ),
            ]
        )

    return specs


def common_position_breakout_expressions(
    windows: tuple[int, ...] = DEFAULT_COMMON_POSITION_WINDOWS,
) -> list[str]:
    """Return position/breakout factor expressions."""
    return _dedupe_expressions([spec.expression for spec in common_position_breakout_factor_specs(windows=windows)])


def common_position_breakout_factor_names(
    windows: tuple[int, ...] = DEFAULT_COMMON_POSITION_WINDOWS,
) -> list[str]:
    """Return position/breakout factor names."""
    return [spec.name for spec in common_position_breakout_factor_specs(windows=windows)]


def common_oscillator_factor_specs(
    windows: tuple[int, ...] = DEFAULT_COMMON_OSCILLATOR_WINDOWS,
) -> list[CommonFactorSpec]:
    """Return classic oscillator and directional-balance factor expressions."""
    specs: list[CommonFactorSpec] = []

    for window in windows:
        rsv = f"(close - ts_min(low, {window})) / (ts_max(high, {window}) - ts_min(low, {window}) + 1e-12)"
        typical_price = "(high + low + close) / 3"
        typical_ma = f"ts_mean(({typical_price}), {window})"
        mean_abs_dev = f"ts_mean(abs(({typical_price}) - ({typical_ma})), {window})"
        cci = f"(({typical_price}) - ({typical_ma})) / (0.015 * ({mean_abs_dev}) + 1e-12)"
        triple_ema = f"ta_ema(ta_ema(ta_ema(close, {window}), {window}), {window})"
        up_move = "greater(high - ts_delay(high, 1), 0)"
        down_move = "greater(ts_delay(low, 1) - low, 0)"
        up_sum = f"ts_sum({up_move}, {window})"
        down_sum = f"ts_sum({down_move}, {window})"

        specs.extend(
            [
                CommonFactorSpec(
                    name=f"common_stoch_k_{window}",
                    expression=rsv,
                    category="oscillator",
                    description=f"{window}-bar stochastic K value",
                ),
                CommonFactorSpec(
                    name=f"common_stoch_d_{window}",
                    expression=f"ts_mean({rsv}, 3)",
                    category="oscillator",
                    description=f"Three-bar average of {window}-bar stochastic K",
                ),
                CommonFactorSpec(
                    name=f"common_williams_r_{window}",
                    expression=f"-1 * (ts_max(high, {window}) - close) / (ts_max(high, {window}) - ts_min(low, {window}) + 1e-12)",
                    category="oscillator",
                    description=f"{window}-bar Williams %R scaled as a negative distance from high",
                ),
                CommonFactorSpec(
                    name=f"common_cci_{window}",
                    expression=cci,
                    category="oscillator",
                    description=f"{window}-bar commodity channel index",
                ),
                CommonFactorSpec(
                    name=f"common_roc_{window}",
                    expression=f"close / (ts_delay(close, {window}) + 1e-12) - 1",
                    category="oscillator",
                    description=f"{window}-bar rate of change",
                ),
                CommonFactorSpec(
                    name=f"common_trix_{window}",
                    expression=f"({triple_ema}) / (ts_delay(({triple_ema}), 1) + 1e-12) - 1",
                    category="oscillator",
                    description=f"{window}-bar triple-EMA momentum",
                ),
                CommonFactorSpec(
                    name=f"common_up_day_ratio_{window}",
                    expression=f"ts_mean(close > ts_delay(close, 1), {window})",
                    category="oscillator",
                    description=f"{window}-bar up-close day ratio",
                ),
                CommonFactorSpec(
                    name=f"common_down_day_ratio_{window}",
                    expression=f"ts_mean(close < ts_delay(close, 1), {window})",
                    category="oscillator",
                    description=f"{window}-bar down-close day ratio",
                ),
                CommonFactorSpec(
                    name=f"common_day_balance_{window}",
                    expression=f"ts_mean(close > ts_delay(close, 1), {window}) - ts_mean(close < ts_delay(close, 1), {window})",
                    category="oscillator",
                    description=f"{window}-bar up/down day balance",
                ),
                CommonFactorSpec(
                    name=f"common_dmi_balance_{window}",
                    expression=f"(({up_sum}) - ({down_sum})) / (({up_sum}) + ({down_sum}) + 1e-12)",
                    category="oscillator",
                    description=f"{window}-bar directional-movement balance proxy",
                ),
                CommonFactorSpec(
                    name=f"common_adx_proxy_{window}",
                    expression=f"abs(({up_sum}) - ({down_sum})) / (({up_sum}) + ({down_sum}) + 1e-12)",
                    category="oscillator",
                    description=f"{window}-bar directional trend-strength proxy",
                ),
            ]
        )

    return specs


def common_oscillator_expressions(
    windows: tuple[int, ...] = DEFAULT_COMMON_OSCILLATOR_WINDOWS,
) -> list[str]:
    """Return oscillator factor expressions."""
    return _dedupe_expressions([spec.expression for spec in common_oscillator_factor_specs(windows=windows)])


def common_oscillator_factor_names(
    windows: tuple[int, ...] = DEFAULT_COMMON_OSCILLATOR_WINDOWS,
) -> list[str]:
    """Return oscillator factor names."""
    return [spec.name for spec in common_oscillator_factor_specs(windows=windows)]


def common_relative_strength_factor_specs(
    windows: tuple[int, ...] = DEFAULT_COMMON_RELATIVE_WINDOWS,
    benchmark_close_column: str = "benchmark_close",
    industry_close_column: str = "industry_close",
) -> list[CommonFactorSpec]:
    """Return optional benchmark/industry relative-strength factors."""
    specs: list[CommonFactorSpec] = []
    stock_return_1 = "close / (ts_delay(close, 1) + 1e-12) - 1"
    benchmark_return_1 = f"{benchmark_close_column} / (ts_delay({benchmark_close_column}, 1) + 1e-12) - 1"
    industry_return_1 = f"{industry_close_column} / (ts_delay({industry_close_column}, 1) + 1e-12) - 1"

    for window in windows:
        stock_return = f"close / (ts_delay(close, {window}) + 1e-12) - 1"
        benchmark_return = (
            f"{benchmark_close_column} / "
            f"(ts_delay({benchmark_close_column}, {window}) + 1e-12) - 1"
        )
        industry_return = (
            f"{industry_close_column} / "
            f"(ts_delay({industry_close_column}, {window}) + 1e-12) - 1"
        )
        benchmark_beta = (
            f"ts_corr({stock_return_1}, {benchmark_return_1}, {window})"
            f" * ts_std({stock_return_1}, {window})"
            f" / (ts_std({benchmark_return_1}, {window}) + 1e-12)"
        )
        industry_beta = (
            f"ts_corr({stock_return_1}, {industry_return_1}, {window})"
            f" * ts_std({stock_return_1}, {window})"
            f" / (ts_std({industry_return_1}, {window}) + 1e-12)"
        )

        specs.extend(
            [
                CommonFactorSpec(
                    name=f"common_relative_benchmark_strength_{window}",
                    expression=f"({stock_return}) - ({benchmark_return})",
                    category="relative_strength",
                    description=f"{window}-bar stock return minus benchmark return",
                    required_columns=(benchmark_close_column,),
                ),
                CommonFactorSpec(
                    name=f"common_relative_industry_strength_{window}",
                    expression=f"({stock_return}) - ({industry_return})",
                    category="relative_strength",
                    description=f"{window}-bar stock return minus industry return",
                    required_columns=(industry_close_column,),
                ),
                CommonFactorSpec(
                    name=f"common_benchmark_beta_{window}",
                    expression=benchmark_beta,
                    category="relative_strength",
                    description=f"{window}-bar beta to benchmark return",
                    required_columns=(benchmark_close_column,),
                ),
                CommonFactorSpec(
                    name=f"common_industry_beta_{window}",
                    expression=industry_beta,
                    category="relative_strength",
                    description=f"{window}-bar beta to industry return",
                    required_columns=(industry_close_column,),
                ),
                CommonFactorSpec(
                    name=f"common_benchmark_residual_momentum_{window}",
                    expression=f"ts_mean(({stock_return_1}) - ({benchmark_beta}) * ({benchmark_return_1}), {window})",
                    category="relative_strength",
                    description=f"{window}-bar average return residual after benchmark beta adjustment",
                    required_columns=(benchmark_close_column,),
                ),
                CommonFactorSpec(
                    name=f"common_industry_residual_momentum_{window}",
                    expression=f"ts_mean(({stock_return_1}) - ({industry_beta}) * ({industry_return_1}), {window})",
                    category="relative_strength",
                    description=f"{window}-bar average return residual after industry beta adjustment",
                    required_columns=(industry_close_column,),
                ),
            ]
        )

    return specs


def common_relative_strength_expressions(
    windows: tuple[int, ...] = DEFAULT_COMMON_RELATIVE_WINDOWS,
    benchmark_close_column: str = "benchmark_close",
    industry_close_column: str = "industry_close",
) -> list[str]:
    """Return relative-strength factor expressions."""
    return _dedupe_expressions(
        [
            spec.expression
            for spec in common_relative_strength_factor_specs(
                windows=windows,
                benchmark_close_column=benchmark_close_column,
                industry_close_column=industry_close_column,
            )
        ]
    )


def common_relative_strength_factor_names(
    windows: tuple[int, ...] = DEFAULT_COMMON_RELATIVE_WINDOWS,
    benchmark_close_column: str = "benchmark_close",
    industry_close_column: str = "industry_close",
) -> list[str]:
    """Return relative-strength factor names."""
    return [
        spec.name
        for spec in common_relative_strength_factor_specs(
            windows=windows,
            benchmark_close_column=benchmark_close_column,
            industry_close_column=industry_close_column,
        )
    ]


def common_a_share_limit_state_factor_specs(
    windows: tuple[int, ...] = DEFAULT_COMMON_LIMIT_STATE_WINDOWS,
    raw_open_column: str = "raw_open",
    raw_high_column: str = "raw_high",
    raw_low_column: str = "raw_low",
    raw_close_column: str = "raw_close",
    up_limit_column: str = "up_limit",
    down_limit_column: str = "down_limit",
    limit_tolerance: float = 0.001,
    near_limit_pct: float = 0.03,
) -> list[CommonFactorSpec]:
    """Return optional A-share limit-up/down state factors."""
    price_columns = (raw_open_column, raw_high_column, raw_low_column, raw_close_column)
    limit_columns = (*price_columns, up_limit_column, down_limit_column)
    close_limit_columns = (raw_close_column, up_limit_column, down_limit_column)
    limit_up_threshold = f"({up_limit_column}) * {1.0 - limit_tolerance:.8f}"
    limit_down_threshold = f"({down_limit_column}) * {1.0 + limit_tolerance:.8f}"
    near_limit_up_threshold = f"({up_limit_column}) * {1.0 - near_limit_pct:.8f}"
    near_limit_down_threshold = f"({down_limit_column}) * {1.0 + near_limit_pct:.8f}"
    is_limit_up = f"({raw_close_column} >= ({limit_up_threshold}))"
    is_limit_down = f"({raw_close_column} <= ({limit_down_threshold}))"
    one_line_limit_up = (
        f"({raw_open_column} >= ({limit_up_threshold}))"
        f" * ({raw_low_column} >= ({limit_up_threshold}))"
        f" * ({raw_close_column} >= ({limit_up_threshold}))"
    )
    limit_up_opened = f"({raw_high_column} >= ({limit_up_threshold})) * ({raw_close_column} < ({limit_up_threshold}))"
    limit_up_resealed = f"({raw_low_column} < ({limit_up_threshold})) * ({raw_close_column} >= ({limit_up_threshold}))"

    specs: list[CommonFactorSpec] = [
        CommonFactorSpec(
            name="common_limit_up_distance",
            expression=f"({up_limit_column}) / ({raw_close_column} + 1e-12) - 1",
            category="a_share_limit_state",
            description="Raw close distance below the A-share limit-up price",
            required_columns=close_limit_columns,
        ),
        CommonFactorSpec(
            name="common_limit_down_distance",
            expression=f"({raw_close_column}) / ({down_limit_column} + 1e-12) - 1",
            category="a_share_limit_state",
            description="Raw close distance above the A-share limit-down price",
            required_columns=close_limit_columns,
        ),
        CommonFactorSpec(
            name="common_near_limit_up_state",
            expression=f"{raw_close_column} >= ({near_limit_up_threshold})",
            category="a_share_limit_state",
            description=f"Binary state for raw close within {near_limit_pct:.1%} of limit-up price",
            required_columns=close_limit_columns,
        ),
        CommonFactorSpec(
            name="common_near_limit_down_state",
            expression=f"{raw_close_column} <= ({near_limit_down_threshold})",
            category="a_share_limit_state",
            description=f"Binary state for raw close within {near_limit_pct:.1%} of limit-down price",
            required_columns=close_limit_columns,
        ),
        CommonFactorSpec(
            name="common_is_limit_up",
            expression=is_limit_up,
            category="a_share_limit_state",
            description="Binary A-share limit-up close state",
            required_columns=close_limit_columns,
        ),
        CommonFactorSpec(
            name="common_is_limit_down",
            expression=is_limit_down,
            category="a_share_limit_state",
            description="Binary A-share limit-down close state",
            required_columns=close_limit_columns,
        ),
        CommonFactorSpec(
            name="common_one_line_limit_up",
            expression=one_line_limit_up,
            category="a_share_limit_state",
            description="Binary one-line limit-up state using raw open, low, and close",
            required_columns=limit_columns,
        ),
        CommonFactorSpec(
            name="common_limit_up_opened_state",
            expression=limit_up_opened,
            category="a_share_limit_state",
            description="Binary state for touching limit-up intraday but closing below it",
            required_columns=limit_columns,
        ),
        CommonFactorSpec(
            name="common_limit_up_resealed_state",
            expression=limit_up_resealed,
            category="a_share_limit_state",
            description="Binary state for closing at limit-up after trading below the limit intraday",
            required_columns=limit_columns,
        ),
    ]

    for window in windows:
        specs.extend(
            [
                CommonFactorSpec(
                    name=f"common_limit_up_count_{window}",
                    expression=f"ts_sum({is_limit_up}, {window})",
                    category="a_share_limit_state",
                    description=f"Recent {window}-bar count of limit-up closes",
                    required_columns=close_limit_columns,
                ),
                CommonFactorSpec(
                    name=f"common_limit_down_count_{window}",
                    expression=f"ts_sum({is_limit_down}, {window})",
                    category="a_share_limit_state",
                    description=f"Recent {window}-bar count of limit-down closes",
                    required_columns=close_limit_columns,
                ),
                CommonFactorSpec(
                    name=f"common_anti_limit_up_heat_{window}",
                    expression=f"-1 * ts_sum({is_limit_up}, {window})",
                    category="a_share_limit_state",
                    description=f"Penalty for recent {window}-bar limit-up heat",
                    required_columns=close_limit_columns,
                ),
                CommonFactorSpec(
                    name=f"common_limit_up_opened_count_{window}",
                    expression=f"ts_sum({limit_up_opened}, {window})",
                    category="a_share_limit_state",
                    description=f"Recent {window}-bar count of limit-up open-board days",
                    required_columns=limit_columns,
                ),
            ]
        )

    return specs


def common_a_share_limit_state_expressions(
    windows: tuple[int, ...] = DEFAULT_COMMON_LIMIT_STATE_WINDOWS,
    raw_open_column: str = "raw_open",
    raw_high_column: str = "raw_high",
    raw_low_column: str = "raw_low",
    raw_close_column: str = "raw_close",
    up_limit_column: str = "up_limit",
    down_limit_column: str = "down_limit",
    limit_tolerance: float = 0.001,
    near_limit_pct: float = 0.03,
) -> list[str]:
    """Return A-share limit-state factor expressions."""
    return _dedupe_expressions(
        [
            spec.expression
            for spec in common_a_share_limit_state_factor_specs(
                windows=windows,
                raw_open_column=raw_open_column,
                raw_high_column=raw_high_column,
                raw_low_column=raw_low_column,
                raw_close_column=raw_close_column,
                up_limit_column=up_limit_column,
                down_limit_column=down_limit_column,
                limit_tolerance=limit_tolerance,
                near_limit_pct=near_limit_pct,
            )
        ]
    )


def common_a_share_limit_state_factor_names(
    windows: tuple[int, ...] = DEFAULT_COMMON_LIMIT_STATE_WINDOWS,
    raw_open_column: str = "raw_open",
    raw_high_column: str = "raw_high",
    raw_low_column: str = "raw_low",
    raw_close_column: str = "raw_close",
    up_limit_column: str = "up_limit",
    down_limit_column: str = "down_limit",
    limit_tolerance: float = 0.001,
    near_limit_pct: float = 0.03,
) -> list[str]:
    """Return A-share limit-state factor names."""
    return [
        spec.name
        for spec in common_a_share_limit_state_factor_specs(
            windows=windows,
            raw_open_column=raw_open_column,
            raw_high_column=raw_high_column,
            raw_low_column=raw_low_column,
            raw_close_column=raw_close_column,
            up_limit_column=up_limit_column,
            down_limit_column=down_limit_column,
            limit_tolerance=limit_tolerance,
            near_limit_pct=near_limit_pct,
        )
    ]


def common_fundamental_factor_specs(
    windows: tuple[int, ...] = DEFAULT_COMMON_FUNDAMENTAL_WINDOWS,
) -> list[CommonFactorSpec]:
    """Return optional valuation, quality, growth, and size factors."""
    specs: list[CommonFactorSpec] = [
        CommonFactorSpec(
            name="common_value_pb",
            expression="-1 * pb",
            category="fundamental",
            description="Negative PB valuation; higher values indicate cheaper PB",
            required_columns=("pb",),
        ),
        CommonFactorSpec(
            name="common_value_pe_ttm",
            expression="-1 * pe_ttm",
            category="fundamental",
            description="Negative PE TTM valuation; higher values indicate cheaper PE",
            required_columns=("pe_ttm",),
        ),
        CommonFactorSpec(
            name="common_value_ps_ttm",
            expression="-1 * ps_ttm",
            category="fundamental",
            description="Negative PS TTM valuation; higher values indicate cheaper PS",
            required_columns=("ps_ttm",),
        ),
        CommonFactorSpec(
            name="common_dividend_yield_ttm",
            expression="dv_ttm",
            category="fundamental",
            description="TTM dividend yield",
            required_columns=("dv_ttm",),
        ),
        CommonFactorSpec(
            name="common_quality_roe",
            expression="fi_roe",
            category="fundamental",
            description="Return on equity",
            required_columns=("fi_roe",),
        ),
        CommonFactorSpec(
            name="common_quality_gross_margin",
            expression="fi_grossprofit_margin",
            category="fundamental",
            description="Gross profit margin",
            required_columns=("fi_grossprofit_margin",),
        ),
        CommonFactorSpec(
            name="common_low_debt_to_assets",
            expression="-1 * fi_debt_to_assets",
            category="fundamental",
            description="Negative debt-to-assets ratio",
            required_columns=("fi_debt_to_assets",),
        ),
        CommonFactorSpec(
            name="common_growth_netprofit_yoy",
            expression="fi_netprofit_yoy",
            category="fundamental",
            description="Net profit year-over-year growth",
            required_columns=("fi_netprofit_yoy",),
        ),
        CommonFactorSpec(
            name="common_growth_or_yoy",
            expression="fi_or_yoy",
            category="fundamental",
            description="Operating revenue year-over-year growth",
            required_columns=("fi_or_yoy",),
        ),
        CommonFactorSpec(
            name="common_log_total_mv",
            expression="log(total_mv + 1)",
            category="fundamental",
            description="Log total market value",
            required_columns=("total_mv",),
        ),
        CommonFactorSpec(
            name="common_log_circ_mv",
            expression="log(circ_mv + 1)",
            category="fundamental",
            description="Log circulating market value",
            required_columns=("circ_mv",),
        ),
        CommonFactorSpec(
            name="common_small_cap_circ_mv",
            expression="-1 * log(circ_mv + 1)",
            category="fundamental",
            description="Negative log circulating market value",
            required_columns=("circ_mv",),
        ),
    ]

    for window in windows:
        specs.extend(
            [
                CommonFactorSpec(
                    name=f"common_pb_percentile_{window}",
                    expression=f"ts_rank(pb, {window})",
                    category="fundamental",
                    description=f"Time-series PB percentile over {window} bars",
                    required_columns=("pb",),
                ),
                CommonFactorSpec(
                    name=f"common_value_pb_percentile_{window}",
                    expression=f"-1 * ts_rank(pb, {window})",
                    category="fundamental",
                    description=f"Negative time-series PB percentile over {window} bars",
                    required_columns=("pb",),
                ),
                CommonFactorSpec(
                    name=f"common_pe_ttm_percentile_{window}",
                    expression=f"ts_rank(pe_ttm, {window})",
                    category="fundamental",
                    description=f"Time-series PE TTM percentile over {window} bars",
                    required_columns=("pe_ttm",),
                ),
                CommonFactorSpec(
                    name=f"common_roe_momentum_{window}",
                    expression=f"fi_roe - ts_delay(fi_roe, {window})",
                    category="fundamental",
                    description=f"{window}-bar ROE change",
                    required_columns=("fi_roe",),
                ),
                CommonFactorSpec(
                    name=f"common_growth_stability_{window}",
                    expression=f"-1 * ts_std(fi_netprofit_yoy, {window})",
                    category="fundamental",
                    description=f"Negative {window}-bar net-profit-growth volatility",
                    required_columns=("fi_netprofit_yoy",),
                ),
                CommonFactorSpec(
                    name=f"common_quality_value_blend_{window}",
                    expression=(
                        f"cs_rank(fi_roe)"
                        f" + cs_rank(fi_grossprofit_margin)"
                        f" + cs_rank(-1 * pb)"
                        f" + cs_rank(-1 * ts_rank(pb, {window}))"
                    ) + " / 4",
                    category="fundamental",
                    description=f"Quality/value blend using ROE, gross margin, PB, and {window}-bar PB percentile",
                    required_columns=("fi_roe", "fi_grossprofit_margin", "pb"),
                ),
            ]
        )

    return specs


def common_fundamental_expressions(
    windows: tuple[int, ...] = DEFAULT_COMMON_FUNDAMENTAL_WINDOWS,
) -> list[str]:
    """Return fundamental factor expressions."""
    return _dedupe_expressions([spec.expression for spec in common_fundamental_factor_specs(windows=windows)])


def common_fundamental_factor_names(
    windows: tuple[int, ...] = DEFAULT_COMMON_FUNDAMENTAL_WINDOWS,
) -> list[str]:
    """Return fundamental factor names."""
    return [spec.name for spec in common_fundamental_factor_specs(windows=windows)]


def common_gmma_factor_specs(
    short_periods: tuple[int, ...] = DEFAULT_GMMA_SHORT_PERIODS,
    long_periods: tuple[int, ...] = DEFAULT_GMMA_LONG_PERIODS,
    slope_window: int = DEFAULT_GMMA_SLOPE_WINDOW,
    min_gap: float = DEFAULT_GMMA_MIN_GAP,
) -> list[CommonFactorSpec]:
    """Return Guppy Multiple Moving Average factor expressions."""
    short_emas = tuple(_ema_expr(period) for period in short_periods)
    long_emas = tuple(_ema_expr(period) for period in long_periods)
    short_label = _period_range_label(short_periods)
    long_label = _period_range_label(long_periods)
    label = f"s{short_label}_l{long_label}"

    short_avg = _average_expr(short_emas)
    long_avg = _average_expr(long_emas)
    short_top = _fold_function_expr("greater", short_emas)
    short_bottom = _fold_function_expr("less", short_emas)
    long_top = _fold_function_expr("greater", long_emas)
    long_bottom = _fold_function_expr("less", long_emas)
    group_gap = f"({short_avg}) / (({long_avg}) + 1e-12) - 1"
    gap_abs = f"abs({group_gap})"
    short_spread = f"(({short_top}) - ({short_bottom})) / (({short_avg}) + 1e-12)"
    long_spread = f"(({long_top}) - ({long_bottom})) / (({long_avg}) + 1e-12)"
    short_slope = f"({short_avg}) / (ts_delay(({short_avg}), {slope_window}) + 1e-12) - 1"
    long_slope = f"({long_avg}) / (ts_delay(({long_avg}), {slope_window}) + 1e-12) - 1"
    conservative_bull = f"({short_bottom}) > ({long_top})"
    cross_bull = f"({short_avg}) > ({long_avg})"
    gap_expanding = f"({gap_abs}) > ts_delay(({gap_abs}), 1)"
    bull_trend = f"({conservative_bull}) * (({short_slope}) > 0) * (({long_slope}) > 0) * (({gap_abs}) >= {min_gap:.8f})"
    anti_group_gap = f"-1 * greater({group_gap}, 0)"
    anti_long_slope = f"-1 * greater({long_slope}, 0)"
    short_pullback_quality = f"-1 * abs(low / (({short_avg}) + 1e-12) - 1)"
    long_pullback_quality = f"-1 * abs(low / (({long_avg}) + 1e-12) - 1)"

    return [
        CommonFactorSpec(
            name=f"common_gmma_group_gap_{label}",
            expression=group_gap,
            category="gmma",
            description="GMMA short-group average distance from long-group average",
        ),
        CommonFactorSpec(
            name=f"common_gmma_short_spread_{label}",
            expression=short_spread,
            category="gmma",
            description="GMMA short-group internal spread normalized by short-group average",
        ),
        CommonFactorSpec(
            name=f"common_gmma_long_spread_{label}",
            expression=long_spread,
            category="gmma",
            description="GMMA long-group internal spread normalized by long-group average",
        ),
        CommonFactorSpec(
            name=f"common_gmma_short_slope_{label}_{slope_window}",
            expression=short_slope,
            category="gmma",
            description="GMMA short-group average slope",
        ),
        CommonFactorSpec(
            name=f"common_gmma_long_slope_{label}_{slope_window}",
            expression=long_slope,
            category="gmma",
            description="GMMA long-group average slope",
        ),
        CommonFactorSpec(
            name=f"common_gmma_conservative_bull_{label}",
            expression=conservative_bull,
            category="gmma",
            description="Binary GMMA conservative bullish alignment: all short MAs above all long MAs",
        ),
        CommonFactorSpec(
            name=f"common_gmma_cross_bull_{label}",
            expression=cross_bull,
            category="gmma",
            description="Binary GMMA bullish state using short average above long average",
        ),
        CommonFactorSpec(
            name=f"common_gmma_gap_expanding_{label}",
            expression=gap_expanding,
            category="gmma",
            description="Binary GMMA group-gap expansion state",
        ),
        CommonFactorSpec(
            name=f"common_gmma_compression_{label}",
            expression=f"-1 * ({gap_abs})",
            category="gmma",
            description="Higher values indicate tighter GMMA short/long group compression",
        ),
        CommonFactorSpec(
            name=f"common_gmma_anti_group_gap_{label}",
            expression=anti_group_gap,
            category="gmma",
            description="Penalizes positive GMMA short/long group overextension for buy-side filters",
        ),
        CommonFactorSpec(
            name=f"common_gmma_anti_long_slope_{label}_{slope_window}",
            expression=anti_long_slope,
            category="gmma",
            description="Penalizes overly steep positive GMMA long-group slope",
        ),
        CommonFactorSpec(
            name=f"common_gmma_pullback_to_short_group_{label}",
            expression=short_pullback_quality,
            category="gmma",
            description="Higher values mean the low pulled back closer to the GMMA short-group average",
        ),
        CommonFactorSpec(
            name=f"common_gmma_pullback_to_long_group_{label}",
            expression=long_pullback_quality,
            category="gmma",
            description="Higher values mean the low pulled back closer to the GMMA long-group average",
        ),
        CommonFactorSpec(
            name=f"common_gmma_bull_trend_{label}_{slope_window}",
            expression=bull_trend,
            category="gmma",
            description="Binary GMMA bullish trend state with alignment, slopes, and minimum group gap",
        ),
        CommonFactorSpec(
            name=f"common_gmma_trend_quality_{label}_{slope_window}",
            expression=(
                f"cs_rank({group_gap})"
                f" + cs_rank({short_slope})"
                f" + cs_rank({long_slope})"
                f" + cs_rank(-1 * ({long_spread}))"
            ) + " / 4",
            category="gmma",
            description="Composite GMMA bullish trend quality from gap, slopes, and long-group compactness",
        ),
        CommonFactorSpec(
            name=f"common_gmma_pullback_trend_quality_{label}_{slope_window}",
            expression=(
                f"cs_rank({anti_group_gap})"
                f" + cs_rank({short_pullback_quality})"
                f" + cs_rank({long_pullback_quality})"
                f" + cs_rank({short_slope})"
            ) + " / 4",
            category="gmma",
            description="Composite GMMA pullback quality that rewards trend context without overextension",
        ),
    ]


def common_gmma_expressions(
    short_periods: tuple[int, ...] = DEFAULT_GMMA_SHORT_PERIODS,
    long_periods: tuple[int, ...] = DEFAULT_GMMA_LONG_PERIODS,
    slope_window: int = DEFAULT_GMMA_SLOPE_WINDOW,
    min_gap: float = DEFAULT_GMMA_MIN_GAP,
) -> list[str]:
    """Return GMMA factor expressions."""
    return _dedupe_expressions(
        [
            spec.expression
            for spec in common_gmma_factor_specs(
                short_periods=short_periods,
                long_periods=long_periods,
                slope_window=slope_window,
                min_gap=min_gap,
            )
        ]
    )


def common_gmma_factor_names(
    short_periods: tuple[int, ...] = DEFAULT_GMMA_SHORT_PERIODS,
    long_periods: tuple[int, ...] = DEFAULT_GMMA_LONG_PERIODS,
    slope_window: int = DEFAULT_GMMA_SLOPE_WINDOW,
    min_gap: float = DEFAULT_GMMA_MIN_GAP,
) -> list[str]:
    """Return GMMA factor names."""
    return [
        spec.name
        for spec in common_gmma_factor_specs(
            short_periods=short_periods,
            long_periods=long_periods,
            slope_window=slope_window,
            min_gap=min_gap,
        )
    ]


def common_vegas_factor_specs(
    ema_filter_period: int = DEFAULT_VEGAS_EMA_FILTER_PERIOD,
    fast_channel: tuple[int, int] = DEFAULT_VEGAS_FAST_CHANNEL,
    slow_channel: tuple[int, int] = DEFAULT_VEGAS_SLOW_CHANNEL,
    slope_window: int = DEFAULT_VEGAS_SLOPE_WINDOW,
    ema_slope_window: int = DEFAULT_VEGAS_EMA_SLOPE_WINDOW,
    prior_high_window: int = DEFAULT_VEGAS_PRIOR_HIGH_WINDOW,
) -> list[CommonFactorSpec]:
    """Return Vegas channel factor expressions."""
    fast_label = f"{fast_channel[0]}_{fast_channel[1]}"
    slow_label = f"{slow_channel[0]}_{slow_channel[1]}"
    label = f"f{fast_label}_s{slow_label}"

    ema_filter = _ema_expr(ema_filter_period)
    fast_top, fast_bottom, fast_mid, fast_width = _channel_expr(fast_channel)
    slow_top, slow_bottom, slow_mid, slow_width = _channel_expr(slow_channel)
    fast_position = f"close / (({fast_mid}) + 1e-12) - 1"
    slow_position = f"close / (({slow_mid}) + 1e-12) - 1"
    trend_gap = f"({fast_mid}) / (({slow_mid}) + 1e-12) - 1"
    fast_slope = f"({fast_mid}) / (ts_delay(({fast_mid}), {slope_window}) + 1e-12) - 1"
    slow_slope = f"({slow_mid}) / (ts_delay(({slow_mid}), {slope_window}) + 1e-12) - 1"
    ema_filter_slope = f"({ema_filter}) / (ts_delay(({ema_filter}), {ema_slope_window}) + 1e-12) - 1"
    ema_filter_bias = f"({ema_filter}) / (({fast_top}) + 1e-12) - 1"
    breakout_strength = f"close / (({fast_top}) + 1e-12) - 1"
    pullback_distance = f"-1 * abs(low / (({fast_top}) + 1e-12) - 1)"
    pullback_to_mid_quality = f"-1 * abs(low / (({fast_mid}) + 1e-12) - 1)"
    anti_breakout_extension = f"-1 * greater({breakout_strength}, 0)"
    anti_fast_position = f"-1 * greater({fast_position}, 0)"
    anti_ema_filter_extension = f"-1 * greater({ema_filter_bias}, 0)"
    fast_top_reclaim_strength = f"(low <= ({fast_top})) * (close > ({fast_top})) * ({breakout_strength})"
    ema_filter_reclaim_strength = (
        f"(low <= ({ema_filter}))"
        f" * (close > ({ema_filter}))"
        f" * (close / (({ema_filter}) + 1e-12) - 1)"
    )
    prior_high_breakout = f"close / (ts_max(ts_delay(high, 1), {prior_high_window}) + 1e-12) - 1"
    long_ready = (
        f"(close > ({fast_top}))"
        f" * (({ema_filter}) > ({fast_top}))"
        f" * (({ema_filter_slope}) > 0)"
        f" * (({fast_top}) > ts_delay(({fast_top}), {slope_window}))"
        f" * (({fast_bottom}) > ts_delay(({fast_bottom}), {slope_window}))"
    )

    return [
        CommonFactorSpec(
            name=f"common_vegas_fast_position_{label}",
            expression=fast_position,
            category="vegas",
            description="Close distance from the Vegas fast-channel midpoint",
        ),
        CommonFactorSpec(
            name=f"common_vegas_fast_width_{label}",
            expression=fast_width,
            category="vegas",
            description="Vegas fast-channel width normalized by midpoint",
        ),
        CommonFactorSpec(
            name=f"common_vegas_slow_position_{label}",
            expression=slow_position,
            category="vegas",
            description="Close distance from the Vegas slow-channel midpoint",
        ),
        CommonFactorSpec(
            name=f"common_vegas_slow_width_{label}",
            expression=slow_width,
            category="vegas",
            description="Vegas slow-channel width normalized by midpoint",
        ),
        CommonFactorSpec(
            name=f"common_vegas_trend_gap_{label}",
            expression=trend_gap,
            category="vegas",
            description="Vegas fast-channel midpoint distance from slow-channel midpoint",
        ),
        CommonFactorSpec(
            name=f"common_vegas_fast_slope_{label}_{slope_window}",
            expression=fast_slope,
            category="vegas",
            description="Vegas fast-channel midpoint slope",
        ),
        CommonFactorSpec(
            name=f"common_vegas_slow_slope_{label}_{slope_window}",
            expression=slow_slope,
            category="vegas",
            description="Vegas slow-channel midpoint slope",
        ),
        CommonFactorSpec(
            name=f"common_vegas_ema_filter_bias_{ema_filter_period}_{fast_label}",
            expression=ema_filter_bias,
            category="vegas",
            description="EMA filter-line distance above the Vegas fast-channel top",
        ),
        CommonFactorSpec(
            name=f"common_vegas_ema_filter_slope_{ema_filter_period}_{ema_slope_window}",
            expression=ema_filter_slope,
            category="vegas",
            description="EMA filter-line slope",
        ),
        CommonFactorSpec(
            name=f"common_vegas_breakout_strength_{label}",
            expression=breakout_strength,
            category="vegas",
            description="Close breakout distance above the Vegas fast-channel top",
        ),
        CommonFactorSpec(
            name=f"common_vegas_anti_breakout_extension_{label}",
            expression=anti_breakout_extension,
            category="vegas",
            description="Penalizes positive close extension above the Vegas fast-channel top",
        ),
        CommonFactorSpec(
            name=f"common_vegas_anti_fast_position_{label}",
            expression=anti_fast_position,
            category="vegas",
            description="Penalizes positive close extension above the Vegas fast-channel midpoint",
        ),
        CommonFactorSpec(
            name=f"common_vegas_anti_ema_filter_extension_{ema_filter_period}_{fast_label}",
            expression=anti_ema_filter_extension,
            category="vegas",
            description="Penalizes EMA filter-line extension above the Vegas fast-channel top",
        ),
        CommonFactorSpec(
            name=f"common_vegas_pullback_distance_{label}",
            expression=pullback_distance,
            category="vegas",
            description="Higher values mean the low stayed closer to the Vegas fast-channel top",
        ),
        CommonFactorSpec(
            name=f"common_vegas_pullback_to_mid_quality_{label}",
            expression=pullback_to_mid_quality,
            category="vegas",
            description="Higher values mean the low pulled back closer to the Vegas fast-channel midpoint",
        ),
        CommonFactorSpec(
            name=f"common_vegas_fast_top_reclaim_strength_{label}",
            expression=fast_top_reclaim_strength,
            category="vegas",
            description="Fast-channel top reclaim strength after an intraday pullback below the top",
        ),
        CommonFactorSpec(
            name=f"common_vegas_ema_filter_reclaim_strength_{ema_filter_period}",
            expression=ema_filter_reclaim_strength,
            category="vegas",
            description="EMA filter-line reclaim strength after an intraday pullback below the filter line",
        ),
        CommonFactorSpec(
            name=f"common_vegas_prior_high_breakout_{prior_high_window}",
            expression=prior_high_breakout,
            category="vegas",
            description="Close distance above the trailing prior high",
        ),
        CommonFactorSpec(
            name=f"common_vegas_long_ready_{ema_filter_period}_{label}",
            expression=long_ready,
            category="vegas",
            description="Binary Vegas long state: close and EMA filter above fast channel with rising channel",
        ),
        CommonFactorSpec(
            name=f"common_vegas_trend_quality_{ema_filter_period}_{label}",
            expression=(
                f"cs_rank({breakout_strength})"
                f" + cs_rank(({ema_filter}) / (({fast_top}) + 1e-12) - 1)"
                f" + cs_rank({fast_slope})"
                f" + cs_rank({trend_gap})"
                f" + cs_rank({prior_high_breakout})"
            ) + " / 5",
            category="vegas",
            description="Composite Vegas bullish trend quality from breakout, EMA filter, slopes, channel gap, and prior-high breakout",
        ),
        CommonFactorSpec(
            name=f"common_vegas_pullback_trend_quality_{ema_filter_period}_{label}",
            expression=(
                f"cs_rank({anti_breakout_extension})"
                f" + cs_rank({pullback_distance})"
                f" + cs_rank({pullback_to_mid_quality})"
                f" + cs_rank({ema_filter_reclaim_strength})"
                f" + cs_rank({fast_slope})"
            ) + " / 5",
            category="vegas",
            description="Composite Vegas pullback quality that rewards reclaim without excessive channel extension",
        ),
    ]


def common_vegas_expressions(
    ema_filter_period: int = DEFAULT_VEGAS_EMA_FILTER_PERIOD,
    fast_channel: tuple[int, int] = DEFAULT_VEGAS_FAST_CHANNEL,
    slow_channel: tuple[int, int] = DEFAULT_VEGAS_SLOW_CHANNEL,
    slope_window: int = DEFAULT_VEGAS_SLOPE_WINDOW,
    ema_slope_window: int = DEFAULT_VEGAS_EMA_SLOPE_WINDOW,
    prior_high_window: int = DEFAULT_VEGAS_PRIOR_HIGH_WINDOW,
) -> list[str]:
    """Return Vegas channel factor expressions."""
    return _dedupe_expressions(
        [
            spec.expression
            for spec in common_vegas_factor_specs(
                ema_filter_period=ema_filter_period,
                fast_channel=fast_channel,
                slow_channel=slow_channel,
                slope_window=slope_window,
                ema_slope_window=ema_slope_window,
                prior_high_window=prior_high_window,
            )
        ]
    )


def common_vegas_factor_names(
    ema_filter_period: int = DEFAULT_VEGAS_EMA_FILTER_PERIOD,
    fast_channel: tuple[int, int] = DEFAULT_VEGAS_FAST_CHANNEL,
    slow_channel: tuple[int, int] = DEFAULT_VEGAS_SLOW_CHANNEL,
    slope_window: int = DEFAULT_VEGAS_SLOPE_WINDOW,
    ema_slope_window: int = DEFAULT_VEGAS_EMA_SLOPE_WINDOW,
    prior_high_window: int = DEFAULT_VEGAS_PRIOR_HIGH_WINDOW,
) -> list[str]:
    """Return Vegas channel factor names."""
    return [
        spec.name
        for spec in common_vegas_factor_specs(
            ema_filter_period=ema_filter_period,
            fast_channel=fast_channel,
            slow_channel=slow_channel,
            slope_window=slope_window,
            ema_slope_window=ema_slope_window,
            prior_high_window=prior_high_window,
        )
    ]


def common_volatility_compression_factor_specs(
    windows: tuple[int, ...] = DEFAULT_VOLATILITY_COMPRESSION_WINDOWS,
    bollinger_std_multiplier: float = DEFAULT_BOLLINGER_STD_MULTIPLIER,
    keltner_atr_multiplier: float = DEFAULT_KELTNER_ATR_MULTIPLIER,
) -> list[CommonFactorSpec]:
    """Return volatility-compression and pre-breakout dry-up factor expressions."""
    ordered_windows = tuple(sorted(set(windows)))
    if not ordered_windows:
        raise ValueError("windows must not be empty")

    specs: list[CommonFactorSpec] = []
    donchian_by_window: dict[int, str] = {}

    for window in ordered_windows:
        bollinger_width = _bollinger_width(window, bollinger_std_multiplier)
        keltner_width = _keltner_width(window, keltner_atr_multiplier)
        donchian_width = _donchian_width(window)
        atr_pct = f"ta_atr(high, low, close, {window}) / (close + 1e-12)"
        squeeze_ratio = f"({bollinger_width}) / (({keltner_width}) + 1e-12)"
        donchian_by_window[window] = donchian_width

        specs.extend(
            [
                CommonFactorSpec(
                    name=f"common_bollinger_width_{window}",
                    expression=bollinger_width,
                    category="volatility_compression",
                    description=f"{window}-bar Bollinger band width normalized by moving average",
                ),
                CommonFactorSpec(
                    name=f"common_bollinger_compression_{window}",
                    expression=f"-1 * ({bollinger_width})",
                    category="volatility_compression",
                    description=f"Higher values indicate tighter {window}-bar Bollinger compression",
                ),
                CommonFactorSpec(
                    name=f"common_keltner_width_{window}",
                    expression=keltner_width,
                    category="volatility_compression",
                    description=f"{window}-bar Keltner channel width normalized by moving average",
                ),
                CommonFactorSpec(
                    name=f"common_keltner_compression_{window}",
                    expression=f"-1 * ({keltner_width})",
                    category="volatility_compression",
                    description=f"Higher values indicate tighter {window}-bar Keltner compression",
                ),
                CommonFactorSpec(
                    name=f"common_squeeze_ratio_{window}",
                    expression=squeeze_ratio,
                    category="volatility_compression",
                    description=f"{window}-bar Bollinger width divided by Keltner width",
                ),
                CommonFactorSpec(
                    name=f"common_squeeze_compression_{window}",
                    expression=f"-1 * ({squeeze_ratio})",
                    category="volatility_compression",
                    description=f"Higher values indicate Bollinger bands are tighter relative to Keltner channel for {window} bars",
                ),
                CommonFactorSpec(
                    name=f"common_donchian_width_{window}",
                    expression=donchian_width,
                    category="volatility_compression",
                    description=f"{window}-bar Donchian range width normalized by moving average",
                ),
                CommonFactorSpec(
                    name=f"common_donchian_compression_{window}",
                    expression=f"-1 * ({donchian_width})",
                    category="volatility_compression",
                    description=f"Higher values indicate tighter {window}-bar Donchian range compression",
                ),
                CommonFactorSpec(
                    name=f"common_atr_percentile_{window}",
                    expression=f"ts_rank({atr_pct}, {window})",
                    category="volatility_compression",
                    description=f"Time-series percentile rank of {window}-bar ATR percentage",
                ),
            ]
        )

    for short_window, long_window in zip(ordered_windows, ordered_windows[1:], strict=False):
        short_donchian = donchian_by_window[short_window]
        long_donchian = donchian_by_window[long_window]
        volume_ratio = (
            f"ts_mean(volume, {short_window}) / "
            f"(ts_mean(volume, {long_window}) + 1e-12)"
        )
        specs.extend(
            [
                CommonFactorSpec(
                    name=f"common_range_compression_{short_window}_{long_window}",
                    expression=f"-1 * (({short_donchian}) / (({long_donchian}) + 1e-12))",
                    category="volatility_compression",
                    description=f"Higher values indicate {short_window}-bar range is compressed versus {long_window}-bar range",
                ),
                CommonFactorSpec(
                    name=f"common_volume_contraction_{short_window}_{long_window}",
                    expression=f"-1 * ({volume_ratio})",
                    category="volatility_compression",
                    description=f"Higher values indicate {short_window}-bar volume has dried up versus {long_window}-bar volume",
                ),
            ]
        )

    return specs


def common_volatility_compression_expressions(
    windows: tuple[int, ...] = DEFAULT_VOLATILITY_COMPRESSION_WINDOWS,
    bollinger_std_multiplier: float = DEFAULT_BOLLINGER_STD_MULTIPLIER,
    keltner_atr_multiplier: float = DEFAULT_KELTNER_ATR_MULTIPLIER,
) -> list[str]:
    """Return volatility-compression factor expressions."""
    return _dedupe_expressions(
        [
            spec.expression
            for spec in common_volatility_compression_factor_specs(
                windows=windows,
                bollinger_std_multiplier=bollinger_std_multiplier,
                keltner_atr_multiplier=keltner_atr_multiplier,
            )
        ]
    )


def common_volatility_compression_factor_names(
    windows: tuple[int, ...] = DEFAULT_VOLATILITY_COMPRESSION_WINDOWS,
    bollinger_std_multiplier: float = DEFAULT_BOLLINGER_STD_MULTIPLIER,
    keltner_atr_multiplier: float = DEFAULT_KELTNER_ATR_MULTIPLIER,
) -> list[str]:
    """Return volatility-compression factor names."""
    return [
        spec.name
        for spec in common_volatility_compression_factor_specs(
            windows=windows,
            bollinger_std_multiplier=bollinger_std_multiplier,
            keltner_atr_multiplier=keltner_atr_multiplier,
        )
    ]


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
