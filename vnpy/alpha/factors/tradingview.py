from __future__ import annotations

import re
from dataclasses import dataclass


TRADINGVIEW_TREND_RSI_FAMILY: str = "trend_rsi"
TRADINGVIEW_TREND_RSI_V2_FAMILY: str = "trend_rsi_v2"


@dataclass(frozen=True)
class TradingViewFactorSpec:
    """Factor translated from a local TradingView/Pine indicator."""

    name: str
    expression: str
    family: str
    source_name: str
    source_line: int
    pine_source: str
    category: str
    description: str
    required_columns: tuple[str, ...] = ()


RawTradingViewSpec = tuple[str, str, str, int]


_TREND_RSI_RAW_SPECS: tuple[RawTradingViewSpec, ...] = (
    ("emaFast", "ta_ema(close, 20)", "emaFast = ta.ema(close, emaFastLen)", 65),
    ("emaMid", "ta_ema(close, 60)", "emaMid  = ta.ema(close, emaMidLen)", 66),
    ("emaSlow", "ta_ema(close, 120)", "emaSlow = ta.ema(close, emaSlowLen)", 67),
    ("rsiVal", "ta_rsi(close, 14)", "rsiVal = ta.rsi(close, rsiLen)", 69),
    ("atrVal", "ta_atr(high, low, close, 14)", "atrVal = ta.atr(atrLen)", 74),
    ("volMA", "ts_mean(volume, 20)", "volMA = ta.sma(volume, volLen)", 76),
    (
        "emaBull",
        "((ta_ema(close, 20) > ta_ema(close, 60)) * (ta_ema(close, 60) > ta_ema(close, 120)))",
        "emaBull = emaFast > emaMid and emaMid > emaSlow",
        83,
    ),
    (
        "emaBear",
        "((ta_ema(close, 20) < ta_ema(close, 60)) * (ta_ema(close, 60) < ta_ema(close, 120)))",
        "emaBear = emaFast < emaMid and emaMid < emaSlow",
        84,
    ),
    (
        "downContext",
        "(((close < ta_ema(close, 60)) + (ta_ema(close, 20) < ta_ema(close, 60)) + (ta_ema(close, 60) < ta_ema(close, 120))) > 0)",
        "downContext = close < emaMid or emaFast < emaMid or emaMid < emaSlow",
        86,
    ),
    (
        "deepDown",
        "((((ta_ema(close, 20) < ta_ema(close, 60)) * (ta_ema(close, 60) < ta_ema(close, 120)))) * (close < ta_ema(close, 20)))",
        "deepDown    = emaBear and close < emaFast",
        87,
    ),
    (
        "upContext",
        "((close > ta_ema(close, 60)) * (ta_ema(close, 20) > ta_ema(close, 60)))",
        "upContext   = close > emaMid and emaFast > emaMid",
        88,
    ),
    (
        "recentOversold",
        "ts_min(ta_rsi(close, 14), 20) <= 30.0",
        "recentOversold = ta.lowest(rsiVal, osLookback) <= rsiOS",
        93,
    ),
    (
        "rsiReturnNow",
        "((ts_min(ta_rsi(close, 14), 20) <= 30.0) * (((ta_rsi(close, 14) > 40.0) * (ts_delay(ta_rsi(close, 14), 1) <= 40.0))))",
        "rsiReturnNow   = recentOversold and ta.crossover(rsiVal, rsiReturnLine)",
        94,
    ),
    (
        "rsiRecovered",
        "((ts_min(ta_rsi(close, 14), 20) <= 30.0) * (ta_rsi(close, 14) > 40.0) * (ta_rsi(close, 14) > ts_delay(ta_rsi(close, 14), 1)))",
        "rsiRecovered   = recentOversold and rsiVal > rsiReturnLine and rsiVal > rsiVal[1]",
        95,
    ),
    ("plRsi", "ts_delay(ta_rsi(close, 14), 5)", "plRsi = rsiVal[divRight]", 100),
    ("priorLowLevel", "ts_min(ts_delay(low, 1), 30)", "priorLowLevel = ta.lowest(low[1], structureLen)", 112),
    ("barRange", "greater(high - low, 1e-12)", "barRange = math.max(high - low, syminfo.mintick)", 115),
    ("realBody", "abs(close - open)", "realBody = math.abs(close - open)", 116),
    ("lowerWick", "less(open, close) - low", "lowerWick = math.min(open, close) - low", 117),
    ("upperWick", "high - greater(open, close)", "upperWick = high - math.max(open, close)", 118),
    ("closePosition", "(close - low) / greater(high - low, 1e-12)", "closePosition = (close - low) / barRange", 120),
    (
        "longLowerWick",
        "((less(open, close) - low >= ta_atr(high, low, close, 14) * 0.7) * ((close - low) / greater(high - low, 1e-12) >= 0.65))",
        "longLowerWick = lowerWick >= atrVal * wickAtrMult and closePosition >= closePosMin",
        122,
    ),
    (
        "strongBullBody",
        "((close > open) * (abs(close - open) >= ta_atr(high, low, close, 14) * 0.9) * ((close - low) / greater(high - low, 1e-12) >= 0.65))",
        "strongBullBody = close > open and realBody >= atrVal * bodyAtrMult and closePosition >= closePosMin",
        123,
    ),
    ("breakHighLevel", "ts_max(ts_delay(high, 1), 10)", "breakHighLevel = ta.highest(high[1], breakoutLen)", 125),
    (
        "crossEma20",
        "((close > ta_ema(close, 20)) * (ts_delay(close, 1) <= ts_delay(ta_ema(close, 20), 1)))",
        "crossEma20 = ta.crossover(close, emaFast)",
        128,
    ),
    (
        "crossEma60",
        "((close > ta_ema(close, 60)) * (ts_delay(close, 1) <= ts_delay(ta_ema(close, 60), 1)))",
        "crossEma60 = ta.crossover(close, emaMid)",
        129,
    ),
)


_TREND_RSI_V2_RAW_SPECS: tuple[RawTradingViewSpec, ...] = (
    ("ema20", "ta_ema(close, 20)", "ema20  = ta.ema(close, emaFastLen)", 271),
    ("ema60", "ta_ema(close, 60)", "ema60  = ta.ema(close, emaMidLen)", 272),
    ("ema120", "ta_ema(close, 120)", "ema120 = ta.ema(close, emaSlowLen)", 273),
    ("rsiVal", "ta_rsi(close, 14)", "rsiVal = ta.rsi(close, rsiLen)", 275),
    ("atrVal", "ta_atr(high, low, close, 14)", "atrVal = ta.atr(atrLen)", 280),
    ("volMA", "ts_mean(volume, 20)", "volMA = ta.sma(volume, volLen)", 282),
    (
        "emaBull",
        "((ta_ema(close, 20) > ta_ema(close, 60)) * (ta_ema(close, 60) > ta_ema(close, 120)))",
        "emaBull = ema20 > ema60 and ema60 > ema120",
        311,
    ),
    (
        "emaBear",
        "((ta_ema(close, 20) < ta_ema(close, 60)) * (ta_ema(close, 60) < ta_ema(close, 120)))",
        "emaBear = ema20 < ema60 and ema60 < ema120",
        312,
    ),
    (
        "downContext",
        "(((close < ta_ema(close, 60)) + (ta_ema(close, 20) < ta_ema(close, 60)) + (ta_ema(close, 60) < ta_ema(close, 120))) > 0)",
        "downContext = close < ema60 or ema20 < ema60 or ema60 < ema120",
        314,
    ),
    (
        "deepDown",
        "((((ta_ema(close, 20) < ta_ema(close, 60)) * (ta_ema(close, 60) < ta_ema(close, 120)))) * (close < ta_ema(close, 20)))",
        "deepDown    = emaBear and close < ema20",
        315,
    ),
    (
        "upContext",
        "((close > ta_ema(close, 60)) * (ta_ema(close, 20) > ta_ema(close, 60)))",
        "upContext   = close > ema60 and ema20 > ema60",
        316,
    ),
    (
        "recentOversold",
        "ts_min(ta_rsi(close, 14), 20) <= 30.0",
        "recentOversold = ta.lowest(rsiVal, osLookback) <= rsiOS",
        321,
    ),
    (
        "rsiReturnNow",
        "((ts_min(ta_rsi(close, 14), 20) <= 30.0) * (((ta_rsi(close, 14) > 40.0) * (ts_delay(ta_rsi(close, 14), 1) <= 40.0))))",
        "rsiReturnNow   = recentOversold and ta.crossover(rsiVal, rsiReturnLine)",
        322,
    ),
    (
        "rsiRecovered",
        "((ts_min(ta_rsi(close, 14), 20) <= 30.0) * (ta_rsi(close, 14) > 40.0) * (ta_rsi(close, 14) > ts_delay(ta_rsi(close, 14), 1)))",
        "rsiRecovered   = recentOversold and rsiVal > rsiReturnLine and rsiVal > rsiVal[1]",
        323,
    ),
    ("plRsi", "ts_delay(ta_rsi(close, 14), 5)", "plRsi = rsiVal[divRight]", 335),
    ("phRsi", "ts_delay(ta_rsi(close, 14), 5)", "phRsi = rsiVal[divRight]", 351),
    ("priorLowLevel", "ts_min(ts_delay(low, 1), 30)", "priorLowLevel = ta.lowest(low[1], structureLen)", 369),
    ("barRange", "greater(high - low, 1e-12)", "barRange = math.max(high - low, syminfo.mintick)", 372),
    ("realBody", "abs(close - open)", "realBody = math.abs(close - open)", 373),
    ("lowerWick", "less(open, close) - low", "lowerWick = math.min(open, close) - low", 374),
    ("upperWick", "high - greater(open, close)", "upperWick = high - math.max(open, close)", 375),
    ("closePosition", "(close - low) / greater(high - low, 1e-12)", "closePosition = (close - low) / barRange", 377),
    (
        "longLowerWick",
        "((less(open, close) - low >= ta_atr(high, low, close, 14) * 0.7) * ((close - low) / greater(high - low, 1e-12) >= 0.65))",
        "longLowerWick = lowerWick >= atrVal * wickAtrMult and closePosition >= closePosMin",
        379,
    ),
    (
        "strongBullBody",
        "((close > open) * (abs(close - open) >= ta_atr(high, low, close, 14) * 0.9) * ((close - low) / greater(high - low, 1e-12) >= 0.65))",
        "strongBullBody = close > open and realBody >= atrVal * bodyAtrMult and closePosition >= closePosMin",
        380,
    ),
    ("breakHighLevel", "ts_max(ts_delay(high, 1), 10)", "breakHighLevel = ta.highest(high[1], breakoutLen)", 382),
    (
        "crossEma20",
        "((close > ta_ema(close, 20)) * (ts_delay(close, 1) <= ts_delay(ta_ema(close, 20), 1)))",
        "crossEma20 = ta.crossover(close, ema20)",
        385,
    ),
    (
        "crossEma60",
        "((close > ta_ema(close, 60)) * (ts_delay(close, 1) <= ts_delay(ta_ema(close, 60), 1)))",
        "crossEma60 = ta.crossover(close, ema60)",
        386,
    ),
    ("rsiRangeLow", "ts_min(low, 120)", "rsiRangeLow  = ta.lowest(low, rsiOverlayLookback)", 1106),
    ("rsiRangeHigh", "ts_max(high, 120)", "rsiRangeHigh = ta.highest(high, rsiOverlayLookback)", 1107),
    (
        "rsiPriceSpan",
        "greater(ts_max(high, 120) - ts_min(low, 120), 1e-12)",
        "rsiPriceSpan = math.max(rsiRangeHigh - rsiRangeLow, syminfo.mintick)",
        1108,
    ),
    (
        "rsiBandBottom",
        "ts_min(low, 120) + greater(ts_max(high, 120) - ts_min(low, 120), 1e-12) * 2.0 / 100.0",
        "rsiBandBottom = rsiRangeLow + rsiPriceSpan * rsiBandOffsetPct / 100.0",
        1110,
    ),
    (
        "rsiBandHeight",
        "greater(ts_max(high, 120) - ts_min(low, 120), 1e-12) * 18.0 / 100.0",
        "rsiBandHeight = rsiPriceSpan * rsiBandHeightPct / 100.0",
        1111,
    ),
)


TRADINGVIEW_SKIPPED_INDICATORS: tuple[RawTradingViewSpec, ...] = (
    ("score", "0", "score = 0", 299),
    ("score", "0", "score = 0", 1007),
    ("bgSoft", "less(86 + 4, 100)", "bgSoft = math.min(bgTransp + 4, 100)", 1082),
)


def _snake_case(value: str) -> str:
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def _category_for(source_name: str) -> str:
    lowered = source_name.lower()
    if "ema" in lowered:
        return "trend"
    if "rsi" in lowered:
        return "rsi"
    if "atr" in lowered:
        return "volatility"
    if "vol" in lowered:
        return "volume"
    if any(token in lowered for token in ("wick", "body", "range", "position", "high", "low")):
        return "structure"
    if "context" in lowered or "bull" in lowered or "bear" in lowered or "deep" in lowered:
        return "regime"
    if "cross" in lowered or "recover" in lowered or "oversold" in lowered:
        return "signal"
    return "tradingview"


def _build_specs(family: str, raw_specs: tuple[RawTradingViewSpec, ...]) -> list[TradingViewFactorSpec]:
    prefix = f"tv_{family}"
    return [
        TradingViewFactorSpec(
            name=f"{prefix}_{_snake_case(source_name)}",
            expression=expression,
            family=family,
            source_name=source_name,
            source_line=line,
            pine_source=pine_source,
            category=_category_for(source_name),
            description=f"TradingView {family} indicator extracted from Pine variable {source_name}.",
        )
        for source_name, expression, pine_source, line in raw_specs
    ]


def tradingview_trend_rsi_factor_specs() -> list[TradingViewFactorSpec]:
    """Return market-derived indicators extracted from trendRsi.pine."""
    return _build_specs(TRADINGVIEW_TREND_RSI_FAMILY, _TREND_RSI_RAW_SPECS)


def tradingview_trend_rsi_v2_factor_specs() -> list[TradingViewFactorSpec]:
    """Return market-derived indicators extracted from trendRsiv2.pine."""
    return _build_specs(TRADINGVIEW_TREND_RSI_V2_FAMILY, _TREND_RSI_V2_RAW_SPECS)


def tradingview_factor_specs(family: str | None = None) -> list[TradingViewFactorSpec]:
    """Return TradingView factor specs, optionally filtered by family."""
    specs = tradingview_trend_rsi_factor_specs() + tradingview_trend_rsi_v2_factor_specs()
    if family is None:
        return specs
    return [spec for spec in specs if spec.family == family]


def tradingview_factor_names(family: str | None = None) -> list[str]:
    """Return TradingView factor names in spec order."""
    return [spec.name for spec in tradingview_factor_specs(family)]


def tradingview_expressions(family: str | None = None) -> list[str]:
    """Return TradingView factor expressions in spec order."""
    return [spec.expression for spec in tradingview_factor_specs(family)]


__all__ = [
    "TRADINGVIEW_SKIPPED_INDICATORS",
    "TRADINGVIEW_TREND_RSI_FAMILY",
    "TRADINGVIEW_TREND_RSI_V2_FAMILY",
    "TradingViewFactorSpec",
    "tradingview_expressions",
    "tradingview_factor_names",
    "tradingview_factor_specs",
    "tradingview_trend_rsi_factor_specs",
    "tradingview_trend_rsi_v2_factor_specs",
]
