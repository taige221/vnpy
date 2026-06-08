from vnpy.alpha.utils.pine import (
    AlphaExpressionToPineTranslator,
    PineExportMode,
    export_factor_specs_to_pine,
    resolve_factor_specs,
)
from vnpy.alpha.factors.generic import CommonFactorSpec


def test_translates_supported_alpha_expression_subset() -> None:
    """Translate common alpha DSL expressions into Pine v6 expressions."""
    translator = AlphaExpressionToPineTranslator()

    result = translator.translate(
        "close / (ts_delay(close, 5) + 1e-12) - 1"
        " + ta_rsi(close, 14) / 100.0"
        " + ta_atr(high, low, close, 14) / close"
        " + greater(ts_max(high, 20), ta_ema(close, 10))"
    )

    assert result.expression is not None
    assert result.reason is None
    assert "close[5]" in result.expression
    assert "ta.rsi(close, 14) / 100.0" in result.expression
    assert "ta.atr(14) / close" in result.expression
    assert "math.max(ta.highest(high, 20), ta.ema(close, 10))" in result.expression


def test_translates_boolean_masks_to_numeric_pine_series() -> None:
    """Convert alpha numeric masks into explicit Pine numeric masks."""
    translator = AlphaExpressionToPineTranslator()

    result = translator.translate("(close > open) * (close > ta_ema(close, 10))")

    assert result.expression == "(mask(close > open) * mask(close > ta.ema(close, 10)))"


def test_reports_unsupported_cross_sectional_and_panel_expressions() -> None:
    """Reject expressions that TradingView cannot calculate on one chart symbol."""
    translator = AlphaExpressionToPineTranslator()

    cross_section = translator.translate("cs_rank(close / ts_delay(close, 20) - 1)")
    panel_column = translator.translate("rank_score >= 50")

    assert cross_section.expression is None
    assert cross_section.reason == "cross-sectional function cs_rank is not supported by single-symbol Pine"
    assert panel_column.expression is None
    assert panel_column.reason == "unknown identifier rank_score"


def test_translates_alpha_quesval_and_pow2_signatures() -> None:
    """Translate alpha helper calls using their real DSL signatures."""
    translator = AlphaExpressionToPineTranslator()

    ternary = translator.translate("quesval(0, close > open, close, open)")
    power = translator.translate("pow2(close, open)")

    assert ternary.expression == "(0 < mask(close > open) ? close : open)"
    assert power.expression == "math.pow(close, open)"


def test_exports_indicator_script_with_unsupported_comments() -> None:
    """Build an indicator script and keep unsupported factors visible."""
    specs = [
        CommonFactorSpec(
            name="momentum_5",
            expression="close / (ts_delay(close, 5) + 1e-12) - 1",
            category="price",
            description="Five-bar momentum",
        ),
        CommonFactorSpec(
            name="ranked_momentum_5",
            expression="cs_rank(close / ts_delay(close, 5) - 1)",
            category="price",
            description="Cross-sectional momentum",
        ),
    ]

    exported = export_factor_specs_to_pine(specs, title="Demo Alpha", mode=PineExportMode.INDICATOR)

    assert 'indicator("Demo Alpha", overlay = false)' in exported.script
    assert "momentum_5 = ((close / (close[5] + 1e-12)) - 1)" in exported.script
    assert 'plot(momentum_5, title = "momentum_5")' in exported.script
    assert "// - ranked_momentum_5: cross-sectional function cs_rank is not supported by single-symbol Pine" in exported.script
    assert len(exported.unsupported) == 1


def test_exports_strategy_script_with_score_threshold() -> None:
    """Build a single-symbol strategy script from translated factor scores."""
    specs = [
        CommonFactorSpec(
            name="ema_bull",
            expression="(close > ta_ema(close, 10)) * (ta_ema(close, 10) > ta_ema(close, 20))",
            category="trend",
            description="EMA bull state",
        ),
    ]

    exported = export_factor_specs_to_pine(
        specs,
        title="Demo Strategy",
        mode=PineExportMode.STRATEGY,
        threshold=0.5,
    )

    assert 'strategy("Demo Strategy", overlay = false, process_orders_on_close = true)' in exported.script
    assert "score = nz(ema_bull) / 1.0" in exported.script
    assert "entrySignal = score >= 0.5" in exported.script
    assert 'strategy.entry("Long", strategy.long)' in exported.script
    assert 'strategy.close("Long")' in exported.script


def test_resolves_existing_factor_families() -> None:
    """Expose existing factor families for CLI usage."""
    common_specs = resolve_factor_specs("common_technical", windows=(5,))
    reclaim_specs = resolve_factor_specs("strategy_box", signal_type="range_reclaim2", windows=(20,))

    assert common_specs[0].name == "common_momentum_5"
    assert any(spec.name == "box_range_reclaim2_quality_20" for spec in reclaim_specs)
