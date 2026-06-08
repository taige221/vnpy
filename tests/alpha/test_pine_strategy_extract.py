from datetime import datetime
from pathlib import Path

import polars as pl

from vnpy.alpha.dataset.utility import calculate_by_expression
from vnpy.alpha.utils.pine_strategy_extract import (
    extract_pine_strategy,
    extract_pine_strategy_file,
)


SIMPLE_PINE = """//@version=6
strategy("Demo Strategy", overlay = true)
groupTrend = "Trend"
len = input.int(20, "EMA Length", minval = 1, group = groupTrend)
src = input.source(close, "Source")
emaFast = ta.ema(src, len)
longSignal = ta.crossover(close, emaFast) and close > emaFast
if longSignal
    strategy.entry("Long", strategy.long, comment = "Buy")
alertcondition(longSignal, title = "Long Alert", message = "Long")
"""


def test_extracts_metadata_inputs_assignments_and_actions() -> None:
    """Extract basic Pine strategy structure."""
    result = extract_pine_strategy(SIMPLE_PINE)

    assert result.metadata["kind"] == "strategy"
    assert result.metadata["title"] == "Demo Strategy"
    assert result.inputs[0].name == "len"
    assert result.inputs[0].input_type == "int"
    assert result.inputs[0].default == 20
    assert result.inputs[0].title == "EMA Length"
    assert result.inputs[0].group == "Trend"
    assert result.assignments[-1].name == "longSignal"
    assert result.actions[0].action_type == "strategy.entry"
    assert result.actions[0].condition == "longSignal"
    assert result.alerts[0].name == "Long Alert"


def test_translates_common_pine_expression_subset() -> None:
    """Translate common technical functions into alpha expressions."""
    result = extract_pine_strategy(SIMPLE_PINE)
    expressions = {item.name: item.expression for item in result.alpha_expressions}

    assert expressions["emaFast"] == "ta_ema(close, 20)"
    assert "ta_ema(close, 20)" in expressions["longSignal"]
    assert "ts_delay(close, 1)" in expressions["longSignal"]
    assert expressions["strategy_entry_Long"] == expressions["longSignal"]
    assert all(item.source != 'groupTrend = "Trend"' for item in result.unsupported)


def test_translates_history_with_input_offset() -> None:
    """Resolve input offsets inside Pine history references."""
    result = extract_pine_strategy(
        "right = input.int(5, \"Right\")\nrsiVal = ta.rsi(close, 14)\nplRsi = rsiVal[right]\n"
    )
    expressions = {item.name: item.expression for item in result.alpha_expressions}

    assert expressions["plRsi"] == "ts_delay(ta_rsi(close, 14), 5)"


def test_crossover_does_not_delay_scalar_threshold() -> None:
    """Keep scalar thresholds scalar in translated crossover expressions."""
    result = extract_pine_strategy("rsiVal = ta.rsi(close, 14)\nrsiReturn = ta.crossover(rsiVal, 40.0)\n")
    expressions = {item.name: item.expression for item in result.alpha_expressions}

    assert "ts_delay(40.0, 1)" not in expressions["rsiReturn"]
    assert "ts_delay(ta_rsi(close, 14), 1) <= 40.0" in expressions["rsiReturn"]


def test_ternary_translation_is_executable_alpha_expression() -> None:
    """Translate Pine ternary masks into the alpha DSL's four-argument quesval."""
    result = extract_pine_strategy("choice = close > open ? close : open\n")
    expressions = {item.name: item.expression for item in result.alpha_expressions}
    bars = pl.DataFrame(
        [
            {
                "datetime": datetime(2024, 1, 1),
                "vt_symbol": "A.TEST",
                "open": 10.0,
                "high": 11.5,
                "low": 9.5,
                "close": 11.0,
                "volume": 100.0,
                "turnover": 1_100.0,
            },
            {
                "datetime": datetime(2024, 1, 2),
                "vt_symbol": "A.TEST",
                "open": 10.0,
                "high": 10.5,
                "low": 8.5,
                "close": 9.0,
                "volume": 100.0,
                "turnover": 900.0,
            },
        ]
    )

    assert expressions["choice"] == "quesval(0, close > open, close, open)"
    assert calculate_by_expression(bars, expressions["choice"])["data"].to_list() == [11.0, 10.0]


def test_reports_unsupported_stateful_pine_features() -> None:
    """Keep stateful Pine constructs in the review report."""
    result = extract_pine_strategy(
        "var bool armed = false\narmed := true\nmode = switch close > open\n    true => close\n"
    )
    reasons = [item.reason for item in result.unsupported]

    assert "stateful var declarations are not translated" in reasons
    assert "reassignment with := is not translated" in reasons
    assert "switch expressions are not translated" in reasons


def test_writes_json_markdown_and_alpha_outputs(tmp_path: Path) -> None:
    """Write extraction artifacts for review and alpha research."""
    source = tmp_path / "demo.pine"
    source.write_text(SIMPLE_PINE, encoding="utf-8")
    output_dir = tmp_path / "reports"

    result = extract_pine_strategy_file(
        source,
        output_dir=output_dir,
        alpha_output=output_dir / "demo_alpha_exprs.json",
    )

    assert result.metadata["title"] == "Demo Strategy"
    assert (output_dir / "demo_pine_extract_report.json").exists()
    assert (output_dir / "demo_pine_extract_report.md").exists()
    assert (output_dir / "demo_alpha_exprs.json").exists()
    assert "Translated Expressions" in (output_dir / "demo_pine_extract_report.md").read_text(encoding="utf-8")
