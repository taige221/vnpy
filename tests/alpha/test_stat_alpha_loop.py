from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from vnpy.alpha.factors import classic_price_expressions
from vnpy.alpha.utils.stat_alpha_loop import (
    StatAlphaLoop,
    dedupe_expressions,
    forward_return_expr,
)
from vnpy.alpha.utils import build_eligibility_from_source_frame


def make_trending_bars(symbols: int = 8, days: int = 90) -> pl.DataFrame:
    """Create deterministic bars where higher recent return predicts higher future return."""
    start = datetime(2020, 1, 1)
    rows: list[dict[str, float | str | datetime]] = []

    for symbol_index in range(symbols):
        vt_symbol = f"S{symbol_index:03d}.TEST"
        drift = 0.001 + symbol_index * 0.0005
        for day_index in range(days):
            current_date = start + timedelta(days=day_index)
            close = 100.0 * ((1.0 + drift) ** day_index)
            volume = 1_000_000.0 + symbol_index * 10_000.0
            rows.append(
                {
                    "datetime": current_date,
                    "vt_symbol": vt_symbol,
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": volume,
                    "turnover": volume * close,
                    "open_interest": 0.0,
                    "vwap": close,
                }
            )

    return pl.DataFrame(rows).sort(["datetime", "vt_symbol"])


def test_forward_return_expr_uses_delayed_entry() -> None:
    """Test label expression construction."""
    assert forward_return_expr(5, entry_lag=1) == "ts_delay(close, -6) / ts_delay(close, -1) - 1"
    assert forward_return_expr(1, entry_lag=0) == "ts_delay(close, -1) / close - 1"


def test_stat_alpha_loop_scores_predictive_expression() -> None:
    """Test a deterministic momentum expression scores strongly in every segment."""
    loop = StatAlphaLoop(
        make_trending_bars(),
        train_period=("2020-01-10", "2020-02-10"),
        valid_period=("2020-02-11", "2020-02-28"),
        test_period=("2020-03-01", "2020-03-20"),
        horizons=(1,),
        min_universe=5,
    )

    records = loop.score("close / ts_delay(close, 1) - 1")

    assert len(records) == 1
    assert records[0]["train_ic"] > 0.99
    assert records[0]["valid_ic"] > 0.99
    assert records[0]["test_ic"] > 0.99
    assert records[0]["test_spread"] > 0
    assert loop.history_frame().height == 1


def test_stat_alpha_loop_applies_eligibility_panel() -> None:
    """Test optional eligibility filters reduce cross-section samples."""
    bars = make_trending_bars()
    eligibility = bars.select(
        "datetime",
        "vt_symbol",
        (pl.col("vt_symbol") != "S000.TEST").alias("eligible"),
    )
    loop = StatAlphaLoop(
        bars,
        train_period=("2020-01-10", "2020-02-10"),
        valid_period=("2020-02-11", "2020-02-28"),
        test_period=("2020-03-01", "2020-03-20"),
        horizons=(1,),
        min_universe=5,
        eligibility_df=eligibility,
    )

    record = loop.score("close / ts_delay(close, 1) - 1")[0]

    assert record["test_samples_mean"] == 7.0


def test_stat_alpha_loop_records_expression_errors() -> None:
    """Test bad DSL expressions are recorded without crashing the loop."""
    loop = StatAlphaLoop(
        make_trending_bars(),
        train_period=("2020-01-10", "2020-02-10"),
        valid_period=("2020-02-11", "2020-02-28"),
        test_period=("2020-03-01", "2020-03-20"),
        horizons=(1,),
        min_universe=5,
    )

    record = loop.score("unknown_operator(close)")[0]

    assert record["error"]
    assert record["test_days"] == 0


def test_candidate_helpers_are_stable() -> None:
    """Test expression helper utilities."""
    assert dedupe_expressions([" close ", "close", "", "volume"]) == ["close", "volume"]
    expressions = classic_price_expressions((5, 10))

    assert expressions
    assert len(expressions) == len(set(expressions))
    assert any("ts_corr(close, volume, 5)" in expr for expr in expressions)


def test_a_share_eligibility_filters_tradability_cases() -> None:
    """Test A-share eligibility filters common untradable rows."""
    dates = [datetime(2020, 6, 1), datetime(2020, 6, 2)]
    rows: list[dict[str, object]] = []
    cases = [
        ("GOOD.SZSE", "Good", datetime(2019, 1, 1), 60000.0, 200000.0, False),
        ("LIMIT.SZSE", "Limit", datetime(2019, 1, 1), 60000.0, 200000.0, True),
        ("STOCK.SZSE", "*ST Stock", datetime(2019, 1, 1), 60000.0, 200000.0, False),
        ("NEW.SZSE", "New", datetime(2020, 5, 15), 60000.0, 200000.0, False),
        ("THIN.SZSE", "Thin", datetime(2019, 1, 1), 1000.0, 2000.0, False),
    ]
    for vt_symbol, name, list_date, amount, circ_mv, entry_limit_up in cases:
        for index, current_date in enumerate(dates):
            raw_close = 10.0
            up_limit = 11.0
            if vt_symbol == "LIMIT.SZSE" and index == 1 and entry_limit_up:
                raw_close = up_limit
            rows.append(
                {
                    "datetime": current_date,
                    "vt_symbol": vt_symbol,
                    "raw_close": raw_close,
                    "amount": amount,
                    "circ_mv": circ_mv,
                    "up_limit": up_limit,
                    "down_limit": 9.0,
                    "name": name,
                    "list_date": list_date,
                    "delist_date": None,
                }
            )

    eligibility = build_eligibility_from_source_frame(
        pl.DataFrame(rows),
        start_date=dates[0],
        end_date=dates[0],
        entry_lag=1,
        exclude_new_listing_days=120,
        include_current_st=False,
        include_entry_limit=False,
        min_amount=50000,
        min_circ_mv=100000,
    )
    signal_day = eligibility.filter(pl.col("datetime") == dates[0]).sort("vt_symbol")
    result = dict(zip(signal_day["vt_symbol"].to_list(), signal_day["eligible"].to_list(), strict=True))

    assert result == {
        "GOOD.SZSE": True,
        "LIMIT.SZSE": False,
        "NEW.SZSE": False,
        "STOCK.SZSE": False,
        "THIN.SZSE": False,
    }
