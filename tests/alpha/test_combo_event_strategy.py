from datetime import datetime
from types import SimpleNamespace
from typing import Any

import polars as pl

from vnpy.alpha.strategy.strategies.combo_event_strategy import ComboEventStrategy
from vnpy.trader.constant import Direction, Exchange, Interval, Offset
from vnpy.trader.object import BarData


class FakeEngine:
    """Small strategy engine double for ComboEventStrategy tests."""

    def __init__(
        self,
        signal: pl.DataFrame,
        cash: float = 1_000_000,
        holding_value: float = 0,
    ) -> None:
        self.signal: pl.DataFrame = signal
        self.cash: float = cash
        self.holding_value: float = holding_value
        self.orders: list[tuple[str, Direction, Offset, float, float]] = []

    def get_signal(self) -> pl.DataFrame:
        """Return current signal slice."""
        return self.signal

    def send_order(
        self,
        strategy: Any,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
    ) -> list[str]:
        """Capture submitted orders and update strategy position state."""
        self.orders.append((vt_symbol, direction, offset, price, volume))
        strategy.update_trade(
            SimpleNamespace(direction=direction, vt_symbol=vt_symbol, volume=volume)
        )
        return [str(len(self.orders))]

    def cancel_order(self, strategy: Any, vt_orderid: str) -> None:
        """Cancel order stub."""

    def get_cash_available(self) -> float:
        """Return test cash."""
        return self.cash

    def get_holding_value(self) -> float:
        """Return test holding value."""
        return self.holding_value

    def write_log(self, msg: str, strategy: Any | None = None) -> None:
        """Collecting logs is not needed for this test."""


def make_bar(vt_symbol: str, close_price: float = 10) -> BarData:
    """Create a daily bar for tests."""
    symbol, exchange = vt_symbol.split(".")
    return BarData(
        symbol=symbol,
        exchange=Exchange(exchange),
        datetime=datetime(2024, 1, 2),
        interval=Interval.DAILY,
        open_price=close_price,
        high_price=close_price,
        low_price=close_price,
        close_price=close_price,
        volume=100_000,
        turnover=close_price * 100_000,
        gateway_name="TEST",
    )


def make_strategy(
    signal: pl.DataFrame,
    setting: dict[str, Any] | None = None,
    cash: float = 1_000_000,
    holding_value: float = 0,
) -> ComboEventStrategy:
    """Create an initialized strategy with a fake engine."""
    engine = FakeEngine(signal, cash=cash, holding_value=holding_value)
    strategy = ComboEventStrategy(engine, "combo", [], setting or {})
    strategy.on_init()
    return strategy


def test_hard_mode_only_opens_entry_signals() -> None:
    """Hard mode requires an entry event for new positions."""
    signal = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "vt_symbol": ["600000.SSE", "000001.SZSE"],
            "signal": [0.7, 1.0],
            "entry_signal": [1.0, 0.0],
        }
    )
    strategy = make_strategy(signal, {"top_k": 2, "cash_ratio": 1.0, "price_add": 0})

    strategy.on_bars(
        {
            "600000.SSE": make_bar("600000.SSE"),
            "000001.SZSE": make_bar("000001.SZSE"),
        }
    )

    assert strategy.get_target("600000.SSE") > 0
    assert strategy.get_target("000001.SZSE") == 0


def test_soft_mode_can_open_high_rank_without_entry_event() -> None:
    """Soft mode keeps combo rank eligible even without an entry event."""
    signal = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "vt_symbol": ["600000.SSE", "000001.SZSE"],
            "signal": [0.7, 1.0],
            "entry_signal": [1.0, 0.0],
        }
    )
    strategy = make_strategy(
        signal,
        {"entry_mode": "soft", "top_k": 1, "cash_ratio": 1.0, "price_add": 0},
    )

    strategy.on_bars(
        {
            "600000.SSE": make_bar("600000.SSE"),
            "000001.SZSE": make_bar("000001.SZSE"),
        }
    )

    assert strategy.get_target("000001.SZSE") > 0
    assert strategy.get_target("600000.SSE") == 0


def test_existing_holding_can_remain_without_entry_event() -> None:
    """Hard mode entry events only gate new positions, not continuing holdings."""
    signal = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2)],
            "vt_symbol": ["600000.SSE"],
            "signal": [1.0],
            "entry_signal": [0.0],
        }
    )
    strategy = make_strategy(signal, {"top_k": 1, "cash_ratio": 1.0, "price_add": 0})
    strategy.pos_data["600000.SSE"] = 1_000
    strategy.target_data["600000.SSE"] = 1_000

    strategy.on_bars({"600000.SSE": make_bar("600000.SSE")})

    assert strategy.get_target("600000.SSE") > 0


def test_exit_signal_sells_existing_position() -> None:
    """Exit signal is an immediate sell rule."""
    signal = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2)],
            "vt_symbol": ["600000.SSE"],
            "signal": [1.0],
            "exit_signal": [1.0],
        }
    )
    strategy = make_strategy(signal, {"price_add": 0})
    strategy.pos_data["600000.SSE"] = 1_000
    strategy.target_data["600000.SSE"] = 1_000

    strategy.on_bars({"600000.SSE": make_bar("600000.SSE")})

    assert strategy.get_target("600000.SSE") == 0


def test_stop_price_sells_existing_position() -> None:
    """Stored stop price is an immediate sell rule."""
    signal = pl.DataFrame(
        {"datetime": [datetime(2024, 1, 2)], "vt_symbol": ["600000.SSE"], "signal": [1.0]}
    )
    strategy = make_strategy(signal, {"price_add": 0})
    strategy.pos_data["600000.SSE"] = 1_000
    strategy.target_data["600000.SSE"] = 1_000
    strategy.stop_prices["600000.SSE"] = 11.0

    strategy.on_bars({"600000.SSE": make_bar("600000.SSE", close_price=10)})

    assert strategy.get_target("600000.SSE") == 0


def test_max_hold_days_sells_after_min_days() -> None:
    """Max holding days exits after the minimum holding period is met."""
    signal = pl.DataFrame(
        {"datetime": [datetime(2024, 1, 2)], "vt_symbol": ["600000.SSE"], "signal": [1.0]}
    )
    strategy = make_strategy(signal, {"min_days": 1, "price_add": 0})
    strategy.pos_data["600000.SSE"] = 1_000
    strategy.target_data["600000.SSE"] = 1_000
    strategy.holding_days["600000.SSE"] = 1
    strategy.max_holding_days["600000.SSE"] = 2

    strategy.on_bars({"600000.SSE": make_bar("600000.SSE")})

    assert strategy.get_target("600000.SSE") == 0


def test_keep_top_k_sells_low_rank_existing_position() -> None:
    """Rank retention sells existing holdings outside keep_top_k."""
    signal = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "vt_symbol": ["600000.SSE", "000001.SZSE"],
            "signal": [0.1, 1.0],
        }
    )
    strategy = make_strategy(signal, {"keep_top_k": 1, "min_days": 1, "price_add": 0})
    strategy.pos_data["600000.SSE"] = 1_000
    strategy.target_data["600000.SSE"] = 1_000
    strategy.holding_days["600000.SSE"] = 1

    strategy.on_bars(
        {
            "600000.SSE": make_bar("600000.SSE"),
            "000001.SZSE": make_bar("000001.SZSE"),
        }
    )

    assert strategy.get_target("600000.SSE") == 0


def test_keep_top_k_retains_existing_position_above_top_k() -> None:
    """Existing holdings inside keep_top_k are not forced out by top_k."""
    signal = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "vt_symbol": ["000001.SZSE", "600000.SSE"],
            "signal": [1.0, 0.9],
        }
    )
    strategy = make_strategy(
        signal,
        {"top_k": 1, "keep_top_k": 2, "min_days": 1, "min_volume": 1, "price_add": 0},
    )
    strategy.pos_data["600000.SSE"] = 1_000
    strategy.target_data["600000.SSE"] = 1_000
    strategy.holding_days["600000.SSE"] = 1

    strategy.on_bars(
        {
            "000001.SZSE": make_bar("000001.SZSE"),
            "600000.SSE": make_bar("600000.SSE"),
        }
    )

    assert strategy.get_target("600000.SSE") > 0
    assert strategy.get_target("000001.SZSE") == 0


def test_partial_sell_keeps_position_metadata_until_flat() -> None:
    """Position metadata is removed only after a close fully exits the symbol."""
    strategy = make_strategy(pl.DataFrame(), {"price_add": 0})
    strategy.pos_data["600000.SSE"] = 1_000
    strategy.holding_days["600000.SSE"] = 3
    strategy.max_holding_days["600000.SSE"] = 8
    strategy.stop_prices["600000.SSE"] = 9.5

    strategy.update_trade(
        SimpleNamespace(direction=Direction.SHORT, vt_symbol="600000.SSE", volume=400)
    )

    assert strategy.pos_data["600000.SSE"] == 600
    assert strategy.holding_days["600000.SSE"] == 3
    assert strategy.max_holding_days["600000.SSE"] == 8
    assert strategy.stop_prices["600000.SSE"] == 9.5

    strategy.update_trade(
        SimpleNamespace(direction=Direction.SHORT, vt_symbol="600000.SSE", volume=600)
    )

    assert strategy.pos_data["600000.SSE"] == 0
    assert "600000.SSE" not in strategy.holding_days
    assert "600000.SSE" not in strategy.max_holding_days
    assert "600000.SSE" not in strategy.stop_prices


def test_position_scale_changes_relative_target_size() -> None:
    """Position scale changes target weights after selection."""
    signal = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "vt_symbol": ["600000.SSE", "000001.SZSE"],
            "signal": [1.0, 0.9],
            "position_scale": [2.0, 1.0],
        }
    )
    strategy = make_strategy(
        signal,
        {"top_k": 2, "cash_ratio": 1.0, "min_volume": 1, "price_add": 0},
    )

    strategy.on_bars(
        {
            "600000.SSE": make_bar("600000.SSE", 10),
            "000001.SZSE": make_bar("000001.SZSE", 10),
        }
    )

    assert strategy.get_target("600000.SSE") == 2 * strategy.get_target("000001.SZSE")


def test_portfolio_scale_reduces_total_target_exposure() -> None:
    """Portfolio scale reduces total target exposure."""
    signal = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2)],
            "vt_symbol": ["600000.SSE"],
            "signal": [1.0],
            "portfolio_scale": [0.5],
        }
    )
    strategy = make_strategy(
        signal,
        {"top_k": 1, "cash_ratio": 1.0, "min_volume": 1, "price_add": 0},
    )

    strategy.on_bars({"600000.SSE": make_bar("600000.SSE", 10)})

    assert strategy.get_target("600000.SSE") == 50_000


def test_unselected_portfolio_scale_does_not_change_target_exposure() -> None:
    """Portfolio scale is taken from selected target rows only."""
    signal = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2), datetime(2024, 1, 2)],
            "vt_symbol": ["600000.SSE", "000001.SZSE"],
            "signal": [1.0, 0.5],
            "portfolio_scale": [0.5, 2.0],
        }
    )
    strategy = make_strategy(
        signal,
        {"top_k": 1, "cash_ratio": 1.0, "min_volume": 1, "price_add": 0},
    )

    strategy.on_bars(
        {
            "600000.SSE": make_bar("600000.SSE", 10),
            "000001.SZSE": make_bar("000001.SZSE", 10),
        }
    )

    assert strategy.get_target("600000.SSE") == 50_000
    assert strategy.get_target("000001.SZSE") == 0


def test_missing_optional_columns_use_defaults() -> None:
    """Only datetime, vt_symbol, and signal are required."""
    signal = pl.DataFrame(
        {"datetime": [datetime(2024, 1, 2)], "vt_symbol": ["600000.SSE"], "signal": [1.0]}
    )
    strategy = make_strategy(
        signal,
        {"top_k": 1, "cash_ratio": 1.0, "min_volume": 1, "price_add": 0},
    )

    strategy.on_bars({"600000.SSE": make_bar("600000.SSE", 10)})

    assert strategy.get_target("600000.SSE") == 100_000
    assert strategy.max_holding_days["600000.SSE"] == strategy.default_max_hold_days
