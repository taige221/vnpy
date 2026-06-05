from datetime import datetime
from typing import Any

import polars as pl

from vnpy.alpha.strategy.strategies import EventSignalStrategy
from vnpy.trader.constant import Direction, Exchange, Interval, Offset
from vnpy.trader.object import BarData


class FakeEngine:
    """Small strategy engine double for EventSignalStrategy tests."""

    def __init__(self, signal: pl.DataFrame, cash: float = 100_000) -> None:
        self.signal: pl.DataFrame = signal
        self.cash: float = cash
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
        """Capture submitted orders."""
        self.orders.append((vt_symbol, direction, offset, price, volume))
        return [str(len(self.orders))]

    def cancel_order(self, strategy: Any, vt_orderid: str) -> None:
        """Cancel order stub."""

    def get_cash_available(self) -> float:
        """Return test cash."""
        return self.cash

    def get_holding_value(self) -> float:
        """Return test holding value."""
        return 0

    def write_log(self, msg: str, strategy: Any | None = None) -> None:
        """Collecting logs is not needed for this test."""


def make_bar(symbol: str, exchange: Exchange, close_price: float) -> BarData:
    """Create a daily bar for tests."""
    return BarData(
        symbol=symbol,
        exchange=exchange,
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


def test_event_signal_strategy_buys_top_signals() -> None:
    """Test that the strategy buys the strongest sparse event signals."""
    signal = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2)] * 3,
            "vt_symbol": ["600000.SSE", "000001.SZSE", "300750.SZSE"],
            "signal": [0.9, 0.7, 0.8],
            "max_hold_days": [5, 5, 5],
        }
    )
    engine = FakeEngine(signal)
    strategy = EventSignalStrategy(engine, "event", [], {"top_k": 2, "price_add": 0})
    strategy.on_init()

    strategy.on_bars(
        {
            "600000.SSE": make_bar("600000", Exchange.SSE, 10),
            "000001.SZSE": make_bar("000001", Exchange.SZSE, 20),
            "300750.SZSE": make_bar("300750", Exchange.SZSE, 50),
        }
    )

    assert [order[0] for order in engine.orders] == ["600000.SSE", "300750.SZSE"]
    assert all(order[1] == Direction.LONG for order in engine.orders)
    assert all(order[2] == Offset.OPEN for order in engine.orders)


def test_event_signal_strategy_sells_after_max_days() -> None:
    """Test that max holding days triggers a sell target."""
    engine = FakeEngine(pl.DataFrame({"datetime": [], "vt_symbol": [], "signal": []}))
    strategy = EventSignalStrategy(engine, "event", [], {"max_days": 2, "price_add": 0})
    strategy.on_init()

    strategy.pos_data["600000.SSE"] = 1_000
    strategy.holding_days["600000.SSE"] = 1

    strategy.on_bars({"600000.SSE": make_bar("600000", Exchange.SSE, 10)})

    assert engine.orders == [("600000.SSE", Direction.SHORT, Offset.CLOSE, 10, 1_000)]


def test_event_signal_strategy_does_not_buy_when_sell_is_blocked_by_min_days() -> None:
    """Test skipped sells do not free a new buy slot."""
    signal = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2)],
            "vt_symbol": ["300750.SZSE"],
            "signal": [0.9],
        }
    )
    engine = FakeEngine(signal)
    strategy = EventSignalStrategy(
        engine,
        "event",
        [],
        {"top_k": 2, "min_days": 3, "price_add": 0},
    )
    strategy.on_init()

    strategy.pos_data["600000.SSE"] = 1_000
    strategy.pos_data["000001.SZSE"] = 1_000
    strategy.target_data["600000.SSE"] = 1_000
    strategy.target_data["000001.SZSE"] = 1_000
    strategy.holding_days["600000.SSE"] = 0
    strategy.holding_days["000001.SZSE"] = 0
    strategy.max_holding_days["600000.SSE"] = 1

    strategy.on_bars(
        {
            "600000.SSE": make_bar("600000", Exchange.SSE, 10),
            "000001.SZSE": make_bar("000001", Exchange.SZSE, 20),
            "300750.SZSE": make_bar("300750", Exchange.SZSE, 50),
        }
    )

    assert engine.orders == []


def test_event_signal_strategy_does_not_buy_when_sell_has_no_bar() -> None:
    """Test missing sell-side bar data does not free a new buy slot."""
    signal = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2)],
            "vt_symbol": ["300750.SZSE"],
            "signal": [0.9],
        }
    )
    engine = FakeEngine(signal)
    strategy = EventSignalStrategy(engine, "event", [], {"top_k": 2, "price_add": 0})
    strategy.on_init()

    strategy.pos_data["600000.SSE"] = 1_000
    strategy.pos_data["000001.SZSE"] = 1_000
    strategy.target_data["600000.SSE"] = 1_000
    strategy.target_data["000001.SZSE"] = 1_000
    strategy.holding_days["600000.SSE"] = 5
    strategy.holding_days["000001.SZSE"] = 0
    strategy.max_holding_days["600000.SSE"] = 1

    strategy.on_bars(
        {
            "000001.SZSE": make_bar("000001", Exchange.SZSE, 20),
            "300750.SZSE": make_bar("300750", Exchange.SZSE, 50),
        }
    )

    assert engine.orders == []


def test_event_signal_strategy_does_not_short_when_cash_is_negative() -> None:
    """Test negative cash cannot create a short target from a buy signal."""
    signal = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2)],
            "vt_symbol": ["000100.SZSE"],
            "signal": [0.9],
        }
    )
    engine = FakeEngine(signal, cash=-1_000)
    strategy = EventSignalStrategy(engine, "event", [], {"top_k": 1, "price_add": 0})
    strategy.on_init()

    strategy.on_bars({"000100.SZSE": make_bar("000100", Exchange.SZSE, 4.34)})

    assert strategy.get_target("000100.SZSE") == 0
    assert engine.orders == []
