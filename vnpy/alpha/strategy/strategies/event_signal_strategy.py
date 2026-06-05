from collections import defaultdict
from typing import Any, cast

import polars as pl

from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Direction
from vnpy.trader.utility import floor_to

from vnpy.alpha import AlphaStrategy


class EventSignalStrategy(AlphaStrategy):
    """Long-only portfolio strategy for sparse external event signals."""

    top_k: int = 20                 # Maximum number of stocks to hold
    min_days: int = 1               # Minimum holding period in days
    max_days: int = 10              # Default maximum holding period in days
    cash_ratio: float = 0.95        # Cash utilization ratio
    min_volume: int = 100           # Minimum trading unit
    open_rate: float = 0.0005       # Opening commission rate
    close_rate: float = 0.0015      # Closing commission rate
    min_commission: int = 5         # Minimum commission value
    price_add: float = 0.05         # Order price adjustment ratio

    def on_init(self) -> None:
        """Strategy initialization callback"""
        self.holding_days: defaultdict[str, int] = defaultdict(int)
        self.max_holding_days: dict[str, int] = {}
        self.stop_prices: dict[str, float] = {}

        self.write_log("Event signal strategy initialized")

    def on_trade(self, trade: TradeData) -> None:
        """Trade execution callback"""
        if trade.direction == Direction.SHORT:
            self.holding_days.pop(trade.vt_symbol, None)
            self.max_holding_days.pop(trade.vt_symbol, None)
            self.stop_prices.pop(trade.vt_symbol, None)
        else:
            self.holding_days.setdefault(trade.vt_symbol, 0)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K-line slice callback"""
        today_signal: pl.DataFrame = self.get_signal().sort("signal", descending=True)
        today_signal = today_signal.filter(pl.col("vt_symbol").is_in(list(bars.keys())))

        pos_symbols: list[str] = [vt_symbol for vt_symbol, pos in self.pos_data.items() if pos]

        for vt_symbol in pos_symbols:
            self.holding_days[vt_symbol] += 1

        sell_symbols: set[str] = self.get_sell_symbols(pos_symbols, bars)

        cash: float = self.get_cash_available()
        executed_sell_symbols: set[str] = set()
        for vt_symbol in sell_symbols:
            if self.holding_days[vt_symbol] < self.min_days:
                continue

            bar: BarData | None = bars.get(vt_symbol)
            if not bar:
                continue

            sell_volume: float = self.get_pos(vt_symbol)
            self.set_target(vt_symbol, target=0)
            executed_sell_symbols.add(vt_symbol)

            turnover: float = bar.close_price * sell_volume
            cost: float = max(turnover * self.close_rate, self.min_commission)
            cash += turnover - cost

        buy_symbols: list[str] = self.get_buy_symbols(today_signal, pos_symbols, executed_sell_symbols)
        if buy_symbols and cash > 0:
            buy_value: float = cash * self.cash_ratio / len(buy_symbols)
            signal_rows: dict[str, dict[str, Any]] = {
                cast(str, row["vt_symbol"]): row
                for row in today_signal.iter_rows(named=True)
            }

            for vt_symbol in buy_symbols:
                bar = bars.get(vt_symbol)
                if not bar or not bar.close_price:
                    continue

                buy_volume: float = floor_to(buy_value / bar.close_price, self.min_volume)
                if buy_volume <= 0:
                    continue

                self.set_target(vt_symbol, buy_volume)
                self.update_signal_metadata(vt_symbol, signal_rows[vt_symbol])

        self.execute_trading(bars, price_add=self.price_add)

    def get_sell_symbols(self, pos_symbols: list[str], bars: dict[str, BarData]) -> set[str]:
        """Get symbols to sell by holding days and optional stop price."""
        sell_symbols: set[str] = set()

        for vt_symbol in pos_symbols:
            holding_days: int = self.holding_days[vt_symbol]
            max_days: int = self.max_holding_days.get(vt_symbol, self.max_days)

            if holding_days >= max_days:
                sell_symbols.add(vt_symbol)
                continue

            stop_price: float | None = self.stop_prices.get(vt_symbol)
            bar: BarData | None = bars.get(vt_symbol)
            if stop_price and bar and bar.close_price <= stop_price:
                sell_symbols.add(vt_symbol)

        return sell_symbols

    def get_buy_symbols(
        self,
        today_signal: pl.DataFrame,
        pos_symbols: list[str],
        sell_symbols: set[str]
    ) -> list[str]:
        """Get new event symbols to buy."""
        pos_set: set[str] = set(pos_symbols)
        holding_after_sell: int = len(pos_set - sell_symbols)
        available_slots: int = max(self.top_k - holding_after_sell, 0)

        if not available_slots:
            return []

        buyable_df: pl.DataFrame = today_signal.filter(~pl.col("vt_symbol").is_in(pos_set))
        return list(buyable_df[:available_slots]["vt_symbol"])

    def update_signal_metadata(self, vt_symbol: str, row: dict[str, Any]) -> None:
        """Store optional per-position metadata from the entry signal row."""
        self.holding_days[vt_symbol] = 0

        max_hold_days: Any | None = row.get("max_hold_days")
        if max_hold_days is not None:
            self.max_holding_days[vt_symbol] = int(max_hold_days)
        else:
            self.max_holding_days[vt_symbol] = self.max_days

        stop_price: Any | None = row.get("stop_price")
        if stop_price is not None:
            self.stop_prices[vt_symbol] = float(stop_price)
