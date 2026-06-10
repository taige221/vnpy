from __future__ import annotations

from collections import defaultdict
from typing import Any, cast

import polars as pl

from vnpy.trader.constant import Direction
from vnpy.trader.object import BarData, TradeData
from vnpy.trader.utility import floor_to

from vnpy.alpha import AlphaStrategy


class ComboEventStrategy(AlphaStrategy):
    """Long-only strategy combining ranking scores with optional event signals."""

    top_k: int = 50
    keep_top_k: int = 100
    entry_mode: str = "hard"
    entry_soft_weight: float = 0.2
    default_max_hold_days: int = 15
    min_days: int = 1
    cash_ratio: float = 0.95
    min_volume: int = 100
    open_rate: float = 0.0005
    close_rate: float = 0.0015
    min_commission: int = 5
    price_add: float = 0.05

    def on_init(self) -> None:
        """Strategy initialization callback."""
        self.holding_days: defaultdict[str, int] = defaultdict(int)
        self.max_holding_days: dict[str, int] = {}
        self.stop_prices: dict[str, float] = {}

        self.write_log("Combo event strategy initialized")

    def on_trade(self, trade: TradeData) -> None:
        """Trade execution callback."""
        if trade.direction == Direction.SHORT:
            if self.pos_data[trade.vt_symbol] <= 0:
                self.holding_days.pop(trade.vt_symbol, None)
                self.max_holding_days.pop(trade.vt_symbol, None)
                self.stop_prices.pop(trade.vt_symbol, None)
        else:
            self.holding_days.setdefault(trade.vt_symbol, 0)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K-line slice callback."""
        today_signal: pl.DataFrame = self.prepare_signal(self.get_signal(), bars)
        pos_symbols: list[str] = [vt_symbol for vt_symbol, pos in self.pos_data.items() if pos]

        for vt_symbol in pos_symbols:
            self.holding_days[vt_symbol] += 1

        sell_symbols: set[str] = self.get_sell_symbols(today_signal, pos_symbols, bars)
        for vt_symbol in sell_symbols:
            self.set_target(vt_symbol, 0)

        if today_signal.is_empty():
            self.execute_trading(bars, price_add=self.price_add)
            return

        target_symbols: list[str] = self.select_target_symbols(today_signal, pos_symbols, sell_symbols)
        self.apply_target_positions(today_signal, target_symbols, bars)
        self.execute_trading(bars, price_add=self.price_add)

    def prepare_signal(self, signal: pl.DataFrame, bars: dict[str, BarData]) -> pl.DataFrame:
        """Normalize current signal rows for generic combo-event logic."""
        if signal.is_empty():
            return pl.DataFrame(
                {
                    "vt_symbol": [],
                    "signal": [],
                    "entry_signal": [],
                    "exit_signal": [],
                    "position_scale": [],
                    "portfolio_scale": [],
                    "max_hold_days": [],
                    "stop_price": [],
                    "adjusted_signal": [],
                }
            )

        required: set[str] = {"vt_symbol", "signal"}
        missing: set[str] = required - set(signal.columns)
        if missing:
            raise ValueError(f"ComboEventStrategy signal_df missing required columns: {sorted(missing)}")

        frame: pl.DataFrame = signal.filter(pl.col("vt_symbol").is_in(list(bars.keys()))).with_columns(
            pl.col("signal").cast(pl.Float64)
        )

        defaults: dict[str, Any] = {
            "entry_signal": 1.0,
            "exit_signal": 0.0,
            "position_scale": 1.0,
            "portfolio_scale": 1.0,
            "max_hold_days": self.default_max_hold_days,
            "stop_price": None,
        }
        for column, default in defaults.items():
            if column not in frame.columns:
                frame = frame.with_columns(pl.lit(default).alias(column))

        frame = frame.with_columns(
            pl.col("entry_signal").cast(pl.Float64).fill_null(1.0),
            pl.col("exit_signal").cast(pl.Float64).fill_null(0.0),
            pl.col("position_scale").cast(pl.Float64).fill_null(1.0),
            pl.col("portfolio_scale").cast(pl.Float64).fill_null(1.0),
            pl.col("max_hold_days").cast(pl.Int64).fill_null(self.default_max_hold_days),
            pl.col("stop_price").cast(pl.Float64),
        )
        return self.add_adjusted_signal(frame)

    def add_adjusted_signal(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add hard/soft mode adjusted ranking score."""
        if self.entry_mode == "hard":
            return frame.with_columns(pl.col("signal").alias("adjusted_signal"))
        if self.entry_mode != "soft":
            raise ValueError("entry_mode must be 'hard' or 'soft'")

        entry_min_raw: Any = frame["entry_signal"].min()
        entry_max_raw: Any = frame["entry_signal"].max()
        entry_min: float | None = float(entry_min_raw) if entry_min_raw is not None else None
        entry_max: float | None = float(entry_max_raw) if entry_max_raw is not None else None
        if entry_min is None or entry_max is None or entry_min == entry_max:
            normalized = pl.lit(0.0)
        else:
            normalized = (pl.col("entry_signal") - entry_min) / (entry_max - entry_min)

        return frame.with_columns(
            (pl.col("signal") + self.entry_soft_weight * normalized).alias("adjusted_signal")
        )

    def get_sell_symbols(
        self,
        today_signal: pl.DataFrame,
        pos_symbols: list[str],
        bars: dict[str, BarData],
    ) -> set[str]:
        """Get symbols to sell by event exits, stops, holding days, and rank retention."""
        sell_symbols: set[str] = set()
        signal_by_symbol: dict[str, dict[str, Any]] = {
            cast(str, row["vt_symbol"]): row
            for row in today_signal.iter_rows(named=True)
        }
        rank_by_symbol: dict[str, int] = self.current_rank_by_symbol(today_signal)

        for vt_symbol in pos_symbols:
            row: dict[str, Any] | None = signal_by_symbol.get(vt_symbol)
            if row and float(row.get("exit_signal", 0.0) or 0.0) > 0:
                sell_symbols.add(vt_symbol)
                continue

            stop_price: float | None = self.stop_prices.get(vt_symbol)
            bar: BarData | None = bars.get(vt_symbol)
            if stop_price is not None and bar and bar.close_price <= stop_price:
                sell_symbols.add(vt_symbol)
                continue

            holding_days: int = self.holding_days[vt_symbol]
            if holding_days < self.min_days:
                continue

            max_days: int = self.max_holding_days.get(vt_symbol, self.default_max_hold_days)
            if holding_days >= max_days:
                sell_symbols.add(vt_symbol)
                continue

            if not today_signal.is_empty():
                rank: int | None = rank_by_symbol.get(vt_symbol)
                if rank is None or rank > self.keep_top_k:
                    sell_symbols.add(vt_symbol)

        return sell_symbols

    def current_rank_by_symbol(self, today_signal: pl.DataFrame) -> dict[str, int]:
        """Return adjusted rank by symbol for current signal rows."""
        if today_signal.is_empty():
            return {}

        ranked: pl.DataFrame = today_signal.sort("adjusted_signal", descending=True).with_row_index(
            "rank",
            offset=1,
        )
        return {
            cast(str, row["vt_symbol"]): int(row["rank"])
            for row in ranked.select("vt_symbol", "rank").iter_rows(named=True)
        }

    def select_target_symbols(
        self,
        today_signal: pl.DataFrame,
        pos_symbols: list[str],
        sell_symbols: set[str],
    ) -> list[str]:
        """Select target holdings from retained positions and new entry candidates."""
        retained: set[str] = set(pos_symbols) - sell_symbols
        if today_signal.is_empty():
            return sorted(retained)

        available_signal: pl.DataFrame = today_signal.filter(~pl.col("vt_symbol").is_in(sell_symbols))
        if available_signal.is_empty():
            return []

        if self.entry_mode == "hard":
            candidates: pl.DataFrame = available_signal.filter(
                (pl.col("entry_signal") > 0) | pl.col("vt_symbol").is_in(retained)
            )
        else:
            candidates = available_signal

        if candidates.is_empty():
            return []

        sorted_symbols: list[str] = [
            cast(str, symbol)
            for symbol in candidates.sort("adjusted_signal", descending=True)
            .get_column("vt_symbol")
            .to_list()
        ]
        retained_symbols: list[str] = [symbol for symbol in sorted_symbols if symbol in retained]
        new_slot_count: int = max(self.top_k - len(retained_symbols), 0)
        new_symbols: list[str] = [
            symbol
            for symbol in sorted_symbols
            if symbol not in retained
        ][:new_slot_count]
        return retained_symbols + new_symbols

    def apply_target_positions(
        self,
        today_signal: pl.DataFrame,
        target_symbols: list[str],
        bars: dict[str, BarData],
    ) -> None:
        """Convert target symbols and scales into target volumes."""
        target_set: set[str] = set(target_symbols)
        for vt_symbol in list(self.target_data):
            if vt_symbol not in target_set:
                self.set_target(vt_symbol, 0)

        if not target_symbols:
            return

        signal_by_symbol: dict[str, dict[str, Any]] = {
            cast(str, row["vt_symbol"]): row
            for row in today_signal.iter_rows(named=True)
        }
        scale_by_symbol: dict[str, float] = {}
        for vt_symbol in target_symbols:
            row: dict[str, Any] = signal_by_symbol[vt_symbol]
            position_scale: float = self.optional_float(row.get("position_scale"), default=1.0)
            scale_by_symbol[vt_symbol] = max(position_scale, 0.0)

        scale_sum: float = sum(scale_by_symbol.values())
        if scale_sum <= 0:
            return

        portfolio_scale: float = max(
            self.optional_float(signal_by_symbol[vt_symbol].get("portfolio_scale"), default=1.0)
            for vt_symbol in target_symbols
        )
        target_value: float = self.get_portfolio_value() * self.cash_ratio * max(portfolio_scale, 0.0)

        for vt_symbol in target_symbols:
            bar: BarData | None = bars.get(vt_symbol)
            if not bar or not bar.close_price:
                continue

            symbol_value: float = target_value * scale_by_symbol[vt_symbol] / scale_sum
            target_volume: float = floor_to(symbol_value / bar.close_price, self.min_volume)
            self.set_target(vt_symbol, target_volume)
            self.update_signal_metadata(vt_symbol, signal_by_symbol[vt_symbol])

    def update_signal_metadata(self, vt_symbol: str, row: dict[str, Any]) -> None:
        """Store optional per-position metadata from current signal rows."""
        if vt_symbol not in self.max_holding_days or row.get("max_hold_days") is not None:
            self.max_holding_days[vt_symbol] = int(row.get("max_hold_days") or self.default_max_hold_days)

        stop_price: Any | None = row.get("stop_price")
        if stop_price is not None:
            self.stop_prices[vt_symbol] = float(stop_price)

    def optional_float(self, value: Any, default: float) -> float:
        """Return a float while preserving explicit zero values."""
        if value is None:
            return default
        parsed = float(value)
        if parsed != parsed:
            return default
        return parsed
