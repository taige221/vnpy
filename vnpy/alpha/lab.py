import json
import shelve
import pickle
from pathlib import Path
from datetime import date, datetime, timedelta
from collections import defaultdict
from functools import lru_cache

import polars as pl

from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval
from vnpy.trader.utility import extract_vt_symbol

from .logger import logger
from .dataset import AlphaDataset, to_datetime
from .model import AlphaModel


class AlphaLab:
    """Alpha Research Laboratory"""

    def __init__(self, lab_path: str) -> None:
        """Constructor"""
        # Set data paths
        self.lab_path: Path = Path(lab_path)

        self.daily_path: Path = self.lab_path.joinpath("daily")
        self.minute_path: Path = self.lab_path.joinpath("minute")
        self.component_path: Path = self.lab_path.joinpath("component")

        self.dataset_path: Path = self.lab_path.joinpath("dataset")
        self.model_path: Path = self.lab_path.joinpath("model")
        self.signal_path: Path = self.lab_path.joinpath("signal")

        self.contract_path: Path = self.lab_path.joinpath("contract.json")

        # Create folders
        for path in [
            self.lab_path,
            self.daily_path,
            self.minute_path,
            self.component_path,
            self.dataset_path,
            self.model_path,
            self.signal_path
        ]:
            if not path.exists():
                path.mkdir(parents=True)

    def save_bar_data(self, bars: list[BarData]) -> None:
        """Save bar data"""
        if not bars:
            return

        # Get file path
        bar: BarData = bars[0]

        if bar.interval == Interval.DAILY:
            file_path: Path = self.daily_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval == Interval.MINUTE:
            file_path = self.minute_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval:
            logger.error(f"Unsupported interval {bar.interval.value}")
            return

        data: list = []
        for bar in bars:
            bar_data: dict = {
                "datetime": bar.datetime.replace(tzinfo=None),
                "open": bar.open_price,
                "high": bar.high_price,
                "low": bar.low_price,
                "close": bar.close_price,
                "volume": bar.volume,
                "turnover": bar.turnover,
                "open_interest": bar.open_interest
            }
            data.append(bar_data)

        new_df: pl.DataFrame = pl.DataFrame(data)

        # If file exists, read and merge
        if file_path.exists():
            old_df: pl.DataFrame = pl.read_parquet(file_path)

            new_df = pl.concat([old_df, new_df])

            new_df = new_df.unique(subset=["datetime"])

            new_df = new_df.sort("datetime")

        # Save to file
        new_df.write_parquet(file_path)

    def load_bar_data(
        self,
        vt_symbol: str,
        interval: Interval | str,
        start: datetime | str,
        end: datetime | str
    ) -> list[BarData]:
        """Load bar data"""
        # Convert types
        if isinstance(interval, str):
            interval = Interval(interval)

        start = to_datetime(start)
        end = to_datetime(end)

        # Get folder path
        if interval == Interval.DAILY:
            folder_path: Path = self.daily_path
        elif interval == Interval.MINUTE:
            folder_path = self.minute_path
        else:
            logger.error(f"Unsupported interval {interval.value}")
            return []

        # Check if file exists
        file_path: Path = folder_path.joinpath(f"{vt_symbol}.parquet")
        if not file_path.exists():
            logger.error(f"File {file_path} does not exist")
            return []

        # Open file
        df: pl.DataFrame = pl.read_parquet(file_path)

        # Filter by date range
        df = df.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))

        # Convert to BarData objects
        bars: list[BarData] = []

        symbol, exchange = extract_vt_symbol(vt_symbol)

        for row in df.iter_rows(named=True):
            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=row["datetime"],
                interval=interval,
                open_price=row["open"],
                high_price=row["high"],
                low_price=row["low"],
                close_price=row["close"],
                volume=row["volume"],
                turnover=row["turnover"],
                open_interest=row["open_interest"],
                gateway_name="DB"
            )
            bars.append(bar)

        return bars

    def load_bar_df(
        self,
        vt_symbols: list[str],
        interval: Interval | str,
        start: datetime | str,
        end: datetime | str,
        extended_days: int
    ) -> pl.DataFrame | None:
        """Load bar data as DataFrame"""
        if not vt_symbols:
            return None

        # Convert types
        if isinstance(interval, str):
            interval = Interval(interval)

        start = to_datetime(start) - timedelta(days=extended_days)
        end = to_datetime(end) + timedelta(days=extended_days // 10)

        # Get folder path
        if interval == Interval.DAILY:
            folder_path: Path = self.daily_path
        elif interval == Interval.MINUTE:
            folder_path = self.minute_path
        else:
            logger.error(f"Unsupported interval {interval.value}")
            return None

        # Read data for each symbol
        dfs: list = []

        for vt_symbol in vt_symbols:
            # Check if file exists
            file_path: Path = folder_path.joinpath(f"{vt_symbol}.parquet")
            if not file_path.exists():
                logger.error(f"File {file_path} does not exist")
                continue

            # Open file
            df: pl.DataFrame = pl.read_parquet(file_path)

            # Filter by date range
            df = df.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))

            # Specify data types
            df = df.with_columns(
                pl.col("open"),
                pl.col("high"),
                pl.col("low"),
                pl.col("close"),
                pl.col("volume"),
                pl.col("turnover"),
                pl.col("open_interest"),
                (pl.col("turnover") / pl.col("volume")).alias("vwap")
            )

            # Check for empty data
            if df.is_empty():
                continue

            # Normalize prices
            close_0: float = df.select(pl.col("close")).item(0, 0)

            df = df.with_columns(
                (pl.col("open") / close_0).alias("open"),
                (pl.col("high") / close_0).alias("high"),
                (pl.col("low") / close_0).alias("low"),
                (pl.col("close") / close_0).alias("close"),
            )

            # Convert zeros to NaN for suspended trading days
            numeric_columns: list = df.columns[1:]                              # Extract numeric columns

            mask: pl.Series = df[numeric_columns].sum_horizontal() == 0         # Sum by row, if 0 then suspended

            df = df.with_columns(                                               # Convert suspended day values to NaN
                [pl.when(mask).then(float("nan")).otherwise(pl.col(col)).alias(col) for col in numeric_columns]
            )

            # Add symbol column
            df = df.with_columns(pl.lit(vt_symbol).alias("vt_symbol"))

            # Cache in list
            dfs.append(df)

        # Concatenate results
        result_df: pl.DataFrame = pl.concat(dfs)
        return result_df

    def save_component_data(
        self,
        index_symbol: str,
        index_components: dict[str, list[str]]
    ) -> None:
        """Save index component data"""
        file_path: Path = self.component_path.joinpath(f"{index_symbol}.parquet")

        if not index_components:
            return

        rows: list[dict[str, date | str]] = []
        replace_dates: list[date] = []
        for trade_date, vt_symbols in index_components.items():
            dt: date = to_datetime(trade_date).date()
            replace_dates.append(dt)
            for vt_symbol in vt_symbols:
                rows.append(
                    {
                        "trade_date": dt,
                        "vt_symbol": vt_symbol,
                    }
                )

        df: pl.DataFrame = pl.DataFrame(
            rows,
            schema={
                "trade_date": pl.Date,
                "vt_symbol": pl.String,
            },
        )
        if file_path.exists():
            old_df: pl.DataFrame = pl.read_parquet(file_path)
            old_df = old_df.filter(~pl.col("trade_date").cast(pl.Date).is_in(replace_dates))
            df = pl.concat([old_df, df])

        df = df.sort(["trade_date", "vt_symbol"])
        df.write_parquet(file_path)
        self.load_component_data.cache_clear()

    @lru_cache      # noqa
    def load_component_data(
        self,
        index_symbol: str,
        start: datetime | str,
        end: datetime | str
    ) -> dict[datetime, list[str]]:
        """Load index component data as DataFrame"""
        start = to_datetime(start)
        end = to_datetime(end)

        parquet_path: Path = self.component_path.joinpath(f"{index_symbol}.parquet")
        if parquet_path.exists():
            return self._load_component_parquet_data(parquet_path, start, end)

        file_path: Path = self.component_path.joinpath(f"{index_symbol}")
        if not self._component_shelve_exists(file_path):
            logger.error(f"Component file {index_symbol} does not exist")
            return {}

        with shelve.open(str(file_path)) as db:
            try:
                keys: list[str] = list(db.keys())
                keys.sort()
            except SystemError:
                keys = [
                    dt.strftime("%Y-%m-%d")
                    for dt in self._load_component_trading_dates(start, end)
                ]

            index_components: dict[datetime, list[str]] = {}
            for key in keys:
                dt: datetime = datetime.strptime(key, "%Y-%m-%d")
                if start <= dt <= end:
                    index_components[dt] = db[key]

            return index_components

    def load_component_symbols(
        self,
        index_symbol: str,
        start: datetime | str,
        end: datetime | str
    ) -> list[str]:
        """Collect index component symbols"""
        start = to_datetime(start)
        end = to_datetime(end)

        parquet_path: Path = self.component_path.joinpath(f"{index_symbol}.parquet")
        if parquet_path.exists():
            return self._load_component_parquet_symbols(parquet_path, start, end)

        index_components: dict[datetime, list[str]] = self.load_component_data(
            index_symbol,
            start,
            end
        )

        component_symbols: set[str] = set()

        for vt_symbols in index_components.values():
            component_symbols.update(vt_symbols)

        return list(component_symbols)

    def load_component_filters(
        self,
        index_symbol: str,
        start: datetime | str,
        end: datetime | str
    ) -> dict[str, list[tuple[datetime, datetime]]]:
        """Collect index component duration filters"""
        start = to_datetime(start)
        end = to_datetime(end)

        parquet_path: Path = self.component_path.joinpath(f"{index_symbol}.parquet")
        if parquet_path.exists():
            intervals = self._load_component_parquet_intervals(parquet_path, start, end)
            interval_filters: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
            for vt_symbol, valid_from, valid_to in intervals:
                interval_filters[vt_symbol].append(
                    (
                        max(valid_from, start),
                        min(valid_to, end),
                    )
                )
            return interval_filters

        index_components: dict[datetime, list[str]] = self.load_component_data(
            index_symbol,
            start,
            end
        )

        # Get all trading dates and sort
        trading_dates: list[datetime] = sorted(index_components.keys())

        # Initialize component duration dictionary
        component_filters: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)

        # Get all component symbols
        all_symbols: set[str] = set()
        for vt_symbols in index_components.values():
            all_symbols.update(vt_symbols)

        # Iterate through each component to identify its duration in the index
        for vt_symbol in all_symbols:
            period_start: datetime | None = None
            period_end: datetime | None = None

            # Iterate through each trading day to identify continuous holding periods
            for trading_date in trading_dates:
                if vt_symbol in index_components[trading_date]:
                    if period_start is None:
                        period_start = trading_date

                    period_end = trading_date
                else:
                    if period_start and period_end:
                        component_filters[vt_symbol].append((period_start, period_end))
                        period_start = None
                        period_end = None

            # Handle the last holding period
            if period_start and period_end:
                component_filters[vt_symbol].append((period_start, period_end))

        return component_filters

    def _component_shelve_exists(self, file_path: Path) -> bool:
        """Check whether an old shelve component file exists."""
        suffixes: tuple[str, ...] = ("", ".db", ".dat", ".dir", ".bak")
        return any(Path(str(file_path) + suffix).exists() for suffix in suffixes)

    def _load_component_parquet_data(
        self,
        file_path: Path,
        start: datetime,
        end: datetime
    ) -> dict[datetime, list[str]]:
        """Load component data from Parquet point or interval format."""
        df: pl.DataFrame = pl.read_parquet(file_path)
        columns: set[str] = set(df.columns)

        if {"valid_from", "valid_to", "vt_symbol"}.issubset(columns):
            return self._load_component_interval_data(df, start, end)

        if {"trade_date", "vt_symbol"}.issubset(columns):
            return self._load_component_point_data(df, start, end)

        logger.error(f"Unsupported component parquet schema {file_path}")
        return {}

    def _load_component_point_data(
        self,
        df: pl.DataFrame,
        start: datetime,
        end: datetime
    ) -> dict[datetime, list[str]]:
        """Load date-symbol component points from Parquet."""
        start_date: date = start.date()
        end_date: date = end.date()

        df = (
            df.with_columns(pl.col("trade_date").cast(pl.Date))
            .filter((pl.col("trade_date") >= start_date) & (pl.col("trade_date") <= end_date))
            .group_by("trade_date")
            .agg(pl.col("vt_symbol").sort())
            .sort("trade_date")
        )

        index_components: dict[datetime, list[str]] = {}
        for row in df.iter_rows(named=True):
            dt: datetime = datetime.combine(row["trade_date"], datetime.min.time())
            index_components[dt] = row["vt_symbol"]

        return index_components

    def _load_component_interval_data(
        self,
        df: pl.DataFrame,
        start: datetime,
        end: datetime
    ) -> dict[datetime, list[str]]:
        """Expand interval component rows into daily component data."""
        intervals: list[tuple[str, datetime, datetime]] = self._component_intervals_from_df(df, start, end)
        trading_dates: list[datetime] = self._load_component_trading_dates(start, end)

        if not trading_dates:
            days: int = (end.date() - start.date()).days
            trading_dates = [
                datetime.combine(start.date() + timedelta(days=i), datetime.min.time())
                for i in range(days + 1)
            ]

        index_components: dict[datetime, list[str]] = {dt: [] for dt in trading_dates}

        for vt_symbol, valid_from, valid_to in intervals:
            for trading_date in trading_dates:
                if valid_from <= trading_date <= valid_to:
                    index_components[trading_date].append(vt_symbol)

        return {
            trading_date: sorted(vt_symbols)
            for trading_date, vt_symbols in index_components.items()
            if vt_symbols
        }

    def _load_component_parquet_symbols(
        self,
        file_path: Path,
        start: datetime,
        end: datetime
    ) -> list[str]:
        """Collect component symbols directly from Parquet."""
        df: pl.DataFrame = pl.read_parquet(file_path)
        columns: set[str] = set(df.columns)
        start_date: date = start.date()
        end_date: date = end.date()

        if {"valid_from", "valid_to", "vt_symbol"}.issubset(columns):
            df = (
                df.with_columns(
                    pl.col("valid_from").cast(pl.Date),
                    pl.col("valid_to").cast(pl.Date),
                )
                .filter((pl.col("valid_to") >= start_date) & (pl.col("valid_from") <= end_date))
            )
        elif {"trade_date", "vt_symbol"}.issubset(columns):
            df = (
                df.with_columns(pl.col("trade_date").cast(pl.Date))
                .filter((pl.col("trade_date") >= start_date) & (pl.col("trade_date") <= end_date))
            )
        else:
            logger.error(f"Unsupported component parquet schema {file_path}")
            return []

        return sorted(df.get_column("vt_symbol").unique().to_list())

    def _load_component_parquet_intervals(
        self,
        file_path: Path,
        start: datetime,
        end: datetime
    ) -> list[tuple[str, datetime, datetime]]:
        """Load component duration intervals from Parquet."""
        df: pl.DataFrame = pl.read_parquet(file_path)
        columns: set[str] = set(df.columns)

        if {"valid_from", "valid_to", "vt_symbol"}.issubset(columns):
            return self._component_intervals_from_df(df, start, end)

        if {"trade_date", "vt_symbol"}.issubset(columns):
            index_components: dict[datetime, list[str]] = self._load_component_point_data(df, start, end)
            return self._component_intervals_from_points(index_components)

        logger.error(f"Unsupported component parquet schema {file_path}")
        return []

    def _component_intervals_from_df(
        self,
        df: pl.DataFrame,
        start: datetime,
        end: datetime
    ) -> list[tuple[str, datetime, datetime]]:
        """Convert interval DataFrame rows to clipped datetime intervals."""
        start_date: date = start.date()
        end_date: date = end.date()

        df = (
            df.with_columns(
                pl.col("valid_from").cast(pl.Date),
                pl.col("valid_to").cast(pl.Date),
            )
            .filter((pl.col("valid_to") >= start_date) & (pl.col("valid_from") <= end_date))
            .sort(["vt_symbol", "valid_from"])
        )

        intervals: list[tuple[str, datetime, datetime]] = []
        for row in df.iter_rows(named=True):
            valid_from: datetime = datetime.combine(row["valid_from"], datetime.min.time())
            valid_to: datetime = datetime.combine(row["valid_to"], datetime.min.time())
            intervals.append((row["vt_symbol"], valid_from, valid_to))

        return intervals

    def _component_intervals_from_points(
        self,
        index_components: dict[datetime, list[str]]
    ) -> list[tuple[str, datetime, datetime]]:
        """Convert daily component points to duration intervals."""
        trading_dates: list[datetime] = sorted(index_components.keys())
        all_symbols: set[str] = set()
        for vt_symbols in index_components.values():
            all_symbols.update(vt_symbols)

        intervals: list[tuple[str, datetime, datetime]] = []
        for vt_symbol in all_symbols:
            period_start: datetime | None = None
            period_end: datetime | None = None

            for trading_date in trading_dates:
                if vt_symbol in index_components[trading_date]:
                    if period_start is None:
                        period_start = trading_date
                    period_end = trading_date
                elif period_start and period_end:
                    intervals.append((vt_symbol, period_start, period_end))
                    period_start = None
                    period_end = None

            if period_start and period_end:
                intervals.append((vt_symbol, period_start, period_end))

        intervals.sort(key=lambda item: (item[0], item[1]))
        return intervals

    def _load_component_trading_dates(
        self,
        start: datetime,
        end: datetime
    ) -> list[datetime]:
        """Load component trading dates if available."""
        file_path: Path = self.component_path.joinpath("trading_dates.parquet")
        if not file_path.exists():
            return []

        start_date: date = start.date()
        end_date: date = end.date()
        df: pl.DataFrame = (
            pl.read_parquet(file_path)
            .with_columns(pl.col("trade_date").cast(pl.Date))
            .filter((pl.col("trade_date") >= start_date) & (pl.col("trade_date") <= end_date))
            .sort("trade_date")
        )

        return [
            datetime.combine(row["trade_date"], datetime.min.time())
            for row in df.iter_rows(named=True)
        ]

    def add_contract_setting(
        self,
        vt_symbol: str,
        long_rate: float,
        short_rate: float,
        size: float,
        pricetick: float
    ) -> None:
        """Add contract information"""
        contracts: dict = {}

        if self.contract_path.exists():
            with open(self.contract_path, encoding="UTF-8") as f:
                contracts = json.load(f)

        contracts[vt_symbol] = {
            "long_rate": long_rate,
            "short_rate": short_rate,
            "size": size,
            "pricetick": pricetick
        }

        with open(self.contract_path, mode="w+", encoding="UTF-8") as f:
            json.dump(
                contracts,
                f,
                indent=4,
                ensure_ascii=False
            )

    def load_contract_setttings(self) -> dict:
        """Load contract settings"""
        contracts: dict = {}

        if self.contract_path.exists():
            with open(self.contract_path, encoding="UTF-8") as f:
                contracts = json.load(f)

        return contracts

    def save_dataset(self, name: str, dataset: AlphaDataset) -> None:
        """Save dataset"""
        file_path: Path = self.dataset_path.joinpath(f"{name}.pkl")

        with open(file_path, mode="wb") as f:
            pickle.dump(dataset, f)

    def load_dataset(self, name: str) -> AlphaDataset | None:
        """Load dataset"""
        file_path: Path = self.dataset_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            logger.error(f"Dataset file {name} does not exist")
            return None

        with open(file_path, mode="rb") as f:
            dataset: AlphaDataset = pickle.load(f)
            return dataset

    def remove_dataset(self, name: str) -> bool:
        """Remove dataset"""
        file_path: Path = self.dataset_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            logger.error(f"Dataset file {name} does not exist")
            return False

        file_path.unlink()
        return True

    def list_all_datasets(self) -> list[str]:
        """List all datasets"""
        return [file.stem for file in self.dataset_path.glob("*.pkl")]

    def save_model(self, name: str, model: AlphaModel) -> None:
        """Save model"""
        file_path: Path = self.model_path.joinpath(f"{name}.pkl")

        with open(file_path, mode="wb") as f:
            pickle.dump(model, f)

    def load_model(self, name: str) -> AlphaModel | None:
        """Load model"""
        file_path: Path = self.model_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            logger.error(f"Model file {name} does not exist")
            return None

        with open(file_path, mode="rb") as f:
            model: AlphaModel = pickle.load(f)
            return model

    def remove_model(self, name: str) -> bool:
        """Remove model"""
        file_path: Path = self.model_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            logger.error(f"Model file {name} does not exist")
            return False

        file_path.unlink()
        return True

    def list_all_models(self) -> list[str]:
        """List all models"""
        return [file.stem for file in self.model_path.glob("*.pkl")]

    def save_signal(self, name: str, signal: pl.DataFrame) -> None:
        """Save signal"""
        file_path: Path = self.signal_path.joinpath(f"{name}.parquet")

        signal.write_parquet(file_path)

    def load_signal(self, name: str) -> pl.DataFrame | None:
        """Load signal"""
        file_path: Path = self.signal_path.joinpath(f"{name}.parquet")
        if not file_path.exists():
            logger.error(f"Signal file {name} does not exist")
            return None

        return pl.read_parquet(file_path)

    def remove_signal(self, name: str) -> bool:
        """Remove signal"""
        file_path: Path = self.signal_path.joinpath(f"{name}.parquet")
        if not file_path.exists():
            logger.error(f"Signal file {name} does not exist")
            return False

        file_path.unlink()
        return True

    def list_all_signals(self) -> list[str]:
        """List all signals"""
        return [file.stem for file in self.signal_path.glob("*.parquet")]
