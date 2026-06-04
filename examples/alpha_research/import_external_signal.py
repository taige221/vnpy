"""
Import external strategy events into an AlphaLab signal parquet.

Example:
    python import_external_signal.py \
        --lab-path ./lab/a_share_research \
        --name w_bottom_neckline075 \
        --input ./signals/w_bottom_neckline075.csv \
        --datetime-col date \
        --symbol-col symbol \
        --signal-col score \
        --extra-columns signal_type max_hold_days stop_price
"""

from pathlib import Path
import argparse

from vnpy.alpha import AlphaLab, import_external_signal


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab-path", required=True, help="AlphaLab folder path")
    parser.add_argument("--name", required=True, help="Saved signal name")
    parser.add_argument("--input", required=True, help="Input CSV or Parquet file")
    parser.add_argument("--datetime-col", help="Datetime/date column name")
    parser.add_argument("--symbol-col", help="Symbol column name")
    parser.add_argument("--signal-col", help="Signal score column name")
    parser.add_argument("--extra-columns", nargs="*", help="Extra columns to preserve")
    return parser.parse_args()


def main() -> None:
    """Import an external signal file into AlphaLab."""
    args = parse_args()

    lab = AlphaLab(args.lab_path)
    signal_df = import_external_signal(
        lab,
        args.name,
        Path(args.input),
        datetime_col=args.datetime_col,
        symbol_col=args.symbol_col,
        signal_col=args.signal_col,
        extra_columns=args.extra_columns,
    )

    print(f"Saved {signal_df.height} signal rows to {lab.signal_path.joinpath(args.name + '.parquet')}")


if __name__ == "__main__":
    main()
