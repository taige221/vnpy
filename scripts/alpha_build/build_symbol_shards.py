"""Build a reusable symbol-to-shard manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.alpha_build.common.symbols import load_daily_symbols, select_symbols


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build a symbol shard manifest")
    parser.add_argument("--lab-path", default="lab/a_share_research")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--symbol-list")
    return parser.parse_args()


def main() -> None:
    """Write the shard manifest."""
    args = parse_args()
    symbols = select_symbols(
        load_daily_symbols(args.lab_path),
        symbol_list=args.symbol_list,
    )
    if not symbols:
        raise RuntimeError("no symbols selected")

    rows = [
        {
            "vt_symbol": symbol,
            "ordinal": ordinal,
            "shard_index": ordinal % args.shard_count,
            "shard_count": args.shard_count,
        }
        for ordinal, symbol in enumerate(symbols)
    ]
    out = pd.DataFrame(rows)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"symbols={len(out)} shard_count={args.shard_count}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
