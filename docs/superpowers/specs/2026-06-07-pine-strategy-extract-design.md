# Pine Strategy Extract Design

## Context

The repository already has `vnpy.alpha` utilities that load bar panels, calculate named factor expressions, and export research artifacts. There is also an untracked TradingView migration workspace under `examples/alpha_research/tradingview/` with raw Pine files, hand-written Vegas replication code, and reports.

This feature adds a first-pass Pine Script extraction tool. It is not a full Pine compiler. It should convert the supported subset into `vnpy.alpha` expressions and produce a complete human review report for everything it sees, including unsupported logic that needs follow-up interpretation.

## Goals

- Parse copied TradingView Pine Script files from local disk.
- Extract strategy or indicator metadata, user inputs, assignments, alerts, and trading calls.
- Translate supported expressions into the existing `vnpy.alpha.dataset.utility.calculate_by_expression()` expression syntax.
- Produce a review report that makes unsupported syntax explicit, so untranslated factors can be handled manually in later iterations.
- Provide a CLI that can be used against the files in `examples/alpha_research/tradingview/pine/`.

## Non-Goals

- Do not execute Pine Script or reproduce TradingView broker semantics.
- Do not scrape TradingView or require browser automation.
- Do not support every Pine language feature in the first version.
- Do not silently approximate unsupported stateful logic such as `var`, `:=`, `switch`, custom functions, arrays, multi-timeframe `request.security()`, or complex `strategy.exit()` risk rules.

## Architecture

Add `vnpy/alpha/utils/pine_strategy_extract.py` as the reusable module and CLI entrypoint.

The module will use a lightweight line-oriented parser plus expression translation. This is deliberate: the first target is a useful migration aid for local Pine files, not a complete grammar implementation. The parser keeps original source snippets and line numbers so the report can be audited.

Core dataclasses:

- `PineInputSpec`: input variable name, Pine input type, default value, title, options, group, and source line.
- `PineAssignmentSpec`: assignment target, source expression, translated expression when available, status, and source line.
- `PineActionSpec`: extracted `strategy.entry`, `strategy.close`, `strategy.exit`, and `alertcondition` calls with their surrounding `if` condition when visible.
- `PineUnsupportedSpec`: unsupported source line, reason, and source text.
- `PineExtractResult`: metadata, inputs, assignments, actions, alerts, unsupported items, and generated alpha expressions.

## Translation Rules

The first version supports the common technical subset:

- Pine identifiers and OHLCV series: `open`, `high`, `low`, `close`, `volume`, `time`.
- Literals: numbers, strings, booleans, `na`.
- Operators: arithmetic, comparisons, parentheses, `and`, `or`, `not`, ternary expressions when both branches translate.
- History references: `x[1]` to `ts_delay(x, 1)`.
- Functions:
  - `ta.sma(x, n)` to `ts_mean(x, n)`
  - `ta.ema(x, n)` to `ta_ema(x, n)`
  - `ta.rsi(x, n)` to `ta_rsi(x, n) / 100.0`
  - `ta.atr(n)` to `ta_atr(high, low, close, n)`
  - `ta.highest(x, n)` to `ts_max(x, n)`
  - `ta.lowest(x, n)` to `ts_min(x, n)`
  - `ta.crossover(a, b)` to `(a > b) * (ts_delay(a, 1) <= ts_delay(b, 1))`
  - `ta.crossunder(a, b)` to `(a < b) * (ts_delay(a, 1) >= ts_delay(b, 1))`
  - `math.max(a, b)` to `greater(a, b)`
  - `math.min(a, b)` to `less(a, b)`
  - `math.abs(x)` to `abs(x)`

Inputs are resolved to their default values before translation when the default is scalar or a known series source. Unsupported input defaults remain symbolic and are reported.

## Output

The CLI writes two files by default:

- `*_pine_extract_report.json`: structured report for automated inspection.
- `*_pine_extract_report.md`: concise human report with translated and untranslated sections.

An optional `--alpha-output` path writes `*_alpha_exprs.json` containing only successfully translated expression records:

```json
[
  {
    "name": "longSignal",
    "expression": "translated vnpy.alpha expression",
    "source": "longSignal = ...",
    "line": 120,
    "kind": "assignment"
  }
]
```

## Error Handling

Parsing should be best-effort. A malformed or unsupported line does not abort the whole extraction unless the input file cannot be read. Unsupported lines are accumulated with reasons. The CLI exits successfully when it produces a report, even if some expressions are unsupported.

Translation errors are localized to the affected assignment or action. The report must clearly separate:

- translated expressions,
- parsed but untranslated expressions,
- ignored display-only statements such as `plot`, `fill`, `plotshape`, and comments,
- unsupported semantics that need manual interpretation.

## Tests

Add focused tests under `tests/alpha/`:

- extract simple inputs and metadata from a small Pine strategy;
- translate EMA, RSI, ATR, highest, lowest, crossover, crossunder, history references, and boolean expressions;
- capture `strategy.entry`, `strategy.close`, and `alertcondition` calls with nearby conditions;
- report unsupported stateful features such as `var`, `:=`, `switch`, `ta.pivotlow`, and `ta.valuewhen`;
- run a smoke extraction against a Vegas-style snippet derived from the local TradingView examples.

## Future Follow-Up

After this first version, untranslated factors from the report can be interpreted manually and promoted into either new translation rules or explicit `vnpy.alpha` factor families. Candidate next additions are `ta.macd`, tuple assignment, `ta.dmi`, `na()` handling, `bar_index`, and limited state-machine extraction for multi-stage entry logic.
