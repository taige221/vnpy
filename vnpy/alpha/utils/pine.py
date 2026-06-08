from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class PineExportMode(str, Enum):
    """TradingView script kind to generate."""

    INDICATOR = "indicator"
    STRATEGY = "strategy"


class FactorSpecLike(Protocol):
    """Minimal factor-spec surface used by the Pine exporter."""

    @property
    def name(self) -> str:
        """Stable factor name."""

    @property
    def expression(self) -> str:
        """Alpha expression string."""


@dataclass(frozen=True)
class PineTranslationResult:
    """Result of translating one alpha expression to Pine."""

    expression: str | None
    reason: str | None


@dataclass(frozen=True)
class PineTranslatedFactor:
    """Translated factor ready to emit into a Pine script."""

    name: str
    expression: str
    source_name: str
    source_expression: str


@dataclass(frozen=True)
class PineExportUnsupported:
    """Expression that could not be exported safely."""

    name: str
    expression: str
    reason: str


@dataclass(frozen=True)
class PineExportSpec:
    """Complete Pine export artifact."""

    script: str
    translated: list[PineTranslatedFactor]
    unsupported: list[PineExportUnsupported]


class _UnsupportedAlphaExpression(Exception):
    """Raised internally when an alpha expression cannot be translated."""


_SAFE_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]+")
_SIMPLE_SERIES_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CALL_SERIES_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*\(.*\)$")
_ALLOWED_COLUMNS: frozenset[str] = frozenset({"open", "high", "low", "close", "volume", "turnover"})
_CROSS_SECTIONAL_FUNCTIONS: frozenset[str] = frozenset(
    {"cs_rank", "cs_mean", "cs_std", "cs_sum", "cs_scale"}
)
_UNSUPPORTED_FUNCTIONS: frozenset[str] = frozenset(
    {
        "ts_argmax",
        "ts_argmin",
        "ts_rank",
        "ts_quantile",
        "ts_rsquare",
        "ts_resi",
        "ts_corr",
        "ts_cov",
        "ts_decay_linear",
        "ts_product",
        "ts_less",
        "ts_greater",
        "ta_macd_dif",
        "ta_macd_dea",
        "ta_macd_hist",
    }
)


class AlphaExpressionToPineTranslator:
    """Translate a conservative `vnpy.alpha` expression subset to Pine v6."""

    def translate(self, expression: str) -> PineTranslationResult:
        """Translate an alpha expression into a Pine expression."""
        try:
            tree = ast.parse(expression, mode="eval")
            return PineTranslationResult(self._convert(tree.body), None)
        except SyntaxError as exc:
            return PineTranslationResult(None, f"invalid alpha expression syntax: {exc.msg}")
        except _UnsupportedAlphaExpression as exc:
            return PineTranslationResult(None, str(exc))

    def _convert(self, node: ast.AST, *, bool_context: bool = False) -> str:
        if isinstance(node, ast.BinOp):
            left = self._convert(node.left)
            right = self._convert(node.right)
            operator = self._convert_binop(node.op)
            return f"({left} {operator} {right})"

        if isinstance(node, ast.UnaryOp):
            operand = self._convert(node.operand)
            if isinstance(node.op, ast.USub):
                return f"-{operand}"
            if isinstance(node.op, ast.UAdd):
                return f"+{operand}"
            if isinstance(node.op, ast.Not):
                condition = self._convert(node.operand, bool_context=True)
                return f"(not {condition})" if bool_context else f"mask(not {condition})"
            raise _UnsupportedAlphaExpression(f"unsupported unary operator {type(node.op).__name__}")

        if isinstance(node, ast.BoolOp):
            values = [self._convert(value, bool_context=bool_context) for value in node.values]
            operator = " and " if isinstance(node.op, ast.And) else " or "
            expression = f"({operator.join(values)})"
            return expression if bool_context else f"mask{expression}"

        if isinstance(node, ast.Compare):
            expression = self._convert_compare(node)
            return f"({expression})" if bool_context else f"mask({expression})"

        if isinstance(node, ast.Call):
            return self._convert_call(node)

        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_COLUMNS:
                return node.id
            raise _UnsupportedAlphaExpression(f"unknown identifier {node.id}")

        if isinstance(node, ast.Constant):
            return self._convert_constant(node.value)

        raise _UnsupportedAlphaExpression(f"unsupported expression node {type(node).__name__}")

    def _convert_call(self, node: ast.Call) -> str:
        name = self._call_name(node.func)
        if name in _CROSS_SECTIONAL_FUNCTIONS:
            raise _UnsupportedAlphaExpression(
                f"cross-sectional function {name} is not supported by single-symbol Pine"
            )
        if name in _UNSUPPORTED_FUNCTIONS:
            raise _UnsupportedAlphaExpression(f"function {name} is not supported by the Pine exporter")
        if node.keywords:
            raise _UnsupportedAlphaExpression(f"keyword arguments are not supported for {name}")

        args = [self._convert(arg) for arg in node.args]

        if name == "ts_delay":
            self._require_arity(name, args, 2)
            return self._history_reference(args[0], args[1])
        if name == "ts_min":
            self._require_arity(name, args, 2)
            return f"ta.lowest({args[0]}, {args[1]})"
        if name == "ts_max":
            self._require_arity(name, args, 2)
            return f"ta.highest({args[0]}, {args[1]})"
        if name == "ts_mean":
            self._require_arity(name, args, 2)
            return f"ta.sma({args[0]}, {args[1]})"
        if name == "ts_std":
            self._require_arity(name, args, 2)
            return f"ta.stdev({args[0]}, {args[1]})"
        if name == "ts_sum":
            self._require_arity(name, args, 2)
            return f"math.sum({args[0]}, {args[1]})"
        if name == "ts_abs":
            self._require_arity(name, args, 1)
            return f"math.abs({args[0]})"
        if name == "ts_log":
            self._require_arity(name, args, 1)
            return f"math.log({args[0]})"

        if name == "ta_ema":
            self._require_arity(name, args, 2)
            return f"ta.ema({args[0]}, {args[1]})"
        if name == "ta_rsi":
            self._require_arity(name, args, 2)
            return f"ta.rsi({args[0]}, {args[1]})"
        if name == "ta_atr":
            self._require_arity(name, args, 4)
            if args[:3] != ["high", "low", "close"]:
                raise _UnsupportedAlphaExpression("ta_atr only supports high, low, close as the first three arguments")
            return f"ta.atr({args[3]})"

        if name == "greater":
            self._require_arity(name, args, 2)
            return f"math.max({args[0]}, {args[1]})"
        if name == "less":
            self._require_arity(name, args, 2)
            return f"math.min({args[0]}, {args[1]})"
        if name == "abs":
            self._require_arity(name, args, 1)
            return f"math.abs({args[0]})"
        if name == "log":
            self._require_arity(name, args, 1)
            return f"math.log({args[0]})"
        if name == "sign":
            self._require_arity(name, args, 1)
            return f"math.sign({args[0]})"
        if name == "pow1":
            self._require_arity(name, args, 2)
            return f"math.pow({args[0]}, {args[1]})"
        if name == "pow2":
            self._require_arity(name, args, 2)
            return f"math.pow({args[0]}, {args[1]})"
        if name in {"quesval", "quesval2"}:
            self._require_arity(name, args, 4)
            condition = f"{args[0]} < {args[1]}"
            return f"({condition} ? {args[2]} : {args[3]})"

        raise _UnsupportedAlphaExpression(f"function {name} is not supported by the Pine exporter")

    def _convert_compare(self, node: ast.Compare) -> str:
        left = self._convert(node.left)
        parts: list[str] = []
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = self._convert(comparator)
            parts.append(f"{left} {self._convert_cmpop(operator)} {right}")
            left = right
        return " and ".join(parts)

    @staticmethod
    def _convert_binop(operator: ast.operator) -> str:
        if isinstance(operator, ast.Add):
            return "+"
        if isinstance(operator, ast.Sub):
            return "-"
        if isinstance(operator, ast.Mult):
            return "*"
        if isinstance(operator, ast.Div):
            return "/"
        if isinstance(operator, ast.Mod):
            return "%"
        if isinstance(operator, ast.Pow):
            raise _UnsupportedAlphaExpression("Python ** power operator is not supported; use pow1 or pow2")
        raise _UnsupportedAlphaExpression(f"unsupported binary operator {type(operator).__name__}")

    @staticmethod
    def _convert_cmpop(operator: ast.cmpop) -> str:
        if isinstance(operator, ast.Gt):
            return ">"
        if isinstance(operator, ast.GtE):
            return ">="
        if isinstance(operator, ast.Lt):
            return "<"
        if isinstance(operator, ast.LtE):
            return "<="
        if isinstance(operator, ast.Eq):
            return "=="
        if isinstance(operator, ast.NotEq):
            return "!="
        raise _UnsupportedAlphaExpression(f"unsupported comparison operator {type(operator).__name__}")

    @staticmethod
    def _convert_constant(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "na"
        if isinstance(value, str):
            raise _UnsupportedAlphaExpression("string literals are not supported in alpha expressions")
        return repr(value)

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        raise _UnsupportedAlphaExpression("only simple function calls are supported")

    @staticmethod
    def _history_reference(source: str, offset: str) -> str:
        if _SIMPLE_SERIES_RE.match(source) or _CALL_SERIES_RE.match(source):
            return f"{source}[{offset}]"
        return f"({source})[{offset}]"

    @staticmethod
    def _require_arity(name: str, args: Sequence[str], expected: int) -> None:
        if len(args) != expected:
            raise _UnsupportedAlphaExpression(f"function {name} expects {expected} arguments")


def export_factor_specs_to_pine(
    specs: Sequence[FactorSpecLike],
    *,
    title: str,
    mode: PineExportMode | str = PineExportMode.INDICATOR,
    threshold: float = 0.0,
    max_plots: int = 20,
) -> PineExportSpec:
    """Export named factor specs into a single-symbol Pine Script v6 file."""
    mode_value = _normalize_mode(mode)
    translator = AlphaExpressionToPineTranslator()
    used_names: set[str] = set()
    translated: list[PineTranslatedFactor] = []
    unsupported: list[PineExportUnsupported] = []

    for spec in specs:
        pine_name = _unique_identifier(_safe_identifier(spec.name), used_names)
        result = translator.translate(spec.expression)
        if result.expression is None:
            unsupported.append(
                PineExportUnsupported(spec.name, spec.expression, result.reason or "unknown translation failure")
            )
            continue
        translated.append(PineTranslatedFactor(pine_name, result.expression, spec.name, spec.expression))

    script = _render_script(
        title=title,
        mode=mode_value,
        translated=translated,
        unsupported=unsupported,
        threshold=threshold,
        max_plots=max_plots,
    )
    return PineExportSpec(script, translated, unsupported)


def resolve_factor_specs(
    family: str,
    *,
    signal_type: str | None = None,
    windows: tuple[int, ...] | None = None,
    include_trade_rule_factors: bool = False,
) -> list[FactorSpecLike]:
    """Resolve an existing factor family into named specs for export."""
    if family == "common_technical":
        from vnpy.alpha.factors.generic import common_technical_factor_specs

        return _as_factor_spec_list(common_technical_factor_specs(windows=windows or (5, 10, 20, 60)))
    if family == "common_gmma":
        from vnpy.alpha.factors.generic import common_gmma_factor_specs

        return _as_factor_spec_list(common_gmma_factor_specs())
    if family == "common_vegas":
        from vnpy.alpha.factors.generic import common_vegas_factor_specs

        return _as_factor_spec_list(common_vegas_factor_specs())
    if family == "strategy_box":
        from vnpy.alpha.factors.strategyByBox import PULLBACK_SIGNAL_TYPES, strategy_box_factor_specs

        specs = list(
            strategy_box_factor_specs(
                windows=windows or (20, 30, 60),
                include_trade_rule_factors=include_trade_rule_factors,
            )
        )
        if signal_type is None:
            return _as_factor_spec_list(specs)
        if signal_type == "pullback_bounce":
            return _as_factor_spec_list(
                [
                    spec
                    for spec in specs
                    if getattr(spec, "signal_type", None) in PULLBACK_SIGNAL_TYPES
                    or getattr(spec, "signal_type", None) == "shared"
                ]
            )
        return _as_factor_spec_list(
            [
                spec
                for spec in specs
                if getattr(spec, "signal_type", None) in {signal_type, "shared"}
            ]
        )
    if family == "tradingview":
        from vnpy.alpha.factors.tradingview import tradingview_factor_specs

        return _as_factor_spec_list(tradingview_factor_specs())

    raise ValueError(f"unknown factor family: {family}")


def _render_script(
    *,
    title: str,
    mode: PineExportMode,
    translated: Sequence[PineTranslatedFactor],
    unsupported: Sequence[PineExportUnsupported],
    threshold: float,
    max_plots: int,
) -> str:
    lines: list[str] = ["//@version=6"]
    escaped_title = _escape_pine_string(title)
    if mode == PineExportMode.STRATEGY:
        lines.append(f'strategy("{escaped_title}", overlay = false, process_orders_on_close = true)')
    else:
        lines.append(f'indicator("{escaped_title}", overlay = false)')

    lines.extend(
        [
            "",
            "mask(condition) => condition ? 1.0 : 0.0",
            "",
        ]
    )

    if translated:
        lines.append("// Translated vnpy.alpha factors")
        for item in translated:
            lines.append(f"{item.name} = {item.expression}")
        lines.append("")

        if len(translated) == 1:
            lines.append(f"score = nz({translated[0].name}) / 1.0")
        else:
            score_terms = " + ".join(f"nz({item.name})" for item in translated)
            lines.append(f"score = ({score_terms}) / {float(len(translated)):.1f}")
        lines.append(f"entrySignal = score >= {_format_float(threshold)}")
        lines.append("")
    else:
        lines.extend(["score = na", "entrySignal = false", ""])

    if mode == PineExportMode.STRATEGY:
        lines.extend(
            [
                "if entrySignal",
                '    strategy.entry("Long", strategy.long)',
                "if not entrySignal",
                '    strategy.close("Long")',
            ]
        )
    else:
        lines.append('plot(score, title = "score", linewidth = 2)')
        for item in translated[:max_plots]:
            lines.append(f'plot({item.name}, title = "{_escape_pine_string(item.source_name)}")')
        lines.append('plotshape(entrySignal, title = "entry", style = shape.triangleup, location = location.bottom)')

    if unsupported:
        lines.extend(["", "// Unsupported expressions"])
        for unsupported_item in unsupported:
            lines.append(f"// - {unsupported_item.name}: {unsupported_item.reason}")
            lines.append(f"//   {unsupported_item.expression}")

    return "\n".join(lines) + "\n"


def _normalize_mode(mode: PineExportMode | str) -> PineExportMode:
    if isinstance(mode, PineExportMode):
        return mode
    try:
        return PineExportMode(mode)
    except ValueError as exc:
        raise ValueError(f"unknown Pine export mode: {mode}") from exc


def _as_factor_spec_list(specs: Iterable[FactorSpecLike]) -> list[FactorSpecLike]:
    return list(specs)


def _safe_identifier(value: str) -> str:
    text = _SAFE_IDENTIFIER_RE.sub("_", value).strip("_")
    if not text:
        return "factor"
    if text[0].isdigit():
        text = f"factor_{text}"
    return text


def _unique_identifier(value: str, used_names: set[str]) -> str:
    if value not in used_names:
        used_names.add(value)
        return value
    index = 2
    while f"{value}_{index}" in used_names:
        index += 1
    unique = f"{value}_{index}"
    used_names.add(unique)
    return unique


def _escape_pine_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_float(value: float) -> str:
    return f"{value:.12g}"


__all__ = [
    "AlphaExpressionToPineTranslator",
    "FactorSpecLike",
    "PineExportMode",
    "PineExportSpec",
    "PineExportUnsupported",
    "PineTranslationResult",
    "PineTranslatedFactor",
    "export_factor_specs_to_pine",
    "resolve_factor_specs",
]
