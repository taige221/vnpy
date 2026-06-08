from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple


@dataclass(frozen=True)
class PineInputSpec:
    """User input extracted from a Pine script."""

    name: str
    input_type: str
    default: Any
    title: str | None
    options: tuple[Any, ...]
    group: str | None
    line: int
    source: str


@dataclass(frozen=True)
class PineAssignmentSpec:
    """Pine assignment and optional alpha-expression translation."""

    name: str
    expression: str
    translated_expression: str | None
    status: str
    line: int
    source: str
    reason: str | None = None


@dataclass(frozen=True)
class PineActionSpec:
    """Trading or alert action extracted from Pine source."""

    action_type: str
    name: str | None
    condition: str | None
    translated_condition: str | None
    line: int
    source: str
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class PineUnsupportedSpec:
    """Unsupported Pine construct that needs manual interpretation."""

    line: int
    reason: str
    source: str


@dataclass(frozen=True)
class PineAlphaExpressionSpec:
    """Successfully translated expression suitable for alpha research."""

    name: str
    expression: str
    source: str
    line: int
    kind: str


@dataclass(frozen=True)
class PineExtractResult:
    """Complete Pine extraction result."""

    metadata: dict[str, Any]
    inputs: list[PineInputSpec]
    assignments: list[PineAssignmentSpec]
    actions: list[PineActionSpec]
    alerts: list[PineActionSpec]
    unsupported: list[PineUnsupportedSpec]
    alpha_expressions: list[PineAlphaExpressionSpec]


class _Statement(NamedTuple):
    line: int
    indent: int
    text: str
    source: str


class _Translation(NamedTuple):
    expression: str | None
    reason: str | None


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*|\[[^\]]+\])\s*=\s*(.+)$")
_INPUT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*input\.([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$")
_METADATA_RE = re.compile(r"^(strategy|indicator)\s*\((.*)\)$")
_ACTION_RE = re.compile(r"^(strategy\.(?:entry|close|exit|close_all))\((.*)\)$")
_ALERT_RE = re.compile(r"^alertcondition\((.*)\)$")
_STRING_RE = re.compile(r"^(['\"])(.*)\1$")
_HISTORY_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\[([A-Za-z_][A-Za-z0-9_]*|\d+)\]")

_DISPLAY_PREFIXES = (
    "plot(",
    "plotshape(",
    "plotchar(",
    "fill(",
    "bgcolor(",
    "barcolor(",
    "hline(",
    "table.",
    "label.",
    "line.",
)
_SUPPORTED_COLUMNS = {"open", "high", "low", "close", "volume", "turnover"}
_ALLOWED_IDENTIFIERS = _SUPPORTED_COLUMNS | {
    "ts_delay",
    "ts_min",
    "ts_max",
    "ts_mean",
    "ta_ema",
    "ta_rsi",
    "ta_atr",
    "greater",
    "less",
    "abs",
    "quesval",
}
_UNSUPPORTED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("request.security", "multi-timeframe request.security is not translated"),
    ("array.", "array operations are not translated"),
    ("ta.pivotlow", "pivot functions are not translated"),
    ("ta.pivothigh", "pivot functions are not translated"),
    ("ta.valuewhen", "valuewhen state lookup is not translated"),
    ("ta.barssince", "barssince state lookup is not translated"),
    ("ta.dmi", "ta.dmi tuple output is not translated"),
    ("ta.macd", "ta.macd tuple output is not translated"),
    ("bar_index", "bar_index state-machine logic is not translated"),
    ("barstate.", "barstate confirmation/display logic is not translated"),
    ("strategy.position_size", "strategy position state is not translated"),
    ("strategy.opentrades", "strategy trade state is not translated"),
    ("time ", "time-based filters are not translated"),
    ("time>", "time-based filters are not translated"),
    ("time<", "time-based filters are not translated"),
    ("time >=", "time-based filters are not translated"),
    ("time <=", "time-based filters are not translated"),
    ("na(", "na() checks are not translated"),
    ("nz(", "nz() replacement is not translated"),
    ("str.", "string formatting is not translated"),
    ("color.", "visual color expressions are not translated"),
    ("table.", "table display logic is not translated"),
    ("label.", "label display logic is not translated"),
)


def extract_pine_strategy(source: str) -> PineExtractResult:
    """Extract supported Pine strategy structure and alpha expressions."""
    statements = _join_logical_statements(source)
    metadata: dict[str, Any] = {}
    inputs: list[PineInputSpec] = []
    assignments: list[PineAssignmentSpec] = []
    actions: list[PineActionSpec] = []
    alerts: list[PineActionSpec] = []
    unsupported: list[PineUnsupportedSpec] = []
    alpha_expressions: list[PineAlphaExpressionSpec] = []

    constants: dict[str, Any] = {}
    translator = PineExpressionTranslator(constants)
    if_stack: list[tuple[int, str]] = []

    for statement in statements:
        text = statement.text
        while if_stack and statement.indent <= if_stack[-1][0]:
            if_stack.pop()

        if not text or text.startswith("//") or text.startswith("//@version"):
            continue
        if _is_display_statement(text):
            continue

        if text.startswith("if "):
            if_stack.append((statement.indent, text[3:].strip()))
            continue
        if text.startswith("else if "):
            if_stack.append((statement.indent, text[8:].strip()))
            continue
        if text in {"else", "else:"}:
            continue

        metadata_match = _METADATA_RE.match(text)
        if metadata_match:
            kind, args_text = metadata_match.groups()
            args, kwargs = _split_call_args(args_text)
            metadata["kind"] = kind
            metadata["title"] = _parse_literal(args[0]) if args else None
            metadata["line"] = statement.line
            metadata["source"] = statement.source
            metadata.update({key: _parse_literal(value) for key, value in kwargs.items()})
            continue

        if text.startswith("var "):
            unsupported.append(
                PineUnsupportedSpec(statement.line, "stateful var declarations are not translated", statement.source)
            )
            continue

        if ":=" in text:
            unsupported.append(PineUnsupportedSpec(statement.line, "reassignment with := is not translated", statement.source))
            continue

        if text.startswith("switch ") or " switch " in text or " = switch " in text:
            unsupported.append(PineUnsupportedSpec(statement.line, "switch expressions are not translated", statement.source))
            continue

        action_match = _ACTION_RE.match(text)
        if action_match:
            action = _build_action(action_match, statement, if_stack, translator)
            actions.append(action)
            if action.translated_condition:
                alpha_expressions.append(
                    PineAlphaExpressionSpec(
                        name=_action_expression_name(action),
                        expression=action.translated_condition,
                        source=action.source,
                        line=action.line,
                        kind=action.action_type,
                    )
                )
            elif action.reason:
                unsupported.append(PineUnsupportedSpec(statement.line, action.reason, statement.source))
            continue

        alert_match = _ALERT_RE.match(text)
        if alert_match:
            alert = _build_alert(alert_match.group(1), statement, translator)
            alerts.append(alert)
            if alert.translated_condition:
                alpha_expressions.append(
                    PineAlphaExpressionSpec(
                        name=_safe_name(alert.name or f"alert_{alert.line}"),
                        expression=alert.translated_condition,
                        source=alert.source,
                        line=alert.line,
                        kind="alertcondition",
                    )
                )
            elif alert.reason:
                unsupported.append(PineUnsupportedSpec(statement.line, alert.reason, statement.source))
            continue

        input_match = _INPUT_RE.match(text)
        if input_match:
            input_spec = _parse_input(input_match, statement, constants)
            inputs.append(input_spec)
            translator.register_input(input_spec)
            constants[input_spec.name] = input_spec.default
            continue

        assignment_match = _ASSIGNMENT_RE.match(text)
        if assignment_match:
            target, expression = assignment_match.groups()
            if target.startswith("["):
                reason = "tuple assignments are not translated"
                unsupported.append(PineUnsupportedSpec(statement.line, reason, statement.source))
                assignments.append(
                    PineAssignmentSpec(target, expression.strip(), None, "unsupported", statement.line, statement.source, reason)
                )
                continue

            literal = _parse_literal(expression.strip())
            if isinstance(literal, str) and _is_quoted(expression.strip()):
                constants[target] = literal
                continue

            translation = translator.translate(expression.strip())
            status = "translated" if translation.expression else "unsupported"
            assignment = PineAssignmentSpec(
                target,
                expression.strip(),
                translation.expression,
                status,
                statement.line,
                statement.source,
                translation.reason,
            )
            assignments.append(assignment)
            if translation.expression:
                translator.register_assignment(target, translation.expression)
                alpha_expressions.append(
                    PineAlphaExpressionSpec(target, translation.expression, statement.source, statement.line, "assignment")
                )
            elif translation.reason:
                unsupported.append(PineUnsupportedSpec(statement.line, translation.reason, statement.source))
            continue

        unsupported_reason: str | None = _unsupported_reason(text)
        if unsupported_reason:
            unsupported.append(PineUnsupportedSpec(statement.line, unsupported_reason, statement.source))

    return PineExtractResult(
        metadata=metadata,
        inputs=inputs,
        assignments=assignments,
        actions=actions,
        alerts=alerts,
        unsupported=_dedupe_unsupported(unsupported),
        alpha_expressions=alpha_expressions,
    )


def extract_pine_strategy_file(
    path: str | Path,
    output_dir: str | Path | None = None,
    alpha_output: str | Path | None = None,
) -> PineExtractResult:
    """Extract a Pine file and write reports when an output directory is provided."""
    source_path = Path(path)
    result = extract_pine_strategy(source_path.read_text(encoding="utf-8"))
    if output_dir is not None or alpha_output is not None:
        write_extract_outputs(result, source_path, Path(output_dir or source_path.parent), _optional_path(alpha_output))
    return result


def write_extract_outputs(
    result: PineExtractResult,
    source_path: Path,
    output_dir: Path,
    alpha_output: Path | None = None,
) -> None:
    """Write JSON, Markdown, and optional alpha-expression outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem
    report_data = _result_to_dict(result)
    (output_dir / f"{stem}_pine_extract_report.json").write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{stem}_pine_extract_report.md").write_text(
        _render_markdown_report(result, source_path),
        encoding="utf-8",
    )

    if alpha_output is not None:
        alpha_output.parent.mkdir(parents=True, exist_ok=True)
        alpha_output.write_text(
            json.dumps([asdict(item) for item in result.alpha_expressions], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class PineExpressionTranslator:
    """Translate a conservative Pine expression subset to vnpy.alpha syntax."""

    def __init__(self, constants: dict[str, Any] | None = None) -> None:
        self.symbols: dict[str, str] = {}
        self.constants = constants if constants is not None else {}

    def register_input(self, input_spec: PineInputSpec) -> None:
        replacement = _input_default_expression(input_spec)
        if replacement is not None:
            self.symbols[input_spec.name] = replacement

    def register_assignment(self, name: str, expression: str) -> None:
        self.symbols[name] = expression

    def translate(self, expression: str) -> _Translation:
        try:
            translated = self._translate(expression.strip())
        except ValueError as exc:
            return _Translation(None, str(exc))

        unresolved = _unresolved_identifiers(translated, set(self.symbols))
        if unresolved:
            joined = ", ".join(unresolved)
            return _Translation(None, f"references unresolved symbols: {joined}")
        return _Translation(translated, None)

    def _translate(self, expression: str) -> str:
        expression = _normalize_expression(expression)
        reason = _unsupported_reason(expression)
        if reason:
            raise ValueError(reason)
        if _is_quoted(expression):
            raise ValueError("string expressions are not alpha expressions")
        if expression == "na":
            raise ValueError("na values are not translated")

        ternary = _split_ternary(expression)
        if ternary:
            condition, true_expr, false_expr = ternary
            return (
                f"quesval(0, {self._translate(condition)}, "
                f"{self._translate(true_expr)}, {self._translate(false_expr)})"
            )

        parts = _split_top_level_operator(expression, "or")
        if len(parts) > 1:
            joined = " + ".join(f"({self._translate(part)})" for part in parts)
            return f"(({joined}) > 0)"

        parts = _split_top_level_operator(expression, "and")
        if len(parts) > 1:
            joined = " * ".join(f"({self._translate(part)})" for part in parts)
            return f"({joined})"

        if expression.startswith("not "):
            return f"(1 - ({self._translate(expression[4:])}))"

        wrapped = _strip_wrapping_parentheses(expression)
        if wrapped != expression:
            return f"({self._translate(wrapped)})"

        expression = _replace_history_references(expression, self)
        expression = expression.replace("syminfo.mintick", "1e-12")
        expression = self._replace_supported_functions(expression)
        expression = _replace_word(expression, "true", "1")
        expression = _replace_word(expression, "false", "0")
        expression = self._replace_symbols(expression)
        expression = _normalize_expression(expression)

        if "=>" in expression:
            raise ValueError("switch branch expressions are not translated")
        return expression

    def _replace_supported_functions(self, expression: str) -> str:
        handlers: dict[str, Callable[[list[str]], str]] = {
            "ta.sma": lambda args: _require_args("ta.sma", args, 2, lambda: f"ts_mean({self._translate(args[0])}, {self._translate(args[1])})"),
            "ta.ema": lambda args: _require_args("ta.ema", args, 2, lambda: f"ta_ema({self._translate(args[0])}, {self._translate(args[1])})"),
            "ta.rsi": lambda args: _require_args("ta.rsi", args, 2, lambda: f"ta_rsi({self._translate(args[0])}, {self._translate(args[1])})"),
            "ta.atr": lambda args: _require_args("ta.atr", args, 1, lambda: f"ta_atr(high, low, close, {self._translate(args[0])})"),
            "ta.highest": lambda args: _require_args("ta.highest", args, 2, lambda: f"ts_max({self._translate(args[0])}, {self._translate(args[1])})"),
            "ta.lowest": lambda args: _require_args("ta.lowest", args, 2, lambda: f"ts_min({self._translate(args[0])}, {self._translate(args[1])})"),
            "ta.crossover": self._translate_crossover,
            "ta.crossunder": self._translate_crossunder,
            "math.max": lambda args: _require_args("math.max", args, 2, lambda: f"greater({self._translate(args[0])}, {self._translate(args[1])})"),
            "math.min": lambda args: _require_args("math.min", args, 2, lambda: f"less({self._translate(args[0])}, {self._translate(args[1])})"),
            "math.abs": lambda args: _require_args("math.abs", args, 1, lambda: f"abs({self._translate(args[0])})"),
        }
        result = expression
        changed = True
        while changed:
            changed = False
            for name, handler in handlers.items():
                replaced = _replace_function_calls(result, name, handler)
                if replaced != result:
                    result = replaced
                    changed = True
        return result

    def _translate_crossover(self, args: list[str]) -> str:
        return _require_args(
            "ta.crossover",
            args,
            2,
            lambda: (
                f"(({self._translate(args[0])} > {self._translate(args[1])}) * "
                f"({_delay_expression(self._translate(args[0]))} <= {_delay_expression(self._translate(args[1]))}))"
            ),
        )

    def _translate_crossunder(self, args: list[str]) -> str:
        return _require_args(
            "ta.crossunder",
            args,
            2,
            lambda: (
                f"(({self._translate(args[0])} < {self._translate(args[1])}) * "
                f"({_delay_expression(self._translate(args[0]))} >= {_delay_expression(self._translate(args[1]))}))"
            ),
        )

    def _replace_symbols(self, expression: str) -> str:
        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            if token in self.symbols:
                return self.symbols[token]
            if token in self.constants:
                value = self.constants[token]
                if isinstance(value, bool):
                    return "1" if value else "0"
                if isinstance(value, int | float):
                    return str(value)
            return token

        return _IDENTIFIER_RE.sub(replace, expression)


def _parse_input(match: re.Match[str], statement: _Statement, constants: dict[str, Any]) -> PineInputSpec:
    name, input_type, args_text = match.groups()
    args, kwargs = _split_call_args(args_text)
    default = _parse_literal(args[0]) if args else None
    title = _parse_literal(args[1]) if len(args) > 1 else _optional_literal(kwargs.get("title"))
    group = _optional_literal(kwargs.get("group"))
    if group in constants:
        group = str(constants[group])
    options = _parse_options(kwargs.get("options"))
    return PineInputSpec(name, input_type, default, title, options, group, statement.line, statement.source)


def _build_action(
    match: re.Match[str],
    statement: _Statement,
    if_stack: list[tuple[int, str]],
    translator: PineExpressionTranslator,
) -> PineActionSpec:
    action_type, args_text = match.groups()
    args, kwargs = _split_call_args(args_text)
    name = _optional_literal(args[0]) if args else None
    condition = if_stack[-1][1] if if_stack else None
    translation = translator.translate(condition) if condition else _Translation(None, "action has no surrounding condition")
    status = "translated" if translation.expression else "unsupported"
    if action_type == "strategy.exit":
        status = "unsupported"
        translation = _Translation(None, "strategy.exit risk orders need manual interpretation")
    if action_type == "strategy.close_all":
        status = "unsupported"
        translation = _Translation(None, "strategy.close_all portfolio state needs manual interpretation")
    if "qty" in kwargs or "qty_percent" in kwargs:
        status = "unsupported"
        translation = _Translation(None, "position sizing arguments need manual interpretation")
    return PineActionSpec(action_type, name, condition, translation.expression, statement.line, statement.source, status, translation.reason)


def _build_alert(args_text: str, statement: _Statement, translator: PineExpressionTranslator) -> PineActionSpec:
    args, kwargs = _split_call_args(args_text)
    condition = args[0] if args else None
    name = _optional_literal(kwargs.get("title"))
    translation = translator.translate(condition) if condition else _Translation(None, "alertcondition has no condition")
    status = "translated" if translation.expression else "unsupported"
    return PineActionSpec("alertcondition", name, condition, translation.expression, statement.line, statement.source, status, translation.reason)


def _join_logical_statements(source: str) -> list[_Statement]:
    statements: list[_Statement] = []
    parts: list[str] = []
    sources: list[str] = []
    start_line = 0
    start_indent = 0

    def flush() -> None:
        nonlocal parts, sources, start_line, start_indent
        if parts:
            text = _normalize_expression(" ".join(parts))
            source_text = "\n".join(sources)
            statements.append(_Statement(start_line, start_indent, text, source_text))
        parts = []
        sources = []
        start_line = 0
        start_indent = 0

    for line_no, raw_line in enumerate(source.splitlines(), start=1):
        stripped_comment = _strip_comment(raw_line).rstrip()
        stripped = stripped_comment.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("//"):
            continue

        indent = len(stripped_comment) - len(stripped_comment.lstrip())
        if not parts:
            start_line = line_no
            start_indent = indent
        parts.append(stripped)
        sources.append(raw_line.rstrip())
        joined = " ".join(parts)
        if _has_open_continuation(joined):
            continue
        flush()

    flush()
    return statements


def _has_open_continuation(text: str) -> bool:
    if _paren_balance(text) > 0:
        return True
    stripped = text.rstrip()
    if stripped.endswith(("=", "+", "-", "*", "/", "?", ":", ",", "and", "or", "not", "(")):
        return True
    if re.search(r"\b(switch|=>)\s*$", stripped):
        return True
    return False


def _paren_balance(text: str) -> int:
    balance = 0
    quote: str | None = None
    escape = False
    for char in text:
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
    return balance


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escape = False
    for index, char in enumerate(line):
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "/" and index + 1 < len(line) and line[index + 1] == "/":
            return line[:index]
    return line


def _split_call_args(text: str) -> tuple[list[str], dict[str, str]]:
    args: list[str] = []
    kwargs: dict[str, str] = {}
    for item in _split_top_level_commas(text):
        if not item:
            continue
        key_value = _split_top_level_keyword(item)
        if key_value:
            key, value = key_value
            kwargs[key.strip()] = value.strip()
        else:
            args.append(item.strip())
    return args, kwargs


def _split_top_level_commas(text: str) -> list[str]:
    return _split_top_level(text, ",")


def _split_top_level_keyword(text: str) -> tuple[str, str] | None:
    quote: str | None = None
    square = 0
    paren = 0
    for index, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "[":
            square += 1
        elif char == "]":
            square -= 1
        elif char == "(":
            paren += 1
        elif char == ")":
            paren -= 1
        elif char == "=" and square == 0 and paren == 0:
            return text[:index], text[index + 1 :]
    return None


def _split_top_level(text: str, separator: str) -> list[str]:
    result: list[str] = []
    quote: str | None = None
    square = 0
    paren = 0
    current: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
        elif char == "[":
            square += 1
            current.append(char)
        elif char == "]":
            square -= 1
            current.append(char)
        elif char == "(":
            paren += 1
            current.append(char)
        elif char == ")":
            paren -= 1
            current.append(char)
        elif text.startswith(separator, index) and square == 0 and paren == 0:
            result.append("".join(current).strip())
            current = []
            index += len(separator) - 1
        else:
            current.append(char)
        index += 1
    result.append("".join(current).strip())
    return result


def _split_top_level_operator(text: str, operator: str) -> list[str]:
    pattern = f" {operator} "
    result: list[str] = []
    quote: str | None = None
    paren = 0
    current: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
        elif char == "(":
            paren += 1
            current.append(char)
        elif char == ")":
            paren -= 1
            current.append(char)
        elif paren == 0 and text.startswith(pattern, index):
            result.append("".join(current).strip())
            current = []
            index += len(pattern) - 1
        else:
            current.append(char)
        index += 1
    result.append("".join(current).strip())
    return result if len(result) > 1 else [text]


def _split_ternary(text: str) -> tuple[str, str, str] | None:
    question = _find_top_level_char(text, "?")
    if question is None:
        return None
    colon = _find_top_level_char(text[question + 1 :], ":")
    if colon is None:
        return None
    colon_index = question + 1 + colon
    return text[:question].strip(), text[question + 1 : colon_index].strip(), text[colon_index + 1 :].strip()


def _find_top_level_char(text: str, needle: str) -> int | None:
    quote: str | None = None
    paren = 0
    square = 0
    for index, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            paren += 1
        elif char == ")":
            paren -= 1
        elif char == "[":
            square += 1
        elif char == "]":
            square -= 1
        elif char == needle and paren == 0 and square == 0:
            return index
    return None


def _replace_function_calls(text: str, name: str, handler: Callable[[list[str]], str]) -> str:
    search = f"{name}("
    index = text.find(search)
    if index < 0:
        return text
    close = _find_matching_paren(text, index + len(name))
    if close is None:
        raise ValueError(f"malformed function call: {name}")
    args_text = text[index + len(search) : close]
    replacement = handler(_split_top_level_commas(args_text))
    return text[:index] + replacement + text[close + 1 :]


def _find_matching_paren(text: str, open_index: int) -> int | None:
    quote: str | None = None
    balance = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
            if balance == 0:
                return index
    return None


def _replace_history_references(expression: str, translator: PineExpressionTranslator) -> str:
    def replace(match: re.Match[str]) -> str:
        name, offset = match.groups()
        base = translator.symbols.get(name, name)
        offset = translator.symbols.get(offset, str(translator.constants.get(offset, offset)))
        if not offset.isdigit():
            raise ValueError(f"history offset is not a resolved integer: {offset}")
        return f"ts_delay({base}, {offset})"

    return _HISTORY_RE.sub(replace, expression)


def _require_args(name: str, args: list[str], count: int, build: Callable[[], str]) -> str:
    if len(args) != count:
        raise ValueError(f"{name} expects {count} arguments")
    return build()


def _delay_expression(expression: str) -> str:
    if re.fullmatch(r"-?\d+(?:\.\d+)?", expression.strip()):
        return expression.strip()
    return f"ts_delay({expression}, 1)"


def _strip_wrapping_parentheses(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("(") or not stripped.endswith(")"):
        return text
    close = _find_matching_paren(stripped, 0)
    if close == len(stripped) - 1:
        return stripped[1:-1].strip()
    return text


def _normalize_expression(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _replace_word(text: str, word: str, replacement: str) -> str:
    return re.sub(rf"\b{re.escape(word)}\b", replacement, text)


def _parse_literal(text: str) -> Any:
    stripped = text.strip()
    if _is_quoted(stripped):
        return stripped[1:-1]
    if stripped == "true":
        return True
    if stripped == "false":
        return False
    if stripped == "na":
        return None
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)
    if re.fullmatch(r"-?\d+\.\d+", stripped):
        return float(stripped)
    return stripped


def _optional_literal(text: str | None) -> str | None:
    if text is None:
        return None
    value = _parse_literal(text)
    return None if value is None else str(value)


def _is_quoted(text: str) -> bool:
    return bool(_STRING_RE.match(text.strip()))


def _parse_options(text: str | None) -> tuple[Any, ...]:
    if not text:
        return ()
    stripped = text.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return ()
    return tuple(_parse_literal(item) for item in _split_top_level_commas(stripped[1:-1]))


def _input_default_expression(input_spec: PineInputSpec) -> str | None:
    value = input_spec.default
    if input_spec.input_type == "source" and isinstance(value, str):
        return value if value in _SUPPORTED_COLUMNS else None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int | float):
        return str(value)
    return None


def _unsupported_reason(text: str) -> str | None:
    for pattern, reason in _UNSUPPORTED_PATTERNS:
        if pattern in text:
            return reason
    if "=>" in text:
        return "switch branch expressions are not translated"
    return None


def _unresolved_identifiers(expression: str, local_symbols: set[str]) -> list[str]:
    unresolved: set[str] = set()
    for token in _IDENTIFIER_RE.findall(_remove_strings(expression)):
        if token in _ALLOWED_IDENTIFIERS:
            continue
        if token in {"true", "false"}:
            continue
        if token in local_symbols:
            continue
        if re.fullmatch(r"\d+", token):
            continue
        unresolved.add(token)
    return sorted(unresolved)


def _remove_strings(text: str) -> str:
    return re.sub(r"(['\"]).*?\1", "", text)


def _is_display_statement(text: str) -> bool:
    return text.startswith(_DISPLAY_PREFIXES)


def _dedupe_unsupported(items: list[PineUnsupportedSpec]) -> list[PineUnsupportedSpec]:
    seen: set[tuple[int, str, str]] = set()
    result: list[PineUnsupportedSpec] = []
    for item in items:
        key = (item.line, item.reason, item.source)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _safe_name(value: str) -> str:
    name = re.sub(r"\W+", "_", value, flags=re.ASCII).strip("_")
    return name or "pine_expression"


def _action_expression_name(action: PineActionSpec) -> str:
    raw = "_".join(part for part in [action.action_type.replace(".", "_"), action.name] if part)
    return _safe_name(raw)


def _optional_path(path: str | Path | None) -> Path | None:
    return None if path is None else Path(path)


def _result_to_dict(result: PineExtractResult) -> dict[str, Any]:
    return {
        "metadata": result.metadata,
        "inputs": [asdict(item) for item in result.inputs],
        "assignments": [asdict(item) for item in result.assignments],
        "actions": [asdict(item) for item in result.actions],
        "alerts": [asdict(item) for item in result.alerts],
        "unsupported": [asdict(item) for item in result.unsupported],
        "alpha_expressions": [asdict(item) for item in result.alpha_expressions],
        "summary": {
            "input_count": len(result.inputs),
            "assignment_count": len(result.assignments),
            "translated_assignment_count": sum(1 for item in result.assignments if item.status == "translated"),
            "action_count": len(result.actions),
            "alert_count": len(result.alerts),
            "unsupported_count": len(result.unsupported),
            "alpha_expression_count": len(result.alpha_expressions),
        },
    }


def _render_markdown_report(result: PineExtractResult, source_path: Path) -> str:
    summary = _result_to_dict(result)["summary"]
    lines = [
        f"# Pine Extract Report: {source_path.name}",
        "",
        "## Summary",
        "",
        f"- Kind: {result.metadata.get('kind', 'unknown')}",
        f"- Title: {result.metadata.get('title', 'unknown')}",
        f"- Inputs: {summary['input_count']}",
        f"- Assignments: {summary['assignment_count']}",
        f"- Translated assignments: {summary['translated_assignment_count']}",
        f"- Actions: {summary['action_count']}",
        f"- Alerts: {summary['alert_count']}",
        f"- Unsupported items: {summary['unsupported_count']}",
        "",
        "## Inputs",
        "",
    ]
    for input_spec in result.inputs:
        group = f" [{input_spec.group}]" if input_spec.group else ""
        lines.append(
            f"- L{input_spec.line}: `{input_spec.name}` {input_spec.input_type} = "
            f"`{input_spec.default}`{group} - {input_spec.title or ''}"
        )

    lines.extend(["", "## Translated Expressions", ""])
    if result.alpha_expressions:
        for alpha_expression in result.alpha_expressions:
            lines.append(f"- L{alpha_expression.line}: `{alpha_expression.name}` ({alpha_expression.kind})")
            lines.append(f"  - `{alpha_expression.expression}`")
    else:
        lines.append("- None")

    lines.extend(["", "## Actions", ""])
    if result.actions:
        for action in result.actions:
            lines.append(f"- L{action.line}: `{action.action_type}` `{action.name}` status={action.status}")
            if action.condition:
                lines.append(f"  - condition: `{action.condition}`")
            if action.reason:
                lines.append(f"  - reason: {action.reason}")
    else:
        lines.append("- None")

    lines.extend(["", "## Alerts", ""])
    if result.alerts:
        for alert in result.alerts:
            lines.append(f"- L{alert.line}: `{alert.name}` status={alert.status}")
            if alert.condition:
                lines.append(f"  - condition: `{alert.condition}`")
            if alert.reason:
                lines.append(f"  - reason: {alert.reason}")
    else:
        lines.append("- None")

    lines.extend(["", "## Unsupported", ""])
    if result.unsupported:
        for unsupported_item in result.unsupported:
            lines.append(f"- L{unsupported_item.line}: {unsupported_item.reason}")
            lines.append(f"  - `{_normalize_expression(unsupported_item.source)}`")
    else:
        lines.append("- None")

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Extract TradingView Pine strategy structure")
    parser.add_argument("paths", nargs="+", help="Pine Script files to extract")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON and Markdown reports")
    parser.add_argument("--alpha-output", help="Alpha-expression JSON path; only valid with one input file")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.alpha_output and len(args.paths) != 1:
        raise SystemExit("--alpha-output can only be used with one input file")

    for raw_path in args.paths:
        path = Path(raw_path)
        alpha_output = Path(args.alpha_output) if args.alpha_output else output_dir / f"{path.stem}_alpha_exprs.json"
        result = extract_pine_strategy_file(path, output_dir=output_dir, alpha_output=alpha_output)
        print(
            f"Extracted {path}: {len(result.alpha_expressions)} translated expressions, "
            f"{len(result.unsupported)} unsupported items"
        )


__all__ = [
    "PineActionSpec",
    "PineAlphaExpressionSpec",
    "PineAssignmentSpec",
    "PineExtractResult",
    "PineInputSpec",
    "PineUnsupportedSpec",
    "extract_pine_strategy",
    "extract_pine_strategy_file",
    "write_extract_outputs",
]


if __name__ == "__main__":
    main()
