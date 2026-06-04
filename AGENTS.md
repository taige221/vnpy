# Repository Guidelines

## Project Structure & Module Organization
VeighNa is an event-driven quant trading framework. The core lives in `vnpy/`:
- `vnpy/event/` — the `EventEngine` message bus that drives the whole system.
- `vnpy/trader/` — platform core. `MainEngine` (`engine.py`) wires together `BaseEngine` subclasses (`OmsEngine`, `LogEngine`, `EmailEngine`, `WechatEngine`) and exposes `add_gateway`/`add_app`/`add_engine`. Shared domain types live in `object.py`/`constant.py`; trading helpers (`ArrayManager`, `BarGenerator`) in `utility.py`; the PySide6 GUI in `ui/`.
- `vnpy/alpha/` — AI/ML multi-factor research stack (`dataset/`, `model/`, `strategy/`, `lab.py`); needs the `alpha` extra.
- `vnpy/chart/` (pyqtgraph K-line) and `vnpy/rpc/` (ZeroMQ distributed deployment).

Trading gateways (CTP, IB…) and apps (CTA strategy…) are **separate `vnpy_*` pip packages**, not in this repo — they are loaded at runtime via `add_gateway`/`add_app`. `examples/` holds runnable setups; `tests/` holds pytest suites.

## Build, Test, and Development Commands
- `bash install.sh` (Linux) / `bash install_osx.sh` (macOS) / `install.bat` (Windows) — full setup including ta-lib.
- `uv pip install -e .[alpha,dev]` — editable dev install. ta-lib needs `--index=https://pypi.vnpy.com`.
- `ruff check .` — lint. `mypy vnpy` — type check. `uv build` — build wheel/sdist.
- `pytest tests/` — run tests; a single test is e.g. `pytest tests/test_alpha101.py::test_<name>`.

## Coding Style & Naming Conventions
Four-space indent, fully typed code (the package ships `py.typed`). Ruff targets py310 with rule sets B/E/F/UP/W (line length `E501` ignored). mypy runs strict (`disallow_untyped_defs`, `disallow_incomplete_defs`, `no_implicit_optional`, …) — annotate every function. Wrap user-facing strings in `_()` from `vnpy.trader.locale`; the English catalog builds from `vnpy/trader/locale/vnpy.pot`.

## Testing Guidelines
Tests use `pytest` and live under `tests/`. Note that CI (`.github/workflows/pythonapp.yml`, Windows + Python 3.13) runs only `ruff`, `mypy`, and `uv build` — it does not run the suite, so run `pytest` locally before submitting.

## Commit & Pull Request Guidelines
Commit subjects use bracketed tags: `[Add]`, `[Mod]`, `[Fix]`, `[Del]` followed by a short description (English or Chinese). Keep PRs small and focused — split large changes (per `PULL_REQUEST_TEMPLATE.md`), list improvements, and reference issues with `Close #<id>`. Target the `master` or `dev` branch.
