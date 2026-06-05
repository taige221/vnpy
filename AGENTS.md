# Repository Guidelines

## Project Structure & Module Organization
VeighNa is an event-driven quant trading framework. Core code lives in `vnpy/`: `vnpy/event/` provides the `EventEngine` message bus, while `vnpy/trader/` contains `MainEngine`, OMS/log/email/wechat engines, shared domain objects, utilities, settings, and the PySide6 UI. `vnpy/alpha/` is the AI/ML research stack, with dataset expressions, processors, models, strategies, and `lab.py` workflow management; install the `alpha` extra before using it. `vnpy/chart/` contains pyqtgraph K-line components, and `vnpy/rpc/` provides ZeroMQ RPC support. Runnable examples live in `examples/`, tests in `tests/`, and documentation in `docs/`. Gateways and trading apps are separate `vnpy_*` packages loaded at runtime through `add_gateway` and `add_app`.

## Build, Test, and Development Commands
- `bash install.sh`, `bash install_osx.sh`, or `install.bat` — platform setup scripts, including TA-Lib installation.
- `uv pip install -e .[alpha,dev] --index=https://pypi.vnpy.com` — editable install with alpha and development dependencies.
- `ruff check .` — run configured lint checks.
- `mypy vnpy` — run strict type checking for the package.
- `pytest tests/` — run the local test suite; a single test can be run as `pytest tests/test_alpha101.py::test_alpha1`.
- `uv build` — build wheel and source distribution.

## Coding Style & Naming Conventions
Use four-space indentation and fully typed Python. The package ships `py.typed`; mypy is configured with strict options including `disallow_untyped_defs`, `disallow_incomplete_defs`, and `no_implicit_optional`. Ruff targets Python 3.10 and enables B/E/F/UP/W rules, with `E501` ignored. Wrap trader-facing strings with `_()` from `vnpy.trader.locale`; locale source files live under `vnpy/trader/locale/`.

## Testing Guidelines
Tests use `pytest` and are named under `tests/`, with alpha-specific tests in `tests/alpha/`. CI (`.github/workflows/pythonapp.yml`) runs on Windows with Python 3.13 and executes `ruff check .`, `mypy vnpy`, and `uv build`; it does not run `pytest`, so run relevant tests locally before submitting behavior changes.

## Commit & Pull Request Guidelines
Recent commits mostly use bracketed tags such as `[Add]`, `[Mod]`, `[Fix]`, and `[Del]`, with some conventional-style subjects like `feat:`, `fix:`, and `chore(ci):`; follow nearby history for the change type. Keep PRs small and focused, as requested by `.github/PULL_REQUEST_TEMPLATE.md`. List improvements, link related issues with `Close #`, and target the `dev` branch per the README contribution flow.
