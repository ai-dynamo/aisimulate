<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Developer Guide

This guide will help you get started with developing the unified `aisimulate`
wheel and its AIConfigurator compatibility surface. We welcome contributions
from the community!

## Initial Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ai-dynamo/aisimulate
cd aisimulate
```

Current performance profiles are checked-in Parquet files. Git LFS is only
needed when developing against retained legacy `*.txt` perf assets or running
their compatibility tests; for that work, install Git LFS and run
`git lfs pull`.

### 2. Install Development Dependencies

```bash
uv sync --project python/aisimulate --extra dev
```

This creates `python/aisimulate/.venv`, builds the unified native extension,
and installs the sole Python distribution together with its development tools.

To activate the environment:

```bash
source python/aisimulate/.venv/bin/activate
```

### 3. Install Pre-Commit Hooks

```bash
pre-commit install --config python/aisimulate/.pre-commit-config.yaml
```

The development environment includes:
- The `aisimulate` package and its `aiconfigurator`, `aiconfigurator_core`, and
  `aisimulate_core` compatibility namespaces in editable mode
- All runtime dependencies
- Development tools: `ruff`, `pre-commit`, `pytest` and related plugins

## AIConfigurator mirror boundaries

Keep upstream AIC Python code/data in
`python/aisimulate/src/aiconfigurator/` and
`python/aisimulate/src/aiconfigurator_core/`. AISimulate-specific compatibility
glue belongs in `python/aisimulate/src/aisimulate_core/`, not in those mirrors.
The corresponding Rust mirror is `crates/core/src/perfmodel/`. See the
repository's [AIC synchronization guide](docs/aic-sync.md) before applying an
upstream AIC commit; packaging and CI changes are adapted manually rather than
mirrored.

### Optional: Install Ruff Extension

If you are using VS Code or one of its forks (e.g. Cursor), you can install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) which will highlight linting issues in your editor. You can also configure your editor to auto-apply formatting when saving files using the instructions [here](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff#:~:text=Taken%20together%2C%20you%20can%20configure%20Ruff%20to%20format%2C%20fix%2C%20and%20organize%20imports%20on%2Dsave%20via%20the%20following%20settings.json%3A).

## Development Workflow

### Code Style and Linting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

#### Run Linting

```bash
# Check for linting issues
ruff check --config python/aisimulate/pyproject.toml python/aisimulate tests

# Auto-fix linting issues
ruff check --fix --config python/aisimulate/pyproject.toml python/aisimulate tests
```

#### Run Formatting

```bash
# Check formatting
ruff format --check --config python/aisimulate/pyproject.toml python/aisimulate tests

# Apply formatting
ruff format --config python/aisimulate/pyproject.toml python/aisimulate tests
```

### Pre-commit Hooks

Pre-commit hooks automatically run checks before each commit.

#### Run Pre-commit Manually

```bash
pre-commit run --all-files --config python/aisimulate/.pre-commit-config.yaml
```

### Running Tests

This project uses [pytest](https://docs.pytest.org/en/stable/) for testing.

```bash
# Run repository-level tests
python -m pytest -c pytest.ini tests

# Run Python package tests
python -m pytest -c python/aisimulate/pytest.ini python/aisimulate/tests

# GitHub PR / build subset (unit + a small stable E2E subset)
python -m pytest -c python/aisimulate/pytest.ini \
  python/aisimulate/tests -m "unit or build"
```

## Data Collection (Advanced)

Data collection is typically not required for development. The repository includes pre-collected performance databases for supported systems.

If you need to collect new data for a new GPU type or framework version, refer
to the [Collector README](python/aisimulate/collector/README.md).

## Contributing

Before contributing, please read:
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines and rules
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community standards

## Common Development Tasks

### Adding a New Model

Refer to [How to Add a New Model](python/aisimulate/docs/add_a_new_model.md).

### Running Automation Scripts

Explore the automation helpers under `python/aisimulate/tools/automation/`.

## Getting Help

- **Documentation**: Check the root `docs/` directory and
  `python/aisimulate/docs/`
- **Issues**: Open an issue on
  [GitHub](https://github.com/ai-dynamo/aisimulate/issues)
- **Examples**: Explore `python/aisimulate/tools/simple_sdk_demo/` for SDK usage
  examples

## License

This project is licensed under Apache 2.0. All contributions must include SPDX license headers and DCO sign-off.
