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

Install Python 3.11–3.13, `uv`, Rust/Cargo, and a C/C++ compiler plus platform
linker first. Maturin builds the native extension during sync; see the
[platform and source-build requirements](docs/installation.md#use-current-source).

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
- The `aisimulate` package and its `aiconfigurator`, `aisimulate_core`, and
  `aisimulate_core` compatibility namespaces in editable mode
- All runtime dependencies
- Development tools: `ruff`, `pre-commit`, `pytest` and related plugins

## AIConfigurator mirror boundaries

Keep upstream AIC Python code/data in
`python/aisimulate/src/aisimulate/` and
`python/aisimulate/src/aisimulate_core/`. AISimulate-specific compatibility
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

For documentation changes, also run the local-destination check used by Fast
CI from the development environment above (`markdown-it-py` is already included):

```bash
python -m unittest discover -s scripts/ci -p test_documentation_links.py
python scripts/ci/check_documentation_links.py
```

It checks inline/image links and reference definitions in the root README,
development/contribution guides, `docs/`, the application README, and
`python/aisimulate/docs/`. It uses Markdown parsing to ignore code examples and HTML
comments. Diagnostics point to the containing Markdown block. Remote URLs,
heading anchors, and HTML links require separate review. Run the actual
documented commands when their behavior changes; link checks do not validate
examples or establish GPU benchmark accuracy.

```bash
# Run repository-level tests
python -m pytest -c pytest.ini tests

# Run Python package tests
python -m pytest -c python/aisimulate/pytest.ini python/aisimulate/tests

# Quick local subset (Full CI also runs unmarked and integration tests)
python -m pytest -c python/aisimulate/pytest.ini \
  python/aisimulate/tests -m "unit or build"
```

The [CI guide](docs/ci.md) explains the Fast/Full/Nightly workflows, code review,
the complete application test inventory, and manual coverage exceptions. The
local marker subset above does not reproduce all Full CI validation.

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

## Daily Slack review digest

The `Slack review digest` workflow runs every day at 17:07
`America/Los_Angeles` (including daylight saving changes). It reports:

- Open non-draft PRs, including approved PRs.
- PRs merged or created since Pacific midnight, through the run's start time.
  Created PRs count even if subsequently closed or merged, including drafts.
- Every open non-draft PR created more than 5 days (120 hours) ago, oldest first, with
  its link, title, author, and age. Age measures creation time, not inactivity
  or time since leaving draft. PR details appear in a thread reply beneath the summary.
  Lists over 35,000 characters fail before delivery to avoid losing entries.

The summary shows merged PRs (`:merged-2472:`), new PRs (`:pr-opened:`), then
open non-draft PRs labeled "PRs waiting for review" (`:reminder-alarm:`). The destination workspace must have the custom
`merged-2472`, `pr-opened`, and `reminder-alarm` emoji for those names to render as icons.

The Workflow Builder Text variable does not parse Slack markup. Messages use
plain text, emoji, and full clickable PR URLs on separate lines; bold and
named hyperlinks are not supported by this template. An acknowledged trigger
means Slack accepted the request; check Slack workflow activity for delivery
failures in subsequent steps.

To enable delivery:

1. In Slack Workflow Builder, create a **From a webhook** workflow. Add a
   Text variables named `message` and `pr_details`. Add **Send a message to a
   channel**, select the destination, and insert `message` into its body.
   Then add **Reply to a message in thread**. For the message to reply to,
   select the message output from the preceding send step; insert `pr_details`
   into the reply body. Leave any option to broadcast the reply to the channel
   disabled. Publish the Slack workflow (republish after changing variables).
2. Save its Web request URL (`https://hooks.slack.com/triggers/...`) under repository **Settings → Secrets and variables → Actions**
   as `SLACK_REVIEW_DIGEST_WEBHOOK_URL`. Never commit the URL.
3. Merge the workflow into the default branch. In **Actions → Slack review
   digest → Run workflow**, leave `dry_run` enabled to preview; disable it
   to send a test message.

No personal GitHub token or Python packages are needed in Actions. The job
uses its read-only repository token. Missing secrets and API errors fail the
job. Runs are not automatically retried; rerunning a sent or partially sent
job can duplicate messages. GitHub schedules may be delayed, so the digest
shows the actual reporting time. Activity after that time is outside the
same-day report.

Local preview (requires an authenticated GitHub CLI and Python 3.9+):
The digest and its offline tests use only the standard library and retain a
Python 3.9 lint target in `scripts/pyproject.toml`.

```bash
GH_TOKEN="$(gh auth token)" python3 scripts/notifications/slack_review_digest.py --dry-run
```

Run the focused offline checks with:

```bash
python3 scripts/notifications/test_slack_review_digest.py
```
