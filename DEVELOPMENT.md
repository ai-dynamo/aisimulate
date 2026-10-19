<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Development setup and validation

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
[platform and source-build requirements](docs/getting-started/installation.md#use-current-source).

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
repository's [AIC synchronization guide](docs/ci/aic-sync.md) before applying an
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

The [CI guide](docs/ci/README.md) explains the Fast/Full/Nightly workflows, code review,
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

Refer to [How to Add a New Model](docs/perf-model/extending.md).

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

## Slack review digest and weekly PIC

The `Slack review digest` workflow runs every Monday and Thursday at 09:00
`America/Los_Angeles` (including daylight saving changes). It reports:

- Open non-draft PRs, including approved PRs.
- PRs merged or created in the past seven local calendar days, through the run's start time.
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
   pair of Text variables named `message` and `pr_details`, and a **User ID**
   variable named `review_pic`. Add **Send a message to a channel**, select the
   destination, and insert `review_pic` as a mention alongside `message` in its
   body. A literal `<@USER_ID>` in the Text variable will not notify the PIC.
   Then add **Reply to a message in thread**. For the message to reply to,
   select the message output from the preceding send step; insert `pr_details`
   into the reply body. Leave any option to broadcast the reply to the channel
   disabled. Publish the Slack workflow (republish after changing variables).
2. Save its Web request URL (`https://hooks.slack.com/triggers/...`) under repository **Settings → Secrets and variables → Actions**
   as `SLACK_REVIEW_DIGEST_WEBHOOK_URL`. Never commit the URL.
3. Create an Actions **variable** named `SLACK_REVIEW_ROTATION` under
   **Settings → Secrets and variables → Actions → Variables**. It holds the
   ordered roster, start date, and OOO coverage, as shown below. Changes to this
   variable apply to the next run without a code change or PR. Keep the webhook
   URL in its secret; configuration variables are not secret or log-masked.
4. Merge the workflow into the default branch. In **Actions → Slack review
   digest → Run workflow**, leave `dry_run` enabled to preview; disable it
   to send a test message.

No personal GitHub token or Python packages are needed in Actions. The job
uses its read-only repository token. Missing secrets and API errors fail the
job. Runs are not automatically retried; rerunning a sent or partially sent
job can duplicate messages. GitHub schedules may be delayed, so the digest
shows the actual reporting time. Activity after that time is outside the
seven-day report.

The PIC changes every Monday at 09:00 Pacific. `start_date` is the Monday of
roster member 1's first week. The sequence repeats after everyone in `roster`
has served one week. Both reminders in a week use the same PIC. The first member
also covers runs before `start_date`. Local calendar dates preserve handoff time
across daylight-saving changes.

Use this format for `SLACK_REVIEW_ROTATION` (replace the example Slack IDs with
real member IDs):

```json
{
  "start_date": "2026-10-19",
  "roster": [
    {"name": "Harry Lee", "slack_id": "U00000002"},
    {"name": "Hongkuan Zhou", "slack_id": "U00000005"},
    {"name": "Jason Zhou", "slack_id": "U00000003"},
    {"name": "Simone Chen", "slack_id": "U00000001"},
    {"name": "Yongming Ding", "slack_id": "U00000004"},
    {"name": "Yueqi Zhao", "slack_id": "U00000006"}
  ],
  "overrides": {
    "2026-11-02": "Yongming Ding",
    "2026-11-09": "Jason Zhou",
    "2026-11-16": "Simone Chen"
  }
}
```

For OOO coverage, edit only `overrides`: keys are the Monday dates of affected
weeks, and values are exact names from `roster`. The example uses alphabetical
order by first name, with Yongming covering November 2, Jason November 9, and
Simone November 16 so everyone serves once in the first six weeks. Overrides affect both
reminders for those weeks without shifting later weeks. Omit `overrides` or use
`{}` for the normal rotation. Names, roster size/order, and the start date are
all editable in this variable. Changing the roster order or start date
recalculates the current rotation, so use dated overrides for temporary swaps.
Missing configuration, invalid dates, unknown coverage names, or
duplicate/invalid roster entries fail before contacting Slack. A missing roster
does not fall back to an unassigned digest: the published Slack workflow requires
the `review_pic` User ID input.

The Actions variable is the source for Slack's PIC. The
[Linear review project](https://linear.app/nvidia/project/external-pr-review-pic-aisim-51559cd6cfe2)
records the responsibilities and a reference schedule; it is not read or updated
by the digest. Update that reference table when changing coverage. The old
`SLACK_REVIEW_PIC_IDS` secret and `SLACK_REVIEW_PIC_OVERRIDES` variable are no
longer read and can be removed after rollout. Configure and republish the Slack
User ID input before enabling the updated sender.

Local preview (requires an authenticated GitHub CLI and Python 3.9+):
The digest and its offline tests use standard-library Python modules and require
IANA time-zone data for `America/Los_Angeles`, supplied by the operating system
or the `tzdata` package. If system time-zone data is missing, install `tzdata`
with `python3 -m pip install tzdata`. Both files retain a Python 3.9 lint target
in `scripts/pyproject.toml`.

```bash
# Read the editable configuration without printing it.
export SLACK_REVIEW_ROTATION="$(gh variable get SLACK_REVIEW_ROTATION --repo ai-dynamo/aisimulate)"
GH_TOKEN="$(gh auth token)" python3 scripts/notifications/slack_review_digest.py --dry-run
```

Run the focused offline checks with:

```bash
python3 scripts/notifications/test_slack_review_digest.py
```
