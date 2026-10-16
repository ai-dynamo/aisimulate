<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Review Parquet changes

Run the commands below from the repository root with the development environment
installed. Git also executes textconv drivers from the repository root, even
when the Git command is invoked from a subdirectory. `origin/main` is the comparison base in these examples.

AIC stores perf tables as compressed parquet files. GitHub cannot render useful
binary diffs for parquet by itself, so use the review helpers in
`python/aisimulate/tools/perf_database`.

For local diffs, install the textconv driver once per clone:

```bash
git config diff.parquet.textconv 'uv run --project python/aisimulate python python/aisimulate/tools/perf_database/parquet_textconv.py'
```

After that, regular git commands show parquet files as CSV-like text:

```bash
git diff origin/main...HEAD -- python/aisimulate/src/aisimulate_core/systems/data
```

For PR review summaries, run:

```bash
uv run --project python/aisimulate python python/aisimulate/tools/perf_database/parquet_diff.py \
  --base-ref origin/main \
  --head-ref HEAD \
  --path-prefix python/aisimulate/src/aisimulate_core/systems/data \
  --output parquet-diff.md \
  --detail-dir parquet-diff-details
```

The summary checks row counts, column names, and Arrow table content hashes.
When a PR replaces `*_perf.txt` with `*.parquet`, the tool compares the new
parquet file against the base branch's legacy text file. A GitHub workflow also
uploads and comments this report on PRs that touch perf data. The artifact
bundle includes `parquet-diff-details/changed-files.csv`, full unified diffs for
every changed perf data file under `parquet-diff-details/diffs/`, and
`parquet-diff-details/summary.csv` with row-level CSV details when the tool can
classify added, removed, or modified rows.
