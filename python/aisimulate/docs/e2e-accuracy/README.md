<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate E2E Accuracy Overview

View the [E2E Accuracy Overview](https://ai-dynamo.org/aisimulate/e2e-accuracy/).
It compares **AISim CLI (new)** and **AIC CLI (legacy)** against measured silicon
so users can assess whether the new CLI is comparable during migration. The AIC
comparison series is temporary and will be removed when the AIC CLI is deprecated.
Successful-point counts matter: AISim errors cover successful engine replays,
while the AIC baseline covers all selected points.

## Branch selection

The Pages build publishes a `branches.json` catalog containing `main` and every
fetched `origin/release/*` branch. Each entry loads the summary committed on that
branch, using the same reviewed UI from main. The browser loads packaged data
from the site, so it does not require access to GitHub's repository API.

The branch containing an artifact and the revision evaluated by that artifact
are separate identities:

- **Evaluated:** the producer recorded a clean checkout, branch, and full commit
  SHA. The page displays that immutable evaluated revision. A snapshot is never
  represented as a live evaluation of the current branch head.
- **Inherited:** a branch contains evidence evaluated on another branch. The
  original evaluated branch and commit remain visible.
- **Historical:** package versions were recorded, but the evaluated branch and
  commit were not. Selecting a branch does not relabel those results as a new run.
- **Unavailable:** the branch has no committed summary. The page shows an empty
  state instead of falling back to main's results.

The existing `summary.json` remains the historical AISimulate 0.12.0 replay
snapshot against the public [InferenceX db-dump/2026-08-24 release](https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-08-24).
This change does not rerun that evaluation or invent branch-specific results.
The measurement release is shown separately from the AISimulate branch selector.

Main-branch changes and manual Pages builds publish immediately after a
successful workflow. A daily main-branch Pages build also picks up release-branch
snapshot updates and newly created release branches. It imports **only JSON**
from release branches, never their HTML or JavaScript. Deleted branches disappear
from the next catalog built with freshly fetched refs.

## Drill down

Expand a workload to see its GPU rows, then select a GPU to open details beside
the matrix (below it on narrow screens). Details include:

- separate AISim CLI and AIC CLI TTFT/TPOT MAPE bars;
- successful replay counts, unsupported points, and failed points;
- a topology selector identifying precision, framework, serving mode,
  speculative method, and parallelism, when exported with the updated builder;
- per-topology TTFT/TPOT curves and a numeric concurrency table, when available;
- a link that preserves the branch, model, workload, GPU, and topology selection.

Curves use latency **relative to the measured value at the lowest concurrency**
within that topology. Measured values and both CLI predictions share the same
anchor, preserving magnitude and shape differences without publishing raw
latencies. Missing predictions remain gaps and explicit statuses. Topologies
are never combined into one curve. Legacy summaries remain usable with GPU
aggregates and explain when detailed evidence has not yet been exported.

These are measurements at specific operating points, not universal support
claims or release gates. The default publication excludes multi-node rows.
Internal run records and exploratory dashboard payloads are not published.

## Regenerate a branch snapshot

The evidence producer supplies matching merged predictions, completed replay
metadata, and a coverage report. For a branch-qualified snapshot, both
`predictions.aisimulate_run.runtime.source_checkout` and
`metadata.aisimulate_run.runtime.source_checkout` must record this identity
**at evaluation time**, from one complete run:

```json
{
  "branch": "release/0.12.0",
  "commit_sha": "<full 40-character evaluated commit SHA>",
  "clean": true
}
```

```bash
python scripts/build_e2e_accuracy_overview.py \
  --predictions /path/to/predictions.json \
  --metadata /path/to/aisimulate_points.meta.json \
  --coverage /path/to/coverage.json \
  --source-url https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-08-24 \
  --branch release/0.12.0 \
  --output python/aisimulate/docs/e2e-accuracy/summary.json
```

Commit the generated summary on the evaluated branch. Use `--branch main` for
main. The builder rejects missing, dirty, mismatched, or mixed incremental run
provenance. Omitting `--branch` preserves the legacy unqualified export path;
it cannot assert branch accuracy. Both export paths include sanitized topology
details. The existing aggregate schema stays compatible.

## Validate and preview

```bash
python -m pytest -c /dev/null tests/test_e2e_accuracy_overview.py tests/test_pages_site.py -q
node --test tests/test_e2e_accuracy_ui.mjs
# Use a fresh output directory. Fetch remote refs first to include releases.
python scripts/build_pages_site.py --accuracy-refs --output-dir /tmp/aisim-site
python -m http.server 8000 --bind 127.0.0.1 --directory /tmp/aisim-site
```

Open `http://127.0.0.1:8000/e2e-accuracy/`. Serving the source documentation tree
directly also works, with a single snapshot when `branches.json` is absent.
