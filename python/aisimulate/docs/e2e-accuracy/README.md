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
The selector explicitly labels **historical only**, **inherited evidence**, and
**no snapshot** entries; an unqualified historical file does not establish that
its containing branch was evaluated.

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

### Catalog contract

`schema_version` is `1`, `default_branch` is `main`, and `branches` contains unique
`main` or `release/<name>` entries. Each entry records:

| Field | Meaning |
| --- | --- |
| `branch` | The branch whose committed snapshot is being published. |
| `status` | `evaluated`, `inherited`, `historical`, or `unavailable`, as defined above. |
| `summary_path` | A site-relative `branches/<16 hex characters>/summary.json` path; `null` for unavailable evidence. A direct source preview uses `summary.json`. |
| `published_from_commit` | The full commit from which the snapshot file was copied, or `null` in a local build without branch refs. This is publication provenance, not the evaluated revision. |
| `evaluated_revision` | Required for evaluated/inherited evidence: the producer-recorded `branch` and full `commit_sha`. Absent or `null` for historical/unavailable evidence. |

The exporter, Pages validator, and browser restrict evaluated branch names to
`main` or `release/[A-Za-z0-9][A-Za-z0-9._/-]*` and commits to 40 lowercase hex
characters. The browser checks catalog status and evaluated identity against the
loaded summary before rendering. Contradictory evidence fails visibly rather
than displaying another branch's results. A missing catalog permits direct
source preview, whose label is derived from the loaded summary itself.

The refreshed `summary.json` evaluates AISimulate main at
[`e46be717175acf06bdbbdeadb7aaf9bb2afdae8d`](https://github.com/ai-dynamo/aisimulate/commit/e46be717175acf06bdbbdeadb7aaf9bb2afdae8d)
against the public [InferenceX db-dump/2026-09-14 release](https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-09-14).
Its legacy AIC baseline uses the `aiconfigurator` CLI bundled in the **same
AISimulate wheel and revision**. The page records the baseline's AISimulate
repository, branch, and commit alongside the replay provenance.
The measurement release is shown separately from the AISimulate branch selector.
Release branches continue to display the evidence committed on those branches.

Both predictions run on remote CPU workers. The AISimulate wheel is built from
a clean source checkout, and the complete campaign records its evaluated branch,
commit, runtime hashes, and input checksum. Replays use the recorded model,
topology, backend version, and reviewed recipe settings where available, with
seed 0, randomized input/output lengths from 80% to 100% of the nominal lengths,
and ten requests per concurrency slot. Unsupported configurations, unresolved
reviewed recipes, and runtime failures remain explicit outcomes.
Replay settings use the public API available on the evaluated commit.

The comparison cohort contains operating points with a successful AIC SILICON
estimate. AISimulate attempts every point in that cohort; the published view
excludes multi-node points. A lower error on a refreshed snapshot does not by
itself prove an improvement on the previous snapshot, because the measurement
release, included points, and successful replay coverage can change.

Main-branch changes and manual Pages builds publish immediately after a
successful workflow. A daily main-branch Pages build also picks up release-branch
snapshot updates and newly created release branches. It imports **only JSON**
from release branches, never their HTML or JavaScript. Deleted branches disappear
from the next catalog built with freshly fetched refs.
This daily publication job does not rerun either predictor; new accuracy results
require a completed prediction campaign and a regenerated summary.

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

All three producer documents must also carry the same completed `aic_run`.
Its `runtime.source_checkout` records the same branch, full commit SHA, and
`clean: true`, plus `repository: "https://github.com/ai-dynamo/aisimulate"`.
Its `runtime.cli_entry_point` is `"aiconfigurator.main:main"`, and its `status`
is `"complete"`. The producer's `aic_commit_sha` identifies that AISimulate
commit. Branch publication rejects a baseline from another repository or
revision, an incomplete baseline, or inconsistent producer documents.

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
