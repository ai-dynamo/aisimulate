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
| `branch` | The branch whose committed snapshot or qualified campaign evidence is being published. |
| `status` | `evaluated`, `inherited`, `historical`, or `unavailable`, as defined above. |
| `summary_path` | A site-relative `branches/<16 hex characters>/summary.json` path; `null` for unavailable evidence. A direct source preview uses `summary.json`. |
| `published_from_commit` | The full commit from which the snapshot file was copied, or `null` for a qualified campaign artifact or a local build without branch refs. Campaign identity is recorded in `evaluated_revision`. This field is publication provenance, not the evaluated revision. |
| `evaluated_revision` | Required for evaluated/inherited evidence: the producer-recorded `branch` and full `commit_sha`. Absent or `null` for historical/unavailable evidence. |

The exporter, Pages validator, and browser restrict evaluated branch names to
`main` or `release/[A-Za-z0-9][A-Za-z0-9._/-]*`, without a trailing slash, and
commits to 40 lowercase hex characters. Nested release names such as
`release/0.13.0/rc1` are allowed. Evaluated snapshots must include matching bundled AIC CLI provenance;
only historical snapshots may omit it. The browser checks catalog status and evaluated identity against the
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
Pages also consumes validated artifacts from the **E2E Accuracy Matrix** workflow.
That workflow runs daily for `main` and every `release/*` branch, using one exact
amd64 wheel per branch for both predictors. Completed matrix runs trigger Pages,
which publishes only successfully qualified branch campaigns.
The matrix supports at most 255 discovered release branches alongside `main`
and fails explicitly if that limit is exceeded.
The daily Pages build itself only republishes available evidence.

## Automated accuracy campaigns

The workflow is `.github/workflows/e2e-accuracy.yml`, scheduled at **10:17 UTC
daily**. It evaluates the scheduled main SHA and the current head of every
`release/*` branch discovered at the start of the run. Each matrix entry pins its
own branch/SHA and runs the reusable `.github/workflows/e2e-accuracy-branch.yml`
campaign, with at most **two branches at once**. A failed branch does not cancel
other campaigns; its failure remains visible in the matrix run.

Main reuses an available amd64 artifact from a successful build job at the same
SHA, even if Nightly CI is still waiting for staging approval. Otherwise, it builds
one wheel for that revision. Release campaigns build their own exact wheels.
Accuracy runs independently of release staging. Each successful branch can
publish while failed branches retain their previous validated evidence.

Manual execution uses the workflow on **main**, with an explicit evaluated
branch and full source SHA:

```bash
gh workflow run e2e-accuracy.yml --repo ai-dynamo/aisimulate --ref main \
  -f branch=release/0.12.0 \
  -f expected_sha=FULL_40_CHARACTER_COMMIT_SHA
```

The SHA must belong to `main` or the selected `release/*` branch. Manual runs build
one wheel from that revision. Reused nightly wheels require matching checksums
and producer provenance. The main-branch campaign code checks both the installed
legacy CLI and native runtime against that wheel. Historical release revisions
must support these public APIs and the manylinux builder; an incompatible revision
fails without replacing its published evidence.

### Measurement and prediction policy

- `.github/e2e-accuracy-dataset.json` pins the InferenceX release, every compressed
  dump part's size and SHA-256, and selection policy. Refresh it in a reviewed PR
  when adopting new measurements. A nightly reruns predictions against this fixed
  silicon dataset; it does not collect new GPU measurements.
- The downloader verifies every part, decompresses the public PostgreSQL archive,
  and reads only `configs`, `benchmark_results`, and `workflow_runs` via COPY text.
  It never executes SQL from the dump. The September 14 release downloads about
  25 GB and requires at least 35 GB of free temporary disk. Decompression streams
  directly into the serial `pg_restore` reader, avoiding an expanded dump on disk.
  Raw data and child logs
  remain on the runner; they are not uploaded as Actions or Pages artifacts.
- Policy `latest-complete-config-run-v1` selects single-turn, single-node,
  non-offloaded points from completed successful measurement runs, with positive
  mean TTFT/TPOT, at most 30 days older than the
  latest measurement for that model/GPU/framework/precision/serving/speculation/
  workload family. Families without recent measurements retain historical evidence.
  Each topology/workload/recipe uses one latest run and
  a consistent image; missing concurrency points are never borrowed from older
  runs. Duplicate IDs or ambiguous curves fail validation.
- The public InferenceX adapter supplies topology and quantization. Speculative
  configurations requiring acceptance-rate overrides and unresolved recipe
  fingerprints are excluded with counts. Both predictors use the bundled CLI's
  resolved performance-database version. These versions are reported; they can
  differ from the measured server image. Replay pins `max_num_seqs=max(256, concurrency)`,
  `max_num_batched_tokens=8192`, `enable_prefix_caching=False`, and
  `aic_forward_model="op_level"`, with
  seed 0, lengths from 80–100% of nominal, and ten requests per concurrency slot.
  This policy differs from the earlier private campaign's reviewed recipe mapping;
  aggregate differences are not evidence of a runtime improvement.
- Six CPU worker processes execute bounded point predictions (180 seconds each).
  Every selected point must have one outcome. Adapter exclusions and failed AIC
  baselines are counted before forming the comparison cohort. Replay failures in
  that cohort remain chart gaps. Missing/duplicate outcomes, a killed/timed-out
  worker, invalid latencies, or no successful matched predictions fail qualification.
- Accuracy values and coverage are advisory. There is no MAPE threshold or claim
  that a lower aggregate on a different cohort is an improvement. Campaign integrity
  is required for publication.

### Artifact and publication contract

Only `summary.json` and `qualification.json` are uploaded in each
`e2e-accuracy-web-<branch-key>` artifact, retained for 90 days. The branch key is
the first 16 hexadecimal characters of SHA-256 of the branch name; wheel artifacts
use the same key to keep branches isolated. They record the evaluated branch/commit, wheel/dataset/input/
cohort/driver hashes, run and attempt, selected/published counts, exclusions, and
completion time. Public data contains derived errors and normalized curves.

Pages runs trusted main code and accepts a branch artifact only when that branch's
qualification job succeeded in the artifact's exact run attempt. The matrix run
may have failed because another branch failed. Rerunning failed jobs can retain
artifacts from already successful branches; Pages verifies each artifact against
its original attempt's metadata and jobs. Legacy `e2e-accuracy-web` artifacts still
require a successful whole workflow run.

Pages checks the producer event/repository/workflow, branch artifact name, ZIP members,
summary checksum, exact source ancestry, complete coverage, and recursive public
field allowlist. Branch HTML and JavaScript never come from artifacts. Newer
evaluated commits supersede older ones; rerunning an older release commit cannot
roll back a newer snapshot. A missing/expired artifact falls back to that branch's
committed evidence; malformed available artifacts fail the Pages build, preserving
the currently deployed site. Main's legacy JSON download and branch catalog update
together. The page's provenance section links the accuracy run and exposes wheel,
dataset, exclusion counts, and prediction database versions.

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
Its `runtime.cli_entry_point` is `"aisimulate.legacy_cli.entrypoint:main"`, and its `status`
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
  --output pages/e2e-accuracy/summary.json
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
