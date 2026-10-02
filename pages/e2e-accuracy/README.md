<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISim E2E Accuracy

View the [E2E Accuracy Overview](https://ai-dynamo.org/aisimulate/e2e-accuracy/).
It compares **AISim** and **AIC (legacy CLI)** against measured silicon
so users can assess whether the new CLI is comparable during migration. The AIC
comparison series is temporary and will be removed when the AIC (legacy CLI) is deprecated.
Successful-point counts matter: AISim errors cover successful engine replays,
while the AIC baseline covers its own successful estimates. Both failures remain
visible in coverage accounting.

## Views and filters

Display names are **AISim** and **AIC (legacy CLI)** throughout the E2E page.
Internal predictor IDs and stored metric fields remain unchanged. Accuracy
cards explain when a snapshot or filter selection has no included predictions;
missing predictions never become zero errors. Research runs without a legacy
CLI evaluation still need that evaluation before branch-qualified publication.

- **Overview(op-based)** shows model/workload/GPU errors, serving/framework
  summaries, and topology drilldowns. Framework summaries group Agg before
  Disagg, with VLLM, SGLANG, then TRTLLM within each group.
  The hardware table below combines Agg and Disagg across models and frameworks,
  showing included AISim point counts and TPOT/TTFT MAPE per GPU SKU. It uses the
  same exclusions and averages individual point errors, not group averages.
- **Details(op-based)** shows one selected topology with model, ISL/OSL, GPU,
  precision, framework, serving mode, and parallelism selectors.
- Each operating point's **InfX CI run** links to its measured silicon run in
  `SemiAnalysisAI/InferenceX` GitHub Actions. The exporter retains the public
  `silicon_github_run_id` as `infx_run_id`; it never uses the dump's internal
  workflow row ID. Older snapshots without this provenance display “—”.
- Both views retain Measured silicon, AISim, and AIC (legacy CLI) series.
  Click a legend to toggle a series; double-click to isolate it. Point markers
  open numeric values and the recorded prediction configuration.
- Branch, topology, exclusions, chart axes, and hidden series are shareable in
  the URL and survive tab switches. Overview retains the selected GPU row.
- Multi-node points are included by default in new campaigns. Optional exclusions
  remove multi-node topologies, anomalous silicon values, or prediction errors
  above 100%. Exactly 100% is retained. Error exclusions apply independently per
  predictor and metric; outliers remain pink in charts. Shape error compares
  curves normalized independently to their first included point.
- Silicon anomalies flag local peaks/dips and lower-concurrency values that
  exceed a later value by more than 5%, within the same topology and metric.
  Exclusions are off by default; they never change the published snapshot.

Charts use milliseconds when recorded. Historical normalized-only snapshots keep
relative curves, and aggregate-only snapshots explicitly disable point filters.
Throughput can show output or total tokens per second per GPU against interactivity,
E2E latency, or TTFT. Measured output throughput stays unavailable when no output
rate was recorded; requested token lengths do not establish measured throughput.
AIC (legacy CLI) total throughput uses the inverse nominal ratio. Replay throughput
uses its recorded token rates. Missing values stay unavailable. Prediction knobs
are not proof that silicon used the same knobs; the point dialog states this.

The exporter also accepts Gym's original `*_tput_per_gpu_output`,
`*_tput_per_gpu_total`, `silicon_e2el_ms`, and `*_request_latency_ms` fields.
These rates are already per GPU and latencies are already in milliseconds.
Explicitly failed AIC predictions remain gaps without dropping the silicon or
AISim point. Re-export from the original prediction records to populate these
charts; rebuilding the UI around a normalized-only summary cannot restore them.
Gym imports without matching branch-qualified producer evidence remain historical
snapshots and must not be labeled as newly evaluated public branch results.

## Local research results

Use the same E2E page to inspect expanded coverage experiments. Export their
recorded predictions with `scripts/build_e2e_accuracy_overview.py
--research-preview --include-multinode` and the usual predictions, metadata,
coverage, source URL, and output arguments. Include both successful and failed
agg/disagg rows. Every row must identify its configuration as `verified` or
`estimated`, and the runtime must record a clean AISim source commit.

The preview labels estimated successes and an unrun AIC (legacy CLI) baseline.
Unrun predictions remain `pending` with missing metrics, and point details show
the configuration evidence tier. This mode cannot take `--branch`, and the
Pages publisher rejects research snapshots. Keep these generated files in the
local preview directory rather than replacing a checked-in qualified snapshot.
When refreshing a preview with a branch catalog, update both `summary.json`
and the selected catalog entry's summary path so the browser loads the new data.

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
`release/0.13.0/rc1` are allowed. Evaluated snapshots must include matching bundled AIC (legacy CLI) provenance;
only historical snapshots may omit it. The browser checks catalog status and evaluated identity against the
loaded summary before rendering. Contradictory evidence fails visibly rather
than displaying another branch's results. A missing catalog permits direct
source preview, whose label is derived from the loaded summary itself.

The refreshed `summary.json` evaluates AISim main at
[`e46be717175acf06bdbbdeadb7aaf9bb2afdae8d`](https://github.com/ai-dynamo/aisimulate/commit/e46be717175acf06bdbbdeadb7aaf9bb2afdae8d)
against the public [InferenceX db-dump/2026-09-14 release](https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-09-14).
Its legacy AIC baseline uses the `aiconfigurator` CLI bundled in the **same
AISim wheel and revision**. The page records the baseline's AISim
repository, branch, and commit alongside the replay provenance.
The measurement release is shown separately from the AISim branch selector.
Release branches continue to display the evidence committed on those branches.

That historical snapshot ran both predictions on remote CPU workers. The AISim wheel is built from
a clean source checkout, and the complete campaign records its evaluated branch,
commit, runtime hashes, and input checksum. Replays use the recorded model,
topology, backend version, and reviewed recipe settings where available, with
seed 0, randomized input/output lengths from 80% to 100% of the nominal lengths,
and ten requests per concurrency slot. Unsupported configurations, unresolved
reviewed recipes, and runtime failures remain explicit outcomes.
Replay settings use the public API available on the evaluated commit.

That historical comparison cohort contains operating points with a successful AIC SILICON
estimate. AISim attempts every point in that cohort; the published view
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

### PR previews for local review

A manual preview runs from an admitted `pull-request/<number>` copy and must
match the open PR's exact SHA, head branch, and repository. It builds that
revision's wheel and runs the same measurement, source-resolution, estimate,
and replay pipeline. It does not publish to Pages.

```bash
gh workflow run e2e-accuracy.yml --repo ai-dynamo/aisimulate --ref pull-request/372 \
  -f branch=simonec/fix-inferencex-ep-gpu-count \
  -f expected_sha=FULL_40_CHARACTER_PR_HEAD_SHA -f preview=true
```

The run produces `e2e-accuracy-preview-<branch-key>` with the summary and
qualification hashes, plus `e2e-accuracy-preview-evidence-<branch-key>-shard-<index>`
with each shard's resolved inputs and per-point outcomes (seven-day retention).
Evidence is saved even when qualification fails, if source resolution completed.
Production runs retain internal shard results and checkpoints; only aggregate
publication artifacts are eligible for Pages. Preview
names and scope markers are rejected by the public artifact importer and site
builder; changing an artifact's name cannot make it publishable.

For local review, download the preview summary artifact and copy the static
files from `pages/e2e-accuracy/` into a separate directory. Replace that copy's
`summary.json` with the downloaded summary and serve it with
`python -m http.server --bind 127.0.0.1 --directory REVIEW_DIRECTORY 8372`.
Do not add a public `branches.json` catalog: standalone preview loading displays
the recorded PR branch and SHA with an explicit preview label. Keep the source
checkout and committed snapshots unchanged.

### Measurement and prediction policy

- Each pipeline resolves the newest published `db-dump/YYYY-MM-DD` release once,
  excluding drafts, prereleases, and unrelated releases. It verifies `SHA256SUMS`
  against GitHub's asset digest and requires every dump part to match the checksum
  list. Missing or inconsistent assets fail the run instead of using older data.
- All branches download the same resolved `e2e-accuracy-dataset` manifest artifact,
  retained for 90 days. The campaign records its release tag and manifest hash.
  Rerunning only failed jobs reuses that manifest; rerunning the whole workflow
  resolves latest again. No new GPU measurements are collected.
- `.github/e2e-accuracy-dataset.json` remains a pinned local reproduction fixture
  and supplies the selection policy, maximum measurement age, and minimum disk
  space. CI replaces its release and parts with the resolved snapshot and requires
  at least the compressed dump size plus 10 GB of free disk space.
- The downloader verifies every part and retries a failed part up to three total
  attempts without redownloading verified parts. It decompresses the public PostgreSQL archive,
  and reads only `configs`, `benchmark_results`, and `workflow_runs` via COPY text.
  It never executes SQL from the dump. The September 14 release downloads about
  25 GB and requires at least 35 GB of free temporary disk. Decompression streams
  directly into the serial `pg_restore` reader, avoiding an expanded dump on disk.
  The extracted measurement tables are shared through a seven-day Actions artifact
  so the four prediction jobs do not repeat the large download and decompression.
  Dump archives and child logs stay on the runner. Raw tables never reach Pages.
- Policy `gym-resolved-config-v2` applies gym's source filters, row deduplication,
  image coherence, and 180-day configuration freshness window. It includes P/D
  and multinode measurements. Every source row is selected or counted as excluded.
  Historical `latest-complete-config-run-v1` artifacts retain their original
  30-day, successful-run, single-node policy; they are never relabeled as v2.
- The repository-only [source resolver](../../scripts/e2e_accuracy/source/README.md)
  joins immutable workflow revisions, reads pinned launchers without executing
  them, verifies reviewed framework defaults, and resolves checkpoint metadata.
  It records the measured framework version separately from the selected
  performance-database version. CI defaults to `configuration_mode: estimated`,
  matching the research preview's `coverage-experiment/1` assumptions after
  verified evidence is applied. The dispatch input can select `verified` to
  exclude missing settings. Reports show verified/estimated counts and per-point
  labels. Conflicts and unsupported mappings remain explicit exclusions.
  Runtime archives are fetched and hash-checked; reviewed, measurement-bound
  observations preserve verified settings when upstream archives expire.
- The public `ResolvedInferenceXSource` adapter supplies the estimate request.
  Replay consumes the same resolved deployment's per-role sequence/token limits,
  block size, memory fraction, KV dtype, prefix caching, chunked prefill, and
  context limit. Request count, length distribution, NumPy sampler, and seed
  come from the resolved workload. Source graph/kernel controls remain evidence
  when the engine does not model them. Assumptions are applied and labeled by
  source preparation before either predictor runs.
- Four independent CI jobs each run two CPU prediction workers (180 seconds per
  point). Selection happens before partitioning; sorted point IDs are distributed
  by stride across shards 0–3, before source resolution. No points are sampled out.
  Every selected point must have one outcome. Estimate and replay run independently:
  a failed baseline does not remove a successful replay. Failed predictions have
  no error value and remain visible in coverage. Missing/duplicate outcomes,
  killed/timed-out workers, or no successful replay fail qualification.
- Full resolved inputs stay in each runner's evidence directory (also retained
  for previews). Completed outcomes are flushed to `results.jsonl` immediately
  and uploaded as internal checkpoints even on failure. Complete shard bundles
  retain per-point metrics and provenance for aggregation. These Actions artifacts
  are not Pages publication inputs; no generated results are committed.
- For a sharded campaign, `cohort_sha256` hashes the ordered list of the four
  resolved-input shard hashes. An unsharded local run retains its direct resolved
  input hash. Compare cohort hashes only with the same partition count and driver;
  matching measurement hashes alone do not establish matching source evidence.
- Accuracy values and coverage are advisory. There is no MAPE threshold or claim
  that a lower aggregate on a different cohort is an improvement. Campaign integrity
  is required for publication.

### Retrying a failed shard

The four jobs use `fail-fast: false`; one failure does not cancel its siblings.
Completed bundles are retained for seven days as
`e2e-accuracy-results-<branch-key>-shard-<index>`. Retry within that window:

```bash
gh run rerun RUN_ID --repo ai-dynamo/aisimulate --failed
```

GitHub reruns failed jobs and their dependent qualification job, reusing successful
jobs' artifacts from the same run. A retry replaces only its shard's bundle.
The failed shard starts its partition again; JSONL checkpoints are diagnostic
records, not automatic resume inputs. A runner loss can also prevent checkpoint
upload, but already uploaded sibling artifacts remain available. If shared inputs
or completed shards have expired, start a new full run.

Qualification requires all four disjoint partitions, every selected point exactly
once, and matching run, commit, wheel, dataset, measurements, driver, and settings.
Earlier successful attempts of the same run are accepted; future attempts and
mixed identities are rejected. The final summary records the qualification job's
current attempt and recomputes statistics from all point results, without averaging
shard summaries. A separate post-prediction check detects missing shard output
even if the runner incorrectly reports the prediction command as successful.

### Artifact and publication contract

Only `summary.json` and `qualification.json` are uploaded in each
`e2e-accuracy-web-<branch-key>` artifact, retained for 90 days. The branch key is
the first 16 hexadecimal characters of SHA-256 of the branch name; wheel artifacts
use the same key to keep branches isolated. They record the evaluated branch/commit, wheel/dataset/input/
cohort/driver hashes, run and attempt, selected/published counts, exclusions, and
completion time. Public data contains serving metrics, derived errors, and normalized curves.

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

- separate AISim and AIC (legacy CLI) TTFT/TPOT MAPE bars;
- successful replay counts, unsupported points, and failed points;
- a topology selector identifying precision, framework, serving mode,
  speculative method, and parallelism, when exported with the updated builder;
- per-topology TTFT/TPOT curves and a numeric concurrency table, when available;
- a link that preserves the branch, model, workload, GPU, and topology selection.

Historical curves use latency **relative to the measured value at the lowest concurrency**
within that topology. Measured values and both CLI predictions share the same
anchor, preserving magnitude and shape differences. Artifacts also retain absolute
latencies and recorded throughput for throughput-versus-latency views. Missing predictions remain gaps and explicit statuses. Topologies
are never combined into one curve. Legacy summaries remain usable with GPU
aggregates and explain when detailed evidence has not yet been exported.

These are measurements at specific operating points, not universal support
claims or release gates. The scope bar reports whether the selected snapshot
includes multi-node rows. Each predictor's errors cover its successful points;
source exclusions remain visible in campaign provenance. Internal run records
and exploratory dashboard payloads are not published.

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
Its `runtime.cli_entry_point` is `"aisimulate.legacy_cli.entrypoint:main"` or
`"aiconfigurator.main:main"` for 0.12 wheels. New producer records include the
matching `baseline_api` and `config_adapter`; the public `aic_source` retains
`cli_entry_point`. Both the site builder and browser validate the supported
entry points, and the provenance panel displays the recorded value. Historical
summaries without this field remain readable. Its `status`
is `"complete"`. The producer's `aic_commit_sha` identifies that AISim
commit. Branch publication rejects a baseline from another repository or
revision, an incomplete baseline, or inconsistent producer documents.

```bash
python scripts/e2e_accuracy/build_e2e_accuracy_overview.py \
  --predictions /path/to/predictions.json \
  --metadata /path/to/aisimulate_points.meta.json \
  --coverage /path/to/coverage.json \
  --source-url https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-08-24 \
  --branch release/0.12.0 --include-multinode \
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
node --test tests/test_e2e_accuracy_ui.mjs tests/test_e2e_accuracy_workflow.mjs
# Requires Playwright and Chromium:
python scripts/check_e2e_accuracy_browser.py
# Use a fresh output directory. Fetch remote refs first to include releases.
python scripts/pages/build_pages_site.py --accuracy-refs --output-dir /tmp/aisim-site
python -m http.server 8000 --bind 127.0.0.1 --directory /tmp/aisim-site
```

Open `http://127.0.0.1:8000/e2e-accuracy/`. Serving the source documentation tree
directly also works, with a single snapshot when `branches.json` is absent.

### Release replay compatibility

The shared evaluator uses the default op-level timing in both legacy release
wheels and current runners. It omits the newer `aic_forward_model` engine
argument because release/0.12.0 and release/0.12.1 do not accept that field.
Validate evaluator changes against actual release wheels as well as main;
a successful main-only campaign does not establish release compatibility.

## Adapter parity audit (2026-10-02)

The adapter and public CI are **not equivalent to recipe-resolved e2e-gym**.
This audit compared AISimulate PR #372 (`b78fb6bdadf87e013fc0a89f5277024ac888a294`)
with e2e-gym main at `2ad1ec4287ad9fb5857680901eb65b4ff0cb4098`.
The local September 28 dump contains 2,564 configs, 85,309 benchmark rows, and
1,238 workflow runs. This is a mapping/selection audit, not a latency rerun.

| Finding | Resolution |
| --- | --- |
| vLLM attention-DP was ignored outside the single-node EP special case, including P/D roles. | Honor the flag for every vLLM worker; preserve worker width and per-rank decode batch. |
| MiniMax-M2.7, Kimi-K2.6, and Kimi-K3 aliases were missing. | Map existing registered models and preserve native FP4 quantization. |
| DB exports use `id`, while provenance read only `config_id`. | Preserve either source ID. |
| Selection and publication trusted the inflated TP × EP count. | Select candidates using the shared width and require the evaluated adapter to confirm it. |
| Replaced/older curves disappeared without exclusion counts. | Count `superseded_curve`; retain every input row in the accounting. |

On the same dump, selection increases from **4,130 to 4,838** points. Benchmark
431553 / config 202 from AIC-2019 changes from excluded to selected and adapts to
four physical GPUs. All 85,309 inputs now equal selected points plus exclusions:
33,140 incomplete runs, 6,104 multinode, 2,261 nonstandard/error, 28,024 stale,
and 10,942 superseded. Adding valid recent EP evidence also moves some families'
freshness cutoff; the difference is not simply a union of the old cohort and EP rows.
The [October 2 accuracy run](https://github.com/ai-dynamo/aisimulate/actions/runs/36995581041)
independently confirms the old 4,130-point selection and the same filter counts.
Its main artifact publishes 1,137 baseline-success points, with 2,468 adapter
exclusions, 218 recipe-required exclusions, and 307 baseline failures. It evaluates
`c62a88ea2ddfcf5a4b85902b1b2f8095f79f0954`, before these corrections.

A differential check against gym's older `mapping.py` helper used its 180-day
window and 2,209 retained points. Of those, 209 were rejected by that helper.
Before these fixes, 507 accepted mappings differed and seven Kimi-K2.6 points
were rejected only by AISimulate. Afterward, those seven adapt and 306 attention-DP
mismatches disappear. The remaining 201 differences are non-vLLM single-node EP
rows for which the older helper still trusts TP × EP. They must not be copied
back into AISimulate. Gym's production `predict.py` instead requires
`deployment.py::resolve_deployment` and immutable recipe evidence.

### Source-resolved policy follow-up

PR #372 replaces the fixed CI settings with the gym source-resolution path:

- **Serving configuration:** pinned launchers, verified framework defaults, and
  available runtime evidence resolve per-role serving controls. Config 202's
  FP8 KV, 0.85 memory fraction, and 16,384-token prefill budget reach replay.
- **Model identity:** checkpoint metadata and quantization evidence are retained;
  estimate and replay materialize the same verified model-config bytes. Unknown
  mappings remain unsupported. Unverified historical checkpoint revisions stay
  explicitly marked in evidence.
- **Versions and workload:** source framework versions and performance-database
  versions remain separate. Replay uses the source request count, length ratio,
  NumPy sampler, and benchmark seed.
- **Cohort and coverage:** the default is gym's 180-day cohort, including P/D and
  multinode evidence. Baseline and replay failures are recorded independently.

On the same September 28 dump, v2 selects exactly the same **2,281 measurement
IDs** as gym revision `2ad1ec4287ad9fb5857680901eb65b4ff0cb4098`. A live differential
check matched complete resolved deployment/evidence outputs for configs **202,
618, 909, and 1553**, including 618's unresolved-knob result. Contract tests cover
all three backends and both serving modes; small native SOL replay smoke tests
completed for vLLM, SGLang, and TensorRT-LLM.

These checks establish input/projection behavior, not measured latency accuracy.
PR tests do not run the scheduled accuracy matrix or replace the published
snapshot. Runtime artifact caches are optional inputs; CI does not fetch them
automatically. Missing dynamic capacity remains unresolved rather than invoking
gym's optional old-revision capacity estimator. Unmodeled graph/kernel and client
behavior remain simulation limitations, as in gym's replay projection. See the
[resolver boundary](../../scripts/e2e_accuracy/source/README.md#campaign-boundary).

Input SHA-256 values for reproduction:

- `configs.json`: `b6071d66377c4762bdac9661077205ac394538b71ce00d24e4761cb4b87223e0`
- `benchmark_results.json`: `e363f2061efbea87ba0d2dd38f765ddd4aabf3aac30e5e0bf0fa6d8ac3df6c10`
- `workflow_runs.json`: `6a86eb6b31e958a17a7a19c61889910cdd1e7808d8dbcc8fd612ab20a34e6310`


## Serving metric artifact contract

New campaigns declare `metric_contract: serving-metrics-v1` in both the summary
and qualification record. Each operating point contains `measured`, `aic`, and
`aisimulate` series with:

- `ttft_ms`, `tpot_ms`, and `e2e_ms`: mean request latencies in milliseconds.
- `interactivity_tok_s`: `1000 / tpot_ms`, in tokens/s/user.
- `output_per_gpu` and `total_per_gpu`: output and input-plus-output throughput,
  in tokens/s/GPU, normalized by the complete physical deployment GPU count.
- `unavailable_metrics`: a map from missing fields to `not_recorded`,
  `prediction_failed`, or `unsupported_by_predictor`.

Replay records native `mean_e2e_latency_ms`, `output_throughput_tok_s`, and
`total_throughput_tok_s`. The AIC baseline records native `request_latency` and
`tokens_per_second`; total-token throughput is unavailable because that API
exposes no native total-token rate. Do not reconstruct throughput or mean E2E
latency from nominal token lengths and average TTFT/TPOT. Measured values are
preserved when present, with seconds converted to milliseconds as needed.

Qualification rejects successful predictions missing TTFT, TPOT, E2E, output
throughput, or interactivity; replay must also include total throughput. Failed
predictions retain their status and null metrics. Optional missing measurements
and unsupported AIC total throughput remain explicit gaps. Historical artifacts
without this contract still load, but do not establish throughput coverage.

These metrics are written to CI artifacts, not committed evaluation JSON.
Pages packages qualified artifacts during production builds; the existing
committed snapshot remains a historical fallback. Chart presentation lives in
the separate op-based UI change (#368).

### Public InferenceX run links

Prediction rows carry the source workflow's `github_run_id` as
`silicon_github_run_id`; summary points export it as `infx_run_id`, a positive
integer string. The details UI links it to the public InferenceX Actions run.
Both resolved-source and legacy selection paths preserve this provenance,
including failed predictions. Missing historical IDs remain null; internal
`workflow_run_id` database keys are never substituted for GitHub run IDs.
The exporter and Pages validators reject malformed IDs. This metadata does not
change predictions, cohort selection, or accuracy metrics.

The selection toolbar uses short parallelism labels without repeating framework,
precision, or serving filters. A short topology ID is shown only when needed to
distinguish otherwise identical choices. The compact evidence line keeps branch,
revision, evaluation date, and failed-update state visible. Detailed migration
notes, filter methodology, and provenance are under the collapsed **About this
comparison** section below the charts.

Detail charts use solid lines for measured silicon and dotted lines for both
AISim and AIC (legacy CLI) predictions. Legend samples match the chart lines.
The three charts share an aligned card grid on desktop and stack on smaller
screens. Throughput controls stay inside their chart card.
