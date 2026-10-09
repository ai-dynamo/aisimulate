<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Accuracy validation and publication

Keep three evidence lanes separate: strict-native FPE support qualification,
whole-forward FPM prediction accuracy, and end-to-end serving accuracy. A
successful query is not an accuracy result; an accuracy result applies only to
the measured model, hardware, backend, topology, and workload.

User-facing coverage is described in [performance-model support](../perf-model/support-matrix.md)
and [Replay features](../replay/features.md). This page owns the jobs, provenance,
publication, and notifications behind those views.

## Prediction regression gate

[The workflow](../../.github/workflows/prediction-regression-gate.yml) compares
the base and candidate predictions using the same controller and inputs. Keep
numerical regression checks separate from measurement-based accuracy: agreeing
with the previous version does not establish agreement with silicon. The active
tooling is under [scripts/prediction_regression](../../scripts/prediction_regression/).
Use the workflow's current admission, sampling and qualification policy rather
than the retired implementation rollout document.

## E2E accuracy campaigns

The independent [E2E Accuracy Matrix](../../.github/workflows/e2e-accuracy.yml)
evaluates the scheduled main SHA and every discovered `release/*` head against
the latest published InferenceX database dump. The resolver freezes one manifest
with verified part checksums for all branches and retains it as a 90-day artifact.
A branch matrix runs at most two campaigns
at once, with separate wheels, artifacts, and provenance. Main reuses a qualified
amd64 wheel for its exact SHA when available, even while nightly staging waits for
approval; otherwise it builds one. Each release builds its own wheel. Both the
AISimulate replay and bundled legacy AIC baseline use their branch's wheel. Manual
runs evaluate one explicit `main` or `release/*` SHA from the trusted main workflow.

Campaign jobs use the configured `CI_JOB_CONTAINER_IMAGE`, Python 3.12, and
`sudo` to install the PostgreSQL 18 client from the official PGDG repository,
plus `zstd` and `libgomp1`. The runner executes container steps as a non-root
user, so a bare PostgreSQL image cannot install these dependencies with `apt-get`.
The evaluated wheel's baseline CLI and config adapter are selected from its own
package layout: `aiconfigurator.*` on 0.12 releases or `aisimulate.*` after the
namespace migration. Ambiguous wheels with both baseline APIs are rejected.
Baseline provenance records the selected API, adapter, and actual console entry point;
the public summary retains that entry point. Wheel byte checks and imports run before downloading measurements.
The client reads the dump without starting a database server. Artifact upload
uses the output directory so the runner container hook remaps the full path.

Each branch prepares measurements once and runs four independent prediction
shards, with two CPU workers per shard and `fail-fast: false`. Results and
incremental checkpoints are retained as internal Actions artifacts for seven days.
`gh run rerun RUN_ID --failed` reuses successful shard artifacts from that run;
only failed partitions repeat. A separate qualification job verifies full coverage
and matching provenance before combining per-point results. Preview runs also
retain resolved source evidence per shard. Actual concurrency depends on runner
capacity (up to eight shard jobs across the two admitted branches).

Complete campaigns upload sanitized `e2e-accuracy-web-<branch-key>` artifacts.
Pages validates the branch's successful qualification job in the artifact's exact
run attempt, producer, revision, coverage, and checksums before combining it with
qualified FPE data. One branch's failure does not cancel other campaigns or block
their publication. Retried jobs preserve successful branches' original run-attempt
provenance. Failed campaigns retain previous evidence. Accuracy numbers are advisory;
incomplete campaigns cannot publish. A release selector may still show a historical
snapshot until that release has a qualified campaign. See the
[accuracy campaign contract](../../pages/e2e-accuracy/README.md)
for pinned scheduler settings, measurement selection, and provenance.

### Daily accuracy Slack report

[Accuracy Slack Daily](../../.github/workflows/accuracy-digest.yml) combines the day's
scheduled E2E and FPM results in one message in **#swdl-dynamo-aisim-daily**.
Failures and comparable regressions appear in that message; separate per-model
and per-GPU E2E tables appear in the main message; FPM coverage and comparison
details remain in its thread. Delivery waits for both pipelines,
with a 09:00 America/Los_Angeles fallback. It is opt-in and has a default dry-run
mode plus an explicit test-send mode. See [setup, comparisons, and testing](#daily-accuracy-report).


## FPM accuracy

`FPM Accuracy Matrix` runs at 10:47 UTC daily and supports manual evaluation of
an exact SHA on main or a release >= 0.12.0. It pins HF data once per campaign,
uses verified exact wheels, and evaluates FPM with KV warmup on/off and regression
on CPU. Branch results remain in Actions artifacts for 90 days. Pages validates
and publishes successful branch results independently; failed refreshes retain
the prior qualified result. Accuracy is advisory, outside PR prediction campaigns
and release staging gates. See [FPM details](../../pages/fpm-accuracy/README.md).

Manual dispatch accepts optional `hf_revision`, a full 40-character dataset commit.
Use it to evaluate a migrated dataset branch with compatible preview consumers
before promoting either change to main. Scheduled runs continue to resolve HF main.
The workflow verifies the requested revision through HF and records the resolved
commit in every artifact. Publish compatible consumers before moving HF main to a
new manifest contract; historical evaluation artifacts are never rewritten.


## FPE support qualification

Native op-level probes qualify the measured query surface; they do not certify Replay, Sweeper, or prediction accuracy.

## Run it

Install a built repository wheel, then generate one op-level FPE shard per
system. The command imports that installed native runtime; it must not prepend
the source-only application package to Python's import path:

```bash
python python/aisimulate/tools/support_matrix/generate_fpe_support_matrix.py \
  --output-dir fpe-support-matrix/b200_sxm \
  --system b200_sxm \
  --forward-model op_level \
  --max-workers 8
```

After all system shards finish, build the split CSV files consumed by the
interactive page:

```bash
python python/aisimulate/tools/support_matrix/build_fpe_support_matrix.py \
  fpe-support-matrix \
  --output-dir python/aisimulate/src/aisimulate_core/systems/fpe_support_matrix
```

Use filters and a deterministic topology cap for a focused smoke run:

```bash
python python/aisimulate/tools/support_matrix/generate_fpe_support_matrix.py \
  --output-dir fpe-support-matrix-smoke \
  --model Qwen/Qwen3-8B \
  --system b200_sxm \
  --backend vllm \
  --forward-model op_level \
  --max-topologies-per-role 1 \
  --max-workers 2
```

Each raw shard contains deterministic JSON, CSV, and Markdown coverage
artifacts plus a separate `run_metrics.json`. The rollup produces mode-neutral
rows for the separate FPE Support Matrix and adds real probe counts, topology
counts, native status counts, phase latencies, and source SHA to the detail view.
Wall time, CPU time, peak RSS, and worker count remain separate from the
deterministic coverage artifacts.


## Parallel execution and memory

The runner opens a bounded thread pool for one system/backend/version group at
a time. Native database loading and hot-path queries release the Python GIL,
but Python-backed model compilation and database memory still limit scaling.
Increase `--max-workers` only with measured memory headroom.

Main branch nightly CI calls the reusable FPE workflow after confirming that `main` has
changed and building its release artifacts. All FPE shards install the exact
amd64 nightly wheel, verified against the artifact checksums, source commit,
and one recorded wheel hash. A manual run requires the full `expected_sha` and
builds one shared wheel. Neither path rebuilds the native runtime in every
shard.

The workflow discovers one shard per curated system/backend pair and runs at
most 20 shards concurrently on the repository-specific AMD64 CPU runner set.
Other CI runs share this pool, so runner availability can reduce concurrency.
Each shard runs only `forward_model=op_level` with an eight-thread local pool. A
final job validates reports before combining them into the split web CSV
artifact. Qualification requires every discovered shard, exact source and wheel
identity, consistent package version and workload, complete role-appropriate
phases, and no unexpected build or query failure. The small required-probe
manifest additionally requires all four phases of at least one native topology
for each known-good model/system/backend identity. An empty or entirely
unsupported report cannot satisfy that requirement. Classified exploratory
coverage gaps remain visible; they do not certify support.

`fpe-qualification.json` records the accepted source, wheel digest, shard count,
and status counts. FPE qualification gates the later GitLab security handoff,
not initial Artifactory staging. The FPE
workflow remains manually dispatchable for an out-of-band refresh. This avoids
leaving runners idle when a small system finishes before the largest systems.
Full runs also suppress repeated SDK warnings at the console while preserving
every classified failure and representative error in the matrix artifacts.
Refresh-time claims must name both runner concurrency and per-runner thread
count, plus the source SHA from the measured run.


## Website publication

GitHub Pages rebuilds after a successful FPE Support Matrix, Main branch nightly CI,
or Release branch nightly CI run and on public-documentation changes. For main, every
deployment selects the retained qualified FPE artifact with the newest tested
source commit in the current main history.
Re-running an older commit cannot displace a newer qualified snapshot. The page
shows the snapshot's source SHA, artifact creation time, and producing CI run.

Pages checks the producing workflow, successful main-branch run, qualification
manifest, row source identities, and complete shard count. It copies only the
indexed CSV data; HTML and scripts always come from main. Pull request previews
use repository data and cannot deploy. If no eligible artifact remains, Pages
fails and preserves the existing website instead of republishing stale committed
data. Run **FPE Support Matrix** on main with the current full main SHA, then
rerun **GitHub Pages** if needed. Web artifacts are retained for 90 days, subject
to the repository's retention policy. Artifacts generated before the qualification
manifest was introduced cannot be published by this path.

Staging waits for the complete matrix. Each shard has a 480-minute timeout;
the 20-shard concurrency limit and runner queues can make the total wait
longer than that per-shard limit. A successful wheel build alone does not make
the nightly available. No release-latency percentile is promised until complete
runs have been measured with this gate enabled.

The final platform-wheel check invokes the package verifier with
`--exercise-engine --exercise-fpe` on each runner. The FPE flag requires the
repository generator and Git checkout; its subprocess runs from an unrelated
temporary directory against the installed wheel. The reduced Docker build
context runs the package/runtime verifier without the repository-only FPE flag.



## Main and release branches

The published matrix's **Branch** selector defaults to `main` and lists the
repository's `release/*` branches. Share a selection using
`?branch=release%2F0.12.0`; the model search (`q`) is preserved when switching.
Each selection loads a separate packaged dataset with its own tested source
SHA, timestamp, and evidence link. Release results never fall back to main's data.

Pages discovers release branches from the fetched `origin` refs. For each
branch, it selects the newest retained qualified artifact with a source SHA in
that branch's history. Eligible producers are successful runs of
`fpe-support-matrix.yml` or `nightly-ci.yml` on the selected branch, or the
main-hosted release workflow described below with explicit release and tooling
provenance. An old-commit rerun cannot displace a newer tested commit.
A release without retained qualification is labeled **unavailable**; an
expired release artifact also removes its data from the next deployment.
The page shows a **Results not available yet** notice with a **Check again**
button and a link to the release nightly runs. Coverage appears after a
successful qualified nightly run and Pages deployment.
Malformed qualification fails the deployment. Main still requires a retained
qualified snapshot before the site can deploy.

**Release branch nightly CI** runs daily at 09:23 UTC from trusted `main`, and can
also be dispatched on `main`. It discovers every fetched `release/<version>`
branch and records its current commit SHA before building. New branches such as
`release/0.13.0` join the next run automatically, without a workflow edit or
backport. Versions may contain letters, digits, dots, underscores, and hyphens,
starting with a letter or digit. An empty inventory skips qualification.

The scheduler calls the same reusable qualification workflow for each release,
one release at a time. It builds one wheel from each unmodified checkout and uses
the release's locked dependencies, curated model inventory, SDK, estimator,
model definitions, and performance tables. Up to 20 system/backend shards run
concurrently with eight probe threads each; the runner pool is shared with
other CI. The serial release matrix keeps this limit at 20 across the release
nightly run. A failed release does not cancel the remaining releases, but
publication requires the entire nightly run to succeed. Every scheduled run
refreshes the evidence, even if the release SHA is unchanged, so retained
artifacts do not silently expire.

The probe harness and required-probe manifest come from the workflow's exact
`main` commit. CI records that tooling SHA separately from the tested release
SHA and wheel digest. Before discovery or probing, it verifies the installed
package bytes and import locations against the shared wheel. The release branch
does not need a workflow backport. This job produces qualification evidence;
it does not stage or publish release packages.

After a release's shards and required probes pass, CI uploads its own
`fpe-support-matrix-web-release-<version>` artifact for 90 days. Wheels and raw
reports also have version-specific names, so releases in one run cannot mix
data. A successful nightly run triggers Pages. The publisher verifies the
trusted producing workflow, its main-history tooling commit, the artifact's release identity,
and the tested source's membership in release history. It ranks snapshots by
tested source history, then artifact ID, while preserving the existing failed
rerun and expired-artifact protections. An older-source rerun cannot replace a
newer tested source.

Release results come only from GitHub Actions artifacts. No manual ZIP,
committed-result fallback, or main-data fallback is used. Until the first
qualified release run succeeds, the release selector shows **unavailable**.
The page links to the CI run and shows both tested source and probe tooling.

To reproduce a release result, check out the recorded tooling commit at the
workspace root and the recorded release source under `release-source/`.
Download the run's `fpe-release-wheel-<version>` artifact into the workspace (preserving
its `fpe-release-wheel/` directory and `fpe-release-shards.json`), install the
release's locked environment and exact wheel as in the workflow. The matrix's
per-cell command uses `release-source/python/aisimulate/.venv/bin/python`
directly to run the verified release wrapper with the recorded source, tooling,
and branch. Each invocation requires a fresh `release-probe-harness/` directory;
remove only that generated directory between reproductions. Raw shard reports
and the wheel are retained for seven
days; the smaller qualified web dataset is retained for 90 days.

The deployed `data/fpe-support-matrix/branches.json` catalog lists available
and unavailable branches. Main retains the existing data path, and release
data lives under `data/fpe-support-matrix/branches/release/<version>/`.
Repository previews package main's committed snapshot only and require no
GitHub credentials or artifact downloads.



## Legacy AIC snapshot provenance

The Legacy AIC Support Matrix displays a **Historical snapshot** and
**Qualification not recorded**. Its **Latest data change** timestamp is the
commit time of the most recent change to its index or an indexed CSV, with a
link to that data commit. It is not the website build time or evidence of a
complete matrix rerun. No full-matrix generation time or qualification report
was recorded for the retained legacy data.

The Pages builder adds this provenance to the packaged
`data/support-matrix/index.json`. Website-only changes do not refresh the data
date. A build with modified/untracked data, a shallow Git history, or no Git
history leaves the date unavailable. Direct source-tree previews also show the
missing-date state because the committed legacy index contains no provenance.

To reproduce the browser checks, install Chromium with
`uv run --python 3.12 --with playwright playwright install chromium`, then run
`uv run --python 3.12 --with playwright python scripts/pages/check_legacy_support_matrix_browser.py`.
The script builds a temporary site and verifies real data, commit links, missing
and malformed metadata, calendar-date boundaries, and unchanged matrix rows.
Use `--browser-executable /path/to/chrome` to reuse an installed browser, or
`--screenshot /path/to/preview.png` to capture the real page before test fixtures.

## Daily accuracy report

`Accuracy Slack Daily` posts one daily top-level message to
**#swdl-dynamo-aisim-daily** (`C0BULBSTXJ6`). Alerts and results share that message;
E2E per-model and per-GPU tables are included in the main message. No mentions are
sent. The bot does not accept commands or start evaluations.

The message title is `:rainbow: *Accuracy Daily · YYYY-MM-DD*`, with a rainbow
emoji and bold text, without a timezone suffix. Immediately below it, a quote
block shows run links with duration and attempt, overview links, and short
alert/recovery summaries. The alert count and comparison-note count appear at
the end of the main message.

## Delivery policy

- Select today's scheduled E2E and FPM runs by their creation date in
  `America/Los_Angeles`. A rerun uses its latest attempt; successful branch
  artifacts retained from an earlier attempt are validated against that attempt.
- Send when both runs finish, including failures/cancellations. Completion of
  either producer wakes the notifier. A scheduled fallback checks at 09:00 local
  time (UTC 16:00 in summer, 17:00 in winter) and reports missing/unfinished runs.
  GitHub schedules can be delayed; 09:00 is the intended cutoff, not a wall-clock SLA.
- Each main message includes both pipeline links. An unstarted pipeline links to
  its workflow. Failed jobs include their job links and failing step names.
- Include every branch evaluated in those runs. Do not substitute yesterday's
  data for a missing branch. Manual campaigns never enter production daily reports.
- After delivery, late completions do not update the frozen report or create
  another daily message. Subsequent daily reports can announce recovery.
- Serialize notifier runs. Identify production messages using Slack metadata
  keyed by repository and local date. Persist the frozen report before posting;
  on retry, restore it and send only missing thread parts. Rechecks at 30-minute
  intervals through 18:30 UTC can resume an interrupted delivery. Do not delete
  the bot's messages or its retained report artifacts; they are delivery records.
- History/thread reads use GET query parameters and omit empty pagination cursors.
- Slack timeouts and `ok:false` fail the notification job. An uncertain POST is
  not blindly retried: the next run checks Slack history/thread metadata first.
  An extended Slack/GitHub outage may prevent delivery; no system can guarantee
  a notification while its delivery service is unavailable.

## Report contents and comparison

The E2E table has six columns: Branch, Overall, vLLM, SGLang, TRT-LLM, Coverage.
Each accuracy cell is **TPOT%/TTFT% MAPE** (for example `24.32%/43.67%`)
for AISimulate, not legacy AIC.
Main-message detail tables independently group by model and by GPU, within each branch,
combining frameworks. They average successful point errors directly; they do
not average already-aggregated MAPEs. Coverage is predicted / eligible points,
not all original source measurements. FPM shows KV-warmup on/off and online
regression MAPE, weighted by successful prediction counts. FPM coverage is in a
thread reply to keep the main table narrow. The parent uses single-line spacing,
shows at most three short alert summaries, and puts comparison notes and full
alert details in the thread. Every numerical MAPE value includes `%`.

For each branch:

- Start with the most recent earlier qualified scheduled campaign in the retained
  90-day history. Download only the requested branch's artifact and comparison
  evidence during this search. Without one, explicitly report an initial baseline.
- Require identical measurement revision/content and evaluation-rule fingerprints.
  The rule fingerprint covers evaluator source and its branch workflow at the
  producer revision, not the evaluated AISimulate package revision. The FPM
  fingerprint includes `scripts/notifications/accuracy_digest.py`, which supplies its point
  codec. A change
  establishes a new baseline and reports comparison unavailable, not recovery.
- Compare each E2E topology/configuration and FPM configuration/predictor on the
  intersection of successfully predicted points. Alert when MAPE increases by
  **at least 2 percentage points AND at least 10% relative**. From zero, an
  increase of at least 2 points qualifies.
- A previously successful point becoming failed, unavailable, or absent alerts
  separately, even if total coverage is unchanged because other points improved.
- Keep the pre-alert baseline while the regression persists. Do not silently
  accept yesterday's regression as today's normal. Successful comparable results
  against that baseline can be marked recovered in a later daily message.
- Initial prediction failures are shown as attention notes; without prior evidence
  they are not labeled new regressions. A green pipeline does not imply every
  prediction succeeded.

Aggregates alone cannot detect exchanged successful and failed points. The FPM
producer writes a separate
`fpm-accuracy-comparison-<branch-key>` artifact containing a hash of observation order
and compressed little-endian float64 percentage-error sequences (-1 for unsuccessful predictions). It binds
to the public summary checksum and exact producer snapshot. Reruns overwrite
both public and comparison artifacts so their attempt identities stay aligned. The notifier verifies
counts and means against that qualified summary. No raw measured/predicted
latencies are exported. Existing Pages artifacts remain unchanged. Old FPM runs
without point evidence can be displayed, but point regression checks are explicitly
unavailable until a comparable pair with evidence exists.

## Slack setup

GitHub Actions runs the notifier; the Slack app supplies its bot identity. No
server, Event Subscriptions, Socket Mode, or slash commands are needed.

At https://api.slack.com/apps, choose **Create New App > From scratch**, name it
`AISimulate Daily`, and select the workspace containing the daily channel.
Under **OAuth & Permissions > Bot Token Scopes**, add:

- `chat:write` to post the main message and replies.
- `channels:history` to check prior messages and resume threads in this public
  channel. If using a private channel later, use `groups:history` as well.
- `metadata.message:read` to read daily-report and thread-part metadata.

Click **Install to Workspace** (or request workspace approval), then copy the
**Bot User OAuth Token** beginning with `xoxb-`. After changing scopes on an
installed app, reinstall it to apply the new permissions.

In **#swdl-dynamo-aisim-daily**, open **Channel details > Integrations > Add apps**
and add `AISimulate Daily`. Store its bot token only as repository
secret `SLACK_ACCURACY_BOT_TOKEN`. Configure repository variables:

| Variable | Value |
| --- | --- |
| `SLACK_ACCURACY_CHANNEL_ID` | `C0BULBSTXJ6` |
| `SLACK_ACCURACY_ENABLED` | `true` only after the test below passes |

The previous `SLACK_ACCURACY_WEBHOOK_URL` and Slack Workflow Builder trigger
are not used by this version. Messages use Slack `mrkdwn`: bold headings,
monospaced tables, and named links. This is not GitHub-flavored Markdown; tables
are rendered inside code blocks rather than as Markdown table syntax.

Both test and production use this same channel. Test messages are labeled
`[TEST]` and use separate metadata, so they do not consume the daily production
slot or update its baseline. Automatic delivery is disabled unless explicitly
enabled. GitHub permissions are read-only (`contents:read`, `actions:read`).
The notifier checks out its own trusted workflow revision; it never executes
code from downloaded artifacts. Pages validation remains strict; explicit manual
preview/test alone permits same-repository producer runs from a non-main branch.

## Dry run and one real Slack test

Open **Actions → Accuracy Slack Daily → Run workflow**. Choose `dry-run`
(the default), and provide completed `e2e_run_id` and `fpm_run_id` values. Manual
accuracy runs are allowed here. For example:

```bash
gh workflow run accuracy-digest.yml --ref main \
  -f mode=dry-run -f e2e_run_id=<e2e-run-id> -f fpm_run_id=<fpm-run-id>
```

For a local preview, run the same Python entrypoint from the checkout with a read-only GitHub credential:

```bash
GH_TOKEN="$(gh auth token)" python3 scripts/notifications/notify_accuracy.py \
  --e2e-run-id <e2e-run-id> --fpm-run-id <fpm-run-id> \
  --output /tmp/accuracy-preview
```

Inspect the Actions Summary and `accuracy-preview` / `accuracy-report` artifacts,
or local `preview.md` / `report.json`. The JSON includes alert reasons, comparison
limitations, and the exact baseline snapshots. Dry run does not read Slack
credentials, post messages, or write production baseline state.

After configuring the bot and channel, run once with `mode=test`:

```bash
gh workflow run accuracy-digest.yml --ref main \
  -f mode=test -f e2e_run_id=<e2e-run-id> -f fpm_run_id=<fpm-run-id>
```

This sends a real `[TEST]` parent message plus its detail thread to the daily
channel. Check desktop/mobile table readability, links, and per-model/per-GPU
sections, then set `SLACK_ACCURACY_ENABLED=true`. Test sending is restricted to
`main`; a branch dry run receives no Slack token. No manual mode sends a
production daily report.

A workflow dry run does not rerun predictors. Use the deterministic tests in
`tests/fpm_accuracy/test_digest.py` to exercise regressions, coverage losses,
changed datasets, missing runs, daylight-saving cutoffs, deduplication, and
interrupted thread delivery without sending fake alarms to Slack.
