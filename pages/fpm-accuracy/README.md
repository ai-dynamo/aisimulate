# FPM Accuracy dashboard

The public [FPM Accuracy Overview](https://ai-dynamo.org/aisimulate/fpm-accuracy/?branch=main)
compares forward-pass predictions with measurements from the public
[nvidia/aisimulate-fpm-dataset](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset).

Details and 3D Visualization are separate tabs with independent controls.

## What is published

- Overview: expandable model/configuration rows and one sortable Best MAPE
  column, with the winning predictor and errors. FPM input links are available
  in Details rather than table cells.
  Model rows average their configurations' best MAPEs equally and say
  “Mixed predictors” when winners differ. Prediction counts, coverage, and
  contributing configuration counts are omitted from Overview/Predictors table
  cells; summary cards retain their contributing counts, and Details/Trends
  retain coverage. The Overall MAPE card averages configuration winners once
  each, independently of model grouping or collapsed rows.
- Predictors: the same branch and table layout, with Regression, FPM (KV warmup
  on), and FPM (KV warmup off) columns. Three cards average each predictor's
  available configuration MAPEs equally; model table rows retain weighting by
  successful prediction count. A reference section explains mechanisms and the
  canonical Python construction, prediction, and regression tuning APIs. Three
  left-aligned reference cards show each mechanism, input source, and explicit
  mode, with expandable Python examples including the selected library root.
  The cards stack on smaller screens and share both dashboard themes. KV
  warmup selects the collected library, not a prediction-time switch.
- Averages exclude unavailable/nonfinite MAPEs and show contributing
  configuration counts. No available MAPE displays as a dash. Best selection
  compares unrounded values; exact ties prefer Regression, warmup on, then
  warmup off. Winning a low MAPE does not imply complete prediction coverage.
- Trends: main-only, rolling 90-day history starting at
  `8dad9634735b6875e22a90927216e542e73ba237`. Each code/population pair
  retains its newest qualified evaluation. Dataset or FPM input changes break
  the series; MAPE is weighted by successful prediction count. Chart labels pair
  each commit ID with its evaluation date (UTC). Hover, focus, or tap a sample
  for a structured tooltip with MAPE, counts, coverage, errors, evaluation time,
  and short revision IDs. Full revisions remain in accessible point labels.
  Larger targets and active markers help selection; connecting lines do not
  intercept sample hover targets. Tooltips stay inside the
  viewport and dismiss with Escape or when leaving the point and tooltip.
  The separate trend-values table is removed; the latest phase summary remains.
- Details: retained branch evaluations, selected FPM inputs, phase summaries,
  measurement-only workload distributions, and prediction-error heatmaps.
  Heatmaps use compact cells and size independently, with scrolling only when
  their contents exceed the available width. A Hugging Face icon identifies the
  pinned configuration and measurement evidence links in a small, muted line.
- 3D Visualization: independent panels, seven workload axes, stable samples,
  full gzip chunks, native rank provenance, camera controls, and PNG export.
  Diagnostic unsynchronized DP groups remain separate from accepted truth.
- Hide configurations with zero measurements and models with no measured
  configurations. Overview counts reflect visible configurations; complete
  evaluation artifacts still retain all configurations.
- Table headings, model/configuration labels, metrics, and evidence links are
  left-aligned across Overview, Predictors, Trends phase summaries, and Details.
- Overview/Predictors configuration rows show the evaluated phases rather than
  the evaluator's inferred `aggregated` worker role. That role combines phase
  populations and does not establish an aggregated serving deployment.
  Each row has expandable test-collection provenance from its measurement
  manifest at the evaluation's exact HF revision. Benchmark truth filenames
  identify self-benchmark point sweeps; declared AgentX collection evidence
  identifies trace replay. These labels describe source artifacts, not a new
  validation of the collection campaign or a train/test independence claim.
  Small collection JSON and named-column window TSV files expose recorded
  job/benchmark IDs, concurrency, duration, and request settings. Trace-defined
  lengths and completed requests are distinguished from fixed ISL/OSL and
  requested `num_req`; missing or unsupported metadata stays explicitly unknown.
  Supporting runs alongside benchmark truth are labeled separately. MAPE remains
  pooled per configuration, with phase detail in Details and run/worker points
  in 3D. Provenance loads from public HF on demand; failures do not hide metrics.
- The E2E accuracy page's compact AISimulate header, branch selector, summary
  cards and table. Light/dark mode shares the `sm-theme`
  preference across the accuracy pages. Filters use compact responsive columns with smaller labels and controls;
  evaluation times use UTC minutes. Trend plots have a bounded size and center
  the point when only one evaluation is available.
- Predictor columns: online Regression, FPM (KV warmup on), then
  FPM (KV warmup off) (KV-off input).
- MAPE over successful predictions, with predicted/measured counts, coverage,
  prediction errors, and regression tuning errors. Cold-start misses count
  against coverage. Missing FPM inputs never remove measurements from coverage.
- Dataset configuration and measurement links pinned to the evaluated HF commit.
- Every tab uses one left-aligned evaluation box below the tabs:
  `Daily evaluation · YYYY-MM-DD · Evaluation run · AISim <commit> · HF <revision>`.
  The date is the evaluation completion date in UTC. All three links belong to
  that snapshot. Separate schedule headers, evaluated-configuration counts,
  revision/timestamp lines, and age-based stale warnings are removed.
- The box follows the loaded snapshot on Overview/Predictors, the selected
  evaluation on Details, and the newest represented evaluation on filtered
  Trends. On 3D Visualization it describes the selected branch's latest
  completed evaluation; the independent measurement catalog still identifies
  its own HF revision. Loading/unavailable states clear prior snapshot links.
  Dataset links remain in configuration/measurement evidence as well.

There is no op-based evaluation or FPM Coverage tab.
FPM variants use the same observations. One winner per KV warmup mode is selected
by coverage descending, MAPE ascending, then artifact ID. This reproduces Gym's
comparison policy; it is not an independent held-out ranking of input libraries.
Regression predicts and scores each observation before tuning on its target;
state is isolated by worker. Worker roles are inferred from scheduled workload
across the case, never latency. This is an offline role-inference policy.
New evaluations select Gym's recommended signed 4×1 lazy regression: 64 retained
observations per store, attention/MoE features, minimum five observations,
ridge 1e-9, no scheduled rebuilds, and updates with 1% / 0.1 ms tolerances,
window 8, trigger 2, cooldown 1, and startup 10. This is an evaluator setting;
shared AISim defaults and the AgentX/ShareGPT/LongBench configurations remain
unchanged. Retained historical evaluations keep their original settings.
Configurations with decode context parallelism (`dcp>1`) retain their measured
coverage and worker regression results, but show native FPM as unsupported.
The evaluator validates DCP identity without treating it as ordinary CP.
Revisions without worker-scoped regression or configurable signed lazy controls
show Regression as unsupported. Their measurements remain in its coverage
denominator; FPM evaluation continues. An older regression policy is not
substituted for the Gym contract.

## Daily data flow

`FPM Accuracy Matrix` runs daily at 10:47 UTC. It pins HF `main` once and evaluates
AISim `main` plus numeric `release/MAJOR.MINOR.PATCH` branches >= `0.12.0`.
Manual dispatch accepts an eligible branch and a full commit belonging to it.
At most two branch jobs run concurrently. A verified exact nightly wheel is
reused for scheduled main when available; otherwise the exact source is built.

Each completed branch uploads `summary.json`, `details.json`, and `qualification.json` as
`fpm-accuracy-web-<branch-key>`, retained for 90 days. Results are not committed.
Upload the output directory as one path so container runners preserve both
files at the archive root; the publisher rejects missing or extra files.
Qualification v2 hashes summary and detail separately; legacy v1 remains
readable for Overview with explicit unavailable detail states.
The main-branch Pages publisher verifies checksums, schema, source ancestry,
producer repository/workflow, evaluator SHA, run attempt, and successful branch
job. It selects the newest eligible source commit, then latest completion time.
Failed branches cannot replace prior valid results or block another successful
branch. Incomplete campaigns do not publish; high MAPE does not fail a campaign
or block release staging. A missing/expired history shows “No completed
evaluation” when no retained qualified artifact remains.
Artifacts deleted or expired after listing (HTTP 404/410) are skipped so earlier
valid results can still publish. Authentication and service errors remain fatal.

Pages serves the JSON alongside the reviewed main-branch HTML/CSS/JS. Browsers
never need GitHub credentials or direct access to Actions artifacts. PR previews
use the unavailable state; browser tests use clearly synthetic fixtures.
FPM loads `../e2e-accuracy/styles.css` for the shared page styles and keeps only
FPM table details in its own stylesheet. Serve the built site or the `pages/`
directory so that both assets are available.

## Local checks and smoke evaluation

Dashboard script URLs carry a shared version (`collection-provenance-1`) so returning
visitors fetch scripts compatible with the unified banner. Bump this version
across all five tabs when changing shared DOM or navigation APIs. The browser
check covers a cached 3D script that still references the removed status header.

```bash
python -m pip install pytest
python -m pip install --require-hashes -r scripts/fpm_accuracy/requirements.txt
python -m pytest -c /dev/null -o cache_dir=.cache/pytest tests/fpm_accuracy
python scripts/pages/build_pages_site.py --output-dir /tmp/aisim-pages
python -m http.server --directory /tmp/aisim-pages 8000
```

Install an exact AISim wheel with **pip** (which records its SHA-256 in
`direct_url.json`) before a real evaluation. Run `python -m scripts.fpm_accuracy.run_fpm_accuracy
--help` for the required source, evaluator, HF, wheel, run-identity, and output
arguments. `--configuration` limits a local smoke to selected configuration paths;
it writes `SMOKE_ONLY.txt` and never writes a qualification manifest. Omit it
for a complete campaign. The output directory must be empty. Only the scheduled
workflow publishes results; the runner reads HF and writes local output.

```bash
python -m pip install playwright==1.63.0
python -m playwright install chromium
python scripts/pages/check_fpm_accuracy_browser.py
node --test tests/test_fpm_accuracy_workflow.mjs
```

## Source attribution

The overview structure and behavior were adapted from NVIDIA
[AISim FPM Gym](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-fpm-gym/-/tree/e8221729db2802e822f6919fd68bc2941743385b/dashboard),
commit `e8221729db2802e822f6919fd68bc2941743385b`, originally
`dashboard/index.html` and `dashboard/assets/gym.css`. Modified for a best-MAPE
overview, a three-column Predictors page (derived from the same overview),
qualified branch snapshots, and public-only provenance. The
visual presentation now uses AISimulate's E2E accuracy stylesheet.
Apache-2.0, with maintainer-confirmed migration permission. The new tabs adapt
Gym behavior from `f934c030afc3a03cb04d8f3ff4709194f7445c98`; the 3D HTML,
JS and CSS derive from `dashboard/3d-visualization.html` and
`dashboard/assets/visualization.{js,css}` at that revision. They are modified
for AISimulate navigation, styling and GitHub artifact data. Plotly.js v3.4.0
is bundled unmodified, loaded only by the 3D page, with its MIT license.
The repository copyright check pins the vendor bundle and MIT license bytes;
updates must refresh those hashes together with the attribution.
See the root THIRD_PARTY_NOTICES.md and LICENSE.

## Latest dataset and storage

Scheduled and manual campaigns resolve HF `main` once to an immutable SHA.
Every branch and the shared `Qualify FPM measurements` job receives that SHA;
HF cache directories include it. The shared job discovers current
measurement snapshots and uploads `fpm-accuracy-measurements` once per campaign.
Branch scoring retains current-snapshot membership. No evaluated results are
committed, and no long-lived Git branch or external database stores history.

A new HF snapshot produces new assets even when the AISim commit is unchanged.
The Pages publisher keeps distinct measurement/FPM populations, checks every
checksum and producer job, and selects visualization data matching the latest
qualified main evaluation's HF revision when available. Otherwise 3D retains
the latest qualified measurement snapshot with its original HF revision and
a stale label; without any retained measurement snapshot it shows unavailable. Failed evaluations retain prior qualified
accuracy results and their original HF revision; results older than 48 hours
are marked stale. Expired/deleted artifacts disappear at the next publication.
Full-point chunks load only when requested; summaries and heatmaps contain
aggregates. Point assets contain public HF measurement evidence, never tokens
or credentials. Browser fixtures are synthetic and are never deployed as data.

## Rollout

After the workflow changes land on main, dispatch `fpm-accuracy.yml` with
`branch=main` and `expected_sha=8dad9634735b6875e22a90927216e542e73ba237`.
This evaluates the exact baseline using that campaign's latest pinned HF
snapshot; it does not recreate historical HF evidence. Confirm the branch and
measurement qualification jobs and the following Pages deployment succeed.
Subsequent daily runs extend history automatically. A failed baseline must be
retried explicitly; do not relabel a newer result as the baseline.

The initial 3D export uses validated current snapshots, matching scoring.
At HF `68fa3add95b32a0399d781b043cb0f1008c8040d`, two archived DeepSeek
manifests lack current-manifest hash bindings. Archived source traversal is
therefore excluded without weakening loader validation. Retained evaluation
history remains available for Details and Trends.

Measurement artifact downloads allow up to 900 MiB, matching the archive bundle
bound. Other Actions responses retain the 64 MiB default; individual assets,
checksums, and archive paths remain validated before publication.

Partial reruns can combine independently qualified attempts from the same campaign
when HF revision, evaluator identity, and measurement membership match. Switching
3D selections does not wait for obsolete downloads; chart mutations remain serialized.

Manual preview campaigns may run the workflow from a development branch while
evaluating an eligible main/release source revision. These artifacts retain the
development evaluator SHA and are excluded from automatic Pages publication.
The container campaign uses an explicit `/tmp/fpm-accuracy-venv/bin/python` for
installation and evaluation so runner path remapping cannot select another Python.

Details initially selects a configuration with accepted measurements and an
available workload. Empty configurations remain labeled and directly linkable.

Details uses the evaluation-selected FPM variant for each method and links its pinned input.
Only Method and Workload controls are shown above the heatmaps. Missing inputs,
unsupported predictors, initialization failures, and workloads without successful
predictions show an explanation in place of the error map; measured distributions remain visible.

Browser visualization fixtures use readable sample/chunk JSON filenames. Test setup
creates SHA-256 asset names, gzip chunks, and the checksum manifest in its temporary
site directory, keeping generated assets out of the repository.

Overview configuration rows link to Details and 3D Viz side by side. The 3D link
selects the matching configuration and snapshot in the left panel when available
in the published measurement catalog.

The browser regression check verifies Sample renders while a superseded Full
download is held, and stays selected after that download and redraw finish.
