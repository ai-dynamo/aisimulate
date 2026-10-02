# FPM Accuracy dashboard

The public [FPM Accuracy Overview](https://ai-dynamo.org/aisimulate/fpm-accuracy/?branch=main)
compares forward-pass predictions with measurements from the public
[nvidia/aisimulate-fpm-dataset](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset).

## What is published

- Overview: expandable model/configuration rows and sortable metrics.
- Trends: main-only, rolling 90-day history starting at
  `8dad9634735b6875e22a90927216e542e73ba237`. Each code/population pair
  retains its newest qualified evaluation. Dataset or FPM input changes break
  the series; MAPE is weighted by successful prediction count.
- Slice Detail: retained branch evaluations, FPM variants, phase summaries,
  measurement-only workload distributions, and prediction-error heatmaps.
- 3D Visualization: independent panels, seven workload axes, stable samples,
  full gzip chunks, native rank provenance, camera controls, and PNG export.
  Diagnostic unsynchronized DP groups remain separate from accepted truth.
- Hide configurations with zero measurements and models with no measured
  configurations. Overview counts reflect visible configurations; complete
  evaluation artifacts still retain all configurations.
- The E2E accuracy page's compact AISimulate header, branch selector, summary
  cards and table. Light/dark mode shares the `sm-theme`
  preference across the accuracy pages.
- Predictor columns: online Regression, FPM (KV warmup on), then
  FPM (KV warmup off) (KV-off input).
- MAPE over successful predictions, with predicted/measured counts, coverage,
  prediction errors, and regression tuning errors. Cold-start misses count
  against coverage. Missing FPM inputs never remove measurements from coverage.
- Dataset configuration and measurement links pinned to the evaluated HF commit.
- The evaluated AISim commit, HF commit, and UTC completion time. Results are
  marked stale after 48 hours or when the selected branch has advanced.

There is no op-based evaluation or FPM Coverage tab.
FPM variants use the same observations. One winner per KV warmup mode is selected
by coverage descending, MAPE ascending, then artifact ID. This reproduces Gym's
comparison policy; it is not an independent held-out ranking of input libraries.
Regression predicts and scores each observation before tuning on its target;
state is isolated by worker. Worker roles are inferred from scheduled workload
across the case, never latency. This is an offline role-inference policy.
Configurations with decode context parallelism (`dcp>1`) retain their measured
coverage and worker regression results, but show native FPM as unsupported.
The evaluator validates DCP identity without treating it as ordinary CP.
Revisions without the worker-scoped regression API (including `release/0.12.0`
at `1f728534`) show Regression as unsupported. Their measurements remain in its
coverage denominator; FPM evaluation continues. Legacy shared regression state
is not substituted for the Gym contract.

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

```bash
python -m pip install pytest
python -m pip install --require-hashes -r scripts/fpm_accuracy/requirements.txt
python -m pytest -c /dev/null -o cache_dir=.cache/pytest tests/fpm_accuracy
python scripts/build_pages_site.py --output-dir /tmp/aisim-pages
python -m http.server --directory /tmp/aisim-pages 8000
```

Install an exact AISim wheel with **pip** (which records its SHA-256 in
`direct_url.json`) before a real evaluation. Run `python scripts/run_fpm_accuracy.py
--help` for the required source, evaluator, HF, wheel, run-identity, and output
arguments. `--configuration` limits a local smoke to selected configuration paths;
it writes `SMOKE_ONLY.txt` and never writes a qualification manifest. Omit it
for a complete campaign. The output directory must be empty. Only the scheduled
workflow publishes results; the runner reads HF and writes local output.

```bash
python -m pip install playwright==1.63.0
python -m playwright install chromium
python scripts/check_fpm_accuracy_browser.py
node --test tests/test_fpm_accuracy_workflow.mjs
```

## Source attribution

The overview structure and behavior were adapted from NVIDIA
[AISim FPM Gym](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-fpm-gym/-/tree/e8221729db2802e822f6919fd68bc2941743385b/dashboard),
commit `e8221729db2802e822f6919fd68bc2941743385b`, originally
`dashboard/index.html` and `dashboard/assets/gym.css`. Modified for a three-column
public overview, qualified branch snapshots, and public-only provenance. The
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
history remains available for Slice Detail and Trends.
