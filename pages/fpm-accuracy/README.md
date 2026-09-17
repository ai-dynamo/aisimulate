# FPM Accuracy Overview

The public [FPM Accuracy Overview](https://ai-dynamo.org/aisimulate/fpm-accuracy/?branch=main)
compares forward-pass predictions with measurements from the public
[nvidia/aisimulate-fpm-dataset](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset).

## What is published

- Only Overview: expandable model/configuration rows and sortable metrics.
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

There is no op-based evaluation or navigation to the internal Gym's other tabs.
FPM variants use the same observations. One winner per KV warmup mode is selected
by coverage descending, MAPE ascending, then artifact ID. This reproduces Gym's
comparison policy; it is not an independent held-out ranking of input libraries.
Regression predicts and scores each observation before tuning on its target;
state is isolated by worker. Worker roles are inferred from scheduled workload
across the case, never latency. This is an offline role-inference policy.
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

Each completed branch uploads `summary.json` and `qualification.json` as
`fpm-accuracy-web-<branch-key>`, retained for 90 days. Results are not committed.
Upload the output directory as one path so container runners preserve both
files at the archive root; the publisher rejects missing or extra files.
The main-branch Pages publisher verifies checksums, schema, source ancestry,
producer repository/workflow, evaluator SHA, run attempt, and successful branch
job. It selects the newest eligible source commit, then latest completion time.
Failed branches cannot replace prior valid results or block another successful
branch. Incomplete campaigns do not publish; high MAPE does not fail a campaign
or block release staging. A missing/expired history shows “No completed
evaluation” when no retained qualified artifact remains.

Pages serves the JSON alongside the reviewed main-branch HTML/CSS/JS. Browsers
never need GitHub credentials or direct access to Actions artifacts. PR previews
use the unavailable state; browser tests use clearly synthetic fixtures.
FPM loads `../e2e-accuracy/styles.css` for the shared page styles and keeps only
FPM table details in its own stylesheet. Serve the built site or the `pages/`
directory so that both assets are available.

## Local checks and smoke evaluation

```bash
python -m pip install pytest -r scripts/fpm_accuracy/requirements.txt
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
Apache-2.0, with maintainer-confirmed migration permission. See the root
THIRD_PARTY_NOTICES.md and LICENSE. Plotly and other tabs are not included.
