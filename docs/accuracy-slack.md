# Daily accuracy Slack report

`Accuracy Slack Daily` posts one daily top-level message to
**#swdl-dynamo-aisim-daily**, through Slack Workflow Builder. Alerts and results share that message;
E2E per-model and per-GPU tables share one reply in the same thread. No mentions are
sent. The workflow does not accept commands or start evaluations.

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
- Serialize notifier runs. Before a production POST, upload an
  `accuracy-attempt-YYYY-MM-DD` artifact reserving the local date. Later triggers
  and reruns skip a reserved day, even if that attempt failed or timed out.
  Dry runs and test sends do not reserve the day.
- Preserve the report before posting. Do not delete the reservation artifact or
  original run: they prevent duplicate production triggers. Reservations expire
  after 90 days, well beyond the current-day delivery window.
- A reservation is intentionally an **at-most-once attempt**, not guaranteed
  delivery: a crash between reserving and posting can leave the day unsent.
  Unknown POST outcomes are never automatically retried.
- Slack `ok:true` confirms webhook acceptance, not completion of the message steps.
  Check Slack Workflow Builder **Activity** if the parent or reply is missing;
  resolve/retry failed steps there. The notifier cannot read channel history or
  resume Slack thread steps without an app token. GitHub baseline state records
  an accepted trigger, not verified message delivery.
- A confirmed unsent report can be sent using explicit `mode=test` (with a TEST
  label). Do not remove a reservation or replay the webhook after an uncertain
  outcome without checking Slack Activity first.

## Report contents and comparison

The E2E table has six columns: Branch, Overall, vLLM, SGLang, TRT-LLM, Coverage.
Each accuracy cell is **TPOT / TTFT MAPE (%)** for AISimulate, not legacy AIC.
Thread tables independently group by model and by GPU, within each branch,
combining frameworks. They average successful point errors directly; they do
not average already-aggregated MAPEs. Coverage is predicted / eligible points,
not all original source measurements. FPM shows KV-warmup on/off and online
regression MAPE, weighted by successful prediction counts, with coverage.

For each branch:

- Start with the most recent earlier qualified scheduled campaign in the retained
  90-day history. Without one, explicitly report an initial baseline.
- Require identical measurement revision/content and evaluation-rule fingerprints.
  The rule fingerprint covers evaluator source and its branch workflow at the
  producer revision, not the evaluated AISimulate package revision. A change
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

FPM previously retained only aggregates, which cannot detect exchanged successful
and failed points. The producer now writes a separate
`fpm-accuracy-comparison-<branch-key>` artifact containing a hash of observation order
and compressed little-endian float64 percentage-error sequences (-1 for unsuccessful predictions). It binds
to the public summary checksum and exact producer snapshot. The notifier verifies
counts and means against that qualified summary. No raw measured/predicted
latencies are exported. Existing Pages artifacts remain unchanged. Old FPM runs
without point evidence can be displayed, but point regression checks are explicitly
unavailable until a comparable pair with evidence exists.

## Slack Workflow Builder setup (no separate Slack app)

1. Create a workflow in Slack **Workflow Builder**, with **From a webhook** as
   its trigger. Use a separate accuracy workflow so the PR review digest remains
   independent.
2. Add two variables, both of type **Text**, with these exact names:

   | Variable | Content |
   | --- | --- |
   | `message` | Daily E2E/FPM summary, alerts, and pipeline links |
   | `accuracy_details` | Separate per-model and per-GPU tables, comparison details |

3. Add **Send a message to a channel**. Select **#swdl-dynamo-aisim-daily** and
   insert the `message` variable as the body.
4. Add **Reply to a message in thread**. Select the message produced by step 3
   as the reply target, and insert `accuracy_details` as its body. Keep the option
   to also send the reply to the channel off. Do not add any mentions.
5. **Publish** the workflow and copy its Web request URL
   (`https://hooks.slack.com/triggers/...`). Store it as repository Actions
   secret **`SLACK_ACCURACY_WEBHOOK_URL`**. No bot token or channel-ID variable
   is needed: the destination lives in the Slack workflow.
6. Leave repository variable **`SLACK_ACCURACY_ENABLED=false`** until the real
   test below passes, then set it to `true`.

Webhook body example (the script supplies both values):

```json
{"message": "AISimulate Accuracy Daily ...", "accuracy_details": "E2E per model ..."}
```

Both test and production use this same workflow/channel. Tests prefix the parent
with `[TEST]` and do not update production baseline state. The Text variables use
plain text, pipe-separated table columns, and complete URLs; they do not depend
on Markdown, named hyperlinks, or monospace alignment. The detail sections share
one thread reply. Each field is capped at 35,000 characters; oversized reports
fail before reserving/sending rather than silently losing rows.

GitHub permissions remain read-only (`contents:read`, `actions:read`). The
notifier checks out its own trusted workflow revision and never executes code
from downloaded artifacts. Pages validation remains strict; explicit manual
preview/test alone permits same-repository producer runs from a non-main branch.

Slack's official [webhook setup guide](https://slack.com/help/articles/360041352714-Build-a-workflow--Create-a-workflow-that-starts-outside-of-Slack)
explains trigger variables and publishing; workspace permissions may control who
can create webhook workflows.

## Dry run and one real Slack test

Open **Actions → Accuracy Slack Daily → Run workflow**. Choose `dry-run`
(the default), and provide completed `e2e_run_id` and `fpm_run_id` values. Manual
accuracy runs are allowed here. For example:

```bash
gh workflow run accuracy-digest.yml --ref main \
  -f mode=dry-run -f e2e_run_id=36643620072 -f fpm_run_id=36558333115
```

Before the workflow is merged/registered, run the same dependency-free Python
entrypoint from its checkout with a read-only GitHub credential:

```bash
GH_TOKEN="$(gh auth token)" python3 scripts/notify_accuracy.py \
  --e2e-run-id 36643620072 --fpm-run-id 36558333115 \
  --output /tmp/accuracy-preview
```

Inspect the Actions Summary and `accuracy-preview` / `accuracy-report` artifacts,
or local `preview.md` / `payload.json` / `report.json`. The JSON includes alert reasons, comparison
limitations, and the exact baseline snapshots. Dry run does not read Slack
credentials, post messages, or write production baseline state.

After merging and configuring the Slack workflow/secret, run once with `mode=test`:

```bash
gh workflow run accuracy-digest.yml --ref main \
  -f mode=test -f e2e_run_id=36643620072 -f fpm_run_id=36558333115
```

This sends a real `[TEST]` parent message plus its detail thread to the daily
channel. Check desktop/mobile table readability, links, and per-model/per-GPU
replies, then set `SLACK_ACCURACY_ENABLED=true`. Test sending is restricted to
`main`; a branch dry run receives no Slack webhook secret. No manual mode sends a
production daily report.

A workflow dry run does not rerun predictors. Use the deterministic tests in
`tests/fpm_accuracy/test_digest.py` to exercise regressions, coverage losses,
changed datasets, missing runs, daylight-saving cutoffs, deduplication, and
uncertain trigger outcomes without sending fake alarms to Slack.
