# Daily accuracy Slack report

`Accuracy Slack Daily` posts one daily top-level message to
**#swdl-dynamo-aisim-daily** (`C0BULBSTXJ6`). Alerts and results share that message;
E2E per-model and per-GPU tables are included in the main message. No mentions are
sent. The bot does not accept commands or start evaluations.

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
or local `preview.md` / `report.json`. The JSON includes alert reasons, comparison
limitations, and the exact baseline snapshots. Dry run does not read Slack
credentials, post messages, or write production baseline state.

After merging and configuring the bot/channel, run once with `mode=test`:

```bash
gh workflow run accuracy-digest.yml --ref main \
  -f mode=test -f e2e_run_id=36643620072 -f fpm_run_id=36558333115
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
