# AGENTS

This file adds explicit repository-wide development guards.

## FPM model onboarding

When asked to onboard a model for FPM simulation on designated hardware, follow
[Onboard with an agent](docs/fpm-self-service.md#onboard-with-an-agent) in the FPM
self-service guide. Use the checkout's `aisimulate onboard` CLI and current help.
Create one `onboarding-checkpoint.json` during stage 1, even while inputs are
incomplete, using the guide's [checkpoint workflow](docs/fpm-self-service.md#checkpoint-and-resume-an-onboarding-session).
Keep it outside every fresh `init --output-dir` root. Save supplied facts,
research with sources and confidence, unresolved questions, user decisions and
per-configuration progress after each meaningful finding, edit or acceptance;
do not wait for a stage to finish or for the user to request a save. The agent
must invoke `onboard checkpoint`: other onboarding commands do not automatically
observe or persist the conversation. When resuming, run
`aisimulate onboard resume --checkpoint PATH` first, reuse the saved context and
inspect any integrity issues before continuing. Report the checkpoint path at
handoff. Use the returned revision for subsequent updates; on a stale-writer
error, reload and reconcile instead of overwriting newer work.
Follow its six stages: inspect the model and target; choose the worker and
review runtime/collection limits; derive, review and save the profile; plan
collection; collect, verify and finalize memory; validate replay and run ordinary
prediction/recommendation. Start by asking only for a missing
Hugging Face model ID (`organization/model-name`) and target GPU platform. Accept
an already supplied local config, profile or checkpoint path instead of requiring
a Hub ID; do not ask for both an ID and a config upfront. For a Hub ID, retrieve
the config using available Hub access as described in stage 1. Reuse supplied
facts and inspect the config/profile before asking for derivable metadata; defer
other questions to their stage. Propose supported runtime settings and worker
topologies with their sources, then let the user review and edit them. An existing
vLLM launch command or configuration is optional evidence; do not require one or
start with a topology/dtype questionnaire. At stage 2 entry, resolve the literal
framework version from supplied evidence, asking only if the pin is missing.
Before asking for derivable inputs, proactively follow
[runtime investigation and precision options](docs/fpm-self-service.md#investigate-runtime-constraints-and-precision-options)
using stage 1's checkpoint revision and hardware, pinned official implementation
and lightweight checkpoint metadata. Use authorized official remote sources when
local evidence is insufficient; a source checkout, launch command, initialization
logs or running server are not prerequisites. Present the required sourced table
of joint precision combinations, distinguishing fixed checkpoint constraints,
runtime defaults, configurable alternatives and unknowns. Runtime support and
current collector compatibility are separate; keep collector-blocked alternatives
explicitly blocked. Defaults are one proposal. Let the user select the combination(s)
to cover and preserve prior explicit choices before profile acceptance. This is
agent investigation using lightweight source and metadata; the existing CLI
consumes prepared inputs without resolving remote runtime capabilities.
At stage 2, inspect `aisimulate onboard init --model-config PATH --suggest-parallel`
with the actual checkpoint/revision, runtime, GPU, interconnect and collection
options. This read-only JSON preview reports model/hardware-aware TP choices and
MoE TP/DEP/TEP alternatives, exact flags, resource sources and missing inputs.
Show the current collector's `comm_quant_mode=half` identity default and its
source; it does not assert a global NCCL tensor dtype. Preserve user overrides
and explain that other communication identities are rejected by the current
collector for new collection. Resolve weight, FMHA and KV precision separately;
do not infer unknown FMHA or KV precision from a checkpoint name or weight
quantization. Ask only for shared precision/layout facts that remain unresolved
after inspection. Only a complete planning estimate can establish
an `estimated_fit` default. It never completes the saved runtime memory profile;
the shortlist is not a performance ranking or runtime qualification. Present the
default and alternatives, help select one or more configurations, and preserve
explicit topology choices. Each generated profile and collection plan still
uses one exact tuple. Standalone interactive setup uses
`--model-config PATH --interactive --output-dir ROOT` for guided comma-separated
selection. Headless setup uses `--model-config PATH --parallel-configs CONFIGS
--output-dir ROOT` for a JSON/YAML list of explicit topology fields and optional
per-entry `resource_overrides`. Retain `--output FILE` for a single request.
If no default exists, explain the missing inputs and select candidates before
observing their runtime memory during collection. Read shared metadata once,
then derive and review each profile independently. Per-rank byte bounds and `cache_groups`
belong to the exact tuple; never transfer them between configurations or place
them in shared overrides for multiple choices. Shared precision/layout and
`cache_block_sizes` may be reused only when valid for the same runtime, backend
and precision; page bytes must be derived per precision and tuple. The CLI rejects
duplicate resolved topologies even with different precision overrides. Use a
separate fresh output root for each selected precision combination and the existing
topology list within it; output separation does not make a blocked variant collectible.
For checkpointed agent sessions, use headless draft generation and save each
configuration's complete draft request, edits and explicit acceptance separately
in the checkpoint. Do not mark an unreviewed configuration accepted because
another configuration was accepted. Use `--accept-profile CONFIG_ID` only after
the user accepts that exact request/profile; a saved draft never implies
acceptance. Publish final requests to fresh paths and verify they match the
accepted drafts before planning. Standalone interactive edits still apply only
to the profile being reviewed; all profiles must be accepted before directory
output is saved, and a later cancellation or invalid profile leaves no new
artifacts. An interactive prompt session does not acquire partial-review
checkpointing automatically. Use a fresh or empty root; directory output rejects
`--overwrite` and never replaces collection results.
Inspect full-attention, sliding-window and supported convolution retention
before choosing linear or grouped cache resources. For grouped resources,
derive available geometry, then investigate the pinned cache implementation and
available allocation metadata for runtime `cache_block_sizes` after choosing the
topology. Ask only for facts still unresolved; their absence from model config is
not a reason to delegate investigation to the user. Missing-field diagnostics in
stage 3 or headless setup follow the same investigate-before-asking rule.
Review/edit the complete `cache_groups` JSON and provenance. Each
`page_size_bytes` is the rank-local aggregate over every layer in that group,
including runtime padding; config-derived packed pages are minimum estimates.
Never replace grouped storage with an averaged `kv_bytes_per_token` or scalar
token capacity. Use the canonical native cache budget for group footprint and
transient prefill admission; unresolved resource assumptions remain explicit.
Derive each configuration's minimum collection GPUs from attention TP times
attention DP (TP4 requires four GPUs). Separate collection runs can reuse the
same GPUs; do not sum their widths into an allocation requirement. Do not ask
for total available GPUs, cluster node allocation or replica budgets during
onboarding. Preserve target hardware, runtime and interconnect characteristics;
verify actual collection resources before execution.
Generated predict/recommend configs validate one worker. Deployment replicas and
optimization GPU budgets belong to ordinary predict/recommend configurations.

Review context, scheduler and prefill capture limits independently of validation
traffic. Fresh config/profile defaults use the smaller of the declared context
and 256,000 tokens, profile scheduler bounds or 8,192 tokens/256 sequences, and
0.90 GPU memory utilization. Reused observed profiles retain their recorded memory
utilization and additionally limit context to their observed `max_model_len`.
New initialization uses
`--prefill-cudagraph-policy runtime`, leaving graph mode/capture sizes and the
exact sample grid unresolved until engine initialization. Show these as proposed
settings, explain their sources, and allow edits; they establish neither capacity
nor timing coverage. `--gpu-memory-utilization` selects a finite fraction in
`(0, 1]`. A numeric `--max-prefill-cudagraph-size` selects `explicit` policy;
`--prefill-cudagraph-policy explicit` without a size uses 2,048. Runtime policy
and a numeric size conflict. In profile review, editing the policy to `runtime`
clears the explicit size. Old requests missing the policy retain explicit capture
with their saved size or the 2,048-token default. Do not require fixed input/output
lengths, concurrency, TTFT or TPOT during intake.
Those flags customize optional synthetic examples only. Maximum sequences does
not reserve maximum context for every sequence or request every prefill batch.
Follow the [shared collection policy](docs/fpm-self-service.md#how-the-collection-grid-is-determined):
AISimulate sets runtime limits and the selected capture policy. Runtime policy
emits no prefill compilation or new-token sample-cap override; the graph-independent
KV-read sample cap remains bounded. Explicit policy sets the reviewed capture
extension and associated sample caps. Dynamo combines these inputs with
initialized engine state, image sampling defaults and feasibility
checks to generate and measure the exact grid. Record actual graph dispatch and
capture sizes with collection evidence; requested settings alone do not establish
them. Capture overrides must match the serving target. A complete generated grid
does not establish AgentX/direct-FPM query coverage; do not invent a separate
AgentX collection grid.

The collector generates and launches benchmark workers through the existing
Kubernetes or Slurm executor; it does not require an already-running HTTP server.
Choose the executor at the collection stage through `onboard collect-fpm
--executor kubernetes|slurm`. Kubernetes retains the existing namespace/PVC and
deployment permissions. Slurm uses a caller-owned `sbatch`/`salloc` allocation,
an explicit `--image`, repeated `--container-mount` options as needed, Pyxis/Enroot,
and a campaign directory shared at the same absolute path across allocated nodes.
The allocation's node count must match the generated collector plan, with enough
GPUs per node for the selected worker. Separate configurations can reuse those
GPUs; do not require allocation details during model intake or claim generic
executor support qualifies every topology. `--transport` still selects GPU
networking, not the executor. Preview, execute and resume with the same deployment
options. Save their exact values in each configuration's existing checkpoint
`inputs.collection_deployment` before registering collection outputs; recover
and explicitly pass those options after `onboard resume`. The checkpoint stores
context and never applies or executes saved options automatically. Deployment
changes invalidate affected acceptance and collection references: preserve the
old campaign, archive superseded references with a reason, and use new output
paths with the existing review workflow.
Before submitting collection jobs, follow
[campaign orchestration](docs/fpm-self-service.md#orchestrate-independent-collection-campaigns):
save a per-configuration execution table with exact commands, isolated paths,
generated allocation geometry, prerequisites and intended concurrency. Independent
formal collection configurations submit independently by default. Use Slurm
`afterany` for intentional sequencing and `afterok` only for a
genuine data prerequisite;
TP publication is not a prerequisite for an independent DEP collection. Keep
worker completion, cell validity/publication, memory finalization and accuracy
separate, retain truthful exit statuses and continue unrelated configurations.
Inspect active jobs and collector evidence before retrying. Recover compatible
post-processing without repeating passed GPU work; guided `--resume` can still
launch unfinished cells and does not retry cells still marked failed. Follow the
guide's lower-level retry procedure when needed. One coordinator records job
history under checkpoint `research`, updates per-configuration progress/artifacts
with revision checks and keeps volatile scheduler state out of accepted inputs.
Formal `collect-fpm` jobs use separate roots and do not update the session
checkpoint. `probe-runtime` (including preview) and `import-observations` do:
the agent must serialize each full invocation with every other writer of that
file, including coordinator saves and profile acceptance. No automatic lock
covers that interval. Save intended commands/paths first, retain new job IDs and
notes in ordinary job logs until safe to merge, and reload the current revision
after completion or failure. Writer coordination does not require an independent
predecessor to succeed. If a save conflicts, preserve produced evidence, reconcile
the checkpoint and use compatible probe resume/import before repeating GPU work.
Validation follows its own configuration's data/profile prerequisites.
Prepare the compatible pinned image, accessible checkpoint and GPUs before
execution. The engine initializes the model/cache
and resolves runtime settings, then Dynamo self-benchmark generates and times the
points. Inspect effective precision, graph policy, cache allocation/padding and
supported seeding behavior during bring-up. Benchmark prefix seeding is separate
from replay's cross-request prefix reuse; preserve the collector's phase-specific
protocol and inspect skipped/fallback evidence instead of copying replay cache
flags into benchmark launches.

When runtime-dependent geometry or capacity remains unresolved, follow
[runtime probing](docs/fpm-self-service.md#resolve-cache-geometry-with-a-runtime-probe).
Use compatible verified evidence or the bundled observer; when no bundle matches,
inspect the exact pinned runtime and vendor patches, proactively author a
campaign-local adapter with source mappings and semantic checks, then preview
and execute within the existing authorized campaign. A missing bundled adapter
does not require an upstream merge or a runtime downgrade. Cover every selected
configuration and both phases, preserve native runtime behavior, and retain
failed attempts and raw worker/scheduler evidence. Import with
`onboard import-observations` before proposing resolved geometry or capacity.
Save bundle/source/evidence/validation progress in the existing checkpoint after
meaningful changes. Review and accept each resulting draft independently;
observed memory establishes neither timing coverage nor serving accuracy.
Ask only for missing access, unresolved user choices or a concrete runtime or
resource constraint that needs a decision.

Do not ask users for activation, runtime, communication or aggregate non-KV
memory bounds. Fresh config-derived profiles keep memory pending, even when
planning estimates exist. Only an explicitly supplied complete four-field
non-KV declaration uses the legacy declared-memory path. Collection accepts
pending memory and observes cache allocation during normal initialization.
For audited vLLM 0.27.0, explain the fixed synchronous-scheduling policy for both
phases in this route: it matches supported prefill benchmarking and is not a vLLM default.

Prediction, recommendation and replay require resolved memory. The initial
observer supports audited vLLM 0.27.0 full/sliding cache interfaces and supported
convolution storage. Other versions retain native timing collection without
enabling the observer; unknown layouts report unresolved memory. Do not replace
that diagnostic with guessed bytes.
After complete formal collection and the stage 5 collection-quality checks
below, run `aisimulate onboard finalize --config
ORIGINAL/request.yaml --output-dir ORIGINAL --resolved-output-dir FRESH`.
It verifies native timing/resource evidence and formal data, then writes a new
profile and simulation plan with the verified pair. It preserves source artifacts,
uses the minimum compatible observed capacity, and rejects different cache layouts
or memory-relevant launch settings, including incompatible phase graph settings.
Review and accept this exact resolved profile separately; finalization never
accepts it automatically in the session checkpoint. Record the fresh request,
profile, data and provenance references while retaining the original collection.
Use the resolved directory for simulation. Observed capacity includes all resident
worker components even when timing models only text-decoder execution.

Make collection quality and matched serving validation part of the standard
onboarding procedure. In stage 5, follow
[Validate collection and serving accuracy](docs/fpm-self-service.md#validate-collection-and-serving-accuracy):
save an editable campaign policy before measurement, then run `onboard
validate-collection` against the original collection directory. Inspect native
validity, actual attention groups, graph configuration and KV initialization;
run the bounded representative repeats only with explicit `--execute`. Preserve
individual attempts and inspect both fresh-sample CV and CV including the
original published sample. Evaluate withheld coordinates through native direct
interpolation; keep unsupported queries separate from numerical errors. The
default policy is five fresh samples per point, at most 12 points per phase cell,
maximum CV 0.05, at most 16 holdouts per phase, seed 42, p95 absolute relative
error 0.20 and zero unsupported holdouts. These are editable initial criteria,
not statistical confidence guarantees. Do not overwrite original collection or
policy snapshots. Threshold changes use fresh assessment directories and can
reuse verified raw samples; changed selection/count/execution requires new
measurements. Old campaigns lacking source/runtime evidence remain incomplete.

After verifying the formal data pair and resolving memory, use `aisimulate onboard validate-fpm`
with a local Weka trace and a separate validation output directory. Follow the
guide's pinned AgentX reference and current cold aggregated, one-lane, HBM-only,
non-speculative scope. Preserve target model projection and strict direct FPM
with denied fallback. Read `validation.json` together with native query coverage
and request completion evidence. Coverage counts native lookup resolutions,
including interpolation and failures; cached timing reuse is not another query.
Missing timing stops replay and preserves partial evidence, which cannot certify
the remainder of the trace. Report coverage and silicon accuracy separately.
Reuse verified v2 collection plans for validation-only changes; legacy v1 plans
need a new directory as described in the guide.

In stage 6, follow the guide's separate-environment installation of the exact
Git-pinned AIPerf revision before using `onboard validate-serving --action
prepare|run|assess` for the matched serving check. Prepare does not launch anything. The caller launches a
fresh ordinary serving worker with the recipe's observation module/settings;
run explicitly sends pinned AIPerf traffic to that endpoint. Use one complete
single-stream play whose recorded predecessor API times are all zero/absent or
whose idle gaps are all zero. Other timing combinations, branches and corpora
remain incomplete; never flatten or edit a trace to make it pass.
Preparation freezes target-tokenized chat payloads and runs a separate ordinary
prediction/coverage check on the derived explicit trace while preserving source
play identities and timing. Keep its artifacts separate from the original Weka
coverage report; requested trace lengths exclude target chat-template overhead.
The producer verifies exact server tokenization before sending frozen payloads.
Inspect TTFT, TPOT and throughput separately and instrument forward timing where available;
missing forward samples are explicitly unavailable. Assessment rechecks source
hashes, native measurements, effective execution and resolved-memory linkage.
Only combined `accuracy: qualified_for_evaluated_scope` permits a bounded
workload-accuracy claim. Missing, stale or failed mandatory gates prevent that
claim; ordinary exploratory predict/recommend remains available. Register the
policy and workload choices in each configuration's `validation_inputs` and
reports with artifact `scope: validation` in the existing single checkpoint.
Validation changes invalidate assessment only; preserve accepted collection.

Grouped cache execution currently requires cold aggregated vLLM with PP1, CP1,
HBM-only storage, no speculative decoding and `prefix_caching: false`. Preserve
that setting in generated predict/recommend and validation configs. Native
allocation shares one byte budget across groups, retains full history or the
declared window, and charges temporary prefill pages. Window eviction does not
shorten logical request progress or FPM query context. Prefix reuse, offload/G3,
disaggregation and scalar capacity overrides are unsupported for grouped caches.

Do not reject a checkpoint solely because it is multimodal. Read its unambiguous
`text_config`, or its flat text-decoder fields, and explain that FPM models only
the text decoder. Config-derived estimates exclude multimodal encoders,
projectors, preprocessing and other non-text components. Observed runtime cache
capacity accounts for all components actually loaded by the worker, while timing
describes the text decoder only. Preserve this scope in the reviewed profile's
provenance. Do not subtract guessed encoder memory from an observation. Keep
validation of incompatible decoder/cache semantics intact.

For directory output, read `onboarding.json` and run each configuration's
`plan_command` for its own `collection/` directory. Inspect
the ordinary plan and preview collection before execution; initialization
does not launch collection. Use each plan's existing collection and validation
commands with separate result paths. The index locates configurations and next
plan commands; it does not replace the session checkpoint.

Report the current stage, its result or blocker, and the next action for each
configuration. Record artifact references and command results in the session
checkpoint; preserve the collector's own checkpoints rather than duplicating
their cell records. Resume from validated artifacts and accepted decisions
instead of repeating shared intake or completed collection. Checkpoint stage
labels and saved command strings are agent-supplied context, not proof of
successful collection, query coverage or accuracy, and must never be executed
automatically. Recheck the corresponding evidence. Relevant input or profile
changes invalidate affected acceptance and downstream progress; keep old
artifacts for inspection and regenerate into new paths. Explicitly mark
superseded artifact references `archived: true` and record the reason; register
current outputs under new names/paths as shown in the guide. Archives preserve
original reference metadata but are historical and unverified, excluded from
current integrity checks. Do not archive unresolved current evidence to hide
failures. Retiring an input reference invalidates dependent acceptance; restoring
an archived reference re-enables its checks. Research-note edits do not require
renewed profile acceptance. Keep validation-only choices separate
from collection inputs so they do not discard valid collected data.
Stage transitions are not additional approval gates. Preserve explicit review
and acceptance of the exact profile, user overrides/provenance, and existing
execution authorization as described in the guide's terminal and headless flows.
The config/profile route requires neither an op-level model class nor
per-operation silicon data. Report planning, collection, simulation and measured
accuracy separately.

## Pull request titles

- Use `<type>: <short description>` for every AISimulate PR title.
- Allowed types: `feat|fix|docs|style|refactor|perf|test|chore|ci|build|revert`.
- Check the title before creating or updating a PR.

## Performance Model Changes

Before changing a performance model, its configuration, or a caller in Rust,
Python, CLI, Sweeper, Replay, or Planner, MUST read and follow
[`perfmodel-api.md`](python/aisimulate/.claude/rules/perfmodel-api.md).
This includes new features and configuration migrations.

## Required First Step

Before making any change under:

- `python/aisimulate/src/aisimulate/generator/**`

MUST read:

- `python/aisimulate/.claude/rules/generator-development.md`

## Required Collector First Step

Before making any change under `python/aisimulate/collector/**` MUST read:

- `python/aisimulate/.claude/rules/collector/layer_permissions.md` (layer
  permission table, module boundary, dispatch-vs-skip rule)
- `python/aisimulate/.claude/rules/collector/failure_handling.md`
  (observe-don't-predict doctrine, escalation decision tree)
- For case YAML work:
  `python/aisimulate/.claude/rules/collector/case_authoring.md`

For adding a new Collector operation, additionally follow
`python/aisimulate/.claude/skills/aic-collector-op-development/SKILL.md`
(consumer-contract, case-identity, deduplication, and validation gates). Skills are procedural
runbooks; if a skill and a `python/aisimulate/.claude/rules/` file ever
disagree, the rule file wins.

## Required Third-Party Attribution

Before adding code or other content copied, adapted, translated, or
substantially derived from an external project, MUST:

- Identify the upstream repository, immutable commit or tag, original path,
  and applicable license. This applies to source code, tests, configuration
  files, patches, fixtures, documentation, and generated derivatives.
- Preserve all applicable upstream copyright, license, attribution, and NOTICE
  material, and mark modified files as modified when the upstream license
  requires it.
- Record the source URL and revision in the file header or an adjacent README.
  For formats that cannot carry comments, use an adjacent attribution or
  license file that is included in distributions.
- Add an entry to the root `THIRD_PARTY_NOTICES.md` identifying the derived
  files, upstream source and immutable revision, copyright owner, applicable
  license, and whether the files were modified. The root notice is canonical;
  keep `python/aisimulate/THIRD_PARTY_NOTICES.md` byte-identical so the notice
  is included in Python distributions. Run
  `python3 scripts/check_packaged_legal_files.py` after either copy changes.
- Do not hand-edit generated attribution artifacts. Update their source or
  generation process instead.

The repository's Apache-2.0 license and NVIDIA copyright header do not replace
an upstream license or attribution requirement. Do not label third-party code
as exclusively NVIDIA-authored. If the source, revision, license, compatibility,
or required notice is unclear, stop and request maintainer or Open Source Review
Board guidance before committing the derived content. This is a required gate,
not optional documentation.

## Cursor Cloud specific instructions

### Project overview

AISimulate is a Python and Rust CLI/SDK for predicting LLM serving behavior and
optimizing inference deployment configurations. See `README.md` for details.

### Environment setup

Python dependencies are managed via `uv` with the
`python/aisimulate/uv.lock` lockfile. The project environment lives at
`python/aisimulate/.venv/`.

- **Install/refresh deps:**
  `uv sync --project python/aisimulate --extra dev`
- **Performance data:** Current op profiles are parquet files under
  `python/aisimulate/src/aisimulate_core/systems/data/<system>/<family>/<backend>/<version>/`
  and are checked in directly. Legacy `*.txt` perf files, when present, use Git
  LFS; run `git lfs pull` only when working with those legacy assets.

### Lint / Test / Run

- **Lint:** `python/aisimulate/.venv/bin/ruff check --config python/aisimulate/pyproject.toml python/aisimulate tests`
  and `python/aisimulate/.venv/bin/ruff format --check --config python/aisimulate/pyproject.toml python/aisimulate tests`
  (see `DEVELOPMENT.md`)
- **Unit tests:** `python/aisimulate/.venv/bin/pytest -c python/aisimulate/pytest.ini python/aisimulate/tests -m unit`
  (no external deps or LFS data needed)
- **Build tests (PR subset):** `python/aisimulate/.venv/bin/pytest -c python/aisimulate/pytest.ini python/aisimulate/tests -m "unit or build"`
  (requires LFS data for the `build`-marked tests)
- **CLI:** `aiconfigurator cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm` (works without LFS data)
### Known environment caveats

1. **Legacy LFS data:** `github-cloud.githubusercontent.com` may be blocked by
   network egress restrictions. If `git lfs pull` fails, tests that explicitly
   exercise legacy text assets may fail; current parquet-backed workflows do
   not depend on those legacy files.
2. **TTY tests:** 4 tests in
   `python/aisimulate/tests/unit/cli/test_plain_output.py` may fail because the
   agent runs in a non-TTY environment.
3. **Rust tests:**
   `python/aisimulate/tests/unit/sdk/test_rust_engine_step.py` requires `cargo`
   with network access to `crates.io`. It will fail if that domain is blocked.
4. **macOS pytest-timeout crash dialogs:** The `timeout = 120` setting in
   `python/aisimulate/pytest.ini` uses SIGALRM by default, which triggers
   "Python unexpectedly quit" crash reporter popups on macOS. Pass
   `-p no:timeout` to disable it locally.
5. **torch-dependent tests:**
   `python/aisimulate/tests/unit/sdk/database/test_moe_dispatch.py` requires
   `torch` (not installed in the default dev venv). Ignore it with
   `--ignore=python/aisimulate/tests/unit/sdk/database/test_moe_dispatch.py`.

## CODEOWNERS

The root `CODEOWNERS` is generated from `.github/codeowners/areas.yaml` - never
hand-edit it; CI fails on drift. Repository rules must require the `codeowners`
check to make failures merge-blocking. If the check fails on a new directory,
claim it in `areas.yaml`, regenerate with
`.github/codeowners/emit_codeowners.py`, and commit every changed source and
generated artifact together. The `aic-codeowners` skill covers all flows (who
reviews a change, gate failures, routing changes, external contributor grants).
