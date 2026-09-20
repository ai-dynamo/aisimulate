<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FPM self-service

`aisimulate onboard` guides onboarding a new model for FPM simulation on your designated hardware platform. It records the model, runtime, target GPU system and interconnect, and lets you select one or more TP, DEP, or TEP worker configurations. Each configuration has its own resource profile, collection plan and minimum collection GPU requirement. You review the resource and collection limits, collect whole-forward timings through Dynamo self-benchmark, then validate their query coverage using ordinary trace replay. Each plan also produces ordinary `predict` and `recommend` configurations for its worker.

Collection limits and validation traffic are separate inputs. AISimulate sets runtime limits and a prefill CUDA graph policy. New requests leave capture configuration to the pinned runtime; an explicit capture extension is available when needed. Dynamo self-benchmark uses the initialized engine, image sampling defaults and runtime feasibility checks to generate and measure the exact grid. AgentX traces exercise the resulting FPM library through replay; their variable request lengths do not require a fixed input/output length or latency target during onboarding.

Planning works before the model has an AISimulate model class or measured FPM timings. With a supplied FPM profile, planning validates the declared deployment identity and estimates memory admission from your resource bounds. Runtime compatibility and data readiness remain **unchecked**, and accuracy is **not assessed**. This setup does not provision GPUs or run target preflight checks.

Ordinary FPM `predict` and `recommend` accept an inline `engine.fpm_profile` with model identity and rank-local resource bounds. Direct interpolation uses measured timings without constructing an op-level model. Registered models retain SOL interpolation. See [Choose the model execution route](#choose-the-model-execution-route) for selection and coverage rules.

## Onboard with an agent

For a request such as "Help me onboard my model for FPM simulation on my target GPUs," follow these six stages. Claude Code reaches them through the root `CLAUDE.md` import of `AGENTS.md`; other agents use the same `AGENTS.md` entry point. Create one [session checkpoint](#checkpoint-and-resume-an-onboarding-session) during stage 1 and update it after meaningful findings and decisions, including unfinished investigation and partial profile review. The sections below remain the detailed CLI reference.

| Stage | Required result |
| --- | --- |
| 1. Inspect the model and target | Accessible config/profile and packaged hardware specification identified; supported metadata read and gaps recorded. |
| 2. Choose the worker and collection limits | Sourced joint precision options and collector limitations reviewed; checkpoint identity/revision, runtime, interconnect, precision combination(s), worker configurations, memory fraction, context, scheduler bounds and capture policy selected. Minimum collection GPUs are derived for each; actual target-runtime compatibility remains unchecked. |
| 3. Derive, review and save the profile | Each configuration's exact resource/precision values and assumptions reviewed and accepted; each final request contains its complete profile and provenance. |
| 4. Plan collection | Validated saved plan, generated configurations and collector preview; sampling scope and remaining execution prerequisites explained. |
| 5. Collect and verify data | Matching formal Parquet/metadata pair verified, with provenance and available phase cells recorded. |
| 6. Validate replay and run predict/recommend | Cold aggregated trace replay checks direct-FPM query coverage; completed and incomplete results are distinguished. Ordinary prediction/recommendation examples remain available; accuracy is assessed separately. |

At each transition or blocker, give a short update: **Stage N/6 — name; result or blocker; next action.** Ask only for missing information needed for the current stage, using facts and decisions already supplied. Continue independent authorized work while an answer is pending. Move on when the required evidence is available; stage transitions do not require another approval. Honor prior authorization, including authorization to execute collection, while preserving the profile acceptance step below.

### 1. Inspect the model and target

Start with only the missing **Hugging Face model ID** (`organization/model-name`) and **target GPU platform**. If neither is known, a first response can be:

> Stage 1/6 — Inspect the model and target. What is your Hugging Face model ID (`organization/model-name`) and target GPU platform? I'll retrieve and inspect the configuration first. If your model isn't on Hugging Face, share a local config or checkpoint path instead.

Omit facts already supplied from this request. Accept a supplied local config, complete FPM profile or checkpoint path without requiring a Hub ID; do not ask for both an ID and a config upfront. With a Hugging Face ID, use available authorized Hub access to retrieve only `config.json` at this stage. Honor any supplied revision and record the repository ID and resolved commit. If no revision was supplied, report the resolved commit for the deployment choice in stage 2. If the config cannot be retrieved, explain the specific gap and ask for a local config. The agent retrieves the file; AISimulate's CLI consumes a local config/profile and does not download it or the checkpoint.

Inspect the accessible config/profile before asking for model kind, architecture, expert count or context limit; derive supported metadata and record its source. Ask about these fields only when the configuration leaves them unresolved. For CLI setup, `--model` identifies the selected Hugging Face repository or actual checkpoint path, `--model-config` names the local JSON file, and `--model-revision` pins the checkpoint separately. A config file hash is not a checkpoint revision. This route does not require an op-level model class or per-operation silicon data. If the hardware lacks a packaged system specification, report that integration gap.

A natively multimodal checkpoint can be onboarded for its text decoder. Explain this scope during inspection and profile review: FPM excludes multimodal encoders, projectors, preprocessing and other non-text components and their resource costs. The resulting profile and timings do not model full multimodal deployment memory or latency. Keep that scope in profile provenance; unknown decoder resource bounds still need explicit input.

The agent handles checkout and environment checks: record the branch/commit, follow [development setup](../DEVELOPMENT.md#initial-setup), activate the environment, and inspect `aisimulate onboard --help` and `aisimulate onboard init --help`. Check later subcommands before using them. Report missing commands/options as a version mismatch, rather than asking the user to supply unsupported inputs.

### 2. Choose the worker and collection limits

At stage 2 entry, reuse stage 1's checkpoint revision and target hardware, and resolve the literal framework version from supplied or available deployment metadata. Ask for the missing version pin only if it remains unknown, then investigate that version before proposing precision choices. Resolve any remaining checkpoint identity or interconnect gaps from available metadata before asking. A model label, config hash or example revision does not establish a checkpoint pin. An existing vLLM launch command or configuration is optional evidence: preserve explicit choices, but do not require one or begin with a questionnaire about every topology and dtype. Do not ask for total available GPUs, node allocation, GPUs per node or replica budgets during onboarding; those are choices for actual prediction or recommendation runs.

Complete the [runtime investigation and options table](#investigate-runtime-constraints-and-precision-options) before asking for derivable precision or cache inputs. New config-based setup proposes `comm_quant_mode: half` with the current collector's identity source; this is not a claim that all NCCL tensors use FP16. The agent investigates and prepares supported inputs; the CLI does not import arbitrary launch arguments or automatically inspect a remote runtime.

Show the [runtime and collection defaults](#runtime-and-collection-limits): 0.90 GPU memory utilization, profile scheduler bounds or 8,192 scheduled tokens and 256 sequences per rank, and `--prefill-cudagraph-policy runtime`. Fresh config/profile-based setup caps context at the smaller of the declared limit and the 256,000-token AgentX reference bound. Runtime graph policy leaves the effective mode/capture list and exact sample grid unresolved until engine initialization. Explain these starting policies and allow edits; they do not establish memory fit or timing coverage. Do not ask for fixed input/output lengths, concurrency, TTFT or TPOT as required collection inputs. Those flags customize optional synthetic validation examples.

With a local model config, inspect the [read-only topology preview](#preview-and-choose-parallelism) for each considered precision combination using the actual identity, runtime, target and collection limits. Supply the precision/layout facts established by inspection, investigate remaining diagnostics, ask only for unresolved choices, and rerun the preview. Present the default and alternatives with their resource assumptions and unresolved fields. A candidate marked `estimated_fit` has a complete declared/estimated byte budget; it is not a performance ranking, runtime qualification or a measurement. If no default is available, explain why and select an exact candidate before resolving its per-rank bounds. Do not silently assume topology or precision.

Help choose one or more TP configurations, or the relevant TP/DEP/TEP configurations for MoE, using the preview's exact flags or the [supported topology fields](#create-the-request). Honor explicit choices even when they lie outside the automatic shortlist. Explain each selected tuple; do not require the user to know every parallelism field upfront. Derive its minimum collection GPUs as attention TP times attention DP: TP4 requires four GPUs, while DEP8 requires eight. This requirement does not declare available capacity or establish runtime placement or compatibility. Use [directory output](#onboard-multiple-parallel-configurations) to prepare several configurations in one session; each profile and collection plan still selects one exact tuple. Collection runs can reuse the same GPUs, so do not add their requirements into an onboarding GPU budget.

#### Investigate runtime constraints and precision options

Perform this investigation proactively once the checkpoint, hardware and framework version are known. A local source checkout, user-supplied launch command, initialization logs and a prestarted server are optional evidence, not prerequisites. Use available local sources first and authorized official remote sources when needed. Do not download weight shards or launch GPU work merely to discover options.

1. Inspect the pinned checkpoint's config, quantization sidecars and lightweight tensor/index metadata where available. Identify quantized and unquantized weight parts and any fixed cache/state constraints; a repository name or one weight-quantization label does not describe every tensor. Distinguish metadata-derived estimates from measured allocations.
2. Read the pinned official framework's model implementation, quantization/backend selection and attention/cache implementation. Trace model dtype validation, backend/kernel dispatch, cache dtype/block sizing and convolution or other state allocation, including version, GPU and topology conditions. Record the relevant flags, environment variables and defaults. For a supplied vendor/custom build, inspect its pinned image/build metadata and known patches; record uncertainty if they are unavailable. Unpatched release behavior alone does not establish that build's limits. General dtype enums or another version's documentation do not establish support for this model. If the pinned build lacks the model or a required backend, report that incompatibility explicitly.
3. Classify each finding as a **fixed checkpoint constraint**, **runtime default**, **configurable alternative**, or **unknown**. Keep model dtype, mixed weight storage, FMHA FPM identity and actual runtime compute/dispatch, attention KV dtype, convolution state and communication identity separate. Trace their dependencies to form plausible joint combinations; do not create a Cartesian product of independently advertised dtypes. Keep unknown FMHA/KV facts unresolved rather than inferring them from weight precision. Record missing evidence and access limits before asking only for facts or choices that investigation cannot resolve.

Present a concise **options table with one row per joint combination**, including the default and evidence-backed alternatives. Do not invent alternatives to fill the table. Use these columns:

| Column | Required content |
| --- | --- |
| Option and joint precision | Distinct weight parts/model dtype, FMHA identity and runtime compute/dispatch, attention KV and fixed state; label fixed, default, configurable or unknown facts. |
| Conditions and controls | Pinned version, GPU, backend and topology dependencies; required runtime flags/environment settings and defaults. |
| Sources and confidence | Checkpoint revision and metadata path; official source URL/path and tag/commit for each claim. Distinguish source-confirmed behavior, inference and unresolved or untested conditions. |
| Runtime support | Supported by the inspected version, conditional/unverified, or ruled out, with the reason. Source review does not establish successful target execution. |
| Collector compatibility | Compatible with the current profile collector's identity checks, blocked, or unresolved, independently of runtime support. |
| Resource effects and limits | Expected changes to weights, attention cache, fixed state, workspaces/padding and memory fit; retain unknown bounds. Accuracy remains unmeasured without matched evidence. |

The current profile collector requires KV dtype to equal its checkpoint-inferred native dtype and checks GEMM, MoE, FMHA and communication identities against its resolved configuration. Those checks restrict collection even when the pinned runtime supports another combination. Mark any blocked alternative with its precise limitation; user selection or a schema-valid profile does not override serving dispatch. Do not relabel checkpoint data or use extra launch arguments to disguise an unsupported identity. See [profile collection constraints](#plan-preview-and-explicitly-execute).

Treat runtime defaults as one proposal and let the user choose the precision combination(s) to cover, preserving prior explicit choices. Record the choice, rationale, sources and unresolved conditions in profile provenance. A blocked selection stays blocked until the required capability is available; do not substitute a default silently. For multiple selected combinations, use a separate fresh output root per precision combination and the existing `--parallel-configs` topology list within each. Duplicate resolved topologies are rejected even if their precision overrides differ. This is agent orchestration over the existing CLI, not a new precision-matrix command or schema. Recompute resources and review each exact precision/topology profile in stage 3 before accepting it.

### 3. Derive, review and save the profile

Use [a local model config](#start-from-a-local-model-config), or [a supplied profile](#provide-identity-and-resource-metadata), for the selected deployment. Both produce a class-independent direct-FPM plan. Derive supported estimates before asking for unresolved resource fields. Explain each value's source and limitations; do not invent missing bounds. Memory values must bound every rank of the exact selected topology. Resolve effective weight, FMHA, communication and KV precision separately; a quantized checkpoint label does not determine all of them. Preserve replacements and their rationale in overrides/provenance. Review all effective values and assumptions using the appropriate flow below.

For several configurations, read shared model, runtime and hardware inputs once, then derive and review each selected precision/topology profile independently. Do not transfer rank-local byte bounds or `cache_groups` between precision combinations or tuples. Put headless overrides under the corresponding `--parallel-configs` entry's `resource_overrides`; interactive edits apply only to the profile being reviewed. Shared `cache_block_sizes` can be reused only when valid for the same runtime, backend and precision, with page bytes derived for each tuple.

For sliding-window or supported convolution state, follow [grouped cache review](#review-grouped-cache-resources). Derive layer geometry first, then resolve runtime block sizes for each chosen worker and review aggregate page bytes, including padding. A scalar bytes-per-token estimate cannot replace these groups.

For agent sessions, use the headless flow below and [checkpoint each configuration's draft and acceptance](#checkpoint-and-resume-an-onboarding-session), even when a terminal is available. This preserves partial review across sessions. The standalone interactive flow remains available, but does not automatically save its prompts or partial acceptance to the session checkpoint.

| Workflow | Review and save behavior |
| --- | --- |
| Checkpointed agent session, with or without a PTY | Supply identity/collection flags and overrides without `--interactive`; use `--parallel-configs` with `--output-dir` for several configurations of one selected precision combination. Missing required inputs exit 2 without saving; use the diagnostics to investigate first and ask only for still-unresolved facts or choices. A successful command writes immediately. Initially write to a separate path such as `draft-request.yaml` or a fresh `draft-onboarding/` directory and show every embedded profile, its sources and scope for user review. Save each configuration's draft, edits and explicit acceptance in the session checkpoint as review progresses. Apply edits in the inputs/overrides and repeat draft review; rerun unchanged accepted inputs to a fresh final path and verify that the profiles match the accepted drafts before planning. Draft names are only a convention; they have no special CLI status. Review a supplied `--fpm-profile` in the same way. |
| Standalone interactive CLI in a terminal | Run `aisimulate onboard init --model-config /path/to/config.json --interactive --output-dir onboarding`, or retain `--output support-request.yaml` for one request. Directory output accepts comma-separated candidate numbers. Inspect missing-field prompts and review the final profile, using `edit` to revise its values before entering `accept`. All selected profiles must be accepted before directory output is saved. Cancellation creates no new artifacts and preserves prior output; partial prompts or acceptance are not persisted in the session checkpoint. This final review is specific to `--model-config --interactive`; supplying `--fpm-profile` does not add it. |

Do not pipe answers into `--interactive`: it requires a terminal. Scripted setup has no built-in acceptance prompt. Keep drafts separate from final requests and use a fresh draft directory for each revision. Preserve prior outputs when revising a deployment.

Model identity, runtime and topology are not profile-review edit fields. If they change, return to stage 2, regenerate dependent estimates and review the new request; use new output paths for the changed deployment. A precision change also requires reconsidering the joint combination and collector compatibility, recomputing dependent resources and reviewing the complete revised profile. Explicit byte overrides can survive CLI edits; rederive any that depend on the changed precision rather than retaining stale bounds.

### 4. Plan collection

Follow [Plan, preview, and explicitly execute](#plan-preview-and-explicitly-execute) using a new output directory. For directory output, read `onboarding.json` and use each emitted `aisimulate onboard plan` command to create that configuration's `collection/` directory. Inspect each `support-plan.json`, embedded/saved profile, generated prediction/recommendation configs and `commands.json`. Run `onboard collect-fpm` without `--execute` to print the collector command; this does not run the collector's own plan or check the target runtime. Inspect the read-only collector plan when its input environment is available and identify any missing prerequisites.

Explain each configuration's minimum collection GPUs, rank-local scheduler/resource envelope and [shared collection policy](#how-the-collection-grid-is-determined). AISimulate configures collection inputs and Dynamo generates the exact points; the plan does not derive a second grid from trace requests. In runtime graph mode, report the capture list and exact point count as unresolved until engine initialization. Each plan selects one parallel tuple and generates single-worker validation configs. Continue stages 5 and 6 separately for each configuration, preserving its data and result paths. Actual collection resources and placement must be checked in the collector environment before execution. The synthetic request count does not bound timing samples or collection duration.

### 5. Collect and verify data

First inspect any existing timing data for a matching deployment. Reuse a verified matching formal pair when available; do not collect again merely to complete a stage. For new collection, follow the [FPM collection guide](../python/aisimulate/docs/fpm/end-to-end-workflow.md) to prepare the compatible pinned Dynamo/vLLM image, accessible checkpoint, GPU resources, Kubernetes namespace and deployment permissions. An already-running HTTP server is not required: the AISimulate collector deploys and launches benchmark workers through the existing Generator/Kubernetes path. The engine initializes the model and cache, resolves its graph configuration, and Dynamo self-benchmark generates and times the admitted points. Use the agreed scope and existing execution authorization, reporting concrete missing prerequisites when blocked. Preview and execute with the same deployment options. `--execute` launches collection; optional `--smoke` is diagnostic and publishes no formal FPM pair. Inspect effective precision, graph mode/capture sizes, cache allocation/padding and supported benchmark seeding during bring-up; preserve checkpoints, initialization logs and raw evidence.

For both reused and new data, [inspect the published pair](../python/aisimulate/docs/fpm/end-to-end-workflow.md#5-inspect-the-published-pair): verify hashes, schema and identities, including actual runtime, topology, precision and available prefill/decode cells. Keep it at the generated configs' local systems path. Record any historical checkpoint-revision uncertainty in provenance; a declared revision does not prove that old measurements used it. A successful preview or smoke run is not formal data, and a matching pair does not prove all simulated queries are covered.

### 6. Validate replay and run predict/recommend

[Validate a local AgentX trace](#validate-fpm-query-coverage-with-agentx-replay) through `onboard validate-fpm`. It writes an inspectable ordinary prediction config, runs cold aggregated replay, and saves coverage evidence separately from the collection plan. Use the selected complete local corpus or one complete play for a first check, and state that selection. Keep `engine.systems_paths`, the target model projection and the direct-FPM profile intact. Report the command's exit status and exact missing coordinates. Missing timing stops replay; a partial report does not audit the rest of the trace.

[Run the generated ordinary configurations](#run-the-generated-ordinary-configurations) when a synthetic prediction or recommendation check is useful. For deployment prediction or optimization, set the desired replicas and GPU budget in the ordinary runtime configs. Do not change precision labels or silently switch timing methods to obtain a result. Return to stage 5 to address missing data, or stage 2 if the selected deployment or collection limits change. A different validation trace alone does not invalidate collected timings. Report the simulation stage as incomplete while required queries fail.

At handoff, provide the session checkpoint path and ensure it records the checkout revision, final request/profile, plan directory, data pair/provenance, exact commands/exit statuses and all result paths. Distinguish estimated memory fit and CPU planning from actual target-runtime checks, formal data/coverage, and completed simulations. For accuracy, report an independent matched silicon comparison if performed, or explicitly **not assessed**. Successful simulation is not evidence of accuracy; an accuracy study is not a mandatory additional collection campaign for onboarding.

### Checkpoint and resume an onboarding session

Use one `onboarding-checkpoint.json` for the whole session, including every selected precision/topology configuration. The agent creates it during stage 1, before all inputs are known, and saves after meaningful findings, user decisions, draft edits, acceptance and command results. Users do not need to request a save. These commands persist the supplied state; they do not observe conversations, investigate remote capabilities or automatically intercept other onboarding commands. The agent is responsible for invoking them.

Keep the file outside fresh `init --output-dir` roots, for example beside `drafts/` and separate final precision directories under `model-onboarding/`. Those directories must still satisfy the existing fresh-output rules. Create the checkpoint with:

```bash
aisimulate onboard checkpoint \
  --file ./model-onboarding/onboarding-checkpoint.json
```

Calling the same command again inspects existing state without replacing it. Each saved revision has a generated revision number. Updates require `--expect-revision` with the number just read; a stale revision rejects the update. Reload and reconcile concurrent changes instead of retrying with an invented revision. Saves use an OS lock and atomic file replacement. The adjacent lock file is an implementation detail, not another user checkpoint; leave it in place.

The JSON document uses `schema_version: "aisimulate-onboarding-checkpoint/v1"`. Its [schema](../python/aisimulate/src/aisimulate/support/checkpoint.py) holds:

| Field | Purpose |
| --- | --- |
| `inputs` | Shared effective model, runtime, hardware and collection choices, including immutable revisions. Incomplete inputs are allowed. |
| `research`, `decisions`, `pending_questions` | Findings with sources/confidence, user rationale and unresolved questions. These are context, not executable instructions or acceptance. |
| `validation_inputs` | Trace selection and other validation-only choices, separate from collection identity. |
| `progress` | Agent-reported stage (`1`–`6`), status (`in_progress`, `blocked` or `complete`), blockers and next action. A saved status does not prove successful execution. |
| `configurations` | Records keyed by a stable user-chosen ID. Each has its own inputs, draft request, validation inputs, progress and artifact references; acceptance is generated by the CLI. |
| `artifacts` | Shared source-file references. Configuration records hold their own profiles, plans, data, results and collector-checkpoint references. |

Both commands return a JSON report containing the full saved `state`, per-configuration `profile_accepted` and `effective_status`, `integrity_issues`, `archived_artifacts`, verification limits and `saved`. Read the current revision from `state.revision`. `effective_status` is `draft`, `accepted` or `needs_attention`; the separately reported stage/status is agent-supplied, not a verified completion result. An update can save successfully while reporting existing artifact integrity issues and exiting 2; check `saved` and the new revision before retrying.

`--update FILE.json` applies a recursive JSON merge patch; `--update -` reads the patch from stdin. Objects merge, arrays replace and `null` removes a field where the schema permits it. Do not edit CLI-owned schema, revision, acceptance or hash fields. Put effective selections in `inputs` or the configuration's `draft_request`, not only in research or decision notes: dependency checks use the former. For example, after initial creation, save supplied facts and unfinished investigation with the current revision:

```bash
aisimulate onboard checkpoint \
  --file ./model-onboarding/onboarding-checkpoint.json \
  --expect-revision 1 --update - <<'JSON'
{
  "inputs": {"model": "organization/model-name", "gpu": "h200_sxm"},
  "research": {"runtime": {"status": "pending", "sources": []}},
  "pending_questions": ["Which pinned vLLM version will collection use?"],
  "progress": {"stage": 1, "status": "in_progress", "next_action": "Retrieve and inspect the model config."}
}
JSON
```

Replace these illustrative inputs with the user's facts. The example revision assumes the preceding creation was the only save; always use the actual current revision.

#### Preserve partial profile review

Use [headless draft generation](#3-derive-review-and-save-the-profile) for checkpointed agent sessions. Save each complete generated request, including its embedded profile and provenance, as that configuration's `draft_request`. Incomplete draft requests may also be saved while resolving inputs, but cannot be accepted. For example, after generating two draft requests at `model-onboarding/draft-a.yaml` and `model-onboarding/draft-b.yaml`, prepare their update without changing the requests:

```bash
python - <<'PY'
import json
from pathlib import Path
import yaml

root = Path("model-onboarding")
update = {"configurations": {}}
for name in ("a", "b"):
    update["configurations"][f"worker-{name}"] = {
        "draft_request": yaml.safe_load((root / f"draft-{name}.yaml").read_text()),
        "progress": {"stage": 3, "status": "in_progress", "next_action": "Review this exact draft request and its assumptions."},
    }
(root / "draft-update.json").write_text(json.dumps(update, indent=2) + "\n")
PY
aisimulate onboard checkpoint \
  --file ./model-onboarding/onboarding-checkpoint.json \
  --expect-revision 2 --update ./model-onboarding/draft-update.json
```

The update file is disposable input; the checkpoint contains the drafts. Show each complete request/profile, source and assumption to the user. Save requested edits before presenting the revised values. Only after the user explicitly accepts `worker-a`'s exact values, record that acceptance:

```bash
aisimulate onboard checkpoint \
  --file ./model-onboarding/onboarding-checkpoint.json \
  --expect-revision 3 --accept-profile worker-a --update - <<'JSON'
{
  "configurations": {
    "worker-a": {
      "progress": {"next_action": "Publish the accepted draft to a fresh final path, verify equality, then plan collection."}
    }
  }
}
JSON
```

The CLI validates the complete request/profile and binds acceptance to its content and relevant shared/configuration inputs. `worker-b` remains a draft. Repeat `--accept-profile` to record multiple explicitly approved configurations together; never use it to infer approval from a saved file, stage label or another configuration's acceptance. Saving or resuming does not publish final requests or launch collection. Regenerate each accepted request to a fresh final path using unchanged reviewed inputs and verify equality before planning; checkpoint acceptance does not replace that comparison. Standalone `init --interactive` remains all-or-nothing and does not save partial prompt/review state automatically.

#### Record artifacts and resume

Artifacts are objects keyed by a descriptive name. Register only existing files, with paths relative to the checkpoint directory where possible. Each reference declares `kind` (`file`, `request`, `profile` or `collector_checkpoint`) and `scope` (`input`, `collection` or `validation`). Shared artifacts require `scope: input`; collection and validation artifacts belong to a configuration. For example, the following configuration fragment registers a final request after publication:

```json
{
  "configurations": {
    "worker-a": {
      "artifacts": {
        "final_request": {"path": "final-a/request.yaml", "kind": "request", "scope": "collection"}
      }
    }
  }
}
```

Immutable files receive a SHA-256 snapshot when registered. `kind: request` and `kind: profile` also check canonical contents against the configuration's draft. Later unrelated saves, including resubmitting the same path, do not silently refresh a snapshot. References are current by default (`archived: false`). Keep old output files for inspection, explicitly set superseded references to `archived: true`, and register current outputs under new names/paths. Archives retain their original path, kind, scope and hashes. The report lists them under `archived_artifacts` as historical and unverified: they do not undergo current integrity or semantic checks, and missing archived files do not block resume. Current outputs still require the usual verification; do not archive unresolved current evidence merely to suppress a failure.

For example, after accepting worker-a's revised draft and publishing its matching request and plan into `final-a-v2/`, archive the old request/plan/collector references while registering the replacements. If the checkpoint now reports revision 7, use the following patch; substitute the actual revision and reference names:

```bash
aisimulate onboard checkpoint \
  --file ./model-onboarding/onboarding-checkpoint.json \
  --expect-revision 7 --update - <<'JSON'
{
  "research": {"worker-a-revision": "Revised collection limits; earlier outputs retained for inspection."},
  "configurations": {
    "worker-a": {
      "artifacts": {
        "final_request": {"archived": true},
        "plan": {"archived": true},
        "collector": {"archived": true},
        "final_request_v2": {"path": "final-a-v2/request.yaml", "kind": "request", "scope": "collection"},
        "plan_v2": {"path": "final-a-v2/collection/support-plan.json", "scope": "collection"}
      }
    }
  }
}
JSON
```

Retiring an `input` reference removes it from the dependency context and invalidates affected acceptance and downstream progress; review and accept the revised inputs explicitly. Setting `archived: false` restores checks against the original snapshot/context, so a stale, modified or missing reference reports an issue again. Deliberately replacing a snapshot at the same path requires removing its reference in one save and registering it again after review; prefer new paths and archived references to preserve history.

Record the existing collector checkpoint as `kind: collector_checkpoint` with `scope: collection`: its mutable cell progress stays in the collector's own file and is not duplicated or treated as immutable drift. The session checks that this file exists and is a readable JSON object; the collector verifies its schema and frozen-plan identity on actual collection resume.

Resume in the same or another agent session using the single entry point:

```bash
aisimulate onboard resume \
  --checkpoint ./model-onboarding/onboarding-checkpoint.json
```

Use the checkpoint's absolute path when resuming from another working directory. Its relative artifact paths resolve from the checkpoint directory. Resume is read-only: it returns saved context, effective per-configuration progress, integrity issues and next actions without accepting a profile, executing saved command strings or starting collection. Missing or modified immutable artifacts produce actionable issues and a nonzero exit status while retaining readable saved context. An unreadable, malformed or unsupported-schema checkpoint fails with a concise error.

Relevant shared inputs affect every configuration; configuration input/draft changes affect only that configuration. Such changes invalidate affected acceptance and downstream progress. Research/rationale-only edits preserve acceptance. Changes under `validation_inputs` invalidate validation progress without discarding valid collection data. Archive superseded references explicitly when replacing stale outputs; leaving them current continues to block readiness even after replacements are registered. Do not reuse stale outputs as current or delete completed collection to make a status pass.

Resume verifies checkpoint integrity and current referenced file snapshots; archived references are historical and unverified. It does not establish runtime compatibility, inspect every plan/data semantic or certify collection, coverage or accuracy from saved stage strings. Inspect the corresponding command results and ordinary plan/data/validation evidence before continuing. Existing collector frozen-plan identity checks remain authoritative when actual collection resumes through `onboard collect-fpm --resume`.

### Resume from existing work

If a session checkpoint exists, run `onboard resume` first, resolve integrity issues and continue each configuration's unfinished work without repeating shared intake or accepted review. If older work has no session checkpoint, inspect the saved request/profile, plan, data pair and results, then record their verified state and any explicit acceptance evidence in a new checkpoint. For directory output, `onboarding.json` locates configurations and next plan commands; it is an output index, not the session checkpoint. A saved file without evidence of acceptance still needs stage 3 review.

Validate that artifacts still match the checkout's CLI, selected deployment and collection settings using the existing checks below. An accepted final request can start at stage 4, a valid plan at stage 5, and a verified matching data pair at stage 6. Preserve completed artifacts when blocked and save the specific missing input and next action. Changed deployment, resource profile or collection bounds require review and a new collection directory. Validation-only changes can reuse the verified collection plan; use a separate results directory for each replay. Collection's `--resume` continues its matching collector campaign, while `onboard resume` supplies the broader session context.

## Create the request

Use an environment installed from this checkout; see [development setup](../DEVELOPMENT.md). Guided setup requires a terminal and starts only when explicitly requested:

```bash
aisimulate onboard init --interactive --output support-request.yaml
```

Enter the actual model identifier or checkpoint path, pinned model revision, dense/MoE kind, pinned vLLM version, GPU system and interconnect. Choose a worker and review its context, scheduler and capture limits. Setup derives and displays the GPUs required for that worker. Supplied options skip their prompts. Enter accepts displayed defaults; invalid values can be corrected; Ctrl-C or end-of-input cancels without saving. Existing files require `--overwrite`.

Both guided and scripted setup use onboarding. `--profile onboarding` is an optional spelling of the same behavior. For automation, supply the identity flags directly:

```bash
aisimulate onboard init \
  --model /models/your-pinned-checkpoint \
  --model-revision YOUR_IMMUTABLE_REVISION \
  --model-kind dense \
  --framework-version YOUR_PINNED_VLLM_VERSION \
  --gpu h200_sxm --interconnect nvswitch \
  --tensor-parallel 2 \
  --output support-request.yaml
```

Replace the model, runtime and hardware inputs with your deployment's values. Framework support currently selects vLLM. Optional tokenizer, chat-template, and AISimulate revisions are recorded only when supplied.

With `--model-config`, omitted parallelism flags trigger model/hardware-aware suggestions. Headless setup selects only a fully assessed default; otherwise it exits without saving. Identifier-only and supplied-profile setup retain TP1 when parallelism is omitted. Review the default limits below and change them through `aisimulate onboard init --help` options or config-based profile review.

The optional `predict/pilot.yaml` and `recommend/pilot.yaml` examples use 1,024 input tokens, 128 output tokens, concurrency 1, four requests, TTFT target 1,000 ms and TPOT target 100 ms. Their workload/SLA flags do not size memory, choose the collector grid or change the collection identity. The example token lengths must still fit the selected runtime context. Guided intake does not prompt for these synthetic workload or SLA fields.

Onboarding takes no GPU-pool or node-allocation inputs. The selected topology establishes a minimum collection requirement, not a reservation or a check that those GPUs are available. Advanced flags include `--request-count`, `--objective`, `--seed`, and `--sm`. Replica counts and optimization budgets remain configurable in ordinary `predict` and `recommend` inputs.

`--tensor-parallel` always means attention TP. Use all MoE dimensions explicitly for DEP and TEP; replicas are independent workers:

| Worker | CLI parallelism flags | `(TP, PP, attention DP, MoE TP, MoE EP, CP)` |
| --- | --- | --- |
| MoE TP4 | `--tensor-parallel 4 --moe-tensor-parallel 4` | `(4, 1, 1, 4, 1, 1)` |
| DEP8 | `--tensor-parallel 1 --attention-data-parallel 8 --moe-tensor-parallel 1 --moe-expert-parallel 8` | `(1, 1, 8, 1, 8, 1)` |
| TEP8 | `--tensor-parallel 8 --attention-data-parallel 1 --moe-tensor-parallel 1 --moe-expert-parallel 8` | `(8, 1, 1, 1, 8, 1)` |

The profile route supports vLLM text decoders, including the text portion of multimodal checkpoints, with PP1, CP1 and linear or grouped cache storage. Groups describe full attention, sliding-window attention and supported short-convolution state. Grouped execution currently requires cold aggregated workers, HBM-only cache, no speculative decoding and `prefix_caching: false`. Prefix reuse, host/G3 offload, disaggregation and scalar cache-capacity overrides are rejected for grouped profiles. AFD, encoder pools, unsupported recurrent state, wide EP and EPLB remain outside this route.

### Runtime and collection limits

These settings define the runtime envelope used for profile sizing, collection and generated configurations. They describe different constraints; none independently proves that a trace is covered.

| Setting | Scope and initial value |
| --- | --- |
| `--context-length` | Maximum input plus output tokens for one request. Fresh config/profile-based setup uses `min(declared context, 256000)`; identifier-only setup uses 256,000 until reviewed. The saved field remains `search.context_length`. |
| `--max-num-tokens` | Scheduled token budget per attention-DP rank. Use the supplied profile's bound, or an initial policy of 8,192. Saved as `collection.max_num_tokens` when explicitly set. |
| `--max-batch-size` | Scheduler sequence bound per attention-DP rank. Use the supplied profile's bound, or an initial policy of 256. It is independent of validation concurrency and does not request every prefill batch up to the bound. Saved as `collection.max_batch_size` when explicitly set. |
| `--gpu-memory-utilization` | Finite fraction of total GPU memory in `(0, 1]`, initially `0.90`. Used consistently for topology/resource admission, collection and generated prediction/recommendation/replay memory budgets. Saved as `collection.gpu_memory_utilization`; it does not establish fit or reserve GPUs. |
| `--prefill-cudagraph-policy` | `runtime` for new initialization: leave prefill compilation settings to the pinned runtime and resolve actual captures at engine initialization. `explicit` selects a reviewed capture extension. Saved as `collection.prefill_cudagraph_policy`. |
| `--max-prefill-cudagraph-size` | Positive prefill capture limit for `explicit` policy. Supplying a number selects `explicit` unless a contradictory `runtime` policy was also supplied. Choosing `explicit` without a number uses 2,048. Saved as `collection.max_prefill_cudagraph_size`; match the target serving configuration. |

For example, add `--gpu-memory-utilization 0.92 --prefill-cudagraph-policy runtime` to review a different memory fraction while retaining engine-selected graphs, or `--max-prefill-cudagraph-size 2048` for the explicit extension. Runtime policy with a numeric capture size is rejected. Old saved requests that omit `collection.prefill_cudagraph_policy` retain the prior explicit policy and 2,048-token default; they are not reinterpreted as runtime-policy requests. An omitted legacy memory fraction retains the prior collector-argument omission and the existing 0.90 simulation budget. Review changes in a new request/collection directory.

Profile resource bounds must cover the selected scheduler envelope. Conflicting declarations fail rather than silently clipping the request. With config-based setup, editing scheduler bounds recomputes dependent estimates; review those estimates again. An explicit byte override remains your declared bound until you edit it.

The 256,000-token context policy is motivated by the [AgentX reference subset](#validate-fpm-query-coverage-with-agentx-replay), whose request limit is 256,000, not 262,144. It is a starting point for review, not an assertion that every maximum-length request fits or that the FPM table covers it. The scheduler sequence limit does not reserve maximum context for all sequences simultaneously. Automatic KV capacity comes from the remaining rank-local GPU memory after non-KV resources and any explicit CUDA graph reservation; replay then applies actual request/cache occupancy.

## Start from a local model config

Use a local Hugging Face-style JSON configuration to fill supported metadata and create the existing FPM profile:

```bash
aisimulate onboard init \
  --model-config /models/your-pinned-checkpoint/config.json \
  --interactive --output support-request.yaml
```

Setup reads that file without downloading a checkpoint, importing model code, constructing an analytical model, or launching GPU work. It displays source information and derived inputs, then asks for unresolved values. Missing model metadata is collected before runtime/collection options so the model's context limit can bound the selected envelope. Config identity hints skip their ordinary prompts; explicit CLI identity options take precedence and conflicts can be corrected. A pinned checkpoint revision, literal runtime version, GPU system and interconnect still need your input when absent. The config's SHA-256 records the local source; it is not a checkpoint revision.

With no parallelism flags, guided setup asks for missing shared precision/layout facts, shows a small topology shortlist with resource reasons, and lets you choose one worker with `--output`, or several with [`--output-dir`](#onboard-multiple-parallel-configurations). Enter selects only the displayed single default, and only when all required profile inputs are resolved and its estimated bytes fit the budget. Otherwise a candidate selection is required before setup asks for the remaining per-rank bounds. Any explicit parallelism flag bypasses suggestions and retains the existing topology validation. Each chosen topology then follows the same profile review, edit, accept and cancellation flow below.

When present, `text_config` must be one nonempty decoder configuration object. Only its geometry is used; wrapper and encoder dimensions are never merged into it. Otherwise, setup reads the flat decoder fields even when vision or audio metadata is present. Nested decoder architecture/model-type declarations select supported decoder validation and resource estimates. The profile preserves the declared architecture used by collection: nested `architectures` when present, otherwise the wrapper architecture. Provenance records both identities when they differ. A wrapper architecture never establishes an unknown nested decoder's resource layout. Shared outer dtype and quantization metadata are inherited only when the text section does not declare that metadata. The checkpoint identity hint and SHA-256 remain those of the original document. `model_max_length` is accepted as a context-limit alias; conflicting context declarations fail explicitly.

Scripted intake and interactive final review state that FPM models the text decoder only. Multimodal encoders, projectors, preprocessing and other non-text components and their resource costs are excluded, so these results do not describe full multimodal deployment memory or latency. The saved profile preserves that notice in provenance through planning and generated prediction/recommendation configurations. Known incompatible text-decoder cache layouts remain rejected, and unknown layouts require explicit resource accounting.

Profile memory quantities describe the largest requirement on any rank of the exact selected TP, DEP, or TEP worker. The prompt explains why each unresolved value needs input. Enter integer bytes or an explicit unit such as `70 GiB`, `512 MiB`, or `1.5 GB`; the conversion must produce a whole number of bytes. Invalid individual values are prompted again. Model revision, runtime version, and deployment conflicts return to the corresponding option while retaining accepted resource answers. During initial input collection, incompatible config facts or combined resource bounds fail with the reason and leave no request; correct the source or overrides and rerun setup. Ctrl-C or end-of-input exits 130 and leaves no partial request or replacement of an existing request.

Once all required inputs are available, guided setup shows the complete profile, sources, and selected deployment for review. Choose `edit` to replace an estimate, a previous answer, or a value from `--resource-overrides`. Choose a field by name, enter its value, and review the updated profile. For example, these illustrative inputs replace the per-rank weight bound and record why:

```text
Review action (accept/edit/cancel): edit
Field to edit: weights_bytes
weights_bytes (per rank; integer bytes or units such as GiB/MiB): 70 GiB
... updated profile and sources ...
Review action (accept/edit/cancel): edit
Field to edit: provenance
provenance: User-declared bound; replace with the actual source of your estimate.
... updated profile and sources ...
Review action (accept/edit/cancel): accept
```

Changing an input recomputes dependent estimates: for example, changing `kv_cache_dtype` updates inferred linear `kv_bytes_per_token` or grouped attention page bytes, and changing `max_num_tokens` updates inferred activation bytes. The review also exposes `runtime_context_length`, `gpu_memory_utilization`, `prefill_cudagraph_policy` and `max_prefill_cudagraph_size` as editable collection settings; `context_length` edits the model profile's declared maximum. Editing the policy to `runtime` clears a saved explicit size; editing the size selects `explicit`. Other explicit values remain in place until you edit those fields themselves. If an edit makes a required estimate unavailable, setup asks for that value before returning to review. Invalid individual answers can be corrected; an edit that conflicts with the config or complete request is rejected with the reason, and the previous profile is retained. Model identity and topology remain the declared deployment.

Enter `accept` to accept the exact reviewed request; Enter alone does not accept it. You can make repeated edits before accepting. Single-file output saves after acceptance; directory output saves after every selected profile has been accepted. Choose `cancel` at topology selection or review, or press Ctrl-C or send end-of-input at any prompt, to exit 130 without creating artifacts or replacing existing output. Single-file output also preserves an existing request with `--overwrite`; directory output rejects `--overwrite`. This review step applies to `--model-config --interactive`; scripted setup never prompts.

For automation, supply missing profile fields through a flat JSON or YAML file. Resource quantities in the file must be integer bytes. This example shows the shape for an explicitly declared BF16 decoder; its numbers are illustrative, not measured bounds for your model. Replace them with justified per-rank bounds and describe their source before planning:

```yaml
# resource-overrides.yaml
gemm_quant_mode: bfloat16
moe_quant_mode: bfloat16
fmha_quant_mode: bfloat16
comm_quant_mode: half
kv_cache_dtype: bfloat16
cache_layout: linear
weights_bytes: 2147483648
activations_bytes: 536870912
runtime_overhead_bytes: 1073741824
comm_overhead_bytes: 268435456
kv_bytes_per_token: 262144
max_num_tokens: 8192
max_batch_size: 256
provenance: Illustrative file shape only; replace values and this explanation with their actual source.
```

```bash
aisimulate onboard init \
  --model-config /models/your-pinned-checkpoint/config.json \
  --resource-overrides resource-overrides.yaml \
  --model /models/your-pinned-checkpoint \
  --model-revision YOUR_IMMUTABLE_REVISION \
  --framework-version YOUR_PINNED_VLLM_VERSION \
  --gpu h200_sxm --interconnect nvswitch \
  --tensor-parallel 4 --output support-request.yaml
```

Replace the model, revision, runtime and hardware inputs with the worker you will actually run. `--model-config` and `--fpm-profile` are mutually exclusive; `--resource-overrides` requires `--model-config` and also works with guided setup. Flat per-rank byte overrides require an explicit topology because the same byte bound cannot be transferred or rescaled across candidate tuples. For multiple configurations, place them in each `--parallel-configs` entry's `resource_overrides` instead. Shared precision/layout overrides can be used for automatic suggestions. Scripted setup never reads terminal input. It exits 2 without writing a request when required inputs remain unresolved or no fully assessed automatic default exists. Validation first lists all missing or invalid target identity and collection options. Once that stage is valid, it lists unresolved profile fields, or candidate-specific gaps and exact topology flags, and asks for the necessary inputs or guided setup.

The flat override fields are `architecture`, `context_length`, `num_experts`, every precision and resource field shown above, `cache_block_sizes`, `cache_groups`, `moe_backend`, `attention_backend`, and optional `provenance`. For grouped resources, replace `cache_layout: linear` and `kv_bytes_per_token` with the declarations described below. Unknown fields, duplicate fields, invalid types, unsupported values, and incompatible cache semantics are rejected. Overrides are recorded with per-field provenance rather than discarded. Profile `context_length` is the model's declared maximum; the CLI `--context-length` selects a runtime limit that cannot exceed it. Scheduler limits are per attention-DP rank, as described in [Runtime and collection limits](#runtime-and-collection-limits).

Derivation is deliberately limited. Unambiguous architecture, context, expert count and model kind can be read independently of memory support. The initial supported estimates are:

| Quantity | Supported derivation and limits |
| --- | --- |
| Weight bytes | Vanilla full-attention Llama, Mistral, Qwen2, Qwen3 and Mixtral BF16 layouts with complete geometry and vocabulary divisible by 64 and TP. Estimates include supported biases, tied/untied embeddings, replicated norms, and the Mixtral router. Quantized storage, unknown padding, auxiliary prediction tensors, shared experts and unsupported custom bias layouts need explicit bounds. |
| KV bytes per token | Full-attention MHA/GQA in those layouts plus Qwen3 MoE and MiniMax-M2, with an explicit supported cache dtype and static tensor storage. KV heads shard or replicate according to TP; attention DP does not divide a rank's cache. MLA/DSA storage and dynamic or per-token quantization scales need explicit accounting. |
| Grouped cache pages | Declared full/sliding attention layers with sufficient MHA/GQA geometry, selected TP, effective KV dtype and explicit runtime block sizes. Supported short-convolution state gets separate windowed groups. Derived page bytes are minimum packed tensor estimates over all group layers; runtime padding and nonstandard storage need explicit review or overrides. |
| Activation bytes | The existing backend estimate for those full-attention layouts, 16-bit FMHA, and TP1/2/4/8, within the declared scheduler envelope. It has a 70 MiB floor and does not cover DSA workspaces or speculative decoding. |
| Runtime overhead | The selected packaged hardware specification's per-rank `misc.other_mem` estimate when present. It excludes the separate CUDA graph reservation. |
| Communication overhead | The exact `misc.nccl_mem` TP entry for a pure-TP worker when present. DEP and TEP require an explicit declaration; total GPU count does not establish their communication storage. |

Source notes identify assumptions and packaged hardware hashes. Weight precision does not establish runtime FMHA or KV precision. Config intake proposes the current collector communication identity `half` with its source. Other schema-valid communication overrides remain recorded for review, with a warning that the current collector rejects them for new collection. Recognized quantization declarations can establish checkpoint/FPM precision identities or an explicit KV dtype independently of the unknown quantized storage size; a checkpoint label does not mean every tensor has that dtype. MiniMax and GLM configs may therefore supply useful identity facts while still requiring weight, cache, activation, or communication bounds. These estimates do not certify runtime memory fit. Unknown layouts must supply the remaining supported semantics explicitly; known incompatible layouts are rejected.

The saved request embeds the complete profile and its provenance. The original model config and override file are no longer needed by `onboard plan`, generated prediction and recommendation configs, or request replay. You can inspect or supply the complete profile directly as described next.

### Review grouped cache resources

Config intake recognizes unambiguous full/sliding layer declarations, including `sliding_window`/`layer_types` and supported `sliding_window_size`/`local_layer_ids` aliases. It also recognizes the supported `use_sconv` and convolution-kernel metadata. It derives available layer/head geometry and records the source. Runtime attention block sizes after page unification, effective KV dtype and allocation padding are not established by the model config. Unknown resource assumptions remain explicit; unsupported recurrent semantics and contradictory layer/window declarations fail.

After selecting precision and topology, investigate the pinned cache/backend implementation and available allocation metadata for block sizes after page unification; ask only for values that remain unresolved. The model config's omission does not imply that the user must supply them. Supply `cache_block_sizes` using the generated group names listed by intake. For example, a mixed full/sliding model might require this JSON answer, or the equivalent mapping in a `--resource-overrides` file:

```json
{"full_attention": 16, "sliding_attention": 16}
```

These sizes are illustrative; use the target runtime's tokens per block after page unification. Supported convolution groups use the pinned kernel width as their default block size, which can also be overridden by group name. Partial answers retain the supplied entries and ask for unresolved attention groups. These declarations describe the cache being modeled; verify that they match the actual serving allocation. Keep config-derived packed minima labeled as estimates until runtime evidence establishes the allocation, including padding; revise the profile and its provenance if bring-up reveals different values.

Review the resulting `cache_groups` and their geometry/provenance. Each `FpmCacheGroup` records `name`, `kind` (`attention` or `convolution`), `num_layers`, `block_size_tokens`, `page_size_bytes`, and optional `sliding_window`. **Page bytes include every layer in that group on one rank, plus runtime padding.** Do not multiply them by `num_layers` again or divide them by attention DP. An omitted or null window retains full history; a positive window limits retained history, and convolution groups require one.

Config-derived pages are minimum packed tensor estimates. To include runtime padding or other known storage, choose `edit`, select `cache_groups`, and enter a complete JSON list of group objects with integer byte values. Update `provenance` with the source and remaining assumptions, then review the revised profile before `accept`. A headless `--resource-overrides` file can supply the same complete list instead of `cache_block_sizes`. For example, this illustrates the cache portion of an override for two full-attention layers and six windowed layers; replace all values with the reviewed layout:

```yaml
cache_layout: grouped
cache_groups:
  - name: full_attention
    kind: attention
    num_layers: 2
    block_size_tokens: 16
    page_size_bytes: 131072
  - name: sliding_attention
    kind: attention
    num_layers: 6
    block_size_tokens: 16
    page_size_bytes: 393216
    sliding_window: 512
provenance: Illustrative cache shape only; replace with the runtime layout, padding and source.
```

Include the remaining precision and non-cache resource bounds requested by intake. Explicit group overrides must preserve the config's known attention/convolution layer counts and retention windows. Do not supply `kv_bytes_per_token` with grouped resources or use a linear override to discard declared windowed/convolution state. Saved profiles preserve the full groups and their provenance through planning, prediction, recommendation and replay.

The native allocator shares one rank-local byte budget across groups and admits their allocations atomically. Full-attention groups retain history; windowed groups retain blocks covering the last `window - 1` computed tokens plus all tokens scheduled for the next forward. Expired completed pages are evicted before subsequent forwards. Temporary prefill pages remain charged through completion, so a long prefill chunk can need more memory than the retained window alone. Logical request progress and FPM query context remain unchanged: a long-context request still needs timing coverage for that logical context even when its retained cache has stopped growing.

### Preview and choose parallelism

`--suggest-parallel` emits one JSON report to stdout without creating a request, plan, timing data or output directory. It requires `--model-config` and the actual target identity; config hints can supply the model label and kind, but checkpoint revision and runtime version are never invented. Use your intended runtime/collection limits and shared precision declarations for meaningful resource accounting:

```bash
aisimulate onboard init \
  --model-config /models/your-pinned-checkpoint/config.json \
  --model /models/your-pinned-checkpoint \
  --model-revision YOUR_IMMUTABLE_REVISION \
  --framework-version YOUR_PINNED_VLLM_VERSION \
  --gpu h200_sxm --interconnect nvswitch \
  --suggest-parallel
```

The report includes hardware/config provenance, exact `cli_flags`, required GPUs, resolved fields and sources, `missing` inputs, rejected candidates and a nullable `default`. Without explicit runtime precision, the example can report `needs_inputs` candidates and no default; use those diagnostics to collect shared facts in a flat `--resource-overrides` file and rerun. The preview does not validate or write `--output`, even if that request path already exists. It cannot be combined with `--interactive`, profile options, explicit topology flags, `--output-dir` or `--parallel-configs`. To save a request, remove `--suggest-parallel`; use a fully assessed automatic default or select a candidate by adding its exact flags. To save several configurations, use directory output below. Rank-local byte overrides require an explicit choice. A pending-input candidate can be selected in guided setup, which asks for its missing bounds before final review.

Dense decoders consider TP; MoE decoders consider pure TP, DEP and TEP. The shortlist contains at most two choices per family and six distinct tuples overall, preferring the smallest estimated fit and the next one. Widths are powers of two within the packaged node domain. A declared NVLink/NVSwitch fabric can use a packaged fast rack domain when its inter-node bandwidth matches or exceeds its intra-node bandwidth; a declared `none` interconnect permits only one GPU automatically. This is hardware architecture, not an available GPU allocation. Explicit topology choices may exceed the shortlist.

The memory precheck compares complete per-rank resource estimates against the selected `--gpu-memory-utilization` fraction of packaged GPU memory, initially 90%. Linear cache reserves one full runtime context plus one cached token per rank for planner headroom. Grouped cache uses the native conservative per-request peak at the selected context and scheduled-token bound, including block alignment and transient prefill pages. Grouped planning requires this checkout's compiled AISimulate extension. Neither path multiplies context by the maximum scheduler sequence count or assumes balanced attention-DP routing. Concurrent requests share the cache capacity left after non-cache resources. Missing precision, weights, cache, activations or reservations prevent a default; a known lower bound can reject an oversized candidate but cannot prove fit. DEP/TEP communication storage, many quantized or custom decoder layouts, and large TP activation envelopes need explicit bounds after selection. CUDA graph reservations and non-text components remain outside these estimates. Only known geometry constraints are checked; generated-plan admission and actual serving-runtime checks remain authoritative. The shortlist does not establish timing coverage, optimal performance or measured accuracy.

### Onboard multiple parallel configurations

Use `--model-config` with `--output-dir` to prepare separate requests and profiles in one session. Guided setup reads shared inputs once, then accepts one or more comma-separated candidate numbers:

```bash
aisimulate onboard init \
  --model-config /models/your-pinned-checkpoint/config.json \
  --interactive --output-dir ./onboarding
```

For example, enter `1,2` to select the first two displayed candidates. Blank input selects only the assessed single default, when one exists. Invalid, repeated or out-of-range numbers are prompted again. Setup derives the resources and asks for unresolved rank-local bounds separately for each selected tuple. Collection settings start from the shared CLI flags and can be changed for each profile in guided review. Review and explicitly `accept` each profile; edits to one profile do not change another. Cancellation, EOF or an invalid later profile leaves no new output, even after an earlier profile was accepted.

For headless setup or explicit choices outside the shortlist, supply `--parallel-configs` as a nonempty JSON or YAML list. Each entry requires a positive integer `tensor_parallel`; optional topology fields are `attention_data_parallel`, `moe_tensor_parallel` and `moe_expert_parallel`. Use these fields to express the TP, DEP and TEP tuples in the [topology table](#create-the-request). PP and CP remain 1. Unknown fields, invalid types, unsupported topologies and duplicate resolved tuples are rejected, including duplicate tuples with different precision overrides. For multiple selected precision combinations, use separate fresh output roots as described in [stage 2](#investigate-runtime-constraints-and-precision-options); this does not expand collector support.

These example files describe two dense TP choices. The byte bounds are illustrative, not measurements; replace them with justified bounds for each configuration. The remaining resource fields must be derivable from a supported BF16 decoder config or supplied in each entry's `resource_overrides`:

```yaml
# shared-overrides.yaml
gemm_quant_mode: bfloat16
moe_quant_mode: bfloat16
fmha_quant_mode: bfloat16
comm_quant_mode: half
kv_cache_dtype: bfloat16
cache_layout: linear
```

```yaml
# parallel-configs.yaml
- tensor_parallel: 2
  resource_overrides:
    weights_bytes: 2147483648
    provenance: Illustrative TP2 bound; replace with the actual source.
- tensor_parallel: 4
  resource_overrides:
    weights_bytes: 1073741824
    provenance: Illustrative TP4 bound; replace with the actual source.
```

```bash
aisimulate onboard init \
  --model-config /models/your-pinned-checkpoint/config.json \
  --model /models/your-pinned-checkpoint \
  --model-revision YOUR_IMMUTABLE_REVISION \
  --framework-version YOUR_PINNED_VLLM_VERSION \
  --gpu h200_sxm --interconnect nvswitch \
  --resource-overrides shared-overrides.yaml \
  --parallel-configs parallel-configs.yaml --output-dir ./onboarding
```

Each entry's `resource_overrides` uses the existing [flat override fields](#start-from-a-local-model-config). Shared `--resource-overrides` may contain precision/layout metadata and `cache_block_sizes` valid for the selected runtime, backend and precision. For multiple choices, shared per-rank byte overrides and complete `cache_groups` are rejected: put them in the corresponding entry or enter them during its guided review. Cache page bytes are derived separately for each precision and tuple, even when block sizes are shared. Do not copy or rescale a reviewed byte bound from another configuration.

`--parallel-configs` requires both `--model-config` and `--output-dir`, and conflicts with explicit topology flags and `--suggest-parallel`. Adding `--interactive` skips candidate selection but still reviews every supplied configuration. Explicit topology flags with `--output-dir` and no configuration list produce one configuration. Existing `--output FILE` behavior and single-file request/profile schemas are unchanged; `--output` and `--output-dir` are mutually exclusive.

Choose a fresh output root or an existing empty directory. Directory output rejects `--overwrite`, nonempty roots and symlink/path conflicts, and does not replace collection results. After every profile validates and, when interactive, is accepted, setup publishes:

```text
onboarding/
  onboarding.json
  tp2-dp1-moe-tp1-moe-ep1/
    request.yaml
    fpm-profile.json
  tp4-dp1-moe-tp1-moe-ep1/
    request.yaml
    fpm-profile.json
```

Each deterministic directory names its resolved tuple; each request embeds one deployment and matches its exported profile. The `configurations` list in `onboarding.json` records each topology, minimum collection GPUs, request/profile paths, intended `collection/` directory and exact next `aisimulate onboard plan` command in `plan_command`. Run those commands to create the ordinary plans, then use each plan's existing `collect-fpm` preview, collection and `validate-fpm` commands with separate result directories. Setup does not launch collection. The selected configurations can be collected in separate runs on the same GPUs; their GPU requirements are not summed. Once saved, the requests and profiles can be planned without the original config or override files.

## Provide identity and resource metadata

For a model without an analytical class, create a JSON or YAML FPM profile and pass `--fpm-profile /path/to/model-profile.yaml` to `onboard init`. The request embeds the complete profile; ordinary prediction and recommendation use the same object under `engine.fpm_profile`. The profile schema is [FpmModelProfile](../python/aisimulate/src/aisimulate_core/fpm_profile.py). It rejects missing fields, unknown fields, conflicting identities and mutable revision placeholders.

| Profile fields | Required meaning |
| --- | --- |
| `schema_version` | Integer `1`. |
| `model`, `model_revision`, `architecture` | Exact timing model identity, pinned checkpoint revision, and architecture identifier. An unknown architecture is valid for direct interpolation. |
| `context_length`, `num_experts`, `provenance` | Declared context limit, routed expert count (`0` for dense), and how the metadata was obtained. |
| `deployments` | One or more exact deployment records, with one precision/resource identity per hardware/runtime/full parallel tuple. |
| Deployment `system`, `backend`, `backend_version` | GPU system, `vllm`, and literal runtime version. |
| Deployment `tp`, `pp`, `dp`, `moe_tp`, `moe_ep`, `cp` | Complete topology. PP and CP default to `1`; other dimensions are explicit. |
| Deployment precision | Exact `gemm_quant_mode`, `moe_quant_mode`, `fmha_quant_mode`, `comm_quant_mode`, `kv_cache_dtype`. They must match the collected cell; FP8 FMHA and BF16 FMHA are separate identities. |
| Deployment runtime hints | `moe_backend`, `attention_backend` (default `auto`), `enable_wideep`, `enable_eplb` (currently both `false`). |
| Deployment `resources` | Conservative bounds described below, specific to this topology. |

Each `resources` record requires integer `weights_bytes`, `activations_bytes`, `runtime_overhead_bytes`, `comm_overhead_bytes`, positive `max_num_tokens` and `max_batch_size`, and a `provenance` explanation. Choose either `cache_layout: linear` with positive `kv_bytes_per_token`, or `cache_layout: grouped` with a nonempty `cache_groups` list and no scalar token rate, as described in [grouped cache review](#review-grouped-cache-resources). Existing linear profiles retain their serialized shape. Obtain these bounds from checkpoint tensor/storage metadata and serving-runtime accounting or conservative explicit estimates. A profile is an input declaration; AISimulate does not certify that its values describe your runtime.

All resource bytes are **per rank**, and must bound every rank. Scheduler envelope limits are also per attention-DP rank; they are not the summed batch and token totals used by the worker's FPM timing query. Do not copy TP resource values into DEP/TEP merely because GPU counts match. Include all non-KV storage, including quantization overheads. Overhead fields exclude the separate `kv_cache.capacity.cuda_graph_reserved_bytes` reservation. Requests above the declared scheduler envelope fail explicitly.

Automatic KV admission subtracts these non-KV bounds and CUDA graph reservation from the configured fraction of GPU memory. Linear cache divides the remainder by its declared bytes per token; grouped cache passes the shared byte budget and group pages to the native allocator. `context_length: max` uses the profile limit. Keep `bytes_per_token: auto` for profile resources; grouped profiles reject explicit scalar rates and block-count capacities. These operations need a hardware specification and profile, but no FPM timing files or model graph. The [canonical cache-budget API](core-api.md#fpm-profile-cache-groups-and-byte-budgets) returns grouped `total_kv_size_bytes` and `request_peak_cache_bytes`; scalar `total_kv_size_tokens` and `kv_size_per_token_bytes` are null. `estimate_num_gpu_blocks` rejects grouped profiles.

The declared checkpoint revision is preserved for review and replay. Existing FPM tables may not pin a checkpoint revision; supplying one in a profile does not prove that the historical measurements came from that exact revision. Record that limitation in `provenance` when validating existing data.

## Plan, preview, and explicitly execute

```bash
aisimulate onboard plan \
  --config support-request.yaml --output-dir ./aisimulate-support

aisimulate onboard collect-fpm \
  --config ./aisimulate-support/request.yaml \
  --output-dir ./aisimulate-support
```

The first command saves the request, `support-plan.json`, `commands.json`, `predict/pilot.yaml`, `recommend/pilot.yaml`, and a local `systems/` directory. When supplied, the profile is also saved as `fpm-model-profile.json`, included in the collector command, and embedded in prediction/recommendation configs. The plan records the CPU resource estimate and effective collection limits with their sources. The second command prints the collector invocation without launching it. Generated command vectors and printed next commands use absolute output paths and preserve spaces or shell punctuation. Use a separate output directory for each deployment or collection envelope.

The v2 plan's `request_id` identifies collection inputs: deployment identity, exact profile and collection settings. Its separate `validation_id` also includes synthetic workload/SLA, recommendation objective and seed. To change only those validation inputs, copy the request to a new file, edit that copy, then run `onboard plan --config UPDATED_REQUEST --output-dir EXISTING_PLAN --overwrite`. The command first validates the original saved request and every existing generated file against the saved plan, then refreshes validation examples and their hashes. It can also repair missing generated files for the same collection. It preserves collected data, checkpoints and prior result directories. Changed collection/profile inputs or modified existing generated files are rejected; do not edit files inside the plan to bypass those checks.

If an AISimulate update changes generated guidance, recreate the plan in a new output directory: validation compares generated files byte for byte and does not migrate existing plans. A validation trace is supplied separately to `validate-fpm`, so selecting another trace does not require regenerating the collection plan.

The plan reports `fpm.collection_gpus_required`, derived as attention TP times attention DP. Collection uses only that worker width and the matching `tp`, `pure_tp`, `dep`, or `tep` preset. The generated prediction and recommendation configs both use one worker. Recommendation has one exact preset and one trial; its `optimization.constraints.max_candidate_gpus` equals the selected worker's requirement so even workers wider than the runtime default are representable. That derived cap describes this validation example and does not declare the user's total available GPUs. There is no replica expansion or deployment optimization during onboarding.

Saved `aisimulate-support-request/v1` requests are read as v2 while preserving their declared context and profile resource bounds; an omitted legacy context retains 16,384. Legacy validation concurrency is not reused as a collection sequence limit. Review these retained and newly independent settings before use. Old `aisimulate-support-plan/v1` directories must be regenerated in a **new directory** because their collection commands depended on the synthetic validation workload. Do not overwrite old timings or checkpoints to reuse the old plan identity.

Some earlier v1 drafts also included `identity.gpu_count`, `identity.node_count`, `identity.gpus_per_node` and `search.max_candidates`; those obsolete fields remain rejected. Copy the request, remove them and review the retained topology/profile before regeneration. Plan summaries use `fpm.collection_gpus_required` and `search.candidates[].required_gpus`. The old `onboard init` allocation flags (`--gpu-count`, `--node-count`, `--gpus-per-node`, `--max-candidates`) are not accepted. Ordinary `predict` and `recommend` replica and GPU-budget APIs are unchanged.

Before execution, prepare the real checkpoint and compatible pinned runtime image using the existing [FPM collection guide](../python/aisimulate/docs/fpm/end-to-end-workflow.md). The packaged collector generates the deployment, creates the benchmark workloads and launches Dynamo/vLLM workers. It needs available GPU resources, Kubernetes access and permissions, deployment configuration, and model/image access; it does not require a prestarted serving process or HTTP endpoint. The command launches workers within that prepared environment rather than provisioning the cluster itself. `commands.json` publishes the guarded `aisimulate onboard collect-fpm --execute` command for collection, alongside a read-only collector planning command.

For a source checkout installed with `uv sync`, activate its environment and expose the collector source package before running generated collector commands. From the repository root:

```bash
source python/aisimulate/.venv/bin/activate
export PYTHONPATH="$PWD/python/aisimulate${PYTHONPATH:+:$PYTHONPATH}"
```

Alternatively, run the collector from the `python/aisimulate` source directory with that environment active. The installed release wheel includes the collector and needs no source-path adjustment. These steps make the command importable; they do not provide the target deployment or GPUs.

After those prerequisites are ready, explicitly launch collection from that environment:

```bash
aisimulate onboard collect-fpm \
  --config ./aisimulate-support/request.yaml \
  --output-dir ./aisimulate-support --execute
```

Execution requires a matching saved plan. Creating the initial plan records model and runtime revisions without downloading a pinned checkpoint or inspecting the running runtime. During profile-based collection execution, the collector checks the observed Pod's vLLM version against the profile's literal backend version before benchmarking. Keep the actual checkpoint consistent with the declared model revision.

For a profile-based campaign, the collector validates its resolved topology and precision against the supplied profile before execution. It does not relabel an FP8 cell as BF16 or change checkpoint quantization to satisfy the profile. A mismatch reports the conflicting field and requires a matching profile/runtime or a supported collector configuration. Resource admission uses the supplied bounds without constructing an analytical model. Profile contents participate in the collector's frozen-plan identity.

New profile-based collection currently uses checkpoint-native weight, FMHA and KV precision without consulting op-level timing tables. The requested KV dtype must match that checkpoint-native dtype. An older FPM profile can still be valid for prediction while being unsuitable for new collection: the historical MiniMax/H200 BF16 FMHA fallback cell differs from its checkpoint-native FP8 inference. Use the historical identity to query those timings and a matching native profile for new collection; the collector rejects that mismatch. Publish the new FP8 campaign into a clean, separate dataset using a new onboarding output directory. Existing cell IDs omit FMHA precision, and publication retains the first published run for a cell ID. If the historical BF16 dataset already holds that cell ID, publication skips the new FP8 run. A collection profile must select one literal runtime version for the target hardware/backend.

Set deployment options directly on `onboard collect-fpm`: `--dynamo-version VERSION`, `--image IMAGE`, `--namespace NAME`, `--model-cache NAME[:MOUNT[:SUBPATH]]`, `--transport nvlink|ib|efa`, and `--image-pull-secret NAME`. The mount, when supplied, is an absolute container path. Prefer an immutable image digest. Supply the same options when previewing, executing, and resuming; deployment settings are part of the collector's frozen-plan identity, so changed settings require a new output directory. Arbitrary collector arguments and engine overrides are not accepted by this command.

For a diagnostic run, add `--execute --smoke`; `--limit N` also requires `--smoke`. Diagnostic smoke and limited runs do not publish formal FPM data. Existing campaign data, raw artifacts, or checkpoints require explicit `--resume` and a readable matching collector checkpoint; otherwise choose a new output directory. A custom `--checkpoint-dir`, if needed, must remain inside the plan's `fpm-checkpoint/` directory. Selecting an empty checkpoint directory does not allow reuse of existing campaign artifacts. Smoke and formal campaigns have separate checkpoints and artifact directories, so an existing smoke run does not prevent the first formal run, or vice versa. The collector verifies the resumed checkpoint's frozen-plan identity.

Planning and collection reject concurrent onboarding operations. The persistent `.support.lock` file uses an OS advisory lock; ownership is released when the process exits, including after an abrupt termination. Leave the file in place. Request validation, saved onboarding-plan checks, and collector input resolution exit 2. Failures after collector execution starts, including a frozen checkpoint identity mismatch, exit 1 with a concise message. Interruption exits 130.

### How the collection grid is determined

AISimulate and Dynamo jointly determine what is collected. For formal collection, their responsibilities are:

| Part of collection policy | Responsibility |
| --- | --- |
| Worker deployment | The AISimulate collector renders the deployment and launches benchmark workers in the prepared GPU/Kubernetes environment. |
| Runtime bounds | AISimulate forwards the reviewed per-request context, scheduled-token and sequence bounds, and configured GPU memory fraction, to prefill and decode workers. |
| Runtime capture policy | With `runtime`, AISimulate emits no prefill compilation override or capture-derived new-token sample cap. Capture sizes and the corresponding new-token axis/counts remain unresolved before engine initialization; the scheduler-derived KV-read sample cap remains bounded. |
| Explicit capture policy | With `explicit`, AISimulate constructs the capture-size list and overrides the prefill engine's compilation configuration. It derives prefill new-token and KV-read sample caps from captures and runtime bounds. These caps are inputs to Dynamo, not an exact point list. |
| Remaining sampling defaults | The collector inherits formal prefill batch-sample and decode-sample caps from the deployed Dynamo image. Inspect that image's defaults; a scheduler sequence bound does not request every prefill batch up to the bound. |
| Engine initialization | vLLM loads the model and allocates cache/workspaces, then resolves supported graph mode and capture sizes for the selected runtime configuration. Inspect effective values in initialization logs and raw runtime artifacts. |
| Exact points and timings | Dynamo self-benchmark combines those inputs with the initialized engine's capture sizes, KV capacity and feasibility checks to generate, admit and time the actual points. |

An explicit prefill compilation override changes the engine being measured, so its capture configuration must match the serving target. Runtime policy also requires checking the actual resolved mode and captures: the same requested settings can resolve differently across model/runtime versions or scheduler bounds. The deployed image's inherited sampling defaults can change the collected set. Inspect the generated configuration, initialized engine and image behavior; runtime bounds alone do not describe the full sampling policy.

The onboarding `support-plan.json` records the reviewed bounds and policy in `fpm.runtime_limits` and summarizes grid generation in `fpm.sampling`. For runtime policy, the collector plan's `point_generation.prefill_sampling` records `cudagraph_policy: runtime` and leaves capture sizes, capture counts and capture-dependent new-token fields null. Explicit policy expands the selected capture list and sample caps. The exact point count remains runtime-determined in either mode. The `point_generation` owner remains Dynamo because Dynamo generates the actual runtime points; that ownership does not mean all collection policy comes from Dynamo.

The onboarding command supplies `--fpm-max-model-len`, `--fpm-max-num-batched-tokens`, `--fpm-max-num-seqs` and the selected `--fpm-prefill-cudagraph-policy`; new requests also supply `--fpm-gpu-memory-utilization`. A supplied numeric capture limit is forwarded as `--fpm-max-prefill-cudagraph-size`; explicit policy without one retains the collector's 2,048-token default. `--fpm-max-prefill-isl` is a total new-token budget, not a per-request context limit. DEP does not multiply these rank-local settings by GPU count. Onboarding does not build a separate trace-derived grid or promise a sample count or collection duration.

Native artifact completion and skip checks account for the generated grid. A complete grid does not guarantee that AgentX replay or another direct-FPM caller can resolve every requested query. Validate those query shapes separately, using the published measurements and supported interpolation.

Direct collector callers can set the same shared runtime flags. Omitting `--fpm-prefill-cudagraph-policy` retains the standalone collector's explicit 2,048-token capture default; select `runtime` deliberately to defer captures. Runtime policy with a numeric capture size is rejected. Omitting `--fpm-gpu-memory-utilization` preserves the existing runtime/default behavior; an explicit value must be finite and in `(0, 1]`. With a supplied FPM profile, omitted scheduler/context limits use that profile's bounds; without one, an omitted context retains the existing vLLM `-1` auto-fit behavior. Explicit limits must be positive and cannot exceed the supplied profile or contradict the prefill sampling bounds. Inspect the generated command and collector plan before committing GPU time. Successful formal collection publishes the FPM Parquet file and metadata pair into the plan's local systems data directory; diagnostic success alone does not provide that pair. Validate the worker's actual FPM query shapes afterward, including summed queries for DEP.

Benchmark prefix seeding and replay cache reuse have different purposes. The collector's phase-specific protocol may retain a prefix cache to prepare timing points outside the measured forward pass; eligible decode strategies can use real KV warm-up when supported by the deployed runtime. Cold replay starts from an empty cache, and grouped replay retains `prefix_caching: false` for cross-request reuse. Do not copy that replay flag into benchmark launches or assume an environment switch proves that real seeding ran. Inspect runtime support, raw row provenance and skipped/fallback evidence. Smoke sampling is diagnostic and uses small caps; it does not establish the formal runtime capture list or point count.

## Validate FPM query coverage with AgentX replay

After verifying a matching formal FPM pair, run the existing cold aggregated replay path against a local Weka JSON or JSONL file:

```bash
aisimulate onboard validate-fpm \
  --config ./aisimulate-support/request.yaml \
  --output-dir ./aisimulate-support \
  --trace /path/to/traces.jsonl \
  --validation-output-dir ./aisimulate-validation/agentx
```

The output directory must be separate from, and neither inside nor above, the collection directory. Use a fresh directory or `--overwrite` to replace prior validation outputs. The command verifies the saved collection plan, writes `predict.yaml`, and invokes ordinary `aisimulate predict` with per-request evidence. It does not download traces or run GPU collection. The reviewed target model, topology, runtime, profile, scheduler limits and systems paths remain fixed; source model labels in a trace do not replace the configured target model.

The current validation scope is **cold aggregated replay, one client lane, HBM-only cache and no speculative decoding**. The cache starts cold. Linear profiles can accumulate normal prefix reuse; grouped profiles retain `prefix_caching: false` and require vLLM with PP1/CP1. Nested timestamps are interpreted relative to the root play (`nested_timestamp_basis: absolute`). The command replays the complete supplied file; selecting one complete play is useful for a first check, but does not establish coverage of a larger corpus. Preserve the full play and its dependencies when preparing a subset. Ordinary prediction's host-memory checks still apply; a larger corpus can require a larger host, and a preflight rejection remains incomplete validation. The current path builds on existing vLLM/SGLang replay support; onboarding collection remains vLLM. Seeded cache snapshots, explicit warmup and expanded replay modes are follow-up work after the relevant AgentX changes, including [#207](https://github.com/ai-dynamo/aisimulate/pull/207) and [#235](https://github.com/ai-dynamo/aisimulate/pull/235); they are not prerequisites for this workflow.

The reference corpus is [semianalysisai/cc-traces-weka-062126-256k](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/tree/8fecd2fc56694469f758f0afbbb6335ad3043740). To download that pinned revision separately, with Hugging Face access available:

```bash
hf download semianalysisai/cc-traces-weka-062126-256k traces.jsonl \
  --type dataset \
  --revision 8fecd2fc56694469f758f0afbbb6335ad3043740 \
  --local-dir ./agentx-reference
```

The expected `traces.jsonl` SHA-256 is `e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`. Record the file hash and selection when using a derived subset. No trace corpus is bundled or fetched automatically by AISimulate.

### Read the coverage result

| Artifact | Meaning |
| --- | --- |
| `predict.yaml` | Inspectable ordinary prediction input with strict direct FPM and native coverage collection enabled. |
| `validation.json` | Overall `covered` or `incomplete` result, issues, collection identity, trace/config/FPM file hashes and replay scope. Accuracy remains `not_assessed`. |
| `prediction/fpm-coverage.json` | Native lookup counts and bounded missing-query details, also saved when a timing lookup fails after replay starts. |
| `prediction/prediction.json` | Ordinary prediction report when produced, including per-request completion, AgentX graph and target-model projection. |

`measured` means the native lookup resolved from an exact measured point; `interpolated` means it used supported direct interpolation; `unsupported` means the native query failed. Counts are **native lookup resolutions**, not requests, replay iterations or tokens. Reusing a cached timing does not make another lookup. Each role records prefill, decode and any mixed-pass decode-baseline counts. Missing entries identify phase, purpose, cell identity, coordinates, reason and occurrence count. Each model retains at most 128 distinct gaps and reports omitted gap occurrences; its total unsupported count remains complete.

An exit status of 0 and overall `validation.json` status `covered` require a nonempty completed replay, all selected requests and plays completed with their requested output, and at least one supported native query with no unsupported queries. An early failure, failed or aborted request, empty query set or interruption cannot pass. Inspect `validation.json` first: an individual native snapshot does not certify replay completion. A missing query stops ordinary prediction and returns nonzero while retaining available evidence. That partial report describes only the attempted portion; it does not identify every missing point in the remainder of the corpus.

For a missing decode query, inspect its exact batch size and total past-KV coordinate together. A high maximum KV value elsewhere in a table does not prove coverage for batch size 1 at that context, or for another batch/capture region. Prefill likewise needs the requested batch and prompt/KV support. Use these gaps to review the collection envelope and matching data, preserving strict direct interpolation and denied fallback. Successful coverage establishes that the selected replay could obtain timings; it does not establish predictive accuracy. Assess accuracy separately with matched silicon measurements.

## Run the generated ordinary configurations

The generated synthetic configurations are optional small prediction/recommendation examples after the selected model route and formal data are available:

```bash
aisimulate predict \
  --config ./aisimulate-support/predict/pilot.yaml \
  --output-dir ./aisimulate-support/predict-results/pilot

aisimulate recommend \
  --config ./aisimulate-support/recommend/pilot.yaml \
  --output-dir ./aisimulate-support/recommend-results/pilot
```

Use fresh result directories and inspect each command's exit status and results. These examples validate one selected worker and workload. For an actual deployment run, copy the generated config and set `engine.workers.aggregated.parallelism.replicas` in prediction, or the replica values in recommendation's `parallelism.preset` and `optimization.constraints.max_candidate_gpus` to your intended deployment budget. Keep the exact worker topology and profile consistent with the measured data. Those runtime choices do not require changing the onboarding request or recollecting the same worker's timings.

The generated configurations select `engine.workers.aggregated.timing.estimation_mode: fpm_interpolation` with `fallback_policy: deny`. With a profile, they also set `estimator_config.fpm_interpolation.method: direct`. `engine.systems_paths` contains the plan's absolute local systems directory, which supplies hardware and collected FPM data. Recommendation preserves the resolved root, interpolation method, and complete profile in exported prediction configs. Moving the plan to another machine requires updating absolute paths or regenerating it there. Existing configurations using the single-root `engine.systems_path` input are normalized to `engine.systems_paths` when saved.

Grouped configurations also set `engine.workers.aggregated.kv_cache.prefix_caching: false`; preserve it in prediction, recommendation and `validate-fpm` replay. The same reviewed groups and byte budget reach native allocation in each path. Changing cache layout does not introduce a new collection grid or change the FPM timing-table coordinates: use the [shared AISimulate/Dynamo collection policy](#how-the-collection-grid-is-determined) and validate the resulting logical-context queries.

Ordinary `predict` and `recommend` retain their existing defaults when no profile is supplied. Generated configs can be edited through the public schema. A recommendation with a profile enumerates only its declared deployment tuples that fit the GPU budget; it preserves the complete inline profile and interpolation choice in exported prediction configs. Default scheduler search ranges may exceed a profile's envelope, so pin or bound those domains explicitly. A successful simulation is not an accuracy result. Compare its output with an independent run of the same model, runtime, topology, and workload to assess accuracy.

## Choose the model execution route

Hand off the saved request, pinned model configuration, and plan. Both routes need a canonical checkpoint identity, effective precision and topology, correct weight and KV-cache accounting, and matching whole-forward FPM measurements. Collected timings alone do not establish memory fit.

- **Registered-model/SOL route:** reuse a compatible analytical class or follow the optional [model-integration procedure and CPU checks](../python/aisimulate/docs/fpm/model-integration.md#2-registered-model-route-reuse-or-implement-the-model-description) when choosing to add one. Verify its operation graph, memory/cache accounting, and native FPM SOL execution.
- **Class-independent direct route:** supply the identity/resource profile and configure the worker as shown below. The guided planner selects this route whenever a profile is supplied. No operation graph is constructed for resources, timing, or recommendation candidates.

```yaml
timing:
  type: default
  estimation_mode: fpm_interpolation
  fallback_policy: deny
  estimator_config:
    fpm_interpolation:
      method: direct
```

Within `estimator_config.fpm_interpolation`, `method: auto` retains SOL for a registered architecture and chooses direct for an unregistered architecture with a profile. Explicit `sol` requires a registered class; explicit `direct` requires a profile. Rust validates and selects the method through `RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig)`, then pins it for queries and exported configurations. A model-construction error does not trigger a silent change of method. The engine's top-level `estimation_mode: auto` retains the normal priority of operation-level estimation, FPM interpolation, then FPM regression; providing a profile does not change that priority.

Direct timing first uses an exact point or interpolation within a measured curve. Prefill interpolation stays at the same batch size, with two measured KV neighbors whose prompt curves both cover the requested token count. Wider KV bracketing removes the old distance limit only when the narrower direct bracket is unavailable. Decode interpolation respects the measured batch/capture domain. Both phases exclude synthetic `fake_fallback` rows, including healed/extrapolated values. Missing two-sided support, unmeasured batches and out-of-domain queries fail explicitly. The direct route does not apply SOL-dependent prefill batch clamping or general extrapolation; 2D interpolation remains experimental.

Per-operation silicon profiling described in the model guide is not required by either FPM route. The workflow collects whole-forward timings, then verifies prediction and recommendation for the exact target deployment. Report timing coverage and interpolation error separately; successful simulation alone does not establish measured accuracy.
