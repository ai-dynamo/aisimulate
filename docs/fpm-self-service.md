<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FPM self-service

`aisimulate onboard` guides onboarding a new model for FPM simulation on your designated hardware platform. It records the model, runtime, target GPU system and interconnect, plans one TP, DEP, or TEP worker, and derives the minimum GPUs required to collect that worker's timings. You review the resource and collection limits, collect whole-forward timings through Dynamo self-benchmark, then validate their query coverage using ordinary trace replay. It also produces ordinary `predict` and `recommend` configurations for the selected worker.

Collection limits and validation traffic are separate inputs. Dynamo self-benchmark generates its sampling grid from CUDA graph sizes and runtime bounds. AgentX traces exercise the resulting FPM library through replay; their variable request lengths do not require a fixed input/output length or latency target during onboarding.

Planning works before the model has an AISimulate model class or measured FPM timings. With a supplied FPM profile, planning validates the declared deployment identity and estimates memory admission from your resource bounds. Runtime compatibility and data readiness remain **unchecked**, and accuracy is **not assessed**. This setup does not provision GPUs or run target preflight checks.

Ordinary FPM `predict` and `recommend` accept an inline `engine.fpm_profile` with model identity and rank-local resource bounds. Direct interpolation uses measured timings without constructing an op-level model. Registered models retain SOL interpolation. See [Choose the model execution route](#choose-the-model-execution-route) for selection and coverage rules.

## Onboard with an agent

For a request such as "Help me onboard my model for FPM simulation on my target GPUs," follow these six stages. Claude Code reaches them through the root `CLAUDE.md` import of `AGENTS.md`; other agents use the same `AGENTS.md` entry point. These stages structure the conversation over the existing CLI; they are not new CLI commands or a persisted stage tracker. The sections below remain the detailed CLI reference.

| Stage | Required result |
| --- | --- |
| 1. Inspect the model and target | Accessible config/profile and packaged hardware specification identified; supported metadata read and gaps recorded. |
| 2. Choose the worker and collection limits | Checkpoint identity/revision, runtime, interconnect, exact worker topology, context and scheduler/capture limits selected; minimum collection GPUs derived. Runtime compatibility remains unchecked. |
| 3. Derive, review and save the profile | Exact resource/precision values and assumptions reviewed and accepted; final request contains the complete profile and provenance. |
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

Resolve only the remaining checkpoint identifier and immutable revision, literal vLLM version, target GPU platform and interconnect. Use available repository/deployment metadata before asking the user. A model label, config hash or example revision does not establish a checkpoint pin. Do not ask for total available GPUs, node allocation, GPUs per node or replica budgets during onboarding; those are choices for actual prediction or recommendation runs.

Review the [runtime and collection defaults](#runtime-and-collection-limits): per-request context, scheduled token budget, maximum sequences and prefill CUDA graph capture limit. Fresh config/profile-based setup caps context at the smaller of the declared limit and the 256,000-token AgentX reference bound. Explain this initial policy and allow edits; it does not establish memory fit or timing coverage. Do not ask for fixed input/output lengths, concurrency, TTFT or TPOT as required collection inputs. Those flags customize optional synthetic validation examples.

With a local model config, inspect the [read-only topology preview](#preview-and-choose-parallelism) using the actual identity, runtime, target and collection limits. Resolve shared precision/layout facts explicitly and rerun the preview. Present the default and alternatives with their resource assumptions and unresolved fields. A candidate marked `estimated_fit` has a complete declared/estimated byte budget; it is not a performance ranking, runtime qualification or a measurement. If no default is available, explain why and select an exact candidate before asking for its per-rank bounds. Do not silently assume topology or precision.

Help choose one initial TP configuration, or the relevant TP/DEP/TEP configuration for MoE, using the preview's exact flags or the [supported topology flags](#create-the-request). Honor an explicit topology even when it lies outside the automatic shortlist. Explain the selected tuple; do not require the user to know every parallelism field upfront. Derive its minimum collection GPUs as attention TP times attention DP: TP4 requires four GPUs, while DEP8 requires eight. This requirement does not declare available capacity or establish runtime placement or compatibility. Each plan collects one exact topology; use separate requests and output directories to investigate alternatives.

### 3. Derive, review and save the profile

Use [a local model config](#start-from-a-local-model-config), or [a supplied profile](#provide-identity-and-resource-metadata), for the selected deployment. Both produce a class-independent direct-FPM plan. Derive supported estimates before asking for unresolved resource fields. Explain each value's source and limitations; do not invent missing bounds. Memory values must bound every rank of the exact selected topology. Resolve effective weight, FMHA, communication and KV precision separately; a quantized checkpoint label does not determine all of them. Preserve replacements and their rationale in overrides/provenance. Review all effective values and assumptions using the appropriate flow below.

| Agent environment | Review and save behavior |
| --- | --- |
| Terminal or agent tool with a PTY | Run `aisimulate onboard init --model-config /path/to/config.json --interactive --output support-request.yaml`. Relay unresolved prompts and the final profile to the user. Apply requested `edit` actions and return the revised profile to the user for review. Enter CLI `accept` only after the user explicitly accepts those exact values; honor any existing explicit acceptance of those same values. `cancel`, Ctrl-C or EOF creates no new request and preserves any existing output, even with `--overwrite`. This final review is specific to `--model-config --interactive`; supplying `--fpm-profile` does not add it. |
| Headless or noninteractive agent | Supply identity/collection flags and `--resource-overrides` as needed, without `--interactive`. Missing required inputs exit 2 without saving; use the diagnostics to ask for the missing facts. A successful command writes immediately. Initially write to a separate path such as `draft-request.yaml` and show the embedded profile, sources and scope for user review. Apply edits in the inputs/overrides and repeat draft review until accepted; then rerun the unchanged reviewed inputs to a new final request path and verify that it matches the accepted draft before planning. The draft name is only a file convention; it has no special CLI status. Review a supplied `--fpm-profile` in the same way. |

Do not pipe answers into `--interactive`: it requires a terminal. Scripted setup has no built-in acceptance prompt. Keep draft files separate from the final request and preserve prior outputs when revising a deployment.

Model identity, runtime and topology are not profile-review edit fields. If they change, return to stage 2, regenerate dependent estimates and review the new request; use new output paths for the changed deployment.

### 4. Plan collection

Follow [Plan, preview, and explicitly execute](#plan-preview-and-explicitly-execute) using a new output directory. Inspect `support-plan.json`, the embedded/saved profile, generated prediction/recommendation configs and `commands.json`. Run `onboard collect-fpm` without `--execute` to print the collector command; this does not run the collector's own plan or check the target runtime. Inspect the read-only collector plan when its input environment is available and identify any missing prerequisites.

Explain the minimum collection GPUs, rank-local scheduler/resource envelope and sampling scope. Dynamo self-benchmark owns the point grid; the plan supplies runtime and capture limits, without deriving a second grid from trace requests. One onboarding plan selects one parallel tuple and generates single-worker validation configs. Use separate requests/output directories for additional tuples. Actual collection resources and placement must be checked in the collector environment before execution. The synthetic request count does not bound timing samples or collection duration.

### 5. Collect and verify data

First inspect any existing timing data for a matching deployment. Reuse a verified matching formal pair when available; do not collect again merely to complete a stage. For new collection, follow the [FPM collection guide](../python/aisimulate/docs/fpm/end-to-end-workflow.md) to prepare the pinned checkpoint, Dynamo/vLLM deployment, model access and GPU resources. Use the agreed scope and existing execution authorization, reporting concrete missing prerequisites when blocked. Preview and execute with the same deployment options. `--execute` launches collection; optional `--smoke` is diagnostic and publishes no formal FPM pair. Preserve checkpoints, logs and raw evidence.

For both reused and new data, [inspect the published pair](../python/aisimulate/docs/fpm/end-to-end-workflow.md#5-inspect-the-published-pair): verify hashes, schema and identities, including actual runtime, topology, precision and available prefill/decode cells. Keep it at the generated configs' local systems path. Record any historical checkpoint-revision uncertainty in provenance; a declared revision does not prove that old measurements used it. A successful preview or smoke run is not formal data, and a matching pair does not prove all simulated queries are covered.

### 6. Validate replay and run predict/recommend

[Validate a local AgentX trace](#validate-fpm-query-coverage-with-agentx-replay) through `onboard validate-fpm`. It writes an inspectable ordinary prediction config, runs cold aggregated replay, and saves coverage evidence separately from the collection plan. Use the selected complete local corpus or one complete play for a first check, and state that selection. Keep `engine.systems_paths`, the target model projection and the direct-FPM profile intact. Report the command's exit status and exact missing coordinates. Missing timing stops replay; a partial report does not audit the rest of the trace.

[Run the generated ordinary configurations](#run-the-generated-ordinary-configurations) when a synthetic prediction or recommendation check is useful. For deployment prediction or optimization, set the desired replicas and GPU budget in the ordinary runtime configs. Do not change precision labels or silently switch timing methods to obtain a result. Return to stage 5 to address missing data, or stage 2 if the selected deployment or collection limits change. A different validation trace alone does not invalidate collected timings. Report the simulation stage as incomplete while required queries fail.

At handoff, include the checkout revision, final request/profile, plan directory, data pair/provenance, exact commands/exit statuses and all result paths. Distinguish estimated memory fit and CPU planning from actual target-runtime checks, formal data/coverage, and completed simulations. For accuracy, report an independent matched silicon comparison if performed, or explicitly **not assessed**. Successful simulation is not evidence of accuracy; an accuracy study is not a mandatory additional collection campaign for onboarding.

### Resume from existing work

Inspect the saved request/profile, plan, data pair and results before deciding where to resume. Validate that they still match the checkout's CLI, selected deployment and collection settings; use the existing plan checks described below. An accepted final request can start at stage 4, a valid plan at stage 5, and a verified matching data pair at stage 6. A draft or a saved file without evidence of acceptance still needs stage 3 review. Do not repeat accepted decisions or rerun completed work without a reason.

Preserve completed artifacts when blocked and report the current stage, specific missing input and next action. Changed deployment, resource profile or collection bounds require review and a new collection directory. Validation-only changes can reuse the verified collection plan as described below; use a separate results directory for each replay. Collection's existing `--resume` is for a matching collector checkpoint as described below, not a general onboarding-stage resume command.

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

The first profile implementation supports vLLM text decoders, including the text portion of multimodal checkpoints, with PP1, CP1 and linear KV storage. AFD, encoder pools, speculative decoding, nonlinear recurrent state, wide EP and EPLB need additional metadata/semantics and are rejected by this route.

### Runtime and collection limits

These settings define the runtime envelope used for profile sizing, collection and generated configurations. They describe different constraints; none independently proves that a trace is covered.

| Setting | Scope and initial value |
| --- | --- |
| `--context-length` | Maximum input plus output tokens for one request. Fresh config/profile-based setup uses `min(declared context, 256000)`; identifier-only setup uses 256,000 until reviewed. The saved field remains `search.context_length`. |
| `--max-num-tokens` | Scheduled token budget per attention-DP rank. Use the supplied profile's bound, or an initial policy of 8,192. Saved as `collection.max_num_tokens` when explicitly set. |
| `--max-batch-size` | Scheduler sequence bound per attention-DP rank. Use the supplied profile's bound, or an initial policy of 256. It is independent of validation concurrency. Saved as `collection.max_batch_size` when explicitly set. |
| `--max-prefill-cudagraph-size` | Prefill CUDA graph capture limit passed to the collector, initially 2,048. Match the target serving configuration. Saved as `collection.max_prefill_cudagraph_size` when explicitly set. |

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

With no parallelism flags, guided setup asks for missing shared precision/layout facts, shows a small topology shortlist with resource reasons, and lets you choose one worker. Enter selects a displayed default only when all required profile inputs are resolved and its estimated bytes fit the budget. Otherwise a candidate number is required before setup asks for its remaining per-rank bounds. Any explicit parallelism flag bypasses suggestions and retains the existing topology validation. The chosen topology then follows the same profile review, edit, accept and cancellation flow below.

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

Changing an input recomputes dependent estimates: for example, changing `kv_cache_dtype` updates inferred `kv_bytes_per_token`, and changing `max_num_tokens` updates inferred activation bytes. The review also exposes `runtime_context_length` and `max_prefill_cudagraph_size` as editable collection settings; `context_length` edits the model profile's declared maximum. Explicit values remain in place until you edit those fields themselves. If an edit makes a required estimate unavailable, setup asks for that value before returning to review. Invalid individual answers can be corrected; an edit that conflicts with the config or complete request is rejected with the reason, and the previous profile is retained. Model identity and topology remain the declared deployment.

Only an explicit `accept` saves the reviewed request; Enter alone does not accept it. You can make repeated edits before accepting. Choose `cancel` at topology selection or review, or press Ctrl-C or send end-of-input at any prompt, to exit 130 without creating the output directory or replacing an existing request, including with `--overwrite`. This review step applies to `--model-config --interactive`; scripted setup never prompts.

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

Replace the model, revision, runtime and hardware inputs with the worker you will actually run. `--model-config` and `--fpm-profile` are mutually exclusive; `--resource-overrides` requires `--model-config` and also works with guided setup. Flat per-rank byte overrides require explicit topology flags because the same byte bound cannot be transferred or rescaled across candidate tuples. Shared precision/layout overrides can be used for automatic suggestions. Scripted setup never reads terminal input. It exits 2 without writing a request when required inputs remain unresolved or no fully assessed automatic default exists. Validation first lists all missing or invalid target identity and collection options. Once that stage is valid, it lists unresolved profile fields, or candidate-specific gaps and exact topology flags, and asks for the necessary inputs or guided setup.

The flat override fields are `architecture`, `context_length`, `num_experts`, every precision and resource field shown above, `moe_backend`, `attention_backend`, and optional `provenance`. Unknown fields, duplicate fields, invalid types, unsupported values, and incompatible cache semantics are rejected. Overrides are recorded with per-field provenance rather than discarded. Profile `context_length` is the model's declared maximum; the CLI `--context-length` selects a runtime limit that cannot exceed it. Scheduler limits are per attention-DP rank, as described in [Runtime and collection limits](#runtime-and-collection-limits).

Derivation is deliberately limited. Unambiguous architecture, context, expert count and model kind can be read independently of memory support. The initial supported estimates are:

| Quantity | Supported derivation and limits |
| --- | --- |
| Weight bytes | Vanilla full-attention Llama, Mistral, Qwen2, Qwen3 and Mixtral BF16 layouts with complete geometry and vocabulary divisible by 64 and TP. Estimates include supported biases, tied/untied embeddings, replicated norms, and the Mixtral router. Quantized storage, unknown padding, auxiliary prediction tensors, shared experts and unsupported custom bias layouts need explicit bounds. |
| KV bytes per token | Full-attention MHA/GQA in those layouts plus Qwen3 MoE and MiniMax-M2, with an explicit supported cache dtype and static tensor storage. KV heads shard or replicate according to TP; attention DP does not divide a rank's cache. MLA/DSA storage and dynamic or per-token quantization scales need explicit accounting. |
| Activation bytes | The existing backend estimate for those full-attention layouts, 16-bit FMHA, and TP1/2/4/8, within the declared scheduler envelope. It has a 70 MiB floor and does not cover DSA workspaces or speculative decoding. |
| Runtime overhead | The selected packaged hardware specification's per-rank `misc.other_mem` estimate when present. It excludes the separate CUDA graph reservation. |
| Communication overhead | The exact `misc.nccl_mem` TP entry for a pure-TP worker when present. DEP and TEP require an explicit declaration; total GPU count does not establish their communication storage. |

Source notes identify assumptions and packaged hardware hashes. Weight precision does not establish runtime FMHA or communication precision. Recognized quantization declarations can establish checkpoint/FPM precision identities or an explicit KV dtype independently of the unknown quantized storage size; a checkpoint label does not mean every tensor has that dtype. MiniMax and GLM configs may therefore supply useful identity facts while still requiring weight, cache, activation, or communication bounds. These estimates do not certify runtime memory fit. Unknown layouts must supply the remaining supported semantics explicitly; known incompatible layouts are rejected.

The saved request embeds the complete profile and its provenance. The original model config and override file are no longer needed by `onboard plan`, generated prediction and recommendation configs, or request replay. You can inspect or supply the complete profile directly as described next.

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

The report includes hardware/config provenance, exact `cli_flags`, required GPUs, resolved fields and sources, `missing` inputs, rejected candidates and a nullable `default`. Without explicit runtime precision, the example can report `needs_inputs` candidates and no default; use those diagnostics to collect shared facts in a flat `--resource-overrides` file and rerun. The preview does not validate or write `--output`, even if that request path already exists. It cannot be combined with `--interactive`, profile options or explicit topology flags. To save a request, remove `--suggest-parallel`; use a fully assessed automatic default or select a candidate by adding its exact flags. Rank-local byte overrides require that explicit choice. A pending-input candidate can be selected in guided setup, which asks for its missing bounds before final review.

Dense decoders consider TP; MoE decoders consider pure TP, DEP and TEP. The shortlist contains at most two choices per family and six distinct tuples overall, preferring the smallest estimated fit and the next one. Widths are powers of two within the packaged node domain. A declared NVLink/NVSwitch fabric can use a packaged fast rack domain when its inter-node bandwidth matches or exceeds its intra-node bandwidth; a declared `none` interconnect permits only one GPU automatically. This is hardware architecture, not an available GPU allocation. Explicit topology choices may exceed the shortlist.

The memory precheck compares complete per-rank resource estimates against 90% of packaged GPU memory. It reserves one full runtime context plus one cached token per rank for planner headroom. It does not multiply context by the maximum scheduler sequence count or assume balanced attention-DP routing. Actual total KV capacity is computed from remaining memory; concurrent requests must share that capacity. Missing precision, weights, cache, activations or reservations prevent a default; a known lower bound can reject an oversized candidate but cannot prove fit. DEP/TEP communication storage, many quantized or custom decoder layouts, and large TP activation envelopes need explicit bounds after selection. CUDA graph reservations and non-text components remain outside these estimates. Only known geometry constraints are checked; generated-plan admission and actual serving-runtime checks remain authoritative. The shortlist does not establish timing coverage, optimal performance or measured accuracy.

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

Each `resources` record requires integer `weights_bytes`, `activations_bytes`, `runtime_overhead_bytes`, `comm_overhead_bytes`, positive `kv_bytes_per_token`, `cache_layout: linear`, positive `max_num_tokens` and `max_batch_size`, and a `provenance` explanation. Obtain these from checkpoint tensor/storage metadata and serving-runtime accounting or conservative explicit estimates. A profile is an input declaration; AISimulate does not certify that its values describe your runtime.

All resource bytes are **per rank**, and must bound every rank. Scheduler envelope limits are also per attention-DP rank; they are not the summed batch and token totals used by the worker's FPM timing query. Do not copy TP resource values into DEP/TEP merely because GPU counts match. Include all non-KV storage, including quantization overheads. Overhead fields exclude the separate `kv_cache.capacity.cuda_graph_reserved_bytes` reservation. Requests above the declared scheduler envelope fail explicitly.

Automatic KV admission subtracts these non-KV bounds and CUDA graph reservation from the configured fraction of GPU memory, then divides by the declared KV bytes per token. `context_length: max` uses the profile limit; `bytes_per_token: auto` uses the exact deployment's cache geometry. These operations need a hardware specification and profile, but no FPM timing files or model graph.

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

Before execution, prepare the real checkpoint and the pinned runtime using the existing [FPM collection guide](../python/aisimulate/docs/fpm/end-to-end-workflow.md). The packaged collector invokes a Generator-resolved Dynamo/vLLM deployment and needs the corresponding GPU resources, deployment configuration, permissions, and model access. Invoking its command locally does not create that environment. `commands.json` publishes the guarded `aisimulate onboard collect-fpm --execute` command for collection, alongside a read-only collector planning command.

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

Dynamo self-benchmark generates the actual point grid after runtime initialization, using CUDA graph sizes and available runtime bounds. The onboarding command supplies `--fpm-max-model-len`, `--fpm-max-num-batched-tokens` and `--fpm-max-num-seqs` to both prefill and decode workers. Prefill sampling uses the reviewed scheduled-token and sequence limits, plus the capture limit; `--fpm-max-prefill-isl` is a total new-token budget, not a per-request context limit. DEP does not multiply these rank-local settings by GPU count. The runtime chooses the actual sampling coordinates; onboarding does not build a separate trace-derived grid or promise a sample count or collection duration.

Direct collector callers can set the same shared runtime flags. With a supplied FPM profile, omitted limits use that profile's bounds; without one, an omitted context retains the existing vLLM `-1` auto-fit behavior. Explicit limits must be positive and cannot exceed the supplied profile or contradict the prefill sampling bounds. Inspect the generated command and collector plan before committing GPU time. Successful formal collection publishes the FPM Parquet file and metadata pair into the plan's local systems data directory; diagnostic success alone does not provide that pair. Validate the worker's actual FPM query shapes afterward, including summed queries for DEP.

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

The current validation scope is **cold aggregated replay, one client lane, HBM-only cache and no speculative decoding**. The cache starts cold and normal prefix reuse can accumulate during replay. Nested timestamps are interpreted relative to the root play (`nested_timestamp_basis: absolute`). The command replays the complete supplied file; selecting one complete play is useful for a first check, but does not establish coverage of a larger corpus. Preserve the full play and its dependencies when preparing a subset. Ordinary prediction's host-memory checks still apply; a larger corpus can require a larger host, and a preflight rejection remains incomplete validation. The current path builds on existing vLLM/SGLang replay support; onboarding collection remains vLLM. Seeded cache snapshots, explicit warmup and expanded replay modes are follow-up work after the relevant AgentX changes, including [#207](https://github.com/ai-dynamo/aisimulate/pull/207) and [#235](https://github.com/ai-dynamo/aisimulate/pull/235); they are not prerequisites for this workflow.

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
