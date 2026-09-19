<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FPM self-service

`aisimulate onboard` guides onboarding a new model for FPM simulation on your designated hardware platform. It records the model, runtime, target GPU system and interconnect, plans one TP, DEP, or TEP worker, and derives the minimum GPUs required to collect that worker's timings. It produces ordinary `predict` and `recommend` configurations for validating that worker against your collected FPM data. It builds on AISimulate's per-worker FPM support and packaged collector.

Planning works before the model has an AISimulate model class or measured FPM timings. With a supplied FPM profile, planning validates the declared deployment identity and estimates memory admission from your resource bounds. Runtime compatibility and data readiness remain **unchecked**, and accuracy is **not assessed**. This setup does not provision GPUs or run target preflight checks.

Ordinary FPM `predict` and `recommend` accept an inline `engine.fpm_profile` with model identity and rank-local resource bounds. Direct interpolation uses measured timings without constructing an op-level model. Registered models retain SOL interpolation. See [Choose the model execution route](#choose-the-model-execution-route) for selection and coverage rules.

## Onboard with an agent

For a request such as "Help me onboard my model for FPM simulation on my target GPUs," follow these six stages. Claude Code reaches them through the root `CLAUDE.md` import of `AGENTS.md`; other agents use the same `AGENTS.md` entry point. These stages structure the conversation over the existing CLI; they are not new CLI commands or a persisted stage tracker. The sections below remain the detailed CLI reference.

| Stage | Required result |
| --- | --- |
| 1. Inspect the model and target | Accessible config/profile and packaged hardware specification identified; supported metadata read and gaps recorded. |
| 2. Choose the worker and workload | Checkpoint identity/revision, runtime, interconnect, exact worker topology and initial workload selected; minimum collection GPUs derived. Runtime compatibility remains unchecked. |
| 3. Derive, review and save the profile | Exact resource/precision values and assumptions reviewed and accepted; final request contains the complete profile and provenance. |
| 4. Plan collection | Validated saved plan, generated configurations and collector preview; sampling scope and remaining execution prerequisites explained. |
| 5. Collect and verify data | Matching formal Parquet/metadata pair verified, with provenance and available phase cells recorded. |
| 6. Run predict/recommend and report | Single-worker prediction and recommendation exercised; results, coverage and failures reported, with accuracy stated separately. |

At each transition or blocker, give a short update: **Stage N/6 — name; result or blocker; next action.** Ask only for missing information needed for the current stage, using facts and decisions already supplied. Continue independent authorized work while an answer is pending. Move on when the required evidence is available; stage transitions do not require another approval. Honor prior authorization, including authorization to execute collection, while preserving the profile acceptance step below.

### 1. Inspect the model and target

Start with only the missing **Hugging Face model ID** (`organization/model-name`) and **target GPU platform**. If neither is known, a first response can be:

> Stage 1/6 — Inspect the model and target. What is your Hugging Face model ID (`organization/model-name`) and target GPU platform? I'll retrieve and inspect the configuration first. If your model isn't on Hugging Face, share a local config or checkpoint path instead.

Omit facts already supplied from this request. Accept a supplied local config, complete FPM profile or checkpoint path without requiring a Hub ID; do not ask for both an ID and a config upfront. With a Hugging Face ID, use available authorized Hub access to retrieve only `config.json` at this stage. Honor any supplied revision and record the repository ID and resolved commit. If no revision was supplied, report the resolved commit for the deployment choice in stage 2. If the config cannot be retrieved, explain the specific gap and ask for a local config. The agent retrieves the file; AISimulate's CLI consumes a local config/profile and does not download it or the checkpoint.

Inspect the accessible config/profile before asking for model kind, architecture, expert count or context limit; derive supported metadata and record its source. Ask about these fields only when the configuration leaves them unresolved. For CLI setup, `--model` identifies the selected Hugging Face repository or actual checkpoint path, `--model-config` names the local JSON file, and `--model-revision` pins the checkpoint separately. A config file hash is not a checkpoint revision. This route does not require an op-level model class or per-operation silicon data. If the hardware lacks a packaged system specification, report that integration gap.

A natively multimodal checkpoint can be onboarded for its text decoder. Explain this scope during inspection and profile review: FPM excludes multimodal encoders, projectors, preprocessing and other non-text components and their resource costs. The resulting profile and timings do not model full multimodal deployment memory or latency. Keep that scope in profile provenance; unknown decoder resource bounds still need explicit input.

The agent handles checkout and environment checks: record the branch/commit, follow [development setup](../DEVELOPMENT.md#initial-setup), activate the environment, and inspect `aisimulate onboard --help` and `aisimulate onboard init --help`. Check later subcommands before using them. Report missing commands/options as a version mismatch, rather than asking the user to supply unsupported inputs.

### 2. Choose the worker and workload

Resolve only the remaining checkpoint identifier and immutable revision, literal vLLM version, target GPU platform and interconnect. Use available repository/deployment metadata before asking the user. A model label, config hash or example revision does not establish a checkpoint pin. Do not ask for total available GPUs, node allocation, GPUs per node or replica budgets during onboarding; those are choices for actual prediction or recommendation runs.

Establish input/output lengths, concurrency, context and latency targets here. If there is no workload preference, show the [planning defaults](#create-the-request) and cap the context at the model's known limit; adjust token lengths if needed to fit that limit. With a local model config, inspect the [read-only topology preview](#preview-and-choose-parallelism) using the actual identity, runtime, target and workload. Resolve shared precision/layout facts explicitly and rerun the preview. Present the default and alternatives with their resource assumptions and unresolved fields. A candidate marked `estimated_fit` has a complete declared/estimated byte budget; it is not a performance ranking, runtime qualification or a measurement. If no default is available, explain why and select an exact candidate before asking for its per-rank bounds. Do not silently assume topology or precision.

Help choose one initial TP configuration, or the relevant TP/DEP/TEP configuration for MoE, using the preview's exact flags or the [supported topology flags](#create-the-request). Honor an explicit topology even when it lies outside the automatic shortlist. Explain the selected tuple; do not require the user to know every parallelism field upfront. Derive its minimum collection GPUs as attention TP times attention DP: TP4 requires four GPUs, while DEP8 requires eight. This requirement does not declare available capacity or establish runtime placement or compatibility. Each plan collects one exact topology; use separate requests and output directories to investigate alternatives.

### 3. Derive, review and save the profile

Use [a local model config](#start-from-a-local-model-config), or [a supplied profile](#provide-identity-and-resource-metadata), for the selected deployment. Both produce a class-independent direct-FPM plan. Derive supported estimates before asking for unresolved resource fields. Explain each value's source and limitations; do not invent missing bounds. Memory values must bound every rank of the exact selected topology. Resolve effective weight, FMHA, communication and KV precision separately; a quantized checkpoint label does not determine all of them. Preserve replacements and their rationale in overrides/provenance. Review all effective values and assumptions using the appropriate flow below.

| Agent environment | Review and save behavior |
| --- | --- |
| Terminal or agent tool with a PTY | Run `aisimulate onboard init --model-config /path/to/config.json --interactive --output support-request.yaml`. Relay unresolved prompts and the final profile to the user. Apply requested `edit` actions and return the revised profile to the user for review. Enter CLI `accept` only after the user explicitly accepts those exact values; honor any existing explicit acceptance of those same values. `cancel`, Ctrl-C or EOF creates no new request and preserves any existing output, even with `--overwrite`. This final review is specific to `--model-config --interactive`; supplying `--fpm-profile` does not add it. |
| Headless or noninteractive agent | Supply identity/workload flags and `--resource-overrides` as needed, without `--interactive`. Missing required inputs exit 2 without saving; use the diagnostics to ask for the missing facts. A successful command writes immediately. Initially write to a separate path such as `draft-request.yaml` and show the embedded profile, sources and scope for user review. Apply edits in the inputs/overrides and repeat draft review until accepted; then rerun the unchanged reviewed inputs to a new final request path and verify that it matches the accepted draft before planning. The draft name is only a file convention; it has no special CLI status. Review a supplied `--fpm-profile` in the same way. |

Do not pipe answers into `--interactive`: it requires a terminal. Scripted setup has no built-in acceptance prompt. Keep draft files separate from the final request and preserve prior outputs when revising a deployment.

Model identity, runtime and topology are not profile-review edit fields. If they change, return to stage 2, regenerate dependent estimates and review the new request; use new output paths for the changed deployment.

### 4. Plan collection

Follow [Plan, preview, and explicitly execute](#plan-preview-and-explicitly-execute) using a new output directory. Inspect `support-plan.json`, the embedded/saved profile, generated prediction/recommendation configs and `commands.json`. Run `onboard collect-fpm` without `--execute` to print the collector command; this does not run the collector's own plan or check the target runtime. Inspect the read-only collector plan when its input environment is available and identify any missing prerequisites.

Explain the minimum collection GPUs, rank-local scheduler/resource envelope and sampling scope. One onboarding plan selects one parallel tuple and generates single-worker validation configs. Use separate requests/output directories for additional tuples. Actual collection resources and placement must be checked in the collector environment before execution. The synthetic request count does not bound timing samples or collection duration.

### 5. Collect and verify data

First inspect any existing timing data for a matching deployment. Reuse a verified matching formal pair when available; do not collect again merely to complete a stage. For new collection, follow the [FPM collection guide](../python/aisimulate/docs/fpm/end-to-end-workflow.md) to prepare the pinned checkpoint, Dynamo/vLLM deployment, model access and GPU resources. Use the agreed scope and existing execution authorization, reporting concrete missing prerequisites when blocked. Preview and execute with the same deployment options. `--execute` launches collection; optional `--smoke` is diagnostic and publishes no formal FPM pair. Preserve checkpoints, logs and raw evidence.

For both reused and new data, [inspect the published pair](../python/aisimulate/docs/fpm/end-to-end-workflow.md#5-inspect-the-published-pair): verify hashes, schema and identities, including actual runtime, topology, precision and available prefill/decode cells. Keep it at the generated configs' local systems path. Record any historical checkpoint-revision uncertainty in provenance; a declared revision does not prove that old measurements used it. A successful preview or smoke run is not formal data, and a matching pair does not prove all simulated queries are covered.

### 6. Run predict/recommend and report

[Run the generated ordinary configurations](#run-the-generated-ordinary-configurations) for single-worker prediction and recommendation validation. Keep `engine.systems_paths` and the direct-FPM profile intact. Report each command's exit status and results; missing cells and out-of-domain queries need their exact coordinates. For deployment prediction or optimization, set the desired replicas and GPU budget in the ordinary runtime configs. Do not change precision labels or silently switch timing methods to obtain a result. Return to stage 5 to address missing data or to stage 2 if the user changes the selected worker/workload scope. Report the simulation stage as incomplete while required queries fail.

At handoff, include the checkout revision, final request/profile, plan directory, data pair/provenance, exact commands/exit statuses and all result paths. Distinguish estimated memory fit and CPU planning from actual target-runtime checks, formal data/coverage, and completed simulations. For accuracy, report an independent matched silicon comparison if performed, or explicitly **not assessed**. Successful simulation is not evidence of accuracy; an accuracy study is not a mandatory additional collection campaign for onboarding.

### Resume from existing work

Inspect the saved request/profile, plan, data pair and results before deciding where to resume. Validate that they still match the checkout's CLI, selected deployment and workload; use the existing plan checks described below. An accepted final request can start at stage 4, a valid plan at stage 5, and a verified matching data pair at stage 6. A draft or a saved file without evidence of acceptance still needs stage 3 review. Do not repeat accepted decisions or rerun completed work without a reason.

Preserve completed artifacts when blocked and report the current stage, specific missing input and next action. Changed deployment or workload inputs invalidate dependent estimates, plans and results; return to the affected stage and use a new output directory. Collection's existing `--resume` is for a matching collector checkpoint as described below, not a general onboarding-stage resume command.

## Create the request

Use an environment installed from this checkout; see [development setup](../DEVELOPMENT.md). Guided setup requires a terminal and starts only when explicitly requested:

```bash
aisimulate onboard init --interactive --output support-request.yaml
```

Enter the actual model identifier or checkpoint path, pinned model revision, dense/MoE kind, pinned vLLM version, GPU system and interconnect. Then choose a TP size and validation workload. Setup derives and displays the GPUs required for the selected worker. Supplied options skip their prompts. Enter accepts displayed defaults; invalid values can be corrected; Ctrl-C or end-of-input cancels without saving. Existing files require `--overwrite`.

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

The default pilot uses 1,024 input tokens, 128 output tokens, concurrency 1, four requests, a 16,384-token context limit, TTFT target 1,000 ms, and TPOT target 100 ms. With `--model-config`, the default pilot context is capped at the config's known context limit, and omitted parallelism flags trigger model/hardware-aware suggestions. Headless setup selects only a fully assessed default; otherwise it exits without saving. Identifier-only and supplied-profile setup retain TP1 when parallelism is omitted. These are planning defaults, not measured model capacity or latency. Change them with the corresponding flags shown by `aisimulate onboard init --help`.

Onboarding takes no GPU-pool or node-allocation inputs. The selected topology establishes a minimum collection requirement, not a reservation or a check that those GPUs are available. Advanced flags include `--request-count`, `--objective`, `--seed`, and `--sm`. Replica counts and optimization budgets remain configurable in ordinary `predict` and `recommend` inputs.

`--tensor-parallel` always means attention TP. Use all MoE dimensions explicitly for DEP and TEP; replicas are independent workers:

| Worker | CLI parallelism flags | `(TP, PP, attention DP, MoE TP, MoE EP, CP)` |
| --- | --- | --- |
| MoE TP4 | `--tensor-parallel 4 --moe-tensor-parallel 4` | `(4, 1, 1, 4, 1, 1)` |
| DEP8 | `--tensor-parallel 1 --attention-data-parallel 8 --moe-tensor-parallel 1 --moe-expert-parallel 8` | `(1, 1, 8, 1, 8, 1)` |
| TEP8 | `--tensor-parallel 8 --attention-data-parallel 1 --moe-tensor-parallel 1 --moe-expert-parallel 8` | `(8, 1, 1, 1, 8, 1)` |

The first profile implementation supports vLLM text decoders, including the text portion of multimodal checkpoints, with PP1, CP1 and linear KV storage. AFD, encoder pools, speculative decoding, nonlinear recurrent state, wide EP and EPLB need additional metadata/semantics and are rejected by this route.

## Start from a local model config

Use a local Hugging Face-style JSON configuration to fill supported metadata and create the existing FPM profile:

```bash
aisimulate onboard init \
  --model-config /models/your-pinned-checkpoint/config.json \
  --interactive --output support-request.yaml
```

Setup reads that file without downloading a checkpoint, importing model code, constructing an analytical model, or launching GPU work. It displays source information and derived inputs, then asks for unresolved values. Missing model metadata is collected before workload options so the model's context limit can bound the validation run. Config identity hints skip their ordinary prompts; explicit CLI identity options take precedence and conflicts can be corrected. A pinned checkpoint revision, literal runtime version, GPU system and interconnect still need your input when absent. The config's SHA-256 records the local source; it is not a checkpoint revision.

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

Changing an input recomputes dependent estimates: for example, changing `kv_cache_dtype` updates inferred `kv_bytes_per_token`, and changing `max_num_tokens` updates inferred activation bytes. Explicit values remain in place until you edit those fields themselves. If an edit makes a required estimate unavailable, setup asks for that value before returning to review. Invalid individual answers can be corrected; an edit that conflicts with the config or complete request is rejected with the reason, and the previous profile is retained. Model identity and topology remain the declared deployment.

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
max_batch_size: 1
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

Replace the model, revision, runtime and hardware inputs with the worker you will actually run. `--model-config` and `--fpm-profile` are mutually exclusive; `--resource-overrides` requires `--model-config` and also works with guided setup. Flat per-rank byte overrides require explicit topology flags because the same byte bound cannot be transferred or rescaled across candidate tuples. Shared precision/layout overrides can be used for automatic suggestions. Scripted setup never reads terminal input. It exits 2 without writing a request when required inputs remain unresolved or no fully assessed automatic default exists. Validation first lists all missing or invalid target identity and workload options. Once that stage is valid, it lists unresolved profile fields, or candidate-specific gaps and exact topology flags, and asks for the necessary inputs or guided setup.

The flat override fields are `architecture`, `context_length`, `num_experts`, every precision and resource field shown above, `moe_backend`, `attention_backend`, and optional `provenance`. Unknown fields, duplicate fields, invalid types, unsupported values, and incompatible cache semantics are rejected. Overrides are recorded with per-field provenance rather than discarded. Profile `context_length` is the model's declared maximum; the CLI `--context-length` selects the smaller pilot limit and cannot exceed it. A scheduler envelope is per attention-DP rank and defaults to 8,192 tokens and the pilot concurrency; `max_num_tokens` and `max_batch_size` can declare different bounds.

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

`--suggest-parallel` emits one JSON report to stdout without creating a request, plan, timing data or output directory. It requires `--model-config` and the actual target identity; config hints can supply the model label and kind, but checkpoint revision and runtime version are never invented. Use your intended workload, context and shared precision declarations for meaningful resource accounting:

```bash
aisimulate onboard init \
  --model-config /models/your-pinned-checkpoint/config.json \
  --model /models/your-pinned-checkpoint \
  --model-revision YOUR_IMMUTABLE_REVISION \
  --framework-version YOUR_PINNED_VLLM_VERSION \
  --gpu h200_sxm --interconnect nvswitch \
  --input-tokens 1024 --output-tokens 128 --concurrency 1 \
  --suggest-parallel
```

The report includes hardware/config provenance, exact `cli_flags`, required GPUs, resolved fields and sources, `missing` inputs, rejected candidates and a nullable `default`. Without explicit runtime precision, the example can report `needs_inputs` candidates and no default; use those diagnostics to collect shared facts in a flat `--resource-overrides` file and rerun. The preview does not validate or write `--output`, even if that request path already exists. It cannot be combined with `--interactive`, profile options or explicit topology flags. To save a request, remove `--suggest-parallel`; use a fully assessed automatic default or select a candidate by adding its exact flags. Rank-local byte overrides require that explicit choice. A pending-input candidate can be selected in guided setup, which asks for its missing bounds before final review.

Dense decoders consider TP; MoE decoders consider pure TP, DEP and TEP. The shortlist contains at most two choices per family and six distinct tuples overall, preferring the smallest estimated fit and the next one. Widths are powers of two within the packaged node domain. A declared NVLink/NVSwitch fabric can use a packaged fast rack domain when its inter-node bandwidth matches or exceeds its intra-node bandwidth; a declared `none` interconnect permits only one GPU automatically. This is hardware architecture, not an available GPU allocation. Explicit topology choices may exceed the shortlist.

The memory precheck compares complete per-rank resource estimates against 90% of packaged GPU memory. It reserves the chosen full context for each concurrently resident sequence on every rank, using the smaller of workload concurrency and the profile's per-rank batch limit, plus one cached token per rank for planner headroom. It does not assume that attention-DP routing balances requests. Missing precision, weights, cache, activations or reservations prevent a default; a known lower bound can reject an oversized candidate but cannot prove fit. DEP/TEP communication storage, many quantized or custom decoder layouts, and large TP activation envelopes need explicit bounds after selection. CUDA graph reservations and non-text components remain outside these estimates. Only known geometry constraints are checked; generated-plan admission and actual serving-runtime checks remain authoritative. The shortlist does not establish timing coverage, optimal performance or measured accuracy.

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

The first command saves the request, `support-plan.json`, `commands.json`, `predict/pilot.yaml`, `recommend/pilot.yaml`, and a local `systems/` directory. When supplied, the profile is also saved as `fpm-model-profile.json`, included in the collector command, and embedded in prediction/recommendation configs. The plan records the CPU resource estimate. The second command prints the collector invocation without launching it. Generated command vectors and printed next commands use absolute output paths and preserve spaces or shell punctuation. Use a separate output directory for each request. On an existing plan, `--overwrite` can repair missing generated files for the identical request; it rejects changed inputs and preserves existing collected data.

If an AISimulate update changes generated guidance, recreate the plan in a new output directory: repair compares generated files byte for byte and does not migrate existing plans. Guidance changes do not change the saved-request identity checks used by collection.

The plan reports `fpm.collection_gpus_required`, derived as attention TP times attention DP. Collection uses only that worker width and the matching `tp`, `pure_tp`, `dep`, or `tep` preset. The generated prediction and recommendation configs both use one worker. Recommendation has one exact preset and one trial; its `optimization.constraints.max_candidate_gpus` equals the selected worker's requirement so even workers wider than the runtime default are representable. That derived cap describes this validation example and does not declare the user's total available GPUs. There is no replica expansion or deployment optimization during onboarding.

Earlier draft `aisimulate-support-request/v1` files included `identity.gpu_count`, `identity.node_count`, `identity.gpus_per_node` and `search.max_candidates`. The draft request schema keeps its version name but now rejects those obsolete fields explicitly. Copy an old request to a new file, remove those fields, review the retained topology/profile and regenerate the plan in a new output directory. Do not edit or overwrite the old plan, timings or checkpoints to reuse its identity. Regenerated plan summaries report `fpm.collection_gpus_required` and `search.candidates[].required_gpus` in place of `fpm.worker_gpus` and `search.candidates[].total_gpus`. The corresponding `onboard init` flags (`--gpu-count`, `--node-count`, `--gpus-per-node`, `--max-candidates`) are no longer accepted. Existing ordinary `predict` and `recommend` configurations remain valid; their replica and GPU-budget APIs are unchanged.

Before execution, prepare the real checkpoint and the pinned runtime using the existing [FPM collection guide](../python/aisimulate/docs/fpm/end-to-end-workflow.md). The packaged collector invokes a Generator-resolved Dynamo/vLLM deployment and needs the corresponding GPU resources, deployment configuration, permissions, and model access. Invoking its command locally does not create that environment. `commands.json` publishes the guarded `aisimulate onboard collect-fpm --execute` command for collection, alongside a read-only collector planning command.

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

The collector narrows initial prefill sampling with the pilot's input-token and concurrency bounds. With an FPM profile, both the sample batch size and total new-token axis are capped by the same rank-local scheduler limits used in the generated prediction and recommendation configs. DEP does not multiply these limits by the GPU count; coverage of the worker's summed FPM queries still requires validation. Decode uses the collector's existing sampling profile within the declared resource envelope; a four-request synthetic pilot does not imply four timing samples or a short decode campaign. Inspect the generated command and collector plan before committing GPU time. Successful formal collection publishes the FPM Parquet file and metadata pair into the plan's local systems data directory; diagnostic success alone does not provide that pair.

## Run the generated ordinary configurations

After the selected model execution route is available and formal data collection is complete:

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

- **Registered-model/SOL route:** reuse a compatible analytical class or follow [How to Add a New Model](../python/aisimulate/docs/add_a_new_model.md) when choosing to add one. Verify its operation graph, memory/cache accounting, and native FPM SOL execution.
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
