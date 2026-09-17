<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FPM self-service

`aisimulate onboard` guides onboarding a new model for FPM simulation on your designated hardware platform. It records the model, runtime, target GPU system and allocation, plans one TP, DEP, or TEP worker, and produces ordinary `predict` and `recommend` configurations that read your collected FPM data. It builds on AISimulate's per-worker FPM support and packaged collector. PR #46 is not a dependency.

Planning works before the model has an AISimulate model class or measured FPM timings. With a supplied FPM profile, planning validates the declared deployment identity and estimates memory admission from your resource bounds. Runtime compatibility and data readiness remain **unchecked**, and accuracy is **not assessed**. This setup does not provision GPUs or run target preflight checks.

Ordinary FPM `predict` and `recommend` accept an inline `engine.fpm_profile` with model identity and rank-local resource bounds. Direct interpolation uses measured timings without constructing an op-level model. Registered models retain SOL interpolation. See [Choose the model execution route](#choose-the-model-execution-route) for selection and coverage rules.

The Inkling pilot targets NVIDIA GB200 with TP, DEP, and TEP through the class-independent route. Its checkpoint, runtime, allocation, profile resource bounds, and strategy degrees must be pinned before collection. This guide does not establish Inkling readiness or GB200 accuracy.

## Create the request

Use an environment installed from this checkout; see [development setup](../DEVELOPMENT.md). Guided setup requires a terminal and starts only when explicitly requested:

```bash
aisimulate onboard init --interactive --output support-request.yaml
```

Enter the actual model identifier or checkpoint path, pinned model revision, dense/MoE kind, pinned vLLM version, GPU system, allocation, and interconnect. Then choose a TP size and pilot workload. Supplied options skip their prompts. Enter accepts displayed defaults; invalid values can be corrected; Ctrl-C or end-of-input cancels without saving. Existing files require `--overwrite`.

Both guided and scripted setup use onboarding. `--profile onboarding` is an optional spelling of the same behavior. For automation, supply the identity flags directly:

```bash
aisimulate onboard init \
  --model /models/your-pinned-checkpoint \
  --model-revision YOUR_IMMUTABLE_REVISION \
  --model-kind dense \
  --framework-version YOUR_PINNED_VLLM_VERSION \
  --gpu h200_sxm --gpu-count 4 --interconnect nvswitch \
  --tensor-parallel 2 \
  --output support-request.yaml
```

Replace the model and runtime placeholders with the actual inputs. The example does not identify an Inkling checkpoint or claim that four H200s can run your model. Framework support currently selects vLLM. Optional tokenizer, chat-template, and AISimulate revisions are recorded only when supplied.

The default pilot uses 1,024 input tokens, 128 output tokens, concurrency 1, four requests, a 16,384-token context limit, TTFT target 1,000 ms, and TPOT target 100 ms. With `--model-config`, the default pilot context is capped at the config's known context limit. Scripted setup defaults to TP1 unless `--tensor-parallel` is supplied. These are planning defaults, not measured model capacity or latency. Change them with the corresponding flags shown by `aisimulate onboard init --help`.

One node is the default; GPUs per node then equals `--gpu-count`. For multiple nodes, supply both `--node-count` and `--gpus-per-node`; their product must equal the total allocation. Each selected worker must fit on one node. Advanced flags include `--request-count`, `--max-candidates`, `--objective`, `--seed`, and `--sm`.

`--tensor-parallel` always means attention TP. Use all MoE dimensions explicitly for DEP and TEP; replicas are independent workers:

| Worker | CLI parallelism flags | `(TP, PP, attention DP, MoE TP, MoE EP, CP)` |
| --- | --- | --- |
| MoE TP4 | `--tensor-parallel 4 --moe-tensor-parallel 4` | `(4, 1, 1, 4, 1, 1)` |
| DEP8 | `--tensor-parallel 1 --attention-data-parallel 8 --moe-tensor-parallel 1 --moe-expert-parallel 8` | `(1, 1, 8, 1, 8, 1)` |
| TEP8 | `--tensor-parallel 8 --attention-data-parallel 1 --moe-tensor-parallel 1 --moe-expert-parallel 8` | `(8, 1, 1, 1, 8, 1)` |

The first profile implementation supports vLLM decoder-only models with PP1, CP1 and linear KV storage. AFD, encoder pools, speculative decoding, nonlinear recurrent state, wide EP and EPLB need additional metadata/semantics and are rejected by this route.

## Start from a local model config

Use a local Hugging Face-style JSON configuration to fill supported metadata and create the existing FPM profile:

```bash
aisimulate onboard init \
  --model-config /models/your-pinned-checkpoint/config.json \
  --interactive --output support-request.yaml
```

Setup reads that file without downloading a checkpoint, importing model code, constructing an analytical model, or launching GPU work. It displays source information and derived inputs, then asks for unresolved values. Missing model metadata is collected before pilot options so the model's context limit can bound the pilot. Config identity hints skip their ordinary prompts; explicit CLI identity options take precedence and conflicts can be corrected. A pinned checkpoint revision, literal runtime version, GPU system, allocation, and interconnect still need your input when absent. The config's SHA-256 records the local source; it is not a checkpoint revision.

Profile memory quantities describe the largest requirement on any rank of the exact selected TP, DEP, or TEP worker. The prompt explains why each unresolved value needs input. Enter integer bytes or an explicit unit such as `70 GiB`, `512 MiB`, or `1.5 GB`; the conversion must produce a whole number of bytes. Invalid individual values are prompted again. Model revision, runtime version, and deployment conflicts return to the corresponding option while retaining accepted resource answers. Incompatible config facts or combined resource bounds fail with the reason and leave no request; correct the source or overrides and rerun setup. Ctrl-C or end-of-input exits 130 and leaves no partial request or replacement of an existing request.

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
  --gpu h200_sxm --gpu-count 4 --interconnect nvswitch \
  --tensor-parallel 4 --output support-request.yaml
```

Replace the model, revision, runtime and hardware inputs with the deployment you will actually run. `--model-config` and `--fpm-profile` are mutually exclusive; `--resource-overrides` requires `--model-config` and also works with guided setup. Scripted setup never reads terminal input. It exits 2 without writing a request when required inputs remain unresolved. Validation first lists all missing or invalid deployment identity and pilot options. Once that stage is valid, it lists all unresolved profile fields, explains why each is unavailable, and asks for `--resource-overrides` or guided setup.

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

## Provide identity and resource metadata

For a model without an analytical class, create a JSON or YAML FPM profile and pass `--fpm-profile /path/to/model-profile.yaml` to `onboard init`. The request embeds the complete profile; ordinary prediction and recommendation use the same object under `engine.fpm_profile`. The profile schema is [FpmModelProfile](../python/aisimulate/src/aiconfigurator_core/sdk/fpm_profile.py). It rejects missing fields, unknown fields, conflicting identities and mutable revision placeholders.

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

If a newer draft revision changes generated guidance, recreate the plan in a new output directory: repair compares generated files byte for byte and does not migrate earlier draft plans. Guidance changes do not change the saved-request identity checks used by collection.

The guided plan uses one selected parallel tuple. By default it evaluates a single worker, so recommendation is not a broad deployment search. `--max-candidates 2` additionally considers the largest count of identical workers that fits the allocation, when that differs from one worker. Each choice gets an independent recommendation config pinned to that replica count with a one-trial budget. The single worker keeps `recommend/pilot.yaml`; the second choice uses `recommend/replicas-N.yaml`, where `N` is its replica count. The plan reports the actual candidate count and lists both config and result paths. Collection uses the matching `tp`, `pure_tp`, `dep`, or `tep` preset and exact worker GPU count.

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

With two candidates, run both replica counts. This loop derives the candidate names from the validated saved request and constructs fixed `aisimulate recommend` commands:

```bash
python3 - <<'PY'
import shlex
import subprocess
from pathlib import Path

from aisimulate.support.plan import check_plan
from aisimulate.support.schema import SupportRequest

root = Path("./aisimulate-support").resolve()
request = SupportRequest.from_yaml(root / "request.yaml")
check_plan(request, root)
max_replicas = request.identity.node_count * (request.identity.gpus_per_node // request.worker_gpus)
replicas = list(dict.fromkeys((1, max_replicas)))[:request.search.max_candidates]
statuses = []
for count in replicas:
    name = "pilot" if count == 1 else f"replicas-{count}"
    command = [
        "aisimulate", "recommend",
        "--config", str(root / f"recommend/{name}.yaml"),
        "--output-dir", str(root / f"recommend-results/{name}"),
        "--format", "json",
    ]
    result = subprocess.run(command, check=False)
    statuses.append(result.returncode)
    print(f"exit {result.returncode}: {shlex.join(command)}", flush=True)
raise SystemExit(1 if any(statuses) else 0)
PY
```

The loop reports each command's exit status and attempts all commands, including when the pilot finds no feasible candidate. It exits 1 after all attempts if any command returned a nonzero status, otherwise 0. Successful outputs remain usable even when the loop exits 1; inspect each result before comparing candidates.

Run either the individual recommendation command or the loop against fresh result directories. The loop also handles a one-candidate plan and does not execute entries from `commands.json`. Results remain separate under `recommend-results/pilot` and, when present, `recommend-results/replicas-N`; compare their objective and latency results for the same workload. This plan does not produce a combined ranking or search additional TP sizes or scheduler settings.

The generated configurations select `engine.workers.aggregated.timing.forward_model: fpm` and set `engine.systems_path` to the plan's absolute local systems directory. The same root supplies hardware and collected FPM data. Recommendation preserves it in exported prediction configs. Moving the plan to another machine requires updating absolute paths or regenerating it there.

Ordinary `predict` and `recommend` retain their existing defaults when no profile is supplied. Generated configs can be edited through the public schema. A recommendation with a profile enumerates only its declared deployment tuples that fit the GPU budget; it preserves the complete inline profile and interpolation choice in exported prediction configs. Default scheduler search ranges may exceed a profile's envelope, so pin or bound those domains explicitly. A successful simulation is not an accuracy result. Compare its output with an independent run of the same model, runtime, topology, and workload to assess accuracy.

## Choose the model execution route

Hand off the saved request, pinned model configuration, and plan. Both routes need a canonical checkpoint identity, effective precision and topology, correct weight and KV-cache accounting, and matching whole-forward FPM measurements. Collected timings alone do not establish memory fit.

- **Registered-model/SOL route:** reuse a compatible analytical class or follow [How to Add a New Model](../python/aisimulate/docs/add_a_new_model.md) when choosing to add one. Verify its operation graph, memory/cache accounting, and native FPM SOL execution.
- **Class-independent direct route:** supply the identity/resource profile and set worker `timing: {type: default, forward_model: fpm, fpm_interpolation: direct}`. The guided planner selects this route whenever a profile is supplied. No operation graph is constructed for resources, timing, or recommendation candidates.

`fpm_interpolation: auto` retains SOL for a registered architecture and chooses direct for an unregistered architecture with a profile. Explicit `sol` requires a registered class; explicit `direct` requires a profile. A model-construction error does not trigger a silent change of method. The interpolation setting applies to default FPM timing only.

Direct timing first uses an exact point or interpolation within a measured curve. Prefill interpolation stays at the same batch size, with two measured KV neighbors whose prompt curves both cover the requested token count. Wider KV bracketing removes the old distance limit only when the narrower direct bracket is unavailable. Decode interpolation respects the measured batch/capture domain. Both phases exclude synthetic `fake_fallback` rows, including healed/extrapolated values. Missing two-sided support, unmeasured batches and out-of-domain queries fail explicitly. The direct route does not apply SOL-dependent prefill batch clamping or general extrapolation; 2D interpolation remains experimental.

Per-operation silicon profiling described in the model guide is not required by either FPM route. The workflow collects whole-forward timings, then verifies prediction and recommendation for the exact target deployment. Existing MiniMax-M2.7/H200 TP4 and GLM-5.2/B200 DEP8/TEP8 data provide functional validation targets. Coverage and interpolation error must be reported separately; these targets do not qualify Inkling or GB200.
