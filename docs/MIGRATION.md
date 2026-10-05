# Migrate to AISimulate

This guide maps existing AIConfigurator commands and Python/Rust APIs to implemented
AISimulate interfaces. A mapping does not imply identical estimation or search
semantics. Workflows without a unified replacement use the bundled compatibility
CLI or SDK.

The guide describes AISimulate 0.13.0. Published AIC and AISimulate 0.12.0 artifacts
retain their original behavior. For source provenance and transfer records, see
[migration history](migration-history.md).

> [!WARNING]
> Replay and Sweeper interfaces are experimental. Validate selected deployments
> on the target hardware; simulation results do not establish serving accuracy.

## Contents

- [Installation and package mapping](#installation-and-package-mapping)
- [Command mapping](#command-mapping)
- [Verified serving examples](#verified-serving-examples)
- [Configuration and workflow mapping](#configuration-and-workflow-mapping)
- [Compatibility-only workflows](#compatibility-only-workflows)
- [Python and Rust API migration](#python-and-rust-api-migration)

## Installation and package mapping

Remove the standalone `aiconfigurator` and `aiconfigurator-core` distributions
before installing the `aisimulate` wheel. Both console commands, `aisimulate` and
`aiconfigurator`, come from that wheel. There is no separate core wheel.
See the [installation guide](installation.md).

| AIConfigurator 0.12.0 surface | AISimulate 0.13.0 surface | Compatibility |
| --- | --- | --- |
| Python distribution `aiconfigurator` | `aisimulate` | The `aiconfigurator` command remains; the Python import namespace is removed in 0.13.0 |
| CLI `aiconfigurator ...` | `aisimulate predict` / `aisimulate recommend` for new simulation workflows; `aiconfigurator ...` for compatibility-only workflows | See [command mapping](#command-mapping); this is not a flag-compatible rename |
| Python distribution `aiconfigurator-core` | included in `aisimulate` | No separate core distribution is installed |
| `aiconfigurator_core` | `aisimulate_core` from the `aisimulate` wheel | Update Python imports for 0.13.0 |
| `aiconfigurator_core.sdk` | `aisimulate_core.sdk` in the same wheel | The old import path is removed in 0.13.0 |
| Rust package/import `aiconfigurator-core` / `aiconfigurator_core` | `aisimulate-core` / `aisimulate_core` | Cargo consumers may temporarily alias the new package under the old dependency key |

Rust consumers may alias the package `aisimulate-core` under the old dependency
key `aiconfigurator-core`. The former compiled-engine module moves to
`aisimulate_core::perfmodel::engine`; `aisimulate_core::engine` is Replay's
scheduler. New code should use the canonical `perfmodel` namespace.
`AicEngineBuilder` and `AicEngine` require the `python` feature; standalone
embedded binaries also require `embed-python` and an importable matching wheel.
Pure Rust performance-model types do not require Python.

## Command mapping

All six `aiconfigurator cli` commands remain in the AISimulate wheel. Keep their
original arguments when using compatibility workflows. There is no automatic
converter for AIC experiment files or a flag-compatible CLI rename.

| AIC command | Implemented destination | Difference / boundary |
| --- | --- | --- |
| `generate` | Bundled `aiconfigurator cli generate` | No unified `generate` command. Deployment files remain a compatibility/generator SDK workflow. |
| `estimate` | `aisimulate predict` | Serving simulation uses arrivals and scheduler-formed batches. Static fixed-batch diagnostics remain in AIC. |
| `support` | Bundled `aiconfigurator cli support` | No unified support-query command. |
| `recommend` | `aisimulate recommend` with `optimization.target: min_gpus` | Smallest qualifying candidate evaluated; not AIC's analytical replica sizing or proof of a global minimum. |
| `default` | `aisimulate recommend` | Choose workload, GPU ceiling, objective, and search budget explicitly. |
| `exp` | Bundled `aiconfigurator cli exp` | Translate individual experiments to `predict` or `recommend`; no equivalent named-file orchestration. |

See the [legacy command reference](cli/legacy-aic-user-guide.md) for compatibility
arguments. `aisimulate onboard` is an additional FPM onboarding workflow, not an
AIC command replacement; see the [onboarding guide](fpm-self-service/implementation.md).

## Verified serving examples

These examples use the offline `engine` stack, H200 op-level performance data,
and vLLM 0.24.0. Run them in one working directory with fresh output directories.
They describe simulated serving traffic, not fixed-batch AIC results.
The original CLI examples and Python imports in this guide were run successfully on
2026-10-02 with AISimulate 0.13.0, built from source revision
[`67b1b9a5`](https://github.com/ai-dynamo/aisimulate/commit/67b1b9a51388bc8202df5d237cd3de4d4f07414f). The AFD example below uses
TRT-LLM 1.3.0rc20. The [captured example results](migration-examples.json) record
the environment, data trees, input hashes, exact commands, output summaries, and
passed request-count, metric, GPU-budget, and artifact checks. These are
execution checks, not a silicon-accuracy or AIC-equivalence claim. Captured
timing values are observations, not golden assertions; results can change with
source or performance-data revisions.
The AFD configuration below was subsequently constrained and rechecked in
[the recommendation search-quality update](https://github.com/ai-dynamo/aisimulate/pull/377);
the archived results retain the earlier input hashes.

### Predict one deployment

AIC's `estimate` fixes an estimator batch. The following input instead keeps up
to 64 requests in flight and lets the scheduler form batches. Save as
`prediction.yaml`:

```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 64}
  stop: {requests: 100}
engine:
  mode: aggregated
  model: meta-llama/Meta-Llama-3.1-8B
  hardware: h200_sxm
  backend: vllm
  backend_version: "0.24.0"
  workers:
    aggregated:
      parallelism: {tensor: 2, replicas: 1}
```

```bash
aisimulate predict --stack engine --config prediction.yaml --output-dir ./prediction-output
```

Inspect `prediction-output/prediction.json` for 100 completed requests, TTFT,
inter-token latency, and output throughput. TP=2 and one replica use two GPUs.

### Search with a GPU ceiling

AIC's `default` performs its own capacity sweep. This replacement samples eight
candidates under fixed serving traffic and an eight-GPU ceiling. Save as
`budget-search.yaml`:

```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 32}
  stop: {requests: 320}
engine:
  mode: aggregated
  model: meta-llama/Meta-Llama-3.1-8B
  hardware: h200_sxm
  backend: vllm
  backend_version: "0.24.0"
  workers:
    aggregated:
      parallelism: {preset: default}
evaluation:
  sla: {ttft_ms: 800, itl_ms: 30}
optimization:
  target: throughput
  strict_sla: true
  constraints: {max_candidate_gpus: 8}
optimizer: {algorithm: random, max_trials: 8, parallelism: 1, seed: 11}
```

```bash
aisimulate recommend --stack engine --config budget-search.yaml --output-dir ./budget-search
aisimulate predict --stack engine --config ./budget-search/recommendations/0001.yaml \
  --output-dir ./budget-selected
```

Inspect `budget-search/recommendation.json` for candidate metrics and rejection
reasons, and `budget-selected/prediction.json` for the selected serving result.
Saved prediction inputs live in `recommendations/`, in rank order.

`throughput` scores output tokens/s. The GPU count is a ceiling; the winner may
use fewer GPUs. Strict SLA filtering checks aggregate-mean latency bounds.
Eight random trials do not enumerate the full domain.

A completed search with no qualifying candidate exits 1 and writes no selected
YAML. Resource-limited candidates cause exit 3, even when some selected YAML is
available. Inspect the ledger and [resource diagnostics](local-resources.md)
before replaying a selected candidate.

## Configuration and workflow mapping

### Common inputs

Use these mappings when translating a workflow; choose traffic and search objectives explicitly.

| AIC input | AISimulate input | Translation note |
|---|---|---|
| `--model-path`, `--system`, `--backend` | `engine.model`, `engine.hardware`, `engine.backend` | Concrete model, hardware, and backend; optional `engine.backend_version` pins the version. |
| `--isl`, `--osl` | `traffic.source.input_tokens`, `traffic.source.output_tokens` | Synthetic request lengths. |
| `--total-gpus` | `optimization.constraints.max_candidate_gpus` | Search ceiling; prediction GPU use follows worker parallelism and replicas. |
| `--target-request-rate` | `traffic.load: {type: constant_rate, requests_per_second: N}` | Offered traffic. |
| `--target-concurrency` | `traffic.load: {type: concurrency, concurrency: N}` | In-flight request cap. |
| `--ttft`, `--tpot` | `evaluation.sla.ttft_ms`, `evaluation.sla.itl_ms` | Goodput uses request-level latency; strict filtering uses aggregate means, including mean TPOT. |
| `--request-latency` | `evaluation.sla.e2e_ms` | Use instead of TTFT/ITL bounds. |
| `--strict-sla` | `optimization.strict_sla: true` | Reject aggregate-mean latency violations before ranking. |

See the [AISimulate input reference](cli/user-guide.md#configuration-model) for full schemas and
[result reference](cli/user-guide.md#outputs) for files and metric units. Analytical EPD uses
aggregate-only SLA semantics, described in its feature guide.

### Traffic, parallelism, and minimum GPUs

| Workflow | Current mapping and limits |
| --- | --- |
| Request-rate sizing | `traffic.load.type: constant_rate` and `requests_per_second` set offered load. `goodput_per_gpu` ranks SLA-qualified output tokens/s/GPU; it does not minimize GPU count. |
| Minimum-GPU sizing | `optimization.target: min_gpus` minimizes GPUs among evaluated qualifying candidates. Fixed request-rate traffic also requires `optimization.constraints.min_goodput_rps`; fixed concurrency does not. |
| Parallelism search | `engine.workers.<role>.parallelism` controls tensor, attention-data, MoE tensor/expert, and replica domains. `optimization.target: pareto` returns throughput/GPU versus throughput/user tradeoffs. |
| Regular P/D | `engine.mode: disaggregated` with `prefill` and `decode` workers replaces AIC disaggregated estimates with scheduled serving traffic. `engine.kv_transfer` controls bandwidth and prompt-KV accounting. |
| Traces and sessions | `traffic.source` accepts supported trace/session inputs. `--capture-per-request` writes individual request evidence. AIC has no equivalent replay command. |
| Dynamo Router/Planner | The optional `dynamo` stack supplies Router/Planner integration. Install its matching dependencies; Planner scaling limits and candidate GPU budgets are separate controls. AIC has no equivalent replay command. |

`min_gpus` enforces aggregate-mean SLA bounds and ranks qualifying candidates by
GPU count, then higher goodput output throughput and lower E2E latency. It finds
the smallest qualifying candidate evaluated, not a global minimum or an
extrapolated replica count. Offered rate and delivered goodput differ because
startup and drain count toward the replay duration. Random `min_gpus` searches
visit the smallest legal backend/topology pairs first, before revisiting their
scheduler domains. The CLI reports suggestion-budget coverage.

Minimum-GPU selection supports static engine pools and fixed synthetic rate or
concurrency traffic. It rejects Dynamo/adapters, traces, sessions, searched loads,
and candidate-relative KV load. Analytical EPD supports fixed concurrency and
aggregate latency only. See [optimization goals](sweeper/optimization-goals.md#minimum-gpus).

### Context parallelism

Unified `predict` exposes two per-role controls under
`engine.workers.<role>.parallelism`; `recommend` does not sweep them.

| AIC input | Prediction input | Meaning |
| --- | --- | --- |
| `*_cp_candidates` | `prefill_context` | Prefill CP: SGLang `--attn-cp-size`, vLLM `-pcp`; adds attention ranks. |
| `dcp_size`, `prefill_dcp_size`, `decode_dcp_size` | `decode_context` | Decode CP: vLLM `-dcp`, SGLang `--dcp-size`; stripes KV over existing attention ranks without adding GPUs. |

Both default to one. Aggregated workers accept at most one above one;
disaggregated roles carry each independently. Support depends on the model
family and backend (`supports_cp` / `supports_dcp`). The canonical performance
model config exposes `cp_size` and `dcp_size`; see
[context-parallel controls](core-api.md#choosing-a-forward-pass-api).

### Cache capacity and shared prefixes

AIC's GPU-memory fraction maps to
`engine.workers.<role>.kv_cache.capacity.memory_fraction`. Other capacity controls
include block size, fixed GPU blocks, and `cuda_graph_reserved_bytes`.

AIC's `--prefix N` assumes a fixed cached-token count for every request. There is
no equivalent serving-cache assumption. `traffic.source.cached_prefix_tokens`
sets shared synthetic input, with a cold first request and later hits governed
by worker placement, block size, eviction, and cache state. Trace/session shared
prefixes likewise describe workload sharing, not guaranteed hits. Enable
`kv_cache.prefix_caching` on supported backends; enabling it alone creates no
shared input.

`kv_cache.host_offload` adds vLLM host-cache simulation, with no AIC CLI flag
counterpart. It requires prefix caching and supports aggregated or token-only
P/D workers, including attention DP. Recommendation keeps host capacity and
bandwidth fixed while searching supported parallelism. Cluster-shared G2 on
both P/D roles requires identical explicit integer tensor/pipeline values.
See the [host-offload contract](cli/user-guide.md#native-vllm-host-offload-prediction)
and [G2 scope](g2-cache-scope.md).

### Estimator selection and data policies

| AIC / saved input | Current input | Boundary |
| --- | --- | --- |
| `--forward-model fpm` | Per-role `timing.estimation_mode: fpm_interpolation` | Requires matching whole-forward profiles and default timing. |
| `--database-mode` | `engine.database_mode` | Preserve the selected data policy; `SOL` uses theoretical estimates. |
| `--transfer-policy` | `engine.transfer_policy` | Controls performance-data transfer, not serving KV bandwidth. |
| `--systems-paths` | `engine.systems_paths` | Ordered roots; `default` includes bundled system data. |
| Regression/correction options | `engine.estimator_config` | Canonical nested estimator configuration; see [estimator controls](core-api.md#estimator-controls). |
| Saved `timing.forward_model` | `timing.estimation_mode` | Preserve the old explicit selection and strict fallback behavior. |

The default selection is `auto` with fallback `deny`. Auto checks
`op_level`, `fpm_interpolation`, then `fpm_regression` during construction, even
with deny. Explicit mode plus deny prevents switching estimators; allow tries
that mode first, then the other modes in global priority order. Invalid inputs
do not trigger fallback. An untrained regression is not ready for offline
simulation, and queries do not switch estimators after construction.

Engine-wide data policies require regular aggregated/disaggregated language
workers with default timing in every role. AFD, analytical encoders, and
fixed/polynomial timing retain their separate paths. Saved recommendation YAML
preserves resolved role data roots, versions, policies, and estimator settings.
External FPM profiles require a matching parquet/metadata pair; see the
[FPM guide](../python/aisimulate/docs/fpm/README.md).

### Analytical EPD

AIC encoder flags map to `engine.workers.encoder`; image inputs map to
`traffic.source.images`. Prediction supports E+aggregated and E+P+D layouts;
recommendation can search encoder/language-worker configurations. These paths
require fixed synthetic images and concurrency, with aggregate-only SLA
semantics. They model encoder capacity and latency without event-level encoder
queueing or embedding transfer. Traces, sessions, and per-request capture are
unsupported. See [EPD inputs and limits](sweeper/epd.md#unified-cli).

### Heterogeneous P/D hardware

AIC `prefill_system_name` and `decode_system_name` map to
`engine.workers.prefill.hardware` and `engine.workers.decode.hardware`.
Omitted roles inherit `engine.hardware`. These are concrete SKUs, not search
domains. P/D shares one model, backend, and backend version; pin a common version
if the SKUs resolve different latest versions.

Each role receives independent parallelism and KV-capacity checks on its
effective SKU. Aggregated candidates use the fallback SKU. Hardware overrides
are rejected on aggregated workers and AFD companions; analytical encoder
hardware has its own setting. The Sweeper SDK retains
`search_space.prefill_hardware_sku` and `search_space.decode_hardware_sku`.
Dynamo Router AIC hooks must consume the effective prefill SKU or use a non-AIC
prefill load model. Heterogeneous deployment generation is unsupported.

### Prediction details and power

`aisimulate predict` supports `--detail summary,memory,time,energy,source`;
`all` requests all five. Run prediction on saved recommendation YAML to inspect
a selected candidate. AIC's `--detail` belongs to `estimate`, not `default`.

| Selector | Available evidence |
| --- | --- |
| `summary` | Serving metrics for the configured traffic. |
| `memory` | Initial per-rank capacity and available components, before native capacity adjustments. |
| `time` | Request latency statistics and, on native op-level paths, scheduled phase/operation timing and SOL comparisons. |
| `source` | Operation source tags and executed measurement substitutions; not full file/row lineage. |
| `energy` | Available scheduled active forward-pass energy, covered latency, and modeled power per GPU. |

JSON keeps complete detail evidence and explicit skipped-section reasons;
`--detail-top-n` limits terminal tables only. Explicit KV blocks or unsupported
providers can omit memory details. Analytical EPD reports available language
memory without claiming a complete encoder breakdown.

Whole-forward FPM and latency-only providers do not expose per-operation
timing/source evidence. EPD/AFD overlays and the external Dynamo Python adapter
do not export a qualified combined operation report. Missing SOL evidence keeps
the measured timing and an explicit reason.

Normal prediction/recommendation summaries include `power_w` and
`power_coverage`, without requiring energy detail. Numeric watts require valid
operation-energy evidence at the 90% latency-coverage gate. Missing coverage is
not zero power, and coverage is not prediction accuracy. Adding energy detail
does not create missing measurements. See the [modeled-power contract](power-model.md)
and [detail output contract](cli/user-guide.md#prediction-details).

### Ngram prompt-lookup speculation

AIC's scalar accepted-token mean maps to explicit workload assumptions, not an
automatically inferred replay distribution. Unified `engine.speculation` uses
`kind: ngram`, a draft-token count, conditional `acceptance_rates`, and a seed.
Each rate is conditional on earlier draft tokens being accepted. The scalar
mean alone does not determine these rates.

Prediction/recommendation support offline engine-stack vLLM aggregated/P/D
language workers with op-level timing. Search preserves draft count and
acceptance assumptions rather than optimizing them. Lookup drafts are assumed
available every round; actual token matching, host lookup latency, and mixed
drafted/draftless rounds are not modeled. KV prefix reuse is separate. See
[ngram combinations](cli/user-guide.md#prompt-lookup-ngram-speculative-decoding).

### Preserve pinned engine and request controls

The unified `predict` and `recommend` configurations accept the following flat
`engine` controls. They use the same canonical estimator interface as estimator
selection and fallback policy.

| AIC control | Unified configuration |
| --- | --- |
| `nextn`, `nextn_accepted` | `engine.nextn`, `engine.nextn_accepted` (both required for MTP) |
| Chunked prefill | `engine.enable_chunked_prefill` (omit for backend default) |
| EPLB and redundant expert slots | `engine.enable_eplb`, `engine.wideep_num_slots` |
| MoE and attention kernel backends | `engine.moe_backend`, `engine.attention_backend` |
| Quantization overrides | `engine.gemm_quant_mode`, `engine.moe_quant_mode`, `engine.kvcache_quant_mode`, `engine.fmha_quant_mode`, `engine.comm_quant_mode` |
| Exact synthetic shared prefix | `traffic.source.cached_prefix_tokens` |
| Maximum sequence length | Existing `engine.context_length` |
| GPU memory fraction | Existing `engine.workers.<role>.kv_cache.capacity.memory_fraction` |

`enable_wideep` is obsolete: topology now determines the MoE execution regime.
The new model controls require default timing on every language role. AFD and
analytical encoder configurations reject them. Backend/model compatibility is
validated by the canonical constructor before simulation. Use `--stack engine`
for these controls and exact synthetic shared prefixes. Older Dynamo adapters
do not support them and fail capability validation before replay. Explicit MTP
expected acceptance also requires an opt-in runner capability; legacy acceptance-rate
payloads retain their existing compatibility. AFD and
AFD+PD reject positive `cached_prefix_tokens`.

Save this bounded recommendation as `controls.yaml` to preserve a shared prefix
and KV quantization while using the existing capacity field:

```yaml
engine:
  mode: aggregated
  model: Qwen/Qwen3-32B
  hardware: h200_sxm
  backend: vllm
  context_length: 4096
  estimation_mode: op_level
  fallback_policy: deny
  kvcache_quant_mode: fp8
  enable_chunked_prefill: true
  workers:
    aggregated:
      kv_cache:
        capacity:
          memory_fraction: 0.85
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
    cached_prefix_tokens: 256
  load:
    type: concurrency
    concurrency: 8
  stop:
    requests: 32
optimization:
  constraints:
    max_candidate_gpus: 8
optimizer: {algorithm: random, max_trials: 2, parallelism: 1, seed: 11}
```

```bash
aisimulate recommend --stack engine --config controls.yaml --output-dir ./controls-search
```

For MTP-capable models, specify both `engine.nextn` and
`engine.nextn_accepted`. The accepted count is a workload assumption;
replay samples fractional accepted-token progress.
Saved candidate YAML retains these values and all engine model controls.
A shared prefix does not mean a prewarmed cache; the first request remains cold.
Only complete cache blocks can be reused, so a shared prefix shorter than the
engine cache block size can produce zero hits.

### AFD translation

AIC AFD batch-size limits map to an explicit `engine.afd.a_batch_size`.
`phase: decode` and `combined_with_pd: true` select decode-side AFD with a regular
prefill companion. The total GPU budget includes attention, FFN, and companion
pools. This is analytical fixed-length synthetic traffic, not event-level AFD.

This walkthrough pins a 2,048-token context to bound KV memory and uses an
illustrative 300 ms ITL limit. It constrains the prefill companion to TP16 and the
AFD attention side to TP8 so this small smoke search has a qualifying topology.
Expand the topology and scheduler domains for an optimization run. Choose limits
appropriate to your workload.
Save as `afd-recommendation.yaml`:

<!-- afd-migration-contract-start -->
```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: constant_rate
    requests_per_second: 4
  stop:
    requests_per_load_unit: 10
engine:
  mode: afd
  model: Qwen/Qwen3-32B
  context_length: 2048
  hardware: h200_sxm
  backend: trtllm
  backend_version: 1.3.0rc20
  afd:
    phase: decode
    combined_with_pd: true
    a_batch_size: 128
    tp_a: 8
    num_microbatches: 3
    pipeline_model: optimistic
  workers:
    prefill:
      parallelism:
        preset:
        - attention_data: 1
          moe_expert: 1
          moe_tensor: 1
          pipeline: 1
          replicas: 1
          tensor: 16
      scheduler:
        max_batched_tokens: 8192
        max_sequences: 16
evaluation:
  sla:
    ttft_ms: 800
    itl_ms: 300
optimization:
  target: throughput_per_gpu
  strict_sla: true
  constraints:
    max_candidate_gpus: 32
optimizer:
  algorithm: random
  max_trials: 2
  parallelism: 1
  seed: 11
```
<!-- afd-migration-contract-end -->

```bash
aisimulate recommend --stack engine --config afd-recommendation.yaml --output-dir ./afd-search
aisimulate predict --stack engine --config ./afd-search/recommendations/0001.yaml \
  --output-dir ./afd-selected
```

Inspect `afd-selected/afd-replay-spec.json` and `afd-qualification.json` for
analytical inputs and GPU accounting. AIC's capped/truncated topology search and
this fixed attention-batch search do not enumerate identical operating points.
See [AFD topology and limits](sweeper/afd-topology.md).

## Compatibility-only workflows

| Workflow | Current boundary |
| --- | --- |
| Static `static`, `static_ctx`, `static_gen` estimates | Keep AIC/SDK for fixed-batch, prefill-only, or decode-only diagnostics. Serving concurrency and scheduler admission limits do not fix every batch size. |
| Fixed cached-token assumption | Keep AIC `--prefix` for controlled cached-prefix estimates. Shared input in Replay preserves cold-cache behavior. |
| Deployment artifacts | Use bundled AIC or the generator SDK. `recommendations/*.yaml` are simulation inputs, not launch scripts or Kubernetes manifests. AIC `default`/`exp` can generate artifacts with `--save-dir`; EPD/AFD and heterogeneous generation are unsupported. |
| Named experiments and support queries | Keep AIC `exp` and `support`. Migrate individual experiments separately. Estimator coverage alone does not establish end-to-end topology support. |
| Pipeline parallelism | Keep PP-dependent workflows on AIC. Unified defaults use PP=1; explicit PP can affect timing, capacity, and GPU accounting but does not establish pipeline scheduling, overlap, or bubble fidelity. |
| Context-parallel candidate sweeps | Keep CP sweeps on AIC; unified `recommend` does not sweep context parallelism. Pinned `predict` supports the per-phase controls described above. |
| Exact legacy search domains | AIC `*_num_gpu_candidates` describes worker GPU counts. Unified default presets use 1/2/4/8/16 per worker; `max_candidate_gpus` limits the whole deployment. |
| Context/request-length sweeps | Unified context length, synthetic input length, and output length stay fixed within a search. Use separate configs or AIC named experiments. |
| Exhaustive capacity sweeps and legacy ranking | Bayesian/random optimizers sample within `max_trials`; a larger budget does not guarantee exhaustive coverage or AIC ranking semantics. |
| Other speculative schemes | EAGLE-3, DFlash, DSpark, and standalone draft models use compatibility/SDK interfaces subject to scheme-specific limits. Unified MTP requires explicit `nextn` and `nextn_accepted`; neither MTP nor ngram predicts acceptance. |

See the [generator SDK](../python/aisimulate/docs/generator_overview.md),
[legacy tuning reference](../python/aisimulate/docs/advanced_tuning.md), and
[speculation limits](../python/aisimulate/src/aisimulate_core/sdk/speculation/README.md).

Recommendation performs search and replay, whereas AIC sizing uses different
analytical semantics. Choose algorithm, trial budget, and concurrency explicitly;
faster search can reduce recommendation quality. Historical runtime ratios are
not measurements of this checkout.

## Python and Rust API migration

### Python imports and resources

The `aiconfigurator` and `aiconfigurator_core` import namespaces are removed in
0.13.0. The `aiconfigurator` executable remains, implemented by
`aisimulate.legacy_cli.entrypoint`. There is no legacy import hook. Convert old
pickles that encode removed module paths in the old environment before upgrading.

| Removed import | Replacement |
|---|---|
| `aiconfigurator_core` | `aisimulate_core` |
| `aiconfigurator_core.sdk` | `aisimulate_core.sdk` |
| `aiconfigurator.sdk.task_v2` | `aisimulate.sdk.task_v2` |
| `aiconfigurator.sdk.config_adapter` | `aisimulate.sdk.config_adapter` |
| `aiconfigurator.sdk.<core module>` | `aisimulate_core.sdk.<core module>` |
| `aiconfigurator.generator` | `aisimulate.generator` |
| `aiconfigurator.cli` | `aisimulate.legacy_cli` (legacy workflow internals) |

For new estimator integrations, prefer the supported
[core SDK facade](core-api.md#stable-python-facade):

```python
from aisimulate_core.sdk import EngineHandle, estimate_kv_cache
from aisimulate.sdk.task_v2 import Task
```

Resource lookup must use `importlib.resources.files("aisimulate_core")` for
`model_configs/` and `systems/`. The old `aic-core/` source symlinks are removed.
Do not infer resource locations from the legacy CLI package.

The `aisimulate` package owns orchestration, configuration adapters, Replay,
Sweeper, the generator, legacy CLI, and `_runtime`. `aisimulate_core` owns the
estimator SDK, model/performance data, and `_native` facade. Application SDK
aliases delegate to canonical core modules to preserve registry, exception, and
cache identity. Core code does not import application orchestration.

`aisimulate_core.fpm_profile` and `aisimulate_core.quantization` own lightweight
metadata types; application exports refer to those same types. Importing
metadata does not load the native extension.

### SDK entry points

| Previous interface | Current interface | Required caller change |
| --- | --- | --- |
| Legacy `sweep_agg`, `sweep_disagg`, `sweep_afd` | `aisimulate.sweeper.Sweeper(...).run(config)` | All three remain callable as deprecated compatibility APIs. For new code, supply a runner factory and typed search configuration; workload/search semantics differ. See the [Sweeper guide](sweeper/overview.md). |
| Flat Rust `build_aic_engine` adapter | `aisimulate_core::perfmodel::AicEngineBuilder` | Construct through the builder; the flat adapter is removed. |
| Positional worker/options and separate estimator constructors | `RustForwardPassPerfModel.best_available(config)` / Rust `ForwardPassPerfModel::best_available(config)` | Pass one complete `ForwardPassPerfModelConfig`; use its Rust constructor or Python SDK config class. |
| Flat `EngineConfig` plus `ForwardPassPerfOptions` | Canonical `ForwardPassPerfModelConfig` | Convert saved legacy values with the explicit migration helper below. |

### Saved performance-model configuration

Use `ForwardPassPerfModelConfig.from_legacy_engine_config(old_config,
worker_type, old_options, allow_regression=False)` to convert a saved flat
EngineConfig and tuning options. It pins the old explicit native mode instead
of changing it to auto. Set `allow_regression=True` only for an old caller that
allowed direct regression fallback; the migration preserves that two-mode
order rather than adding interpolation. Legacy `forward_model: fpm` maps to
`fpm_interpolation`, and `fallback_policy: error` maps to deny. The deprecated
`regression` policy remains readable for these saved direct-fallback requests.
Legacy `extra.fpm_profile` and `extra.fpm_interpolation` migrate to the full
canonical profile and nested interpolation method; newly exported configuration
uses only the canonical fields.

Migration merges saved estimator controls with explicitly supplied legacy
options before applying ordinary defaults. It preserves disjoint settings and
accepts agreeing overlaps; contradictory explicit values report the canonical
setting's path. The same rule applies to legacy and canonical interpolation
methods, including an explicit `auto`. An omitted field does not override a
saved value. `ForwardPassPerfOptions.to_dict()` serializes only arguments
explicitly supplied to that legacy options object, including explicit defaults.

The migration adapter rejects any non-null `prefill_graph_profile`, `prefill_graph_profile_id`, or `decode_workload_distribution` field, including an orphan profile ID. These selectors require the canonical `ForwardPassPerfModelConfig.estimator_config.op_level` configuration; pass a saved canonical configuration directly to `RustForwardPassPerfModel.best_available` to preserve its profile identity and supported API restrictions. Profile-free legacy configurations continue to migrate normally.

Previously saved CLI timing with `forward_model` retains explicit selection
and deny. Newly authored requests without a selection use auto. `ForwardPassPerfOptions`
is retained as a legacy migration value type; new construction has one complete
config and no separate options argument. The raw PyO3 class also exposes
`normalize_config` and migration helpers for JSON-oriented consumers.

The [FPM regression design](../python/aisimulate/docs/fpm/aic-fpm-regression-design.md)
explains the retained workload routing and feature mathematics.

### Rust resource and memory literals

- `KvCacheEstimateRequest` struct literals must supply
  `cuda_graph_reserved_bytes: 0` when no additional reservation is needed.
  Exhaustive `MemoryBreakdown` literals/patterns must include the same field.
  Missing serialized values default to zero.
- `FpmResourceConfig` literals must wrap legacy byte values in `Some(...)` and
  supply `runtime_memory: None` when using declared memory. Runtime memory and
  the four legacy non-KV fields cannot coexist. See the
  [resource contract](core-api.md#fpm-profile-cache-groups-and-byte-budgets).

### Agentic report source migration

The replay report additions belong to the coordinated 0.13.0 wheel/crate
version and must not be backported to 0.12.x. Downstream exhaustive Rust `ReplayReport` literals must supply
`agentic_phases: None` for a cold run (or its prepared phase evidence).
Exhaustive `PerRequestRecord` literals must supply `agentic_phase: None` for
cold replay, or `Some(AgenticReplayPhase::Profile)` for measured warmed requests.
Exhaustive destructuring must name these fields or use `..`.

The external-consumer compile fixture `rebuild_replay_report_literals` constructs
both public structs exhaustively against this boundary. JSON consumers retain
the existing cold shape: absent optional phase evidence is not serialized.
This source migration does not change the engine-config/spec or FPM wire schemas.

### Offload replay API migration

Cluster-shared G2 ([G2 host-cache scope](g2-cache-scope.md)) extends the public
Rust types without changing JSON that omits the new fields:

- `engine::NativeHostOffloadConfig` adds `scope: G2Scope`,
  `shared_d2h_bandwidth_gbps`, `shared_h2d_bandwidth_gbps`,
  `latency_to_first_byte_ms` and `kv_layout_id: Option<String>`, and is no
  longer `Copy`; clone it where it was copied. It stays `#[non_exhaustive]`:
  build it with `new(..)`, `with_bandwidths(..)` and `cluster_shared(layout_id)`.
  Fields at their defaults are not serialized, so existing descriptors keep
  their shape. `G2Scope` is separate from `G3Scope`; G2 rejects `worker_local`.
- `engine::KvEvent` adds `tier: KvEventTier` (`Device` or `HostPinned`).
  Device events omit `tier` on output and a missing `tier` deserializes as
  `Device`; exhaustive literals must add `tier: KvEventTier::Device`.
  `HostPinned` `Stored` events set `start_position` to the prompt index of
  their first block.
- `ReplayReport` adds `g2_domains: Vec<replay::G2DomainStats>`; it is
  serialized only when nonempty. Exhaustive literals must supply it.
- `engine::G3Stats` adds `bypassed_restores: u64`, serialized only when
  nonzero. Exhaustive literals must supply it or use `..Default::default()`.
- A raw `cluster_shared` descriptor must set a
  `native_host_offload.kv_layout_id` that is not empty or whitespace-only;
  public YAML derives it. Engines built directly with `EngineFactory` cannot
  join a shared pool and fail at construction; use `ReplaySpec`.

### Release and downstream contracts

Wheel and crate versions must match. Namespace changes do not preserve removed
constructor signatures. Exact-wheel qualification of Dynamo Router, Planner,
Mocker, and deployment adapters remains separate from this documentation's
example checks; see the [release gates](../.github/release-gates.json).

Existing `aic_*` replay transport fields, native `AicEngine` names, and FPM
resource labels remain distinct downstream contracts. Do not rename them as
part of updating Python imports. See the [public API contract](core-api.md) for
wire-schema and compatibility requirements.
