<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Introduction
Data collection is a standalone process for collecting the database for aiconfigurator. By default, you don't have to collect the data by yourself.
Small versions of database will not introduce huge perf difference. Say, you can use 1.0.0rc3 data of trtllm on h200_sxm and deploy the generated
configs with Dynamo + trtllm 1.0.0rc4 worker.

If you want to go through the process, you can try belowing commands. However, you need to prepare the env by yourself such as installing a specific trtllm version.
This process is not well verified, you need to debug sometimes.

For a framework-version or GPU-platform upgrade, follow the
[Collector Upgrade Playbook](../../../docs/perf-model/collector/upgrade.md).
The repo-tracked [`aic-auto-collect`](../.claude/skills/aic-auto-collect/SKILL.md)
skill applies that workflow during long, resumable collection runs.
SGLang 0.5.14 Hopper/Blackwell follow-up work must also consult the
platform alignment ledger, a local campaign record kept outside the repo
(`docs/perf_database/sglang-0.5.14-hopper-blackwell-ledger.md` in the
campaign workspace).

# Dynamo 1.6 refresh candidate

The stock TensorRT-LLM collector runtime is pinned to `1.3.0rc29`, including
MoE and MSA. WideEP retains its independent runtime. This declaration enables
qualification; it does not promote queryable data versions or establish GPU
coverage. Collect and validate each target system before promoting its data.

The rc29 source (`c76f4a856447820aa1f994b053da7580d69b657d`) moves attention
submodules to `_torch.attention.backends` and MoE to `_torch.moe.fused_moe`.
The affected stock collectors use those paths and declare exact rc29
compatibility. The old attention package only re-exports selected public names;
it does not preserve imports such as `attention_backend.interface`. These
source fixes still require GPU qualification of inputs and kernel dispatch.
MoE also uses rc29's explicit activation payload: GPT-OSS preserves its
`1.702 / 1.0 / 7.0` SwiGLU constants, and Nemotron-3 preserves ReLU². The removed
activation keyword arguments must not silently fall back to ordinary SwiGLU.
Explicit MoE backend requests reject rc29 backend substitution, preserving a
failure record instead of labeling a replacement kernel as the requested one.
Full stock runs must select stock registry ops from the retained model plan:
that plan still names WideEP `moe_ep`, whose independent runtime cannot be mixed
with rc29 in one container.

DeepSeek-V4 attention uses rc29's `aux_stream_dict` constructor argument,
with separate attention, compressor, indexer, and auxiliary-indexer streams,
matching the native model. The removed single `aux_stream` argument fails
before measurement. Cache construction also calls the native model-default
selector: Hopper requires the packed `fp8_ds_mla` layout, while Blackwell
keeps the native default. Both layouts use FP8 quantization. Qualify all four
CSA/HCA context/decode paths after these changes.

GDN remains blocked: rc29's default recurrent-state cache is BF16, while
the existing TRT-LLM data reader assumes FP32 state-sensitive rows. Its native
packed-QKV convolution and recurrence APIs also differ. Resolve the producer
and consumer contract before publishing these rows. Likewise, current mHC
collection measures unfused pre/post mapping, while rc29 enables fused HC by
default. The existing rc23 exception does not cover rc29; an unfused smoke
result alone does not qualify rc29's default serving path.

vLLM `compute_scale` admits 0.31.0 after a source API audit against
[`db9527a4`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/_custom_ops.py#L1823).
It retains dynamic per-token quantization, static per-tensor quantization with
`group_shape=(-1, -1)`, and five calls per timed component. The two outputs
remain `max(0, dynamic - static)` and the static latency. This permits GPU
qualification; it does not approve historical timing reuse. Compare both
components when the difference is clipped to zero. Other newly intervening
release versions remain excluded pending their own API audit.


vLLM 0.31 recurrent qualification uses the same pinned source:

- GDN keeps the native prefill resolver and FP32 recurrent state. The native
  [packed decoder](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py#L454)
  now splits large grids; the collector no longer rejects them before launch.
  Context callbacks emit both convolution and chunk rows; generation callbacks
  emit convolution and packed-recurrence rows. Compare donor convolution rows
  separately from the newly measured recurrence implementations. The manifest
  contract test checks the audited 0.31.0 compatibility bound and retains the
  exclusions for unqualified intervening releases.
- The Kimi KDA benchmark callbacks support the target's native dtype-aware
  [resolvers and prefill buffers](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/models/kimi_k3/nvidia/kda.py#L217).
  FlashKDA output/state/workspace allocation stays outside timing. Default
  generation records only the native-selected path; a failed selected backend
  raises rather than substituting another kernel. Preview callback behavior is
  retained. The Kimi entry point admits B200 0.31 context, nonspeculative generation
  and fixed non-RecoverSSM verify with SD convolution state, BF16 inputs and
  FP32 recurrence.
  Target prefill reserves null cache slot 0 and assigns real requests slots
  1..batch, as required by the native convolution; preview slot allocation is
  unchanged. Callbacks fix gate bound −5 and omit RecoverSSM. Prefill does not
  export checkpoints. The native layout probe rejects DS; other hardware
  remains unaudited for this target route and raises explicit failures.
  Preview and GLM routes are unchanged.
  Existing labels retain their meaning: native decode includes convolution and
  gated RMSNorm, while prefill emits convolution and recurrence separately.
  Kimi structural keys (d_model 7168, heads 12/24/48/96) do not overlap current
  GLM keys (d_model 4096, heads 8/16/32/64); model name is metadata, so a second
  state/layout/speculation profile at the same key needs a separate contract.
  Default framework versions, case grids and consumer routing are unchanged.
  Admission permits qualification/collection; publication still requires complete
  numerical and timing validation of the selected cases. Three-way prefill
  convolution and speculative verify need their own numerical evidence.
  Verify uses the [native checkpoint and packed-input contract](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/models/kimi_k3/nvidia/kda.py#L954):
  query width S in 2/4/8 means num_spec=S−1, accepted=S, positive checkpoint
  IDs 1..B*S with null 0 reserved, and conv extent S+2. Native dtype/shape
  calculators plus `MambaBase.bind_kv_cache` create unpadded per-layer shared
  pages; packed Q/K/V views consume actual conv output, with shared recurrence
  output and native projection/beta strides. This is pure uniform verify,
  not mixed scheduling or a full-model page-padding claim. The four existing
  H96/48/24/12 callbacks retain 108 shape points and 216 conv/recurrence rows.
  Separate native timer callbacks, row keys and point failures are preserved.
  CPU tests execute both callbacks and check IDs, strides, view sharing,
  preview preservation and selected-backend failure propagation. This repair
  enables qualification; fresh auto-collector smoke/full evidence is still
  required before publication. Missing historical allocator provenance
  prevents reuse approval. No forced fallback profile is added: the approved
  B200 auto/nonspec decode cases all select native fused decode.
  Prefill releases the completed convolution input and its views before
  allocating independent core inputs, and releases native output/state/workspace
  buffers after every point, including failures. This removes setup overlap;
  it does not reduce the [native workspace](https://github.com/vllm-project/FlashKDA/blob/17a037d98da546deb4591e967cf961a43c034d8b/csrc/flash_kda.cpp#L11)
  or change callback grids, inputs, dispatch, or timing. Oversized points still
  run and retain explicit failure evidence; partial convolution rows do not
  make a failed callback complete. CPU lifetime tests cover both prefill paths
  and continuation after a core failure. GPU capacity remains a runtime check.
- DeepSeek-V4 mHC remains gated on 0.31: its pre rows do not model the fused
  default serving path, and shared mHC identities omit architecture. A standalone
  post comparison may use an isolated research artifact, but cannot establish
  full mHC coverage or permit mixed DeepSeek/GLM publication.

CPU tests cover resolver arguments, target native decode output/state contracts,
selected-extension failures, and preservation of the publication gates. They do
not establish numerical correctness, GPU launch validity, or timing reuse.

vLLM0.31 MiniMax-M3 MSA constructs both main and indexer caches through the
native layout resolver and byte-stride view builder. Public common metadata
retains absolute positions, prefix lengths, and the current token in decode
sequence lengths. Both layers use native cache-binding hooks; the indexer
removes its singleton head axis. The model's own backend/implementation
selector remains authoritative. On B200, default MSA decode is Triton;
CUTLASS decode is an opt-in configuration and is not qualified by default runs.
See [target cache binding](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/models/minimax_m3/common/indexer.py#L141)
and [default decode selection](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/attention.py#L43).
These adapters enable GPU qualification; older MSA timing data remains subject
to the refresh audit.

MSA dummy-weight initialization preserves the native attention Q/K/V/probability
scale buffers and their host values. These are runtime attention scales, not
projection weight scales. The target fused FP8 cache writer uses identity scale;
changing device scales alone would make Triton decode rescale stored K/V while
MSA prefill reads the same bytes at identity scale. See
[native scale initialization](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/attention/attention.py#L133)
and [fused cache conversion](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu#L215).
Projection weight-scale initialization and native backend selection are unchanged.

vLLM 0.31 MiniMax-M3 MoE uses the native packed `swigluoai_uninterleave`
activation with checkpoint clamp/alpha/beta (defaults 7, 1.702, 1). GPT-OSS
keeps its interleaved activation. The NVFP4 checkpoint uses
`ModelOptMixedPrecisionConfig` with an actual routed-expert prefix, so native
resolution constructs `ModelOptNvFp4FusedMoE`; a dummy prefix would miss the
per-layer mapping. The native selector still chooses the kernel. See
[MiniMax serving arguments](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/models/minimax_m3/nvidia/model.py#L253),
[activation defaults](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/transformers_utils/configs/minimax_m3.py#L36),
and [mixed-precision resolution](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/quantization/modelopt.py#L1789).
This changes no case arguments, precision keys, token sweep or output columns.
Earlier MiniMax timings using the interleaved/no-clamp contract are not serving
measurements and need fresh qualification. Older vLLM constructor behavior is
unchanged; the new path still requires native numerical and smoke validation.

# Preparation

The vLLM 0.31.0 MoE collector probes the callable `FusedMoEFactory` for native
SiTU parameters and forwards both model beta values. Factories without these
parameters retain the existing, explicitly labeled `_situ_as_silu` Marlin
approximation. Native SiTU measurements require fresh qualification; the old
approximation is not evidence of native activation or backend equivalence.
The stock compressed-tensors CUTLASS MXFP4 path quantizes activations to MXFP4
(W4A4). It now fails before timing when requested under the `w4a16_mxfp4`
label. B200 Kimi-K3 native SiTU needs a separate approved precision contract;
do not force Marlin or publish W4A4 timings as W4A16 to bypass this failure.
For vLLM 0.31.0, the collector also checks both native activation quantization
descriptors after weight loading. OCP GPT-OSS can select MXFP8 activations on
B200 despite a declared W4A16 case; that selection now raises before timing.
Explicitly unquantized descriptors remain valid. Unknown descriptors raise an
unverified-precision error. Each queued geometry is checked independently;
this is not a hardware exclusion, backend override or new data identity.

The vLLM 0.31.0 collector also rejects an actual selected
`TrtLlmBf16ExpertsMonolithic` leaf when the model requires
`apply_router_weight_on_input=True`. Native Llama4 BF16 diagnostics on B200
showed output weighting instead of the declared input weighting. The guard
runs after per-token native leaf selection and raises before timing; queued
cases remain explicit failures. Other leaves and output-weighted models retain
their paths. It does not force a fallback or change the model contract.
The source evidence is vLLM
[`db9527a` BF16 apply](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py#L257)
(drops the input-weight flag) and FlashInfer
[`946200de` BF16 launcher](https://github.com/flashinfer-ai/flashinfer/blob/946200de1ae94fc93fdd0926f0a13afd1fa7f0f1/csrc/trtllm_fused_moe_kernel_launcher.cu#L5304)
(passes false). A native repair and numerical requalification are required
before retiring this version-scoped guard; unchanged class names alone are
not evidence that an upgraded implementation still has the defect.
CPU regression checks include the strict import-surface tests and the shared
getter/GLM case-plan tests. Their fake vLLM tree includes the BF16 expert module
import without running or replacing its GPU implementation.

SGLang 0.5.14 standalone MLA rows distinguish kernel compute from KV storage.
On B200/B300, `trtllm_mla` quantizes Q/K/V internally when the KV cache is FP8,
so those rows use `mla_dtype=fp8, kv_cache_dtype=fp8`. On H200, the absorbed
576-dimensional FA3 path uses BF16 compute, including when KV storage is FP8.
BF16 inputs to the collector do not determine the kernel's compute precision.
Compute-precision labels are restricted to the audited `trtllm_mla`, `fa3`,
and `triton` backends; an unknown backend raises instead of receiving a BF16 label.
Legacy Blackwell rows labeled `bfloat16/fp8` need fresh measurements with the
correct label; do not copy their timings into a second precision key.

Before collecting the data, make sure you own the whole node and no interfierence happens.
Next, please enable persistent-mode and lock frequency of the node. Make sure the cooling system of the node is working well.
```bash
sudo nvidia-smi -pm 1
```
```bash
sudo nvidia-smi -ac yyy,xxx
```
xxx, yyy frequency can be queried by nvidia-smi -q -i 0, refer to the Max Clocks part, xxx is SM frequency, yyy is Memory frequency.
A script to set frequency:
```
#!/bin/bash

# Run nvidia-smi query and extract SM and Memory frequencies from Max Clocks
sm_freq=$(nvidia-smi -q -i 0 | grep -A 4 "Max Clocks" | grep "SM " | grep -o "[0-9]\+ MHz" | grep -o "[0-9]\+")
mem_freq=$(nvidia-smi -q -i 0 | grep -A 4 "Max Clocks" | grep "Memory " | grep -o "[0-9]\+ MHz" | grep -o "[0-9]\+")

# Check if frequencies were successfully extracted
if [ -z "$sm_freq" ] || [ -z "$mem_freq" ]; then
    echo "Error: Could not extract SM or Memory frequency from Max Clocks."
    exit 1
fi

# Generate the command
echo "sudo nvidia-smi -ac $mem_freq,$sm_freq"
```
Prepare a clean env with the target framework and nccl lib installed.

# Collect comm data
```bash
export PATH=$PATH:${NCCL_TEST_BIN_PATH}/
network/collect_comm.sh #all_reduce data will be collected using default trtllm backend
network/collect_comm.sh --all_reduce_backend vllm #all_reduce data will be collected using vllm backend
network/collect_comm.sh --all_reduce_backend vllm --device xpu #all_reduce data will be collected using vllm backend on XPU
```
The communication driver stops on the first failed subprocess and checks all four
NCCL binaries before beginning. Previously a failed child could be followed by a
misleading completion message. NCCL subprocess stdout/stderr, command and exit
status are retained under `communication_evidence/`, including failures. Its
performance sweep still uses the historical `-c 0` timing contract; run separate
`nccl-tests -c 1` correctness probes before accepting the sweep.

For CUDA vLLM, `<perf-filename>.rank<N>.evidence.jsonl` records the Python
communicator calls that actually return outputs during graph capture or an
untimed same-shape eager probe. These distinguish native FlashInfer, custom AR,
PyNccl and symmetric-memory paths without changing perf keys or forcing a backend.
They identify the Python dispatch path, not a device-kernel binary or tactic.
The captured graph outputs and eager probe must reduce the benchmark's all-ones
input to `world_size` on every rank. Checks and tracing are outside timing; graph
replay counts and eager clone costs are unchanged. Preserve these sidecars with
library/image provenance, and validate a rank-distinct smoke input separately
when checking rank wiring. A missing dispatch observation or wrong result stops
publication. Existing generic `kernel_source` labels remain for compatibility.

Today we only collect intra-node comm. This script will collect custom allreduce data for trtllm within a node.
It will also collect nccl allreduce, all_gather, all2all, reduce_scatter using nccl.
The standalone scripts stage `nccl_perf.txt`, `oneccl_perf.txt`, and
`custom_allreduce_perf.txt`. Collector finalization converts accepted output
to parquet under
`python/aisimulate/src/aisimulate_core/systems/data/<system>/comm/<backend>/<version>/`.

# Model-centric cases and healing runs

The model-centric case layer (introduced as "collector v2") is how collection
coverage is declared, and it is unchanged under Collector V3 — V3 adds
op-centric *management* on top (per-family runtime pins, provenance sidecars,
declared reuse; see `docs/perf_database/collector-v3-op-centric-design.md`),
it does not replace how cases are declared or healed. A healing run collects
cases for a specific model/GPU pair instead of running every op bucket and
hoping the support matrix improves. Use `--model-cases-full` when you want
the case YAML to define a full model-centric run. Omitting all model-case
flags runs the backend registry directly without a model-centric case plan.

```bash
# Heal one model on one GPU type.
python3 collect.py --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --gpu b200_sxm

# Inspect the resolved model plan without collecting.
python3 collect.py --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --gpu b200_sxm \
  --plan-only

# Select the architecture case file directly.
python3 collect.py --backend trtllm \
  --model-architecture Qwen3MoeForCausalLM \
  --gpu b200_sxm \
  --plan-only

# Full model-centric run: aggregate base ops plus every model case YAML file.
python3 collect.py --backend trtllm --model-cases-full

# Raw backend registry run: no model-centric case plan.
python3 collect.py --backend trtllm

# Ephemeral subset filter (healing): substring match, repeatable.
python3 collect.py --backend trtllm --ops moe --case-filter "tp=4"

# Predeclared campaign exclusions, applied after inclusion and before sampling.
python3 collect.py --backend vllm --ops gemm --case-filter "bfloat16" \
  --case-exclude-filter ", 99, 32]"
```

Case files (see `cases/README.md` for the full authoring guide):

```text
cases/base_ops/<op>.yaml               — shared sweep recipes (interpolation grid)
cases/models/<architecture>_cases.yaml — op activation + model shape values
cases/capabilities.yaml                — positive hardware floors (dtype/op -> min SM)
cases/denylist.yaml                    — hang/node-killer cases only
model_cases.py                         — resolves which ops a model plan collects
capabilities.py                        — generation-time capability/denylist filter
```

The plan is one equation: cases = dedup(base grid ∪ model shapes), then
intersected with hardware capability floors and minus the hang denylist.

TRT-LLM MLA precision and GPT-OSS TP profile collection evidence is recorded
in [the September 17 campaign](trtllm/gym-profiles-20260917.md).

## Whole-forward FPM campaign

Use the dedicated `python3 -m collector.fpm_forward` entry point for vLLM
whole-forward campaigns. The compatibility route
`python3 collector/collect.py --ops fpm_forward` remains supported and must be
selected alone. Both routes require a resolved model, `--gpu`, and
`--fpm-max-gpus`. The campaign derives a latency-blind prefill/decode
design from the model's AIC attention cases, asks Generator for three rendered
artifacts per cell — a keepalive Pod, LeaderWorkerSet, or Grove PodCliqueSet
manifest plus `fpm_env.sh` and a thin engine-launch `run.sh` — stages them with
its own in-pod runtime `fpm_exec.sh` (etcd lifecycle, result gate and checker,
completion barrier, exit-code policy), and publishes the formal table only
after every frozen cell passes.
For multinode runs, select the installed controller with
`--fpm-orchestrator lws|grove`. Start with `--plan-only` or a one-cell
`--smoke --limit 1` run.

```bash
python3 -m collector.fpm_forward \
  --model-path nvidia/GLM-5.2-NVFP4 \
  --gpu b200_sxm \
  --fpm-max-gpus 8 \
  --plan-only
```

For profile-based FPM collection, the profile's `backend_version` is the dataset
identity label. It may be a custom string such as `my-vllm-patch-3` and need not
match the installed vLLM package version. The collector retains the actual
container version separately in `collector-provenance.json` under
`runtime.backend_version`; runtime capability, benchmark-schema, topology and
precision checks still apply. An optional Dynamo version is provenance only and
does not select FPM templates or resolve the backend version. See the
[onboarding version contract](../../../docs/perf-model/fpm-self-service/implementation.md#plan-preview-and-explicitly-execute)
for guided setup, omitted-version detection inside the target image, and
publication identity.

The low-level entry point also accepts `--fpm-backend-version LABEL` for dataset
identity and `--fpm-runtime-backend-version VERSION` for an independently observed
package-version pin. Guided onboarding supplies that pin after its image
preflight. The low-level collector does not launch this separate CPU preflight:
without an observed pin it can collect timings and record actual package
provenance, but it cannot select audited version-specific memory/execution hooks.
Supply the observed pin or accepted runtime instrumentation when those hooks are
required. The op-level collector's runtime manifest and version-routing rules
below are unchanged.

Generation-time admission may omit a topology only when a concrete AIC
size-vs-capacity estimate proves its configured token envelope cannot fit.
Missing performance data, bootstrap architectures, and structural estimator
errors remain runnable so the target runtime supplies the observation.

Model identity is resolved from a local checkpoint/config, the packaged AIC
config cache, or Hugging Face. A real model architecture that is not registered
by AIC uses a config-derived bootstrap template; an unresolved or empty config
fails before planning. When the checkpoint exists only inside the runtime Pod,
pass its real local metadata explicitly with
`--fpm-model-config /path/to/config.json` (a directory containing `config.json`
is also accepted). The resolved config snapshot and SHA-256 are frozen into the
collection plan.

Generator and Collector share the Dynamo-native benchmark result schema
version through the rendered `fpm_env.sh`; the staged `fpm_exec.sh` validates
that contract in-pod without a Collector-side text patch. Formal
`fpm_forward_perf.parquet` output uses metadata schema v6 and records
`collector_attempt_id`, `runtime_run_id`, `runtime_grid_digest`, and
`kv_seed_regime` on every row. The seed regime is `real_kv`, `fake_fallback`,
an approved `skip:<reason>`, `legacy`, or `n/a`; non-benign warm-up skips
remain `fake_fallback` rather than acceptable evidence. Republishing the same
attempt is idempotent. If a cell already exists under a different attempt or
runtime identity, first publisher wins: that entire cell is skipped while
non-overlapping cells in the same publication still land.

The default publication gate requires every frozen cell to pass.
`--fpm-publish-partial` explicitly publishes passed cells, records the exact
`missing_cells` in the checkpoint, and still returns a nonzero incomplete-run
result. The database checkpoint also records
`skipped_first_publisher_wins`. A validated completed schema-v6 database is
terminal on resume even if raw cell artifacts have since been pruned.
`run-manifest.json` schema v2 is invocation-scoped: its `attempts` include the
attempt ID, timestamps, final status, and partial phase timings for failed
attempts; a no-op resume writes no historical attempts.

## Failure philosophy: observe, don't predict

There is no declarative expected-failure layer. A case that cannot run on the
current framework version fails fast and is recorded in `errors_<module>.json`
and `collection_summary_<backend>.json` with a classification and a
(model, dtype) group label. The summary aggregates failures by group: a whole
group failing is a fix-me signal (collector bug, unverified combo, framework
gap), not something to tolerate or auto-skip. Missing points are tolerated
downstream by interpolation and fallback. Escalation rules — when a failure
deserves a denylist entry, a registry `unverified` marker, or a capability
floor — live in `.claude/rules/collector/failure_handling.md`.

`--gpu <type>` resolves the SM version from
`src/aisimulate_core/systems/<gpu>.yaml`; use `--sm <version>` on an
unregistered GPU. Without either, the local device capability is detected
automatically.

To add a new architecture, create one `cases/models/<architecture>_cases.yaml`
file. To add a new model in an existing architecture, add the model path to
that architecture's `model_paths` list. Add shared op sweeps to the matching
`cases/base_ops/<op>.yaml` file. Add a new op collector only when the existing
ops cannot generate the needed data points.

# Version Management

## Overview

Each backend (trtllm, vllm, sglang) has a **registry** (`registry.py`) that maps ops to collector modules, and a **version resolver** (`version_resolver.py`) that picks the right module at runtime. Individual collector files declare their compatibility via `__compat__`. The current collector framework versions and runtime images are declared in `framework_manifest.yaml`.

```text
framework_manifest.yaml — current collector framework versions and images (schema v2)
framework_manifest.py   — manifest loader/validator/resolver
op_catalog.py         — table→family identity map (op_backend_catalog.yaml)
model_cases.py       — model-centric model/SM case-plan resolver
registry.py          — declares which module handles which version range
version_resolver.py  — routes runtime version → module (packaging.version)
collect.py/collect_ops — validates __compat__ and fails incompatible ops
__compat__           — per-file metadata declaring supported framework versions
cases/               — model-centric case manifests and SM exceptions
wideep/              — WideEP collector namespace for special images/runtimes
wideep/*/registry.py — WideEP-only ops appended when the v2 plan requests them
network/             — collective communication collectors and Slurm network jobs
```

`framework_manifest.yaml` is `schema_version: 2`: each framework entry has a
`default` runtime (`version` + `images`) plus an optional `families:` map of
per-family overrides. Resolution for a given `(framework, op)` walks
`table → catalog family → families[family] or default` — exactly one runtime
per op, always. `families:` overrides require the op catalog
(`op_backend_catalog.yaml`); without it, family-scoped pins fail closed rather
than silently falling back to `default`.

WideEP runtimes (`wideep_sglang`, `wideep_trtllm`) are flattened into peer
framework entries rather than nested under a `wideep:` key, so the same
per-op resolution logic applies uniformly. Each WideEP entry adds
`base_framework` (the stock framework it forks from), `data_backend` (which
table namespace its output is written under), and `collector_dir`.
`collect.py` requires the exact public version for a run and rejects
mixed-pin selections across ops.

Image digests are policy, not decoration: the rule is syntactic — any image
reference containing `/` must be pinned with `@sha256:<64 hex>`; only bare
internal image names (no `/`, e.g. `deepseek-v4-blackwell`) are exempt. In
practice that means every public-registry reference is digest-pinned.
When bumping a version pin, re-fetch and update the digest for every image
variant touched — a stale digest silently keeps pulling the old image. Always
pin the manifest-list (index) digest, never a platform-child digest —
`docker buildx imagetools inspect <image>` prints the index digest; a child
digest pins one architecture and breaks the other platforms.

Use `collector.framework_manifest.resolve_op_runtime(framework, op)` to
resolve a single op's runtime, and
`collector.framework_manifest.validate_resolution()` to check the whole
manifest against every backend registry at once — it returns a list of error
strings (empty = valid) and is the fail-closed CI gate asserting every
registered op resolves to exactly one pinned runtime. See
`docs/perf_database/collector-v3-op-centric-design.md` §4 for the full
design.

## B200 vLLM 0.31 qualification campaign source

The composed qualification source retains the case declarations and generator
from `ff8b7a30d75d09e4db07fe892149ef9762e69fb1`. Its MLA/DSA, DSV4 and MSA
adapters change native runtime metadata/cache handling; they preserve the
ordered case plans. It also includes communication dispatch/failure evidence
and the version-route-aware CI change detector.

Keep each run's immutable source SHA, runtime manifest hash and explicit campaign
exclusions together. Existing native SiTU handling and include/exclude filters
remain part of this source. Pin a new source revision only for a new qualification
run; an in-flight run retains its original source and case identity. Each newly
adapted family still requires its own numerical and runtime gates before full
collection or publication.

## vLLM 0.31.0 MLA BMM qualification

`--ops mla_bmm_gen_pre` selects the 0.31.0 route when an external, hash-bound
runtime manifest declares that version. The repository's default runtime pins
remain unchanged. The B200 qualification manifest is
`collector/campaigns/vllm031-b200-runtime.json` (relative to `python/aisimulate`).
Supply its absolute in-container path through `AISIM_COLLECTOR_RUNTIME_MANIFEST`
and its file SHA-256 through `AISIM_COLLECTOR_RUNTIME_MANIFEST_SHA256`.
The campaign must separately select `b200_sxm` and `mla_bmm_gen_pre`; the manifest
declares runtime identity and does not grant compatibility to other ops or GPUs.
The new route measures BF16 `torch.bmm` with token-major
output storage, matching the unpadded serving branch at
[vLLM db9527a4](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/attention/mla_attention.py#L1080).
Older qualified versions retain head-major pre-BMM output. The 0.31.0
`--ops mla_bmm_gen_post` route shares the existing post-BMM implementation:
its token-major output view and timed operation are unchanged. This enables
the 48 B200 reuse probes; it does not approve performance reuse before their
results are reviewed. The padded `CUTLASS_MLA` branch is outside this dataset's scope.

The full pre-BMM sweep contains 300 cases: 25 token counts × 12 head counts,
BF16 only, with two warmups and ten measured runs under CUDA graph replay.
Explicit `--ops` uses this complete base sweep without `--model-cases-full`.
For the B200 refresh, first smoke-test the pinned target image, then remove the
sample limit for the full sweep. CPU tests qualify layout and routing, not GPU
performance. Retain failures and provenance; do not reuse checkpoints from the
old output layout or claim post-BMM coverage from a pre-only table.

### B200 native dense attention qualification

`attention_context` and `attention_generation` select the exact 0.31.0 adapter.
It accepts native `FLASHINFER`, `FLASH_ATTN` and `TRITON_ATTN` selection, with
no forced backend or fallback. FlashInfer's own metadata builder chooses its
native or TRTLLM prefill/decode path; FlashAttention chooses its own FA version.
Other selected backends still fail explicitly. Case grids and old routes stay
unchanged.

The adapter delegates layout negotiation and packed cache views to vLLM and
populates the current
[serving metadata fields](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/worker/gpu_model_runner.py#L2426).
The existing single-layer parameters match
[the native FlashInfer builder](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/flashinfer.py#L882).
Query conversion follows the selected implementation's quantized-query capability;
metadata construction and history population remain outside timing. The measured
boundary remains the production KV update plus forward, with unchanged warmup,
repeats, CUDA graph timing and perf keys. `kernel_source` records the actual
FlashInfer phase-metadata class or selected FlashAttention version; retain native
FlashInfer logs identifying `trtllm-gen`, `xqa` or `flashinfer-native` as well.

This enables qualification; it does not establish correctness or full coverage.
Before a full sweep, use exact case filters to smoke every planned head-size,
KV-dtype and window group, check the native dispatch/cache layout and validate
outputs outside timing. Include context length one: FlashInfer classifies that
input as decode even though the collector writes a context row. Keep every
failure visible. A successful Triton or encoder test does not qualify dense
FlashInfer/FA4. Existing 16 Triton smoke cases are fresh-path measurements because
[vLLM 0.31 changed the packed K/V cache contract](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/triton_attn.py#L615).

`encoder_attention` also has an exact 0.31.0 route for the existing BF16 grid.
It retains the native ViT selector, selected FlashAttention version, and wrapper
calls, whose signatures match the target source. The target wrapper adds
[output initialization during CUDA graph capture](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/ops/vit_attn_wrappers.py#L64);
collect fresh measurements after its smoke gate. FP8 encoder attention is
outside this grid. A backend unsupported by the collector remains an explicit
failure instead of being replaced.

### B200 MLA and DSA module qualification

The four `mla_*_module` and `dsa_*_module` operations select an exact 0.31.0
adapter through the campaign manifest. Older runtimes keep their legacy routes.
The adapter fills the current public common metadata, delegates cache layout and
views to vLLM, calls each production layer's `bind_kv_cache`, and populates cached
MLA history through the native `update_kv_cache` method. The indexer retains its
native quantized cache producer. See the target
[MLA cache contract](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/attention/mla_attention.py#L733)
and [indexer binding](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/models/deepseek_v2.py#L647).
Metadata builders use vLLM's serving `AttentionGroup` factory, including the
DSA indexer's required, aligned `block_table_width`. This avoids reconstructing
backend-specific constructor arguments in the collector.

Model construction, native backend selection, precision checks, case identities,
and the complete module-forward CUDA graph timing remain unchanged. The existing
explicit FP8-prefill configuration is retained. HiSparse/offloaded cache state
raises an explicit error because this harness allocates a single-device cache.
CPU tests cover cache binding, history slots, worker forwarding, legacy routing,
and provenance. Run exact small cases for each model and precision before a
full sweep; a source-qualified adapter does not establish runtime support for
all declared cases. Failures remain visible and must not become broad skips.

NVFP4 module inputs retain ModelOpt's native W4A4 checkpoint contract. vLLM
0.31 stores it in `ModelOptLinearMethod` with the exact static-NVFP4 weight and
dynamic-NVFP4 activation keys; older runtimes keep `ModelOptNvFp4LinearMethod`.
The initializer reads the native 16-element group size, fills the existing
packed weights, block scales and dequantizing global scales, then lets vLLM
perform its own post-load layout conversion and kernel selection. W4A16 and
other unified ModelOpt formats are not mistaken for W4A4. See the
[native format resolution](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/quantization/modelopt.py#L2621).
This compatibility repair does not qualify ModelOpt numerical output from
CompressedTensors GEMM evidence; module-specific native numerical checks remain
required. Case grids, output keys and timing are unchanged.

### B200 DSV4 qualification

The six `dsv4_*` operations have an exact 0.31.0 route. It retains the existing
case grids, native attention selector, TP shard simulation, and timing calls.
Full CSA/HCA and isolated HCA use vLLM's native cache groups, shared allocation,
layer binding, kernel block sizes, and metadata builders. Each group receives
disjoint physical page IDs in the shared buffer; layers packed within a group
use the same page table. This avoids overwriting attention KV with compressor
state. See the native [allocator](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/worker/utils.py#L395)
and [group placement](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/kv_cache_utils.py#L1669).

Isolated FlashMLA HCA preserves the full model's CSA indexer layout constraint
([native declaration](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/mla/indexer.py#L254)).
Without that constraint, the isolated layer selects layer-outer `LBNHC`, and
native page unification rejects its unequal MLA pages before execution.
The adapter includes the native indexer's layout declaration when resolving
HCA's layout; it does not instantiate an indexer or change attention dispatch.
Explicit incompatible layouts still fail. CPU tests check HCA/CSA preference
and rejection; numerical GPU qualification remains required after this fix.

Standalone paged MQA retains its FP8 packed-cache inputs and native DeepGEMM
wrapper call, whose target signature remains compatible. All six operations
require fresh target measurements and a GPU smoke gate. CPU checks verify page
and slot mappings, shared-allocation delegation, capacity restoration on error,
version routing, and unchanged case/measurement fields. Use fresh checkpoint
identities after this adapter change; no performance reuse is implied.

## File Naming Convention

- **No version fork**: `collect_{op}.py` (no suffix)
- **Version forks**: `collect_{op}_v1.py`, `collect_{op}_v2.py`, ... (sequential)
- The number is just an ordering key; the actual version range lives in `__compat__`

## `__compat__` Format

Every collector file must declare `__compat__` after the license header:

```python
# SPDX-FileCopyrightText: ...
# SPDX-License-Identifier: Apache-2.0

__compat__ = "trtllm>=1.1.0"
```

The format is `<framework><constraints>` using PEP 440-style operators:
- `"trtllm>=1.1.0"` — open-ended (latest version)
- `"trtllm>=0.21.0,<1.1.0"` — bounded range
- `"sglang>=0.5.5"` — sglang example

Pre-release ordering is respected: `1.1.0rc2 < 1.1.0 < 1.1.0.post1`.

## Registry Format

Each entry in `registry.py` is an `OpEntry` dataclass (defined in `collector/registry_types.py`).
Exactly one of `module` (unversioned) or `versions` (versioned) must be provided — this is
validated at construction time.

```python
from collector.registry_types import OpEntry, VersionRoute

# Unversioned (no fork):
OpEntry(op="gemm", module="collector.trtllm.collect_gemm", get_func="...", run_func="...")

# Versioned (has forks) — VersionRoutes in descending min_version order:
OpEntry(op="myop", get_func="...", run_func="...", versions=(
    VersionRoute("X.Y.Z", "collector.<backend>.collect_myop_v2"),
    VersionRoute("0.0.0", "collector.<backend>.collect_myop_v1"),
))
```

The resolver picks the first `VersionRoute` where `min_version <= runtime_version`.

## Adding a New Op

1. Create `collector/<backend>/collect_myop.py`
2. Add `__compat__ = "<backend>>=X.Y.Z"` after the license header
3. Export `get_myop_test_cases()` and `run_myop()` (or `run_myop_torch()`)
4. Add an entry to `collector/<backend>/registry.py`:
   ```python
   OpEntry(op="myop", module="collector.<backend>.collect_myop", get_func="get_myop_test_cases", run_func="run_myop")
   ```
5. Run `pytest tests/unit/collector/ -m unit` to verify registry integrity

## Handling a Framework API Change

When upstream framework `X.Y.Z` changes an API that a collector depends on:

1. Rename the original file to `collect_{op}_v1.py` (if not already versioned)
2. Create `collect_{op}_v2.py` with the new API calls
3. Add `__compat__ = "<backend>>=X.Y.Z"` to the new file
4. Add `__compat__` upper bound to the old file if it doesn't have one (e.g. change `">=1.1.0"` to `">=1.1.0,<X.Y.Z"`)
5. Convert the registry entry from unversioned to versioned (or prepend a new `VersionRoute`):
   ```python
   # Before (unversioned):
   OpEntry(op="gemm", module="collector.trtllm.collect_gemm", ...)

   # After (versioned — all forks carry explicit _vN suffix):
   OpEntry(op="gemm", versions=(
       VersionRoute("X.Y.Z", "collector.<backend>.collect_gemm_v2"),
       VersionRoute("0.0.0", "collector.<backend>.collect_gemm_v1"),
   ), ...)
   ```
6. Run tests to validate

## Runtime Behavior

`compute_scale` produces both `computescale_perf` and its static-quantization
companion `scale_matrix_perf`. Finalization and recovery bind both tables to
the same registered producer and checkpoint. Both staging files are required
before either is published; the attempted ledger closes only after both tables
and their provenance commit.

TRT-LLM quantized MoE construction uses the BF16 model dtype and selects weight
quantization through `QuantConfig`, matching rc29 serving. Passing FP8 as the
model dtype is rejected by rc29's CUTLASS implementation resolver.

The TRT-LLM all-to-all Slurm launcher uses the same rc29 image digest as the
stock runtime manifest. Explicit image overrides must match that runtime's
package-version gate; the launcher default is covered by the network tests.

At collection time:
1. `collect.py` reads the backend registry
2. `build_collections()` resolves each op to the correct module for the detected framework version
3. If the resolved module declares `__compat__`, it is validated against the runtime version — mismatches fail that op explicitly and are recorded in the error summary
4. Unsupported versions (no matching entry) are skipped with a warning

## Tests

```bash
pytest tests/unit/collector/ -m unit
```

Covers: version parsing (PEP 440 with rc/post), `__compat__` constraint evaluation, module routing, and structural validation of all three registries.

# Collect gemm/attention/moe data/etc.

## Smoke Test

Use `--smoke` to quickly verify the collector runs end-to-end. It randomly samples 4 test cases per op instead of running the full suite.

```bash
# Smoke test for sglang
python3 collect.py --backend sglang --smoke

# Smoke a specific op
python3 collect.py --backend trtllm --ops moe --smoke
```

## Power Monitoring (Optional)

The collector supports GPU power monitoring during kernel execution using NVML. This feature is optional and disabled by default.

### Enable Power Monitoring
```bash
# Basic power monitoring
python3 collect.py --backend trtllm --measure_power

# With custom minimum duration (default: 1.0s)
python3 collect.py --backend trtllm --measure_power --power_test_duration_sec 2.0
```

### Options
- `--measure_power`: Enable NVML-based power monitoring (samples at 100ms intervals)
- `--power_test_duration_sec`: Minimum test duration for accurate power readings (default: 1.0s)

### Output
When power monitoring is enabled, performance CSV files will include additional columns:
- `power`: Average power consumption during kernel execution (Watts)
- `power_limit`: GPU power management limit (Watts)

**Example output:**
```csv
framework,version,device,op_name,kernel_source,gemm_dtype,m,n,k,latency,power,power_limit
TRTLLM,1.2.0,NVIDIA H200 SXM,gemm,torch_flow,bfloat16,1024,4096,4096,0.234,523.4,700.0
```

### Requirements
Power monitoring requires:
- `pynvml` Python package: `pip install pynvml`
- NVML support (NVIDIA drivers)

If unavailable, a warning is logged and execution continues without power data.

### Notes
- Power monitoring adds minimal overhead (<1%)
- Kernel iterations are automatically adjusted to meet minimum duration for accurate measurements
- Backward compatible: without `--measure_power`, CSVs remain unchanged

## CUDA Graph Fallback Support

The `benchmark_with_power` helper function now supports graceful fallback to eager execution when CUDA graph capture fails. This is particularly useful for complex operations like MOE (Mixture of Experts) with large batch sizes.

### Features
- **Automatic fallback**: When `allow_graph_fail=True`, CUDA graph capture failures trigger eager execution instead of raising exceptions
- **Power measurement in both paths**: Power monitoring works correctly in both graph replay and eager execution modes
- **Memory safety**: Automatic `torch.cuda.empty_cache()` call on graph capture failure to prevent memory fragmentation
- **Transparency**: Results include `used_cuda_graph` flag to indicate which execution path was used

### Usage Example
```python
from helper import benchmark_with_power

def my_kernel():
    # Your kernel code here
    moe.forward(hidden_states, logits)

# Use benchmark_with_power with fallback support
with benchmark_with_power(
    device=device,
    kernel_func=my_kernel,
    num_warmups=3,
    num_runs=6,
    repeat_n=1,
    allow_graph_fail=True,  # Enable graceful fallback
) as results:
    latency = results["latency_ms"]
    power_stats = results["power_stats"]  # Available in both paths

    # Check which execution path was used
    if not results["used_cuda_graph"]:
        print("CUDA graph capture failed, used eager execution")
```

### When to Use
- **Complex operations**: MOE, dynamic memory patterns, or operations that may not be graph-compatible
- **Large batch sizes**: When graph capture may fail due to memory constraints
- **Development/debugging**: To ensure collection continues even if graph capture fails

### Backward Compatibility
- Default behavior unchanged: `allow_graph_fail=False` maintains existing behavior
- Existing collectors work without modifications
- Only opt-in when needed for specific use cases

## For TensorRT-LLM
### Optional( Read if collecting for mxfp4 kernels)
To collect performance data for mxfp4 kernels used in GPTOSS models, depending on the version of TensorRT-LLM, you might need to manually install triton-kernels. If your version is **>= 1.3.0rc2**, nothing needs to be done and you can run the collection process directly. For any version before **1.3.0rc2**. Follow the below instructions to install triton-kernels and expose them to TensorRT-LLM
1. Build and install Triton (tested with the commit below):
```bash
git clone https://github.com/triton-lang/triton.git
cd triton
# Specific commit verified with TensorRT-LLM
git checkout f3067cd3bd0c29065fa4ecdb724b6f29cbabea5f
pip install -r python/requirements.txt # build-time dependencies
pip install wheel build
python3 setup.py bdist_wheel # if this step fails due cuda related error, you might need to set CUDA_HOME following the optional step
pip install ./dist/*.whl
```
2. (Optional) You may need to set CUDA_HOME env variable to successfully build triton
```bash
export CUDA_HOME=/usr/local/cuda
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
python3 setup.py bdist_wheel
```
3. Expose the Triton kernels to TensorRT-LLM The kernels are not packaged in the wheel, so set the environment variable TRITON_ROOT to your Triton clone:
```bash
export TRITON_ROOT=/local/user/triton
# TensorRT-LLM expects the kernels at:
#   $TRITON_ROOT/python/triton_kernels
```
### Run collection for all available operations
```bash
python3 collect.py --backend trtllm
```
For trtllm, the whole collecting process takes about 30 gpu-hours. On 8-gpu, it takes 3-4 hours.
Please note that the whole process will report a lot of missing datapoints with errors. But it's okay. Our system is kindof robust to fair amount of missing data.
Once everything is done, you might see mutliple xxx.txt files under the same folder. Refer to src/aisimulate_core/systems/ folder to prepare the database including
how many files are needed accordingly.

## Resume Collection (Checkpoint)

Checkpoint files are always written so an interrupted run can be resumed later:

```bash
# Normal run (writes checkpoint to .collector_checkpoint/)
python3 collect.py --backend trtllm

# Resume an interrupted run (skips already-attempted tasks)
python3 collect.py --backend trtllm --resume

# Custom checkpoint directory
python3 collect.py --backend trtllm --resume --checkpoint-dir /path/to/checkpoints
```

A task is marked **done** once it is attempted (success or failure).
Only tasks that never finished are re-queued on `--resume`;
`--resume-retry-failed` additionally re-runs previously failed tasks.
Running without `--resume` always starts fresh (overwrites old checkpoint).

A live snapshot can contain a measurement appended before its callback reached
the checkpoint. On resume, that callback can append another measurement. Default
Parquet finalization keeps the last row per existing measurement identity, even
when no Parquet exists yet, and attests the unique current-row count. It does not
choose the fastest timing or combine different kernel identities. Keep the raw
snapshot for audit; `delete_source=False` preserves staging when using the helper.
Explicit `merge_existing=False` retains its overwrite behavior.
Compatible merges preserve the validated Arrow field types, nullability, and
metadata through pandas conversion, including `string` versus `large_string`
identities. This prevents pandas 3 string inference from turning a later resume
into a schema-replacement write. Genuine incoming identity-schema mismatches
retain their existing warning and replacement behavior; metric null/numeric
reconciliation and latest-measurement selection are unchanged.

## For SGLang

Use the pinned stock image, `lmsysorg/sglang:v0.5.14` (or its `-cu130`
variant), for normal SGLang collection.

```bash
python3 collect.py --backend sglang
```

This collects the stock SGLang ops, including:

- GEMM operations (FP8, BF16, INT8, INT4)
- MLA (Multi-head Latent Attention) for context and generation
- MLA BMM (Batch Matrix Multiplication) operations
- MoE (Mixture of Experts) operations
- Normal attention operations

Large-EP local expert compute is available as the separately routed `moe_ep`
operator under `wideep/sglang/collect_deepep_moe.py`; run it explicitly in its
pinned SGLang 0.5.10 image. WideEP MLA is not registered because its legacy
wrapper now reaches the stock 0.5.14-only module implementation.

### DeepEP multi-node collector
For models with DeepEP MoE, inter-node communication data requires a separate
multi-node setup. It is collected by the standalone
`wideep/sglang/collect_moe_a2a.py` collector, launched one Slurm job per
world size by `network/slurm/submit_moe_a2a.sh` — see section 4 of
`network/slurm/README.md`. The older manual log-scraping pipeline under
`wideep/sglang/deepep/` is deprecated (its README carries the retirement
story); do not add new data through it.

# Supporting a new large-EP (WideEP) model

Large-EP MoE uses stock `moe_perf` for modeled local expert compute and
`moe_a2a_perf` for measured dispatch/combine communication. To support model X:

1. **Declare the shapes.** Add the model to
   `cases/models/<Architecture>_cases.yaml` (new architecture: one new file;
   new model in an existing architecture: append to `model_paths`) and mark
   its `model_case_values.moe` rows with `wideep: true`. Correlated MoE
   dimensions (hidden/inter size, topk, expert count) stay together in one row;
   never cross another model's values. The `wideep` flag activates standalone
   A2A shape population.

2. **Ensure stock `moe_perf` coverage.** Local compute is modeled as a
   uniformly balanced stream over EP-local experts, using TP=1 stock MoE
   coordinates. Cross-version stock-table reuse is allowed.

3. **Collect `moe_a2a` across nodes.** From `network/slurm/`:
   ```bash
   bash submit_moe_a2a.sh                      # sglang DeepEP HT+LL, one job per world size
   bash submit_trtllm_alltoall.sh              # trtllm NVLink alltoall (single/multi-GPU NVL)
   ```
   The comm tables have no model column — a new model whose
   (hidden_size, topk, num_experts) tuple is already covered needs no new
   runs; a new physical shape lands in the sweep automatically once declared
   in step 1.

4. **Publish with sidecars.** Finalized communication parquet goes into
   `.../<system>/comm/<backend>/<version>/moe_a2a_perf.parquet` together
   with its `collection_meta.yaml` entry — never a parquet without its
   provenance. The per-world `moe_a2a` outputs need the cross-job merge
   procedure in `network/slurm/README.md` section 4.3.

## MoE table units and caveats

`moe_a2a_perf` records latency in microseconds; `load_moe_a2a_data` converts
leaves to milliseconds. Stock `moe_perf` retains its existing timing contract.

Stock `moe_perf` also supports optional Boolean `default_eligible` selection metadata. The [Core API contract](../../../docs/perf-model/configuration.md#choosing-a-forward-pass-api) defines automatic and exact-source selection. Finalization validates non-null Boolean flags and the named source required by `false`, excludes the flag from measurement identity, and preserves existing annotations when merging a legacy collection that omits the column. Collectors do not infer eligibility from kernel labels or row order.

**Legacy-overwrite caveats.** A new-schema row replaces a legacy-adapted
leaf only at the *same* key, and the legacy adapters derive their node/EP
geometry rather than reading it:

- Legacy sglang DeepEP comm tables carry no `ep_size`; the adapter assumes
  `ep_size = node_num * 8` (8-GPU HGX fleets). New `moe_a2a` worlds
  therefore overwrite those leaves only when collected at 8 GPUs/node — a
  4-GPU/node (GB200) world with the same node count keys differently and
  **coexists** with the legacy data instead of replacing it.
- The legacy trtllm alltoall table (GB200 NVL4) carries no `num_nodes`
  column; the adapter derives `node_num = max(1, moe_ep_size // 4)`. New
  rows overwrite only at 4 GPUs/node; a row with an explicit `num_nodes`
  column is honored as written.

**`log_perf` freezes the CSV header from the first row**, so optional
columns are all-or-nothing per file. In particular, power columns exist in a
staging CSV only if the very first logged row had them: flipping
`--measure_power` across a `--resume` of the same staging file appends rows
whose fields no longer match the frozen header — misaligned columns or empty
cells that fail at load time. Collect any one file with a single power
setting end to end. This property is also why the `moe_a2a` collector ships
**no power column at all** (owner ruling during PR review): its low-latency
timing covers one round-trip rather than a per-phase region, and meaningful
per-phase power would need winning-config re-runs — a measurement-method
design for hardware, not a column to fake. The loaders treat absent (or
null) power as 0.0.

# Test
Rebuild and install the new aiconfigurator. Please make sure you have your new system definition file prepared. It's src/aisimulate_core/systems/xxx.yaml

# Validate the correctness
Today, we have limited method to validate the database. You can try tools/sanity_check to validate the database a little bit. But it highly depends on your understanding
of the GPU system and kernel optimization.

# Known Issues

## NFS File Locking (Worker Deadlock)

**Symptom**: Collection stalls after a few test cases with no error messages.

**Cause**: `fcntl.flock()` doesn't work reliably on NFS. Workers deadlock when writing to shared output files.

**Solution**: Use `/tmp/` for output files, then copy results after collection.

# Support Matrix

Browse the [Legacy AIC Support Matrix](https://ai-dynamo.org/aisimulate/support-matrix/)
for CLI compatibility coverage, or inspect the
[per-system CSV files](../src/aisimulate_core/systems/support_matrix/).
For strict-native estimator coverage, use the
[FPE Support Matrix](https://ai-dynamo.org/aisimulate/fpe-support-matrix/).

## SGLang ordinary MLA context modules

For uniform-projection checkpoints, `run_attention_torch(..., ordinary_mla=True)`
collects the complete context attention module into
`mla_context_module_perf.txt`. It invokes SGLang's real `prepare_qkv_latent`
and creates fresh `AttentionInputs` on every warmup and timed iteration, so
Q/KV down-projection is included. Prefill is timed eagerly. The loaded projection
weights must match the requested single `gemm_type`; mixed BF16/NVFP4 modules
are rejected instead of mislabeled.

The CLI flag `--ordinary-mla` requires `--mode context --attn-type mla`.
Other combinations fail during argument parsing, before collection starts.
If any dispatch fails, the CLI reports its traceback, finishes the remaining
dispatches, and exits with status 1 so partial collection is not reported as success.

Pass `--target-tp-size 4` (or `8`) to derive the per-rank head count from the
model. It must be positive and divide the model's native head count. If
`--num-heads` is also supplied, it must equal native heads divided by TP.
Without an explicit TP size, ordinary MLA derives TP separately for each
per-rank head count. Python callers pass the actual per-rank head count and
`target_tp_size`. Set
`--chunked-prefill-size 16384` on the CLI, or `chunked_prefill_size=16384`
on `run_mla_module()` or `run_mla_module_worker()`, to reproduce a serving
chunk limit of 16,384 tokens. The worker forwards the limit into its subprocess.
Explicit limits must be positive for one-shot module collection.
Omitting the option keeps SGLang's default. The existing
wide-EP and DSA paths keep their separate behavior. The ordinary context table
uses the existing consumer schema; this does not add mixed-precision modeling.

## TRT-LLM MoE autotuning failures

TRT-LLM MoE collection requires either a loaded tactic cache or at least one
successful autotuning attempt before measuring a tuning-enabled case. Ordinary
failures may try smaller eligible token counts; if every attempt fails, the
case raises before timing or publishing rows. CUDA accelerator errors propagate
immediately. The successful collection path and historical profile values are
unchanged by this failure guard.

Campaign scope filters are literal substrings, with OR semantics within each list.
Exclusion wins over inclusion. Preserve the filter arguments with the campaign
manifest and account for excluded backlog separately; never derive exclusions
from failed cases or count them as successful collection. Case identities, native
callbacks, checkpoint compatibility and failure classification are unchanged.

## Pinned B200 communication qualification runtime

Use `collector/campaigns/vllm031-b200-comm-runtime.json` for the B200/vLLM
0.31.0 communication smoke and subsequent full collection. Its SHA256 is
`3e9a843bd12cf4d5a87b86494344651ad6496ef9572820a51c5244f998cb1b9f`.
The provider image preserves all 40 layers of the official amd64 vLLM image
`sha256:5db2b6de97e4f1ca6424f3319f1edffaee0091e2c8b75744fc4be5e380e90cc2`.
It adds nccl-tests v2.17.9 at source
`2535da805b34e96d1dc08be66289be1a6d57f5ad`, built against the existing
NCCL 2.30.7; it does not replace framework libraries. The derived image digest
is separate from vLLM's source commit and original image digest.

In the auto collector, set `AISIM_COMM_MODE=smoke`, `AISIM_OPS=comm`,
`AISIM_GPU_NUM=8`, and `AISIM_COLLECT_NCCL=1`, with the committed manifest
path/hash and its exact image digest. The smoke runs 24 NCCL correctness commands
(half/int8, four primitives, TP2/4/8) and three native vLLM commands covering
eager/graph BF16 at five sizes. Native numerical and dispatch evidence must pass
on every rank. Smoke output is qualification evidence, never full coverage or
automatic publication. Use a separate `full` run after reviewing the smoke.
CPU build/CI success does not establish GPU correctness or performance.

### B200 campaign composition and original MSA seeds

The campaign branch composes the native postload W4A16 precision guard
(`843e3e2`), actual-leaf BF16 input-weighting guard (`ed5d5016`), its getter
stub fix (`d67e3ce06`), and Arrow schema-preserving finalization (`6ca136f`).
The guards reject only the demonstrated unsupported MoE contracts before timing;
MiniMax integration, case grids, and benchmark timing remain unchanged.

For the 184 MSA seed measurements, keep the original `9b4de9d` source mapping,
CSV, checkpoint, and provenance JSON unchanged. Compared with campaign `362953b`,
only `_merge_perf_rows` changes in the shared helper: finalization retains the
original Arrow identity schema and metadata through pandas conversion. MSA native
setup, dispatch, timing, runtime manifest, and generated case identities are
unchanged. The collector content hash still changes because `helper.py` is in
its shared closure:

- Original measurement closure: `cf179f90ba916ebb5cb162896b4458facad46fb66eb289fbd759779a5e0a9797`.
- Campaign `362953b` closure: `e2eb81ce4bf514c4188e02bea3a75f6d1adb5135513e51515d3650bc1f0493c4`.
- Composed MSA closure: `5ad4823f1b772fb554491f065579653ad7c40581a5a672b410ecafc65d69306a`.

Checkpoint loading checks schema, framework, module, callback, framework version,
and SM version; it does not establish source equivalence from these fields alone.
The source comparison above supports scoped seed continuation with the same
runtime, while new finalization metadata must report the new closure honestly.
Retain the per-case original-source mapping and all 44 recorded failures; do not
rewrite seed sidecars or claim these 184 measurements were collected by the new
source. Existing requests pinned to `362953b` with pandas 2.3.3 remain unchanged.
