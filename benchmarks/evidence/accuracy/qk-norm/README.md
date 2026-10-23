<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Decode Q/K normalization accounting

CPU-only reproduction and source audit for [AISimulate issue #434](https://github.com/ai-dynamo/aisimulate/issues/434).
The baseline is `c9e7363bf5983b980fe6622f7b064ee7adff5f5d`. Both prediction runs
use checked-in H200/vLLM 0.24.0 profiles, BF16 activations/KV, tensor parallelism
1, and `RustForwardPassPerfModel.best_available` with explicit `op_level`
estimation. No GPU experiments or model weights are needed.

## Backend audit

A universal one-kernel assumption is **not supported**:

| Backend and immutable source | Q/K normalization path |
| --- | --- |
| [vLLM 0.24.0](https://github.com/vllm-project/vllm/blob/ee0da84ab9e04ac7610e28580af62c365e898389/vllm/model_executor/models/qwen3.py#L147) | Qwen3 makes separate Q and K RMSNorm calls. The optional Q/K-norm + RoPE fusion is [off by default](https://github.com/vllm-project/vllm/blob/ee0da84ab9e04ac7610e28580af62c365e898389/docs/design/fusions.md#qk-norm--rope-enable_qk_norm_rope_fusion). |
| [vLLM 0.31.0](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/vllm.py#L267) | Q/K-norm + RoPE fusion remains disabled in the optimization defaults. This tag is an adjacent source reference; the issue's 0.31.1-dev runtime has no exact source SHA. |
| [SGLang 0.5.14](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/models/utils.py#L438) | `apply_qk_norm` can call a fused in-place Q/K kernel on eligible CUDA inputs. Deterministic inference, compiler choice, dtype/head size and other eligibility checks affect selection. Its fallback calls Q and K norms separately, optionally on different streams. |
| [TensorRT-LLM 1.3.0rc20](https://github.com/NVIDIA/TensorRT-LLM/blob/c25c23f71786bad54d192893d696ce8043426eca/tensorrt_llm/_torch/modules/qk_norm_attention.py#L154) | `QKNormRoPEAttention` defaults to fused Q/K norm + RoPE. Disabling fusion takes a separate-Q/K path, which may execute the calls in parallel. |

CUDA-graph replay does not establish a single normalization kernel or zero
normalization execution time. These implementations provide fused and unfused
paths; a backend name and version alone do not identify the active path.
[backend-review.json](backend-review.json) retains immutable commits, inspected
source-file fingerprints and the profile evidence boundary. No upstream code
is copied into this change.

The vLLM collector times its low-level attention closure with already-produced
Q/K/V inputs (`collector/vllm/collect_attn.py`, `run`), optionally including KV
cache update. The SGLang and TensorRT-LLM collectors likewise pass Q/K/V to
attention rather than timing the enclosing Qwen normalization layers. The
shared helper defaults to CUDA-graph timing. These describe the inspected
collector code, not an assertion that a legacy parquet records its exact
historical collector revision or graph mode.

The H200 attention parquet has 63,500 rows, legacy collection provenance,
shape/dtype/kernel-source coordinates and latency. It has no Q/K-normalization
or fusion-policy coordinate, per-norm timing, or norm launch count. The issue
reports whole-step H200 measurements, not a per-kernel normalization trace.
Neither source can calibrate one generic memory-op intercept per physical norm
kernel.

## Correction and boundary

The old decode extra charges the empirical memory-op intercept 8.8 times per
layer, giving 26.4 us on H200 before accounting for bytes. Its byte term omits
the query-token count.

The corrected extra models **bandwidth-only normalization traffic**:

```text
query_tokens = sequence_batch * max(verify_query_tokens, 1)
bytes = query_tokens * (local_query_heads + local_kv_heads) * head_dim * 2 * 4
extra_ms = 1.1 * bytes / effective_bandwidth * 1000
```

The existing four-pass BF16 traffic allowance and 1.1 correction are retained.
They are analytical modeling choices, not a claim that the source audit proves
four physical DRAM passes. Passes and tokens scale bytes. No fixed intercept
is inferred from an unknown normalization kernel count. This uses the issue's
suggested option of dropping the generic constant for the decode extra.

The component is deliberately incomplete: it does not predict separate fixed
norm-kernel execution costs or distinguish fused/unfused runtime configurations.
It does not assert that those costs are zero. A calibrated residual term would
need normalization-specific timing and execution identity. No model-, system-
or backend-specific fitted offset is introduced, and the common formula applies
to each backend. These results are prediction changes, not certification of
whole-model accuracy on every backend.

Normalization uses local heads before the DCP query gather and counts all real
new speculative query tokens. The FPM SOL leaf keeps fractional inputs and the
same traffic allowance at peak bandwidth. Prefill and standalone memory-op
intercepts are unchanged. No schema, public config field or profile data changes.

Host-cost assessment: the changed path performs one bounded scalar traffic
calculation per generation-attention query, replacing the baseline's two
memory-op queries and extra result composition. It adds no allocations,
caches, indexes, preparation or retained memory; its host work does not loop
over tokens. Normalization-disabled paths do not call the new helper. This is
a source-level cost bound, not a measured host speedup. Table lookup and
preparation, first-query and warm-query cache boundaries remain unchanged.

## Reproduction

Build each matching revision, then run from the repository root:

```bash
uv sync --project python/aisimulate --extra dev --reinstall-package aisimulate
python/aisimulate/.venv/bin/python benchmarks/evidence/accuracy/qk-norm/reproduce.py
```

The baseline predates the script; save it outside the checkout before switching
revisions. `before.jsonl` and `after.jsonl` retain 12 paired points: two Qwen3
models, batches 1 and 8, and KV lengths 64, 576 and 4096.

`attention_ms` and `decode_step_ms` are predictions. `nearest_kernel_ms` is the
first attention-table row at or above KV-1 times layer count, excluding Q/K
norm and other model ops. It is not the exact smoothed/interpolated query; KV
576 uses table step 1023. Its difference from a prediction is not a measured
full-model error or exact norm cost. The issue's serving runtime/workload differ
from these fixed-shape vLLM 0.24.0 queries.

Profile SHA-256 (unchanged):

```text
h200_sxm.yaml
1d7359c1a6615efb6c09bb0edcac05fd35ba0a3743a9908e041c4a6c1ec3909d
generation_attention_perf.parquet (h200_sxm/attention/vllm/0.24.0)
ea1b6f802bb7b0792a6eaa9b533d5728c2b230749b9673f6553743ddb78837b9
```

## Regression checks

The expected norm traffic is hand-counted independently: Qwen3-8B has 5,120 Q+K
BF16 elements per new token. The four-pass allowance moves 40,960 bytes/token;
at H200's 3.84 TB/s effective bandwidth and the 1.1 factor, this is
0.0000117333333333 ms/token. There is no fixed intercept. Standalone memory
operations retain their 0.003 ms intercept at zero bytes.

The Rust checks cover SILICON, HYBRID, EMPIRICAL, SOL and SOL_FULL; three batches
and three KV lengths; layer/imbalance scaling; energy/provenance; SOL memory
components; DCP local heads; FP8 KV with BF16 Q/K; speculative query widths;
and fractional FPM inputs. Python checks exercise the real native bindings,
canonical configuration round trips and speculative consumers.

The golden refresh uses `pin_goldens.py --refresh`; record source commits are
retained in `post_freeze_pins`, with no test-matrix or tolerance changes.
[golden-review.json](golden-review.json) records the complete value audit and
validation counts. In the five per-op cases, only generation-attention latency
changes; op sets, other latencies, source tags and energies are identical.

The refreshed Qwen3.8 vLLM records also contain four pre-existing discrepancies:
3 us in each `static_ctx` value and 5 us in each `disagg_ttft` value. An isolated
build of the original source and the corrected release return identical prefill
values. The audit retains both values and the old goldens; these small shifts
are unrelated to the decode correction.


For batch 1 and KV length 576, the paired decode-step predictions are:

| Model | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Qwen3-8B | 6.005822 | 5.055422 |
| Qwen3-0.6B | 2.280446 | 1.541246 |

All 12 pairs retain the same nearest-table references and source tags. The
change in total decode prediction equals the generation-attention change.
These are CPU-computed predictions from unchanged GPU profiles.

The original refresh updates 119 engine-step records, 21 compile-engine
references and five per-op cases. Of the 119 engine-step records, 118 change
numerically; the rounded AFD control remains identical to the original
baseline. That original model and refresh use source revision
`88a4f8b40ee7370174e5c6ef13304d418a056432`.

After incorporating upstream main at `4730b2c1006ec19b9513f09e4c9ad93e67c41265`,
the four `qwen3-30b-a3b-b60-vllm-scan-coverage` records are an exception: they
combine upstream's vLLM 0.28.0/XPU calibration with the unchanged Q/K formula
and were selectively pinned from `173a390abbb25269e6bd74267ec415f54e2f1ea7`.
Compared with upstream, B60 static context remains 133.227 ms and generation
changes from 9.784 to 8.517 ms. The combined model at
`d6722c4a9bec8101946f4a63cc1de94b72283067` passed both parity suites (390 tests).

The existing 16-case B200 native stability manifest now records that combined
model as its source. Only its eight decode expectations change; all eight
prefill expectations, query inputs, and tolerances are unchanged. These are
proposed prediction references, not measured hardware accuracy or maintainer
acceptance. The source was rebuilt in release mode, and its imported native
binary and numerical environment were verified before capturing all 16 values.
`golden-review.json` records each old/new value and its independent derivation.
This source is a PR-branch merge commit. Its exact-SHA fetch succeeds in a
fresh shallow PR checkout, but that does not guarantee availability after a
squash merge. The [canonical baseline handoff](../../../../docs/ci/README.md#numerical-and-installed-package-evidence)
requires revalidation at the actual main-branch commit and a follow-up manifest
repin. Keep this capture SHA as historical evidence; the durability follow-up
must preserve the recorded expectations and tolerances unless a separate,
explained modeling change is reviewed.

B200's effective bandwidth is 6.16 TB/s. The old expression charges 8.8 generic
3 us intercepts per layer, or 0.0264 ms. Let `t` be the hand-counted four-pass
BF16 Q/K traffic per token at that bandwidth, including the retained 1.1 factor.
For these non-speculative shapes, the predicted change is:

```text
delta_ms = layers * (osl - 1) * ((batch - 1) * t - 0.0264)
```

Qwen3-32B has 64 layers and local Q/K heads 16/2 at TP4; MiniMax-M2.5 has
62 layers and local heads 6/1 at TP8. Both use head dimension 128. Their `t`
values are 0.0000032914285714285715 and 0.00000128 ms, respectively.

| Model and covered backends | Batch 1, OSL 2 delta (ms) | Batch 8, OSL 256 delta (ms) |
| --- | ---: | ---: |
| Qwen3-32B: vLLM, TRT-LLM, SGLang | -1.6896 | -430.4719872 |
| MiniMax-M2.5: vLLM | -1.6368 | -417.2423424 |

The long-output cases total 255 decode steps. All eight deltas agree with the
native changes within 0.000000000001 ms. The original PR and combined model
produce identical B200 values; the Rust performance-model and B200 profile
trees are unchanged between the PR's base and incorporated upstream main.
Thus these shifts come from the intended formula change, rather than the B60
promotion. This algebraic check does not calibrate the fixed normalization
costs omitted by the approximation described above.

Reproduce the existing numerical check with the matching native build:

```bash
python/aisimulate/.venv/bin/python scripts/prediction_regression/check_prediction_numerics.py \
  --output numerical-sentinels.json
```

Validation on the original implementation (with its refreshed goldens):

```bash
cargo test -p aisimulate-core --lib perfmodel::operators
# 318 passed

python/aisimulate/.venv/bin/python -m pytest -q -p no:timeout \
  python/aisimulate/tests/unit/sdk/database/test_attention_lanes.py \
  python/aisimulate/tests/unit/sdk/test_forward_pass_speculation.py \
  python/aisimulate/tests/cross_package/test_single_oracle_contract.py
# 100 passed

python/aisimulate/.venv/bin/python -m pytest -q -p no:timeout \
  python/aisimulate/tests/unit/sdk/test_rust_engine_step.py -k canonical
# 26 passed, 65 deselected

python/aisimulate/.venv/bin/python -m pytest -q -n 4 -p no:timeout \
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py \
  crates/core/parity_tests/perfmodel/test_compile_engine_parity.py
# 390 passed
```

Rust formatting, repository Ruff checks/formatting, documentation links,
copyright, packaged legal files and generated CODEOWNERS checks also passed
locally. Hosted CI and independent reviews are separate PR gates.
