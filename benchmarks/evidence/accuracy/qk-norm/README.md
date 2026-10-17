<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Decode Q/K normalization accounting

This CPU-only reproduction addresses [AISimulate issue #434](https://github.com/ai-dynamo/aisimulate/issues/434).
The base is `c9e7363bf5983b980fe6622f7b064ee7adff5f5d`. Both runs use the checked-in
H200/vLLM 0.24.0 profiles, BF16 activations/KV, tensor parallelism 1, and the
canonical `RustForwardPassPerfModel.best_available` interface with explicit
`op_level` estimation. No GPU experiment or model weights are required.

## Accounting

The old decode extra multiplies the complete empirical memory-op latency,
including its constant, by memory-pass counts. Two Q/K contributions, each
scaled by 2, then by 2 and 1.1, charge the constant 8.8 times. On H200 this
alone adds 26.4 us per layer. Its byte term also omits the query-token count.

The corrected model retains the existing four BF16 memory passes and 1.1
correction. It combines their traffic before making one memory-op query:

```text
query_tokens = sequence_batch * max(verify_query_tokens, 1)
bytes = query_tokens * (local_query_heads + local_kv_heads) * head_dim * 2 * 4
extra_ms = 1.1 * (bytes / effective_bandwidth + constant_seconds) * 1000
```

One constant models the combined fused Q/K extra per layer. Memory passes and
query tokens multiply bytes, rather than launches. This is an analytical
one-launch assumption, not a measured claim about every backend's kernel
fusion. The common decode formula changes for every backend; there is no
Qwen-, H200-, or vLLM-specific fitted offset. Maintainer review may prefer a
backend-specific kernel-count policy when such data becomes available.

Q/K activations are normalized on local heads before the DCP query gather.
Speculative attention counts every real new query token even though KV reads
use the request count. The FPM SOL leaf retains fractional coordinates and the
same byte count, with peak bandwidth and no empirical constant. Prefill and
standalone memory-op accounting are unchanged.

## Reproduction and interpretation

From the repository root, build each matching source revision and run:

```bash
uv sync --project python/aisimulate --extra dev --reinstall-package aisimulate
python/aisimulate/.venv/bin/python benchmarks/evidence/accuracy/qk-norm/reproduce.py
```

`before.jsonl` and `after.jsonl` retain all 12 paired points: two Qwen3 models,
batch sizes 1 and 8, and KV lengths 64, 576 and 4096. The base predates this
script, so save the script outside the checkout before switching revisions.

`attention_ms` and `decode_step_ms` are simulator predictions. The
`nearest_kernel_ms` reference comes from the first collected attention-table
step at or above KV-1, multiplied by layer count. It excludes Q/K norm and
other model operations, and is not the exact interpolated/smoothed table query.
In particular KV 576 uses table step 1023. Differences from this reference
must not be interpreted as full-model prediction error or exact norm cost.
The issue's serving measurements use a different vLLM runtime and a workload
distribution; these fixed-shape CPU results do not reproduce its serving MAPE.

Profile SHA-256 fingerprints (unchanged between runs):

```text
h200_sxm.yaml
1d7359c1a6615efb6c09bb0edcac05fd35ba0a3743a9908e041c4a6c1ec3909d
generation_attention_perf.parquet (h200_sxm/attention/vllm/0.24.0)
ea1b6f802bb7b0792a6eaa9b533d5728c2b230749b9673f6553743ddb78837b9
```

At batch 1 and KV 576, the full decode prediction changes as follows:

| Model | Before (ms) | After (ms) | Reduction (ms) |
| --- | ---: | ---: | ---: |
| Qwen3-8B | 6.0058218615 | 5.1742218615 | 0.8316 |
| Qwen3-0.6B | 2.2804456185 | 1.6336456185 | 0.6468 |

Across all 12 points, changes to the full decode prediction equal the changes
to `generation_attention`; the nearest table rows and source tags are unchanged.

## Regression checks

Before the correction, both new Rust Q/K norm tests and both parametrized
Python binding cases fail. The binding observes 0.0264117333 ms per layer at
both batch sizes, instead of the independently counted 0.0033117333 ms (batch
1) and 0.0033938667 ms (batch 8).

The Rust checks cover SILICON, HYBRID, EMPIRICAL, SOL and SOL_FULL, three batch
sizes and three KV lengths, layer/imbalance scaling, energy/provenance, SOL
memory components, local heads under DCP, FP8 KV with BF16 Q/K activations,
real speculative query widths, and fractional FPM inputs.

Validation on corrected model revision
`66327986403c1d6ef9ccf345e8d9d3eaeec176b3`:

```bash
# 318 passed
cargo test -p aisimulate-core --lib perfmodel::operators

# 100 passed
python/aisimulate/.venv/bin/python -m pytest -q -p no:timeout \
  python/aisimulate/tests/unit/sdk/database/test_attention_lanes.py \
  python/aisimulate/tests/unit/sdk/test_forward_pass_speculation.py \
  python/aisimulate/tests/cross_package/test_single_oracle_contract.py

# 26 passed, 65 deselected
python/aisimulate/.venv/bin/python -m pytest -q -p no:timeout \
  python/aisimulate/tests/unit/sdk/test_rust_engine_step.py -k canonical

# 390 passed, after the deliberate golden refresh
python/aisimulate/.venv/bin/python -m pytest -q -n 4 -p no:timeout \
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py \
  crates/core/parity_tests/perfmodel/test_compile_engine_parity.py
```

The first parity run reported 125 failures and 265 passes. The selective
`pin_goldens.py --refresh` run updates 119 engine-step records, 21 compiled
references and five per-op records; this includes affected predictions below
the existing parity tolerance. Each refreshed key is identified by the above
source revision in the golden files' `post_freeze_pins` maps. Test matrices and
tolerances are unchanged. A fresh parity run passes all 390 tests.

[golden-review.json](golden-review.json) records the value audit, build hashes
and the two Qwen3.8 vLLM prefill controls. Among the five per-op cases, only
`generation_attention` latency changes; op names, other latencies, source tags
and all energies remain identical. The complete engine-step audit preserves
error records and energies.

The refresh also picks up four pre-existing discrepancies in the two Qwen3.8
vLLM records: 3 us in each `static_ctx` value and 5 us in each `disagg_ttft`
value. These were checked using an isolated debug build of the original
`c9e7363b` source and the corrected release extension, in separate processes
with identical current Python code and profiles. Both builds return exactly
the same prefill values. The audit records both results and the old golden
numbers; these small prefill shifts are unrelated to the decode correction.
