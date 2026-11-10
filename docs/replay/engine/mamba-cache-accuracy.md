<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen hybrid-cache accuracy regression

The checkpoint-aware cache eliminates all **18 observed cache-admission errors**
in this capture: exact prefix matches improve from **77/95 to 95/95 requests**.
Client TTFT improves modestly and still has substantial error; this is a cache
correctness result, not complete end-to-end accuracy acceptance.

| Metric | Population | Before | After |
| --- | --- | ---: | ---: |
| Exact cached-prefix matches | All 95 requests | 81.05% (77/95) | 100% (95/95) |
| Cached-prefix mean absolute error | All 95 requests | 663.58 tokens | 0 tokens |
| Uncached prefill-work WAPE | All 95 requests | 2.6205% | 0% |
| Conditional first-prefill step MAPE | All 95 requests | 7.3890% | 0.02392% |
| Client TTFT MAPE | 94 profiling requests | 50.8630% | 47.1290% |
| Client TPOT MAPE | 94 profiling requests | 2.6444% | 2.6444% |

The machine-readable measurements, input hashes, binary hashes, and controls are
in [mamba-cache-accuracy.json](mamba-cache-accuracy.json).

## What was measured

The native reference is GPU collection `qwen-tp4-c1-source24-5563558`:
Qwen3.5-397B-A17B FP8, 4×B200, SGLang TP4 / attention-DP1, one lane, STP,
page size 64, and chunked prefill size 16,384. The diagnostic workload contains
95 requests from 24 selected source traces: one warmup and 94 profiling requests.
It uses shortened windows with output length capped at 128 and gaps capped at
one second. It is not the unmodified published AgentX benchmark.

Both runs execute the real Replay engine on the same captured prompt-prefix
equivalence graph and the same frozen FPM artifact. The only executed
configuration difference is enabling:

```json
{"mamba_cache":{"prefill_checkpoint_interval":64,"decode_checkpoint_interval":256,"max_states":994}}
```

This is an explicit opt-in configuration. Ordinary attention cache behavior is
also exercised by running the candidate binary with this setting absent.
The measured cohort completes without entering unsupported state/KV eviction
pressure. See [supported configuration and limits](kv-cache.md).

The exact-prefix score compares each Replay request's
`first_admission_reused_input_tokens` with the native observer's first prefill
`past_kv_lengths`. Every native request must appear in both runs. WAPE is
`sum(abs(predicted_uncached - native_uncached)) / sum(native_uncached)`;
it does not divide by total prompt length or discard misses.

The first-prefill latency score evaluates the **unchanged frozen FPM** at each
run's predicted admission geometry and compares it with the corresponding native
first-prefill duration. The scorer requires singleton native prefill batches and
verifies the chunk geometry. This is a conditional step score over 95 first
steps, not a score over every iteration or a claim that all later batches align.
The forest was trained on the same capture: **11,401 decode + 210 prefill rows,
zero independent test rows**. Its scalar interpretation was cross-checked against
all 210 saved prefill predictions with zero numerical difference.

Client TTFT/TPOT use the actual Replay request report and the native client
metrics for the same 94 profiling requests. Those numbers include the remaining
client-boundary and scheduling differences. Warmup is excluded from these client
metrics on both sides. The residual **47.13% TTFT MAPE** remains unresolved.

## First request that exposed the bug

Request `3d3a962a-26ec-4d69-8cc6-d6d9fd90a8ec` has 46,462 prompt tokens:

| Quantity | Native | Before | After |
| --- | ---: | ---: | ---: |
| Cached prefix | 44,736 | 46,400 | 44,736 |
| First prefill work | 1,726 | 62 | 1,726 |
| Conditional step latency | 66.4594 ms | 34.0676 ms | 66.4594 ms |

Its conditional step error falls from 48.739% to approximately zero. The 95/95
prefix result comes from replaying the cohort lifecycle; no per-request prefix
answer or checkpoint oracle is supplied to the engine. Native generated output
IDs and the complete live checkpoint inventory were not captured, so this does
not independently validate every aspect of the native cache lifecycle.

<details>
<summary>Reproduce the comparison and controls</summary>

The native capture and fitted forest are not distributed in this repository.
Use the captured files identified by the hashes in the JSON receipt. The trace
header must bind the exact native prompt map, and the execution descriptor must
point to its captured trace and frozen forest. No GPU rerun is required.

Build the baseline revision and the candidate in separate checkouts, using the
same toolchain and build flags in both:

```bash
cargo +1.99.0 build --release --locked -p aisim-core --features extension-module
```

Set `MAMBA_BASE_RUNTIME` and `MAMBA_FIXED_RUNTIME` to those checkouts'
`target/release/libaisimulate_core.so` files. Set `MAMBA_EXECUTION` to the
captured `execution.json`, `MAMBA_NATIVE` to the native postprocess directory,
and `MAMBA_RESULTS` to a fresh output parent. Run from the candidate checkout:

```bash
python3 scripts/prediction_regression/compare_mamba_cache_accuracy.py run \
  --runtime "$MAMBA_BASE_RUNTIME" --revision 4ff7bd2ab8748c4c843db8a1755eb0bc59c8a0c6 \
  --execution "$MAMBA_EXECUTION" --output "$MAMBA_RESULTS/before"

python3 scripts/prediction_regression/compare_mamba_cache_accuracy.py run \
  --runtime "$MAMBA_FIXED_RUNTIME" --revision 93a9c551555331b0b9825b9f6c332bcc1819abfe \
  --execution "$MAMBA_EXECUTION" --output "$MAMBA_RESULTS/after" \
  --mamba-config '{"prefill_checkpoint_interval":64,"decode_checkpoint_interval":256,"max_states":994}'

python3 scripts/prediction_regression/compare_mamba_cache_accuracy.py compare \
  --before "$MAMBA_RESULTS/before" --after "$MAMBA_RESULTS/after" \
  --native-canonical "$MAMBA_NATIVE/canonical/canonical-fpm.jsonl" \
  --native-prompts "$MAMBA_NATIVE/prompt-map/native-request-token-map.jsonl" \
  --native-client-metrics "$MAMBA_NATIVE/request-metrics.json" \
  --output "$MAMBA_RESULTS/accuracy.json"
```

Each run saves its executed payload, report, runner snapshot, and hashes.
Comparison rejects input drift, a changed forest, a different prompt capture,
missing requests, or unsupported native first-step geometry. The old
observer-only `experimental_learned_query_capture` field is removed from both
archived descriptors because the released runtime rejects it; it affects no
predictions. Both sides use `run_replay_json`, avoiding unsupported KV-event
publishers enabled by the artifact API.

Repeat each run into another fresh directory, then use:

```bash
python3 scripts/prediction_regression/compare_mamba_cache_accuracy.py parity \
  --first "$MAMBA_RESULTS/after" --second "$MAMBA_RESULTS/after-repeat" \
  --output "$MAMBA_RESULTS/repeat-parity.json"
```

Parity compares the entire report, including request timings and cache reuse,
excluding only three measured host execution-cost fields: `wall_time_ms`,
`processed_tokens_per_s`, and `processed_output_tokens_per_s`. The same check
compares the baseline with the candidate run **without** `--mamba-config`.

The checked-in scalar controls require no captured data:

```bash
pytest -q tests/test_mamba_cache_accuracy_metrics.py
```

</details>
