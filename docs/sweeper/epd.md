<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Analytical EPD search

The Sweeper SDK can search a dedicated encoder pool in front of aggregate
language workers (E+agg) or prefill/decode workers (E+P+D). This is an analytical
integration with the AIC EPD implementation already shipped inside AISimulate,
not an event-level encoder simulation or a replacement for all AIC sweep semantics.

The model and composition source is [AIC PR #1340](https://github.com/ai-dynamo/aiconfigurator/pull/1340),
merged as `f8f2341cb5761877bda694ab954cb6f5eff78fd4` and migrated into this repository.
The integration calls the current in-tree `_get_encoder_worker_candidates` and
`_overlay_encoder_stage` helpers; it does not fork those modeling formulas.
It replaces the integration approach of closed AISimulate PR #63 without its
historical estimator, heterogeneous-P/D, or parallel-search dependency stack.

## Unified CLI

`aisimulate predict` and `recommend` support this same analytical model on the
offline `engine` stack. The public input pairs `traffic.source.images` (positive
height, width, count) with `engine.workers.encoder`. Text `input_tokens` excludes
visual tokens; preprocessing adds them exactly once before language replay.

```bash
aisimulate predict -c examples/cli/epd-predict-aggregated.yaml --output-dir /tmp/epd-agg
aisimulate predict -c examples/cli/epd-predict-disaggregated.yaml --output-dir /tmp/epd-disagg
aisimulate recommend -c examples/cli/epd-recommend.yaml --output-dir /tmp/epd-search
aisimulate predict -c /tmp/epd-search/recommendations/0001.yaml --output-dir /tmp/epd-selected
```

Use a fresh output directory, or deliberately pass `--overwrite` for known outputs.
Encoder `tensor`, `replicas`, and `batch_size` are positive scalars for prediction,
and scalars or `{choices: [...]}` for recommendation. Defaults are 1; batch size
is at most 8. Encoder `hardware` defaults to language hardware. Optional
`backend_version` selects encoder data independently; the backend follows the
language backend. `latency_correction` defaults to 1.0 and `rate_degradation`
defaults to 0.9. Saved YAML pins the resolved encoder hardware/data version,
shape and both factors, rather than storing editable timing estimates.

CLI EPD requires `source.type: synthetic`, a fixed integer concurrency, and an
absolute request count or `requests_per_load_unit`. Traces, sessions, arrival-rate
loads, concurrency search, adapters, online execution and per-request capture
are rejected. Language timing must be default `op_level`, with static workers.
Recommendation GPU budgets include all encoder and language GPUs; minimum-GPU
constraints and goodput objectives remain unsupported. Recommendation SLA bounds
require `strict_sla: true`; prediction preserves these as aggregate-mean bounds,
never per-request goodput.

`prediction.json` contains aggregate `summary` metrics and `metadata` identifying
`analytical_epd_overlay`, the complete resolved encoder, GPU totals and SLA
semantics. JSON stdout also identifies the approximation. It contains no raw
native report, percentiles, per-request records or total-deployment power claims.
The recommendation ledger retains the same provenance. Lossless prediction YAML
is not a deployment manifest; deployment generation still rejects EPD.

## Sweeper SDK

Use [the SDK example](../../examples/sweeper/epd.yaml):

```python
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper import SmartSearchConfig, Sweeper

if __name__ == "__main__":
    config = SmartSearchConfig.from_yaml("examples/sweeper/epd.yaml")
    result = Sweeper(runner_factory=EngineReplayRunnerFactory()).run(config)
    print(result.to_json())
```

`workload.images` is a fixed positive height, width and image count on every
synthetic request. Text ISL excludes the visual tokens, which are derived by
the same model-specific AIC preprocessing contract and added to language context.
`search_space.encoder` searches TP, batch size (at most 8), and worker count.
Optional `hardware_sku` and `backend_version` select encoder data independently;
the encoder backend always follows the language backend. Only available SILICON
data is admitted, and AIC's encoder geometry and memory gates remain authoritative.

## Semantics and limits

Language scheduling is replayed with the visual context tokens. The AIC analytical
overlay then caps throughput by degraded encoder-pool capacity and adds raw
encoder batch latency to aggregate TTFT and E2E latency. This matches AIC's
single-point composition convention (factor 1), not the disaggregated capacity
sweep's additional TTFT correction. The language replay already models queueing.
Encoder backpressure does not change language scheduling in this approximation.

The output is marked `analytical_epd_overlay`. Its duration is a rate-derived
accounting interval for the same completed request count, not an EPD event
timeline; GPU-hours use that interval and all language plus encoder GPUs.
The original language replay duration is retained in provenance.
Mean inter-token/user throughput remains the language result. No EPD percentile,
per-request goodput, raw trace, or telemetry claim is made.

Candidates retain the resolved encoder model, system, backend/version, TP, batch,
workers, timing, memory and available power evidence. Missing encoder power is
`null` in provenance and omitted from numeric metrics, never reported as zero
measured watts. Encoder power is not a prediction of total deployment power.

The first integration intentionally rejects traces, variable token lengths,
sessions, prefix-sharing workloads, KV-load search, custom language timing,
whole-model FPM, dynamic pools, adapters and per-request goodput objectives.
Aggregate SLA bounds require `strict_sla: true`. Encoder queueing, CPU overhead,
embedding transfer, variable-image traces and deployment generation remain out
of scope. Estimation support does not establish silicon accuracy.

The SDK example uses its own schema, distinct from public CLI YAML above.
Prediction-config callbacks must preserve the resolved encoder identity, fixed
image/text workload and GPU topology or the candidate is rejected. Deployment
generation still rejects EPD candidates rather than dropping the encoder pool.
The compatibility `aiconfigurator` CLI remains available for AIC EPD workflows.

AFD and analytical EPD cannot be combined. EPD requires an `agg` or `disagg` language deployment; searches and direct replay specifications reject AFD combinations before evaluation.
