<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AgentX with vLLM host offload

AgentX accepts the existing `engine.workers.<role>.kv_cache.host_offload`
configuration through `aisimulate predict`, Python `ReplaySpec`, and native
replay JSON. The built-in Engine runner uses the public compatibility checks.
The native entrypoint also enforces the deployment limits. ReplaySpec versions
and omitted default fields are unchanged.

The supported G2 deployment is one aggregated worker or one prefill plus one
decode worker, with vLLM and attention DP=1 on every role. Tensor parallelism may
use multiple GPUs. Worker pools are static, speculative decoding is disabled,
and G3 is excluded. Existing HBM-only vLLM/SGLang support is unchanged.

Each role may omit host offload, use `dp_rank_local`, or use `cluster_shared`.
Local pools are independent. Shared roles join the same deployment-level pool:
`num_host_blocks` is its total capacity, not capacity per participant. Shared
capacity, shared directional bandwidths, and KV layout must agree; incompatible
participants are rejected. Layout includes model/backend, parallel shape, KV
dtype and block geometry. The runtime registry is recreated for each replay.
Per-role D2H/H2D bandwidth and first-byte latency may differ. Shared bandwidth
limits apply to concurrent transfers in addition to each role's link limit.

## Integration scope

AISimulate #381 opens and qualifies AgentX G2 through the existing Engine
runner and native JSON entrypoint. It adds no execution bridge, package version
API or consumer build plumbing.

Dynamo's G2 transport, event handling and routing are implemented separately in
[Dynamo #15647](https://github.com/ai-dynamo/dynamo/pull/15647). AgentX-specific
Dynamo integration and qualification belong to
[Dynamo #15626](https://github.com/ai-dynamo/dynamo/pull/15626), based on that PR.
Use that consumer's dependency requirements and installation instructions once
its AgentX support is qualified. AISimulate's Engine acceptance results do not
qualify a Dynamo installation.

## Run the original functional fixture

From the repository root with a source build containing this feature:

```bash
aisimulate predict --stack engine --config examples/cli/agentx-g2-local.yaml \
  --capture-per-request --output-dir /tmp/agentx-g2-local --format json
aisimulate predict --stack engine --config examples/cli/agentx-g2-shared-pd.yaml \
  --capture-per-request --output-dir /tmp/agentx-g2-shared --format json
```

The small Weka fixture is authored for this repository. Its child request
evicts the parent's prefix from the three-block GPU cache. The parent resumes
after the child and restores eight tokens from host memory. Fixed pass timing
and explicit KV bytes make this an offline functional test; no weights or GPU
are required. These numbers are not measured model performance.

For a local/shared control, change only `host_offload.scope`. For a bandwidth
control, lower `h2d_bandwidth_gbps`; for a capacity control, lower
`num_host_blocks` to 1 (on both roles of a shared pool). To disable G2, remove
each role's `host_offload` mapping. No separate connector configuration is used.

Python callers compile the same configuration with
`prediction_to_replay_spec(CorePredictionConfig.from_yaml(path))`, then execute
it with `EngineReplayRunnerFactory().create(0).run(spec)`. Request
`ReplayOutputRequirements(include_raw_report=True, capture_per_request=True)`
to retain the native report. The native JSON form uses
`spec.engine.rank.native_host_offload` for aggregated workers and
`spec.engine.{prefill,decode}.rank.native_host_offload` for P/D; Python supplies
the shared `kv_layout_id` from the configured geometry. Direct native JSON
callers **must supply a nonempty `kv_layout_id` for `cluster_shared`**. Use the
same value only for genuinely compatible layouts; the registry also validates
physical block geometry, TP, capacity and shared bandwidths.

## Result interpretation

Inspect `first_admission_g1_reused_input_tokens`,
`first_admission_host_reused_input_tokens`, and `admission_history` in the
per-request report. History identifies the admission `pool`; P/D observations
are not added twice to request-first reuse. `g2_domains` reports shared capacity
once. P→D handoff and G1↔G2 transfers remain separate operations. A pending
store or restore cannot supply available cache early; H2D wait delays admission
and first token.

Warmup waits for necessary engine work and retains its G1/G2 state across the
measurement barrier. Tool gaps, dependency joins, lane recycling and play cache
identities retain their AgentX semantics. Duration/cancellation do not force a
background-transfer drain beyond the existing measurement contract.

Qualification remains `functional_only`. This implementation reuses the native
vLLM G2 model and does not establish performance equivalence with SemiAnalysis's
Mooncake recipe, SGLang offload or TRT-LLM offload.
