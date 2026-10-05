<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AgentX with vLLM host offload

AgentX accepts the existing `engine.workers.<role>.kv_cache.host_offload`
configuration through `aisimulate predict`, Python `ReplaySpec`, and native
replay JSON. The built-in Engine runner and matching Dynamo replay adapter use
the same public compatibility checks. The native entrypoint also enforces the
deployment limits. ReplaySpec versions and omitted default fields are unchanged.

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

## Merge order and source qualification

Both G2 PRs are based on their repositories' `main` branches. AISimulate #381
includes the generic canonical JSON execution entrypoint used by Dynamo #15626;
the consumer includes the corresponding Python/native bridge. Conversation
lineage, session/sibling affinity and their policy lifecycle changes remain
separate work. Neither complete routing PR (#306 or #15240) is a prerequisite.

After AISimulate #381 merges, update Dynamo #15626 to the resulting commit on
`main`, regenerate its existing lockfiles, and build/install the matching
AISimulate Python wheel. When consuming a published package, use an actual
release/nightly containing that commit and update Python, Rust and container
requirements together. Repeat installation and end-to-end validation, then
update the documented source pair before merging Dynamo #15626.

The branch commits below are immutable **pre-merge qualification snapshots**.
They include the review fixes and reproduce the paired integration run; they
are not a final release or proof that the consumer is ready to merge. A squash
merge produces a different main-branch SHA, so keeping these snapshots does not
satisfy the post-merge pin update. Do not guess a future SHA or replace the pin
with a floating branch.

## Install a pre-merge qualification snapshot

The source pair is AISimulate `7f47667dc1066d4afa019f8f0700c17ecd3ec57b`
and Dynamo `f31031a725067f5705a476a4e619fa63cfd9c447`. These are development
source builds, not a published release. Keep the strict Python/core version
checks; do not substitute an older wheel with the same package version.
Use Python 3.12, `uv`, Rust 1.96.1 and a C/C++ compiler/linker. See
[source installation](installation.md#use-current-source) for AISimulate's
platform requirements and Dynamo's source-build requirements in its checkout.

```bash
agentx_g2_dir=/tmp/agentx-g2-install
mkdir -p "$agentx_g2_dir"
uv venv --python 3.12 "$agentx_g2_dir/venv"
uv pip install --python "$agentx_g2_dir/venv/bin/python" 'maturin>=1.12,<2' patchelf
git clone https://github.com/ai-dynamo/dynamo.git "$agentx_g2_dir/dynamo"
git -C "$agentx_g2_dir/dynamo" checkout --detach f31031a725067f5705a476a4e619fa63cfd9c447
git clone https://github.com/ai-dynamo/aisimulate.git "$agentx_g2_dir/aisimulate"
git -C "$agentx_g2_dir/aisimulate" checkout --detach 7f47667dc1066d4afa019f8f0700c17ecd3ec57b
cd "$agentx_g2_dir/aisimulate/python/aisimulate"
RUSTUP_TOOLCHAIN=1.96.1 "$agentx_g2_dir/venv/bin/maturin" build \
  --locked --profile dev --out "$agentx_g2_dir/wheels"
uv pip install --python "$agentx_g2_dir/venv/bin/python" "$agentx_g2_dir"/wheels/aisimulate-*.whl
cd "$agentx_g2_dir/dynamo/lib/bindings/python"
RUSTUP_TOOLCHAIN=1.96.1 "$agentx_g2_dir/venv/bin/maturin" build \
  --locked --profile dev --features ais-forward-pass --out "$agentx_g2_dir/wheels"
uv pip install --python "$agentx_g2_dir/venv/bin/python" \
  "$agentx_g2_dir"/wheels/ai_dynamo_runtime-*.whl "$agentx_g2_dir/dynamo"
uv pip check --python "$agentx_g2_dir/venv/bin/python"
export PATH="$agentx_g2_dir/venv/bin:$PATH"
cd "$agentx_g2_dir/aisimulate"
```

Dynamo's manifests, existing lockfiles and container source requirement pin
the same AISimulate commit. This pair was built with `--locked` without Cargo
patches and installed into an isolated environment. Twelve CLI runs cover
aggregated/P-D, HBM/local/shared and native Engine/Dynamo KV routing. G2
restoration and timing agree across runners, and host-only reuse does not
become a GPU routing hit.

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

## Dynamo and result interpretation

Use a matching AISimulate/Dynamo source pair containing the AgentX G2 adapter
changes, with the same AISimulate source revision for its Python wheel and
Dynamo's compiled core. Older HBM-only builds do not include these changes.
No additional distribution, connector, or runtime service is
required. With that pair installed, add the existing `dynamo.router` adapter
using `router.policy: kv_router` and run with `--stack dynamo`; the adapter
uses canonical ReplaySpec execution whenever an active role enables G2,
independent of trace format or profile settings. This also preserves
G2 parameters for ordinary Dynamo, Mooncake, Mooncake-delta and synthetic
traffic. Static worker pools and `router.policy: kv_router` remain required by
this adapter path. Ordinary non-agentic inputs retain native G2 topology
validation; AgentX's single-worker/1P1D and DP=1 restrictions apply when the
native loader identifies an agentic workload. It retains the existing routing
strategy.

```bash
aisimulate predict --stack dynamo --config examples/cli/agentx-g2-shared-pd.yaml \
  --set router.policy=kv_router --capture-per-request \
  --output-dir /tmp/agentx-g2-dynamo --format json
```

Read the complete `prediction.json` in the output directory for per-request
and shared-pool evidence; Dynamo's JSON stdout retains its summary envelope.

GPU and pinned-host events preserve their tiers. The Dynamo G1 routing index
ignores host events, so G2 presence does not imply GPU residency. Shared G2
does not introduce hit-aware worker selection.

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
