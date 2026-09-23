<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Start an AgentX simulation

This walkthrough replays a Weka agentic workload on a simulated disaggregated
deployment: two prefill workers and four decode workers, using eight H200 GPUs
in total. It prepares KV caches from seeded snapshots, then measures the
remaining requests and recycles completed plays to keep two client lanes occupied
for a 3,600-second simulated admission window. The simulator runs offline on your
CPU; you do not need to allocate those GPUs or download model weights.

Use the combined AISimulate source from
[PR #306](https://github.com/ai-dynamo/aisimulate/pull/306), which includes the
continuous profiles from [PR #307](https://github.com/ai-dynamo/aisimulate/pull/307),
and the matching Dynamo source below. Routing uses the existing `dynamo` runner
and `dynamo.router` adapter supplied by Dynamo. AISimulate keeps its existing
`aisimulate` wheel and `aisimulate-core` crate; it does not build another Python
package or link Dynamo into its native extension.

The YAML below sets `traffic.load.agentic_profile.duration_seconds: 3600`.
This controls simulated time for issuing profile requests, not CPU wall time.
See [continuous agentic profiles](agentic-profile.md) for the detailed contract.
Functional simulation does not establish hardware performance accuracy.

## 1. Install matching source builds

The existing published Dynamo packages use AISimulate 0.12 and do not provide the
conversation/profile adapter used here. A generic AISimulate 0.13 nightly is also
not evidence that it contains both required features. Use the exact source pair:
the Dynamo checkout pins the AISimulate core revision, and the same revision must
supply the Python package. This source-built integration is not a published
release.

You need Python 3.12, `uv`, Rust 1.96.1, and Dynamo's CPU build prerequisites,
including a C/C++ compiler and linker, `pkg-config`, OpenSSL development headers,
CMake and protobuf compiler. The build uses Dynamo's existing `ai-dynamo` and
`ai-dynamo-runtime` distributions. No running Dynamo service or GPU is needed.

```bash
mkdir -p /tmp/agentx-quickstart
rustup toolchain install 1.96.1 --profile minimal
uv venv --python 3.12 /tmp/agentx-quickstart/venv
uv pip install --python /tmp/agentx-quickstart/venv/bin/python 'maturin>=1.12,<2' patchelf

git clone https://github.com/ai-dynamo/dynamo.git /tmp/agentx-quickstart/dynamo
git -C /tmp/agentx-quickstart/dynamo checkout --detach origin/harrli/aic-1817-existing-adapter
agentx_core_rev=$(/tmp/agentx-quickstart/venv/bin/python -c \
  'import pathlib,tomllib; print(tomllib.loads(pathlib.Path("/tmp/agentx-quickstart/dynamo/Cargo.toml").read_text())["workspace"]["dependencies"]["aisimulate-core"]["rev"])')
git clone https://github.com/ai-dynamo/aisimulate.git /tmp/agentx-quickstart/aisimulate
git -C /tmp/agentx-quickstart/aisimulate checkout --detach "$agentx_core_rev"
cd /tmp/agentx-quickstart/aisimulate/python/aisimulate
RUSTUP_TOOLCHAIN=1.96.1 /tmp/agentx-quickstart/venv/bin/maturin build \
  --locked --profile dev --out /tmp/agentx-quickstart/wheels
uv pip install --python /tmp/agentx-quickstart/venv/bin/python \
  /tmp/agentx-quickstart/wheels/aisimulate-*.whl
cd /tmp/agentx-quickstart/dynamo/lib/bindings/python
RUSTUP_TOOLCHAIN=1.96.1 /tmp/agentx-quickstart/venv/bin/maturin build \
  --locked --profile dev --no-default-features --features aic-forward-pass \
  --out /tmp/agentx-quickstart/wheels
uv pip install --python /tmp/agentx-quickstart/venv/bin/python \
  /tmp/agentx-quickstart/wheels/ai_dynamo_runtime-*.whl \
  /tmp/agentx-quickstart/dynamo
uv pip check --python /tmp/agentx-quickstart/venv/bin/python
git -C /tmp/agentx-quickstart/dynamo rev-parse HEAD
git -C /tmp/agentx-quickstart/aisimulate rev-parse HEAD
```

Keep both source revisions with the results. Rust, Python and container dependency
checks require the same AISimulate version and immutable Rust source. The runner
also checks the installed Python version against the compiled native core and
replay API before executing the configured policy. This does not use local Cargo
patches, package overrides or a separate policy wheel. Dynamo #15149 is not a
prerequisite; the existing native selector and affinity APIs are reused.
These commands use development builds. Their host CPU execution speed is not a
release-build benchmark; modeled GPU timing comes from the configured timing model.
Internet access is needed for initial dependencies, trace download and model
metadata. Simulation itself runs offline on your CPU.

## 2. Download a reproducible Weka workload

The input is SemiAnalysis's
[Weka trace dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/tree/8fecd2fc56694469f758f0afbbb6335ad3043740),
original file `traces.jsonl`, revision
`8fecd2fc56694469f758f0afbbb6335ad3043740`. Its upstream dataset card declares
Apache-2.0. Download the card alongside the data to retain its source information.

```bash
mkdir -p /tmp/agentx-quickstart
uv tool run --from huggingface_hub hf download \
  semianalysisai/cc-traces-weka-062126-256k traces.jsonl README.md \
  --repo-type dataset --revision 8fecd2fc56694469f758f0afbbb6335ad3043740 \
  --local-dir /tmp/agentx-quickstart
cp /tmp/agentx-quickstart/README.md /tmp/agentx-quickstart/UPSTREAM_DATASET_CARD.md
head -n 2 /tmp/agentx-quickstart/traces.jsonl \
  > /tmp/agentx-quickstart/plays-0000-0001.jsonl
```

The full download is about 569 MB. Each JSONL line is a complete source play;
taking the first two lines preserves complete dependency trees, timestamps,
lengths, and prefix hashes. This subset contains two plays and 162 model
requests, and occupies 1,198,197 bytes. Its SHA-256 is
`e3a34f0617457a004694be52885d58748b998b6d3c22cf344ff6572a78757d5a`.
Verify the downloaded subset before running:

```bash
/tmp/agentx-quickstart/venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
trace = Path("/tmp/agentx-quickstart/plays-0000-0001.jsonl")
assert hashlib.sha256(trace.read_bytes()).hexdigest() == "e3a34f0617457a004694be52885d58748b998b6d3c22cf344ff6572a78757d5a"
print("Weka subset verified")
PY
```

The original trace names Claude models. This example projects its workload onto
`Qwen/Qwen3-4B-Instruct-2507`; it does not reproduce Claude performance. The
selected requests require up to 255,034 input-plus-output tokens, so the target
uses a 262,144-token context window.

## 3. Configure the model, GPUs, workers, and traffic

Save the following as `/tmp/agentx-quickstart/agentx.yaml`:

```yaml
traffic:
  source:
    type: trace
    format: weka
    block_size: 64
    paths: [/tmp/agentx-quickstart/plays-0000-0001.jsonl]
  load:
    type: trace_timestamps
    agentic_lanes: 2
    agentic_snapshot: {seed: 42}
    agentic_warmup: true
    agentic_profile:
      duration_seconds: 3600
router:
  policy: kv_router
  affinity:
    mode: sibling_group  # or session
    ttl_seconds: 3600
engine:
  mode: disaggregated
  model: Qwen/Qwen3-4B-Instruct-2507
  hardware: h200_sxm
  backend: vllm
  backend_version: 0.24.0
  context_length: 262144
  kv_transfer:
    bandwidth_gb_per_second: 400
    timing_mode: destination_missing
  workers:
    prefill:
      parallelism: {replicas: 2, tensor: 2, pipeline: 1, attention_data: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 64}
      kv_cache:
        block_size: 64
        prefix_caching: true
        capacity: {type: default, memory_fraction: 0.9}
      timing: {type: default}
    decode:
      parallelism: {replicas: 4, tensor: 1, pipeline: 1, attention_data: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      kv_cache:
        block_size: 64
        prefix_caching: true
        capacity: {type: default, memory_fraction: 0.9}
      timing: {type: default}
execution:
  resources:
    memory_limit_gb: 4
```

`router.policy` selects native Dynamo KV-aware worker selection. `affinity.mode`
independently selects the binding key: `session` binds each conversation, while
`sibling_group` binds children of the same parent conversation together. Parents
keep their own keys, and separate plays cannot share an affinity key. Bindings
include the worker and attention-DP rank in each routing pool. The integration uses
native hard affinity and commits a binding only after the engine accepts dispatch.
The TTL starts when the last active request releases its lease, using simulated
time; supported TTL values are 1 through 31,536,000 seconds, including fractions.

YAML with `router` automatically selects the existing `dynamo` stack and its
`dynamo.router` adapter. Explicit `--stack dynamo` selects the same integration.
An incompatible explicit stack, unavailable Dynamo package, mismatched native
core or invalid routing option fails with an error. With no routing section and
no explicit stack, the engine default is unchanged.

The trace's embedded hash blocks contain 64 tokens. The explicit
`traffic.source.block_size: 64` keeps host resource inspection aligned with
those blocks; it is separate from the worker KV-cache block setting.
Use a host with at least 5 GB of available RAM: the example allows a 4 GB
execution-process budget and keeps the default 1 GB host reserve. Initial trace
materialization is estimated at about 2.81 GB. The full profile has no qualified
static peak-memory bound because recycled plays retain lifecycle evidence.
The CLI runs it under live resource supervision and can stop with
`resource_limited` if that budget is exhausted. Increasing the duration or lane
count does not guarantee completion within the same budget.

The GPU count comes from the worker configuration:

| Role | Workers (`replicas`) | GPUs per worker (`tensor`, with other dimensions set to 1) | Total GPUs |
| --- | --- | --- | --- |
| Prefill | 2 | 2 | 4 |
| Decode | 4 | 1 | 4 |
| Total | 6 | | 8 |

`agentic_lanes: 2` means two concurrent play instances, independently of worker
or GPU counts. Here the initial lanes select each of the two source plays once.
`seed: 42` makes snapshot selection reproducible for the same input and sampling
version. Requests before each initial snapshot boundary become history; profile
measurement starts with the remaining suffix. When a play and its descendants
reach their client terminal states, its lane takes the next play from the shared
corpus cursor, wrapping after the last source play. Replacement plays start at
turn zero with fresh play, request, conversation, and cache identities.

`duration_seconds: 3600` starts at the preparation barrier. Until that deadline,
lanes can recycle repeatedly through the two-play corpus. The default idle guards
cap idle waits at 300 seconds per tree and 10 seconds across the client workload,
while preserving dependencies and relative delays.

Default timing uses the AIC timing provider; default KV capacity is derived
from the model and hardware at the selected memory fraction. The 400 GB/s KV
transfer bandwidth is an example assumption, not a measured link speed.

## 4. Run the simulation

```bash
/tmp/agentx-quickstart/venv/bin/aisimulate predict \
  --config /tmp/agentx-quickstart/agentx.yaml \
  --capture-per-request \
  --output-dir /tmp/agentx-quickstart/output \
  --format json
```

For another run, choose a new output directory or add `--overwrite` to replace
the previous output.

To change the admission window to 600 simulated seconds without editing the
YAML, use a CLI override:

```bash
/tmp/agentx-quickstart/venv/bin/aisimulate predict \
  --config /tmp/agentx-quickstart/agentx.yaml \
  --set traffic.load.agentic_profile.duration_seconds=600 \
  --capture-per-request \
  --output-dir /tmp/agentx-quickstart/output-600s \
  --format json
```

With warmup enabled, the run has three stages:

1. **Primer:** feed each live conversation's last historical full input into
   the simulated engine, requesting one output token. This populates native KV
   cache using the same play identity that the measured requests will use.
2. **Warmup:** run ten requests per lane, repeating its primer inputs in
   deterministic order. A lane without history uses its earliest retained
   request. Each warmup produces one output token.
3. **Profile:** after preparation succeeds and both worker pools and their KV
   transfers settle, resume the saved suffix. Preserve the caches, and start
   measurement and the admission clock at this barrier. Profile requests use
   their original planned input and output lengths. Recycle lanes into new
   plays until the configured admission deadline.

At the deadline, stop issuing requests and creating replacement plays.
Already-issued requests have a default 30-second response grace period. Then
cancel remaining client requests and allow up to 10 seconds for cancellation
acknowledgements. Client completion does not guarantee that all simulated server
work has settled; the report records any remaining server work separately.

Historical, primer, and warmup requests do not count as measured requests.
Preparation can improve reuse, but routing, capacity, and eviction still affect
actual cache hits. See [warmup inputs, outputs, and barrier semantics](agentic-warmup.md)
for the detailed contract and qualification boundaries.

## 5. Read the results

| File under `/tmp/agentx-quickstart/output` | Contents |
| --- | --- |
| `prediction.json` | Aggregate predictions, snapshot information, preparation/barrier audit, profile duration/cutoff accounting, and play outcomes |
| `requests.jsonl` | Measured profile requests, including per-request timing and identity |

Check `dynamo_policy.native_policy: true` and
`dynamo_policy.routing_provider: dynamo.DefaultWorkerSelector` to identify the
native implementation actually called; retain the source revisions recorded at
installation alongside the report. `dynamo_policy.decisions` records request/group, pool, worker, DP rank and
native cache overlap; `physical_kv_events` counts actual engine cache events fed
to the native index. Compare decisions with each request's `routing_history`.
Worker selection alone does not prove physical cache reuse.
Detailed decisions are retained only with `--capture-per-request`; otherwise
`decision_count`, `decisions_by_role`, and physical KV event counts remain
available, with an empty `decisions` array.

Check `agentic_phases` for preparation success and barrier state, and
`agentic_play_outcomes` for completed or incomplete plays. Check `agentic_profile`
for the resolved duration and grace settings, admission cutoff, recycled play
counts, cancellations, never-issued requests, and unsettled server work.

Throughput uses the observed successful-request cohort, including successful
responses during grace. Its interval runs from the earliest arrival in that
cohort to the latest successful response; it can be shorter or longer than the
configured admission duration. Preparation and canceled requests do not extend
that interval. CPU wall time is separate from both simulated durations.

Actual prefix reuse is recorded by `first_admission_prefix_cache_reused_ratio`;
router overlap is not a substitute for cache hits. The 162 source requests
include initial snapshot history, and the corpus can be replayed repeatedly as
lanes recycle, so this is not the expected measured request count.

Acceptance results for the existing Dynamo adapter are recorded with the exact
source pair above. Compare native routing decisions with actual worker/DP
placement and physical cache reuse; request counts and cache percentages can
vary with native selection and are not fixed expected values or hardware
accuracy claims.

To compare against a cold snapshot, repeat the command with a separate output
directory and add:

```bash
--set traffic.load.agentic_warmup=false
```

Keep the duration, seed, trace, lanes, and deployment unchanged for this
comparison. Without warmup, the admission clock starts at simulation start.

To run the initial snapshot suffixes once without recycling, remove
`agentic_profile` from the YAML. To replay those finite plays from turn zero,
also remove `agentic_snapshot` and `agentic_warmup`. Continuous profiles require
seeded snapshots for the initial lanes.

## Capability boundaries

This integration supports offline vLLM/SGLang aggregated and P/D simulation,
including attention DP, seeded snapshots, saved-frontier warmup and duration
profiles. Workload and snapshot seeds remain supported. Conversation/profile
replay requires static pools and cannot be combined with Planner scaling,
online execution, or unsupported backends. Existing Dynamo paths keep their own
capabilities; configuration combinations outside the selected path fail explicitly.

The KV index receives physical simulated engine store/remove events. Cache reuse
is still simulated behavior, not measured GPU performance. The existing
saved-frontier warmup and snapshot sampling differ from the live AgentX harness;
this guide does not claim full recipe or bitwise parity.
