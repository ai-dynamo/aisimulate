<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Start an AgentX simulation

This development walkthrough replays a Weka agentic workload on a simulated
disaggregated deployment: two prefill workers and four decode workers, using
eight H200 GPUs in total. It prepares KV caches from seeded snapshots, then measures the
remaining requests and recycles completed plays to keep two client lanes occupied
for a 3,600-second simulated admission window. The simulator runs offline on your
CPU; you do not need to allocate those GPUs or download model weights.

AISimulate owns the complete simulation: YAML → the built-in `engine` runner
and `engine.router` config adapter → AISimulate's native replay composition →
Dynamo's production worker selector and affinity policy → simulated engines
and reports. The policy runs inside the AISimulate process. No Dynamo service,
Dynamo Python package, or Dynamo replay runner is required.

The YAML below sets `traffic.load.agentic_profile.duration_seconds: 3600`.
This controls simulated admission time, not CPU wall time. See
[agentic load controls](../workloads.md#agentic-load-controls) for every option and
[continuous agentic profiles](continuous-profiles.md) for its detailed contract.
Functional simulation does not establish hardware performance accuracy.

## Source and package status

[Dynamo #15631](https://github.com/ai-dynamo/dynamo/pull/15631) is merged at
`094aaca91941421bb0dcb645b0f700a42d6de3ef`. The native policy dependency is
pinned to `f24c59b936ba23c933cc35d2f8c7f9ff71f54c3d`, the merged revision of
[Dynamo #16490](https://github.com/ai-dynamo/dynamo/pull/16490) on Dynamo main.
It exposes an admission barrier so AISimulate can drive the production
scheduler at each simulation timestamp.
[AISimulate #306](https://github.com/ai-dynamo/aisimulate/pull/306) provides
simulation, CLI configuration, and reports.
[Dynamo #15240](https://github.com/ai-dynamo/dynamo/pull/15240) updates the
optional Dynamo integration to consume AISimulate's interfaces; it is not a
prerequisite for this example.

The development wheel builds its native routing extension against pinned
Dynamo source without importing Dynamo's mocker or LLM crate. The extension
calls Dynamo's `LocalScheduler` for admission, queueing, selection and booking,
and `SessionAffinity` for affinity acquisition and binding. AISimulate supplies
simulation time, requests, KV observations and lifecycle events. The published
Rust core remains independent of Dynamo; no Dynamo release or registry
publication is required. No published AISimulate version is claimed here to
contain this new path.

## 1. Install the standalone AISimulate development wheel

You need Python 3.12, `uv`, the AISimulate Rust toolchain, a C/C++ compiler,
`pkg-config`, OpenSSL development headers and a protobuf compiler. Build and
install AISimulate only. See [source installation](../../getting-started/installation.md#use-current-source)
for platform details. Record the resolved source revision and wheel hash;
PR branch heads can change.

```bash
mkdir -p /tmp/agentx-quickstart
uv venv --python 3.12 /tmp/agentx-quickstart/venv
uv pip install --python /tmp/agentx-quickstart/venv/bin/python 'maturin>=1.12,<2' patchelf

git clone https://github.com/ai-dynamo/aisimulate.git /tmp/agentx-quickstart/aisimulate
git -C /tmp/agentx-quickstart/aisimulate fetch origin pull/306/head
git -C /tmp/agentx-quickstart/aisimulate checkout --detach FETCH_HEAD
git -C /tmp/agentx-quickstart/aisimulate rev-parse HEAD > /tmp/agentx-quickstart/source-revision.txt
cd /tmp/agentx-quickstart/aisimulate/python/aisimulate
/tmp/agentx-quickstart/venv/bin/maturin build --locked --release \
  --interpreter /tmp/agentx-quickstart/venv/bin/python \
  --out /tmp/agentx-quickstart/wheels
uv pip install --python /tmp/agentx-quickstart/venv/bin/python \
  /tmp/agentx-quickstart/wheels/aisimulate-*.whl
uv pip check --python /tmp/agentx-quickstart/venv/bin/python
cd /tmp/agentx-quickstart
sha256sum wheels/*.whl > wheel-sha256.txt
/tmp/agentx-quickstart/venv/bin/python - <<'PYTHON'
from importlib import metadata
from aisimulate.stack import resolve_runner_factory
from aisimulate.config_adapter import resolve_config_adapters
for name in ("ai-dynamo", "ai-dynamo-runtime"):
    try:
        metadata.distribution(name)
    except metadata.PackageNotFoundError:
        continue
    raise RuntimeError(f"Standalone qualification requires an environment without {name}")
print(resolve_runner_factory("engine").capabilities())
print(resolve_config_adapters(["engine.router"])["engine.router"].name)
PYTHON
```

The final checks verify installation and adapter discovery. Run the prediction
below to exercise native routing and inspect `routing_policy` in its report.
Do not substitute the older Dynamo source-pair qualification for validation of
this standalone wheel.

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

Save the following complete example as `/tmp/agentx-quickstart/agentx.yaml`:

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

`router.policy` selects native Dynamo worker selection. Aggregated and prefill
selection use KV/cache credit; native plain decode selection is load-based.
Affinity binds the worker/DP actually selected in each pool. `affinity.mode`
independently selects the binding key: `session` binds each conversation, while
`sibling_group` binds children of the same parent conversation together. Parents
keep their own keys, and separate plays cannot share an affinity key. Bindings
include the worker and attention-DP rank in each routing pool. The integration uses
native hard affinity and commits a binding only after the engine accepts dispatch.
The TTL starts when the last active request releases its lease, using simulated
time; supported TTL values are 1 through 31,536,000 seconds, including fractions.

The default `engine` stack accepts `router` through its built-in adapter.
Explicit `--stack engine` uses the same path. `--stack dynamo` is an optional
integration for Dynamo-specific capabilities such as Planner. A native build
without the required routing feature or an unsupported configuration fails
explicitly; it never silently substitutes round-robin.

The trace's embedded hash blocks contain 64 tokens. The explicit
`traffic.source.block_size: 64` keeps host resource inspection aligned with
those blocks; it is separate from the worker KV-cache block setting.
Allow headroom for the 4 GB execution-process budget, the default 1 GB host
reserve, and the CLI coordinator. Use the verified two-play subset and block size;
the complete Weka corpus needs much more memory. The full profile has no qualified
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

Default timing uses the AISimulate timing provider; default KV capacity is derived
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
actual cache hits. See [agentic warmup](warmup.md) for the detailed contract and
qualification boundaries.

## 5. Read the results

| File under `/tmp/agentx-quickstart/output` | Contents |
| --- | --- |
| `prediction.json` | Aggregate predictions, snapshot information, preparation/barrier audit, profile duration/cutoff accounting, and play outcomes |
| `requests.jsonl` | Measured profile requests, including per-request timing and identity |

`routing_policy.roles` records each pool's `native_policy`
(`dynamo.DefaultWorkerSelector`) and `dynamo_revision`. Its `decisions` contain
`request_id`, `group_key`, `worker_id`, `dp_rank`, `binding_reused` and
`overlap_blocks`; match measured decisions to each request's `routing_history`.
`decision_count` also includes preparation requests; `post_dispatch_checks` counts
accepted dispatches and `dispatch_aborts` counts failed dispatches.
`physical_kv_events` counts simulated engine events delivered to that pool's
router index. Plain P/D decode selection is load-based and reports zero such
events; this does not imply zero destination-cache reuse. Compare
`destination_activated_ms - destination_reserved_ms` between `destination_missing`
and `full_prompt` transfer timing, with a cache-disabled control. Imported KV
reuse can leave `decode_reused_input_tokens` at zero on this path.
Worker selection or an overlap score alone does not prove cache reuse.

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

The [historical source qualification](../../../benchmarks/evidence/accuracy/replay-evidence.md#agentx-native-routing-source-qualification)
records earlier Dynamo-runner results and their immutable revisions. Those
measurements do not qualify this AISimulate-owned path or hardware accuracy.

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

The standalone path supports offline vLLM/SGLang aggregated and P/D simulation,
including attention DP, seeded snapshots, saved-frontier warmup and duration
profiles. `predict` and `recommend` can use a fixed routing configuration;
searching router parameters is unsupported. Conversation affinity requires
static workers, device-only KV cache, default scoring, and speculative decoding
disabled. Planner scaling, custom prefill estimators or policy classes, authored
DP pins, online execution and unsupported backends are rejected with affinity.
Canonical selector seeds require a separate `replay-bench` build. Workload/snapshot seeds remain supported. The optional Dynamo
stack has separate integration and compatibility requirements.

The KV index receives physical simulated engine store/remove events. Cache reuse
is still simulated behavior, not measured GPU performance. The existing
saved-frontier warmup and snapshot sampling differ from the live AgentX harness;
this guide does not claim full recipe or bitwise parity.
