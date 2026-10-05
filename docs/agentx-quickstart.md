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

**Development source pair, verified October 4, 2026:** this CLI path uses
AISimulate's existing engine runner and Dynamo's native policy from the existing
`ai-dynamo-runtime` wheel. Both PRs are still under review; these source builds
are not an official paired release.

The call chain is YAML → AISimulate's shared replay executor → the optional
`dynamo.llm.NativeReplayPolicy` provider → Dynamo's production `SelectionCore`
and `SessionAffinity` → simulated engines and the normal AISimulate report.
AISimulate supplies real engine KV events and dispatch/terminal callbacks; it
does not implement Dynamo's scoring or affinity algorithms.

The YAML below sets `traffic.load.agentic_profile.duration_seconds: 3600`.
This controls simulated admission time, not CPU wall time. See
[continuous agentic profiles](agentic-profile.md) for its detailed contract.
Functional simulation does not establish hardware performance accuracy.

## Delivery status and merge order

[AISimulate #307](https://github.com/ai-dynamo/aisimulate/pull/307) has merged as
`a9c358c5cd0f053054800a72137a4db5f95a1671`. The remaining order is:

1. **A — [Dynamo #15631](https://github.com/ai-dynamo/dynamo/pull/15631):** expose
   the native policy provider in `ai-dynamo-runtime`, including virtual time and
   affinity lifecycle. This change does not require the new AISimulate API or
   upgrade Dynamo's existing AISimulate dependency.
2. **B — [AISimulate #306](https://github.com/ai-dynamo/aisimulate/pull/306):**
   consume A through its existing Python extension and shared executor. Before
   merging B, qualify the actual merged A revision. Publish matching AISimulate
   Python/Rust packages containing both duration and routing support.
3. **C — [Dynamo #15240](https://github.com/ai-dynamo/dynamo/pull/15240):** reduce
   the existing adapter to a consumer of B, pin the actual published pair across
   Rust, Python and containers, and validate the installed full integration.

[#15149](https://github.com/ai-dynamo/dynamo/pull/15149) is closed.
[#15625](https://github.com/ai-dynamo/dynamo/pull/15625)'s queue helpers are
closed as superseded; they are not the policy provider consumed by AISimulate.
Source merge, published matching packages, required CI/review, and final C
installation are separate gates. Do not substitute an arbitrary nightly based
on its date. This walkthrough needs only A and B, so it does not install the
`ai-dynamo` consumer package or depend on #15240.

## 1. Install the verified development source pair

Use these exact source revisions together:

| Existing distribution | Source revision | Status |
| --- | --- | --- |
| `ai-dynamo-runtime` | `2cee4fe9fb00baf2e0a029ccf4768e31982272d4` (#15631) | Development build, not merged/released |
| `aisimulate` | `a298db2eae5b8c5ead28137c8715e5bcafaa1ae3` (#306, including merged #307) | Development build, not merged/released |

You need Python 3.12, `uv`, Rust 1.96.1 for Dynamo, Rust 1.91.0 for AISimulate,
and Dynamo's CPU build prerequisites,
including a C/C++ compiler and linker, `pkg-config`, OpenSSL development headers,
CMake and protobuf compiler. No running Dynamo service or physical GPU is needed.
The two existing distributions are built normally; no extra policy wheel, binding
crate, Cargo lockfile, editable installation or dependency override is used.

```bash
mkdir -p /tmp/agentx-quickstart
rustup toolchain install 1.96.1 1.91.0 --profile minimal
uv venv --python 3.12 /tmp/agentx-quickstart/venv
uv pip install --python /tmp/agentx-quickstart/venv/bin/python 'maturin>=1.12,<2' patchelf

agentx_dynamo_rev=2cee4fe9fb00baf2e0a029ccf4768e31982272d4
agentx_aisim_rev=a298db2eae5b8c5ead28137c8715e5bcafaa1ae3
git clone https://github.com/ai-dynamo/dynamo.git /tmp/agentx-quickstart/dynamo
git -C /tmp/agentx-quickstart/dynamo checkout --detach "$agentx_dynamo_rev"
git clone https://github.com/ai-dynamo/aisimulate.git /tmp/agentx-quickstart/aisimulate
git -C /tmp/agentx-quickstart/aisimulate checkout --detach "$agentx_aisim_rev"

cd /tmp/agentx-quickstart/dynamo/lib/bindings/python
CARGO_PROFILE_DEV_DEBUG=0 CARGO_BUILD_JOBS=2 RUSTUP_TOOLCHAIN=1.96.1 \
  /tmp/agentx-quickstart/venv/bin/maturin build --locked --profile dev \
  --interpreter /tmp/agentx-quickstart/venv/bin/python \
  --out /tmp/agentx-quickstart/wheels
cd /tmp/agentx-quickstart/aisimulate/python/aisimulate
RUSTUP_TOOLCHAIN=1.91.0 /tmp/agentx-quickstart/venv/bin/maturin build \
  --locked --release --interpreter /tmp/agentx-quickstart/venv/bin/python \
  --out /tmp/agentx-quickstart/wheels
uv pip install --python /tmp/agentx-quickstart/venv/bin/python \
  /tmp/agentx-quickstart/wheels/ai_dynamo_runtime-*.whl \
  /tmp/agentx-quickstart/wheels/aisimulate-*.whl
uv pip check --python /tmp/agentx-quickstart/venv/bin/python
cd /tmp/agentx-quickstart
/tmp/agentx-quickstart/venv/bin/python - <<'PYTHON'
from importlib.metadata import version
from aisimulate import _runtime
from dynamo.llm import NativeReplayPolicy
assert NativeReplayPolicy.contract()["api_version"] == 1
assert _runtime.native_replay_policy_contract()["api_version"] == 1
print("aisimulate", version("aisimulate"))
print("ai-dynamo-runtime", version("ai-dynamo-runtime"))
print(NativeReplayPolicy.contract())
print(_runtime.native_replay_policy_contract())
PYTHON
```

Keep both immutable revisions and wheel hashes with your results. AISimulate
checks its installed Python package against its compiled core; the bridge and
Dynamo provider each require contract API version 1. A version number alone does
not identify an unreleased source build. Dynamo's pre-existing Rust dependency
on its older published AISimulate core remains unchanged in A; the new native
policy provider does not call it. The isolated A wheel has been tested without
an AISimulate Python package installed.

These commands reproduce the tested profiles: a Dynamo development build and an
AISimulate release build. The extensions exchange versioned JSON through Python,
not a shared Rust ABI, so their toolchains can differ. The tested Dynamo build
has a null revision field; its source and wheel hashes identify it. CPU execution
speed is not a hardware-performance measurement. Internet access is needed for
initial dependencies, trace download and model metadata; simulation runs offline.

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

YAML with `router` loads the native provider through the built-in `engine` stack;
`--stack engine` selects this same path explicitly. Without a routing section,
the existing engine default remains unchanged. Missing `ai-dynamo-runtime`, an
incompatible contract/core version, or unsupported configuration produces an
error rather than silently switching to round-robin. Explicit selection of other
installed stacks remains supported; the full `--stack dynamo` consumer is C and
has its own pending qualification.

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
actual cache hits. See [warmup inputs, outputs, and barrier semantics](agentic-warmup.md)
for the detailed contract and qualification boundaries.

## 5. Read the results

| File under `/tmp/agentx-quickstart/output` | Contents |
| --- | --- |
| `prediction.json` | Aggregate predictions, snapshot information, preparation/barrier audit, profile duration/cutoff accounting, and play outcomes |
| `requests.jsonl` | Measured profile requests, including per-request timing and identity |

Inspect `routing_policy.roles` in `prediction.json`. Each active role
(`aggregated`, or `prefill` and `decode`) reports
`native_policy: "dynamo.SelectionCore"`, `decision_count`, `physical_kv_events`,
and optional captured `decisions`. Each decision records the request, group,
role, worker, DP rank and `binding_reused` state. `dynamo_revision` is optional
build provenance; when absent, retain the source/wheel receipt from installation.
Compare these native decisions with each measured request's `routing_history`.
The physical-event count records actual engine KV events supplied to Dynamo's
index. Worker selection or an overlap score alone does not prove cache reuse.
Detailed decisions are retained only with `--capture-per-request`; otherwise
counters remain available and the `decisions` arrays stay empty.

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

The source pair above passes ten installed CLI cases covering vLLM/SGLang,
aggregated/P-D, session/sibling affinity, duration, real worker/DP bindings and
physical cache reuse. Cache-disabled controls report zero actual reuse. The
standalone Dynamo provider additionally passes thirteen installed API tests without
AISimulate installed. These are local functional results, separate from remote
CI, merge approval and publication.

The exact YAML and two-play subset above were also run with this source pair:

| Admission window | Completed requests | Actual prefix reuse | Matched worker/DP routes | Canceled / unsettled |
| --- | --- | --- | --- | --- |
| 600 seconds | 42 | 96.1709% | 84 | 0 / 0 |
| 3,600 seconds | 219 | 95.6497% | 438 | 0 / 0 |

All 128/482 native decisions correspond to measured or preparation requests;
22 preparation requests are excluded from each measured cohort. At 3,600 seconds,
14 group/role pairs keep stable native worker/DP bindings, including two sibling
families with multiple conversations in each routing pool. The physical KV event
counts are 81/302 for prefill and 480/2,740 for decode at 600/3,600 seconds.
Actual reuse was recomputed from engine admission records. A fully cached vLLM
prompt still recomputes its final block, so raw router overlap can exceed actual
reuse by that block. These are observed source-pair results, not fixed expected
values, hardware-accuracy measurements, or acceptance of the pending C consumer.

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

This development source pair supports offline vLLM/SGLang aggregated and P/D
simulation, including attention DP, seeded snapshots, saved-frontier warmup and
duration profiles. The built-in routing path supports offline `predict` with
static pools. Routing with `recommend`, Planner scaling, periodic telemetry,
authored DP pins, custom policy classes, selector seeds, online execution or
unsupported backends is rejected. Workload/snapshot seeds remain supported. Other explicit
stacks retain their own capabilities; C still needs independent compatibility and
installation acceptance. This feature has not yet been published as a paired release.

The KV index receives physical simulated engine store/remove events. Cache reuse
is still simulated behavior, not measured GPU performance. The existing
saved-frontier warmup and snapshot sampling differ from the live AgentX harness;
this guide does not claim full recipe or bitwise parity.
