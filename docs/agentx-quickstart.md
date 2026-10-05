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

**Integration status:** the complete routing example requires the existing
Dynamo replay runner and router adapter delivered by
[Dynamo #15240](https://github.com/ai-dynamo/dynamo/pull/15240) (C), built against
the matching AISimulate source. A and B alone do not enable this example.

The call chain is YAML → the existing `dynamo` runner/config-adapter entry points
→ Dynamo's Rust replay composition → the shared AISimulate executor → Dynamo's
native worker selection and session affinity → simulated engines and reports.
AISimulate exposes generic placement lifecycle hooks and conversation lineage;
its engine runner does not import Dynamo or bridge policy calls through Python.

The YAML below sets `traffic.load.agentic_profile.duration_seconds: 3600`.
This controls simulated admission time, not CPU wall time. See
[continuous agentic profiles](agentic-profile.md) for its detailed contract.
Functional simulation does not establish hardware performance accuracy.

## Delivery status and merge order

[AISimulate #307](https://github.com/ai-dynamo/aisimulate/pull/307) has merged as
`a9c358c5cd0f053054800a72137a4db5f95a1671`. The remaining order is:

1. **A — [Dynamo #15631](https://github.com/ai-dynamo/dynamo/pull/15631):** expose
   the native policy lifecycle needed by simulation consumers, without upgrading
   Dynamo's AISimulate dependency.
2. **B — [AISimulate #306](https://github.com/ai-dynamo/aisimulate/pull/306):**
   provide generic dispatch/time hooks, conversation lineage and the shared replay
   executor, and select the existing Dynamo plugin from routing configuration.
   AISimulate does not depend on or package Dynamo policy code.
3. **C — [Dynamo #15240](https://github.com/ai-dynamo/dynamo/pull/15240):** wire
   those APIs into the existing native adapter, align Rust/Python/container
   dependencies, and qualify the full installed CLI including duration and routing.

[#15149](https://github.com/ai-dynamo/dynamo/pull/15149) and
[#15625](https://github.com/ai-dynamo/dynamo/pull/15625) are closed. Source merge,
published matching packages, required CI/review, and installed CLI qualification
are separate gates. No published version is claimed here to contain the complete
A/B/C integration. Earlier A/B Python-bridge results do not validate this path.

## 1. Install a matching C development build

Until C is qualified and matching packages are published, this is a development
installation procedure. Use a C revision that pins the new B API. Installing A's
runtime wheel alone or an arbitrary nightly does not supply the required runner.
The three distributions below already exist: `aisimulate`, `ai-dynamo-runtime`,
and `ai-dynamo`. No additional policy package or Cargo lockfile is needed.

You need Python 3.12, `uv`, the Rust toolchains required by the checked-out
repositories, and Dynamo's CPU build prerequisites (C/C++ compiler and linker,
`pkg-config`, OpenSSL development headers, CMake and protobuf compiler). No running
Dynamo service, model weights, or physical GPU is required.

The commands capture the current C source and build the exact AISimulate revision
it pins. Retain both immutable revisions and wheel hashes with the run results.
A version number alone does not identify an unreleased source build.

```bash
mkdir -p /tmp/agentx-quickstart
uv venv --python 3.12 /tmp/agentx-quickstart/venv
uv pip install --python /tmp/agentx-quickstart/venv/bin/python 'maturin>=1.12,<2' patchelf

git clone https://github.com/ai-dynamo/dynamo.git /tmp/agentx-quickstart/dynamo
git -C /tmp/agentx-quickstart/dynamo fetch origin pull/15240/head
git -C /tmp/agentx-quickstart/dynamo checkout --detach FETCH_HEAD
agentx_dynamo_rev=$(git -C /tmp/agentx-quickstart/dynamo rev-parse HEAD)
agentx_aisim_rev=$(/tmp/agentx-quickstart/venv/bin/python - <<'PYTHON'
import tomllib
from pathlib import Path
manifest = tomllib.loads(Path("/tmp/agentx-quickstart/dynamo/Cargo.toml").read_text())
print(manifest["workspace"]["dependencies"]["aisimulate-core"]["rev"])
PYTHON
)
git clone https://github.com/ai-dynamo/aisimulate.git /tmp/agentx-quickstart/aisimulate
git -C /tmp/agentx-quickstart/aisimulate checkout --detach "$agentx_aisim_rev"

cd /tmp/agentx-quickstart/aisimulate/python/aisimulate
/tmp/agentx-quickstart/venv/bin/maturin build --locked --release \
  --interpreter /tmp/agentx-quickstart/venv/bin/python \
  --out /tmp/agentx-quickstart/wheels
cd /tmp/agentx-quickstart/dynamo/lib/bindings/python
/tmp/agentx-quickstart/venv/bin/maturin build --locked --release \
  --features ais-forward-pass \
  --interpreter /tmp/agentx-quickstart/venv/bin/python \
  --out /tmp/agentx-quickstart/wheels
uv build --wheel --out-dir /tmp/agentx-quickstart/wheels /tmp/agentx-quickstart/dynamo
uv pip install --python /tmp/agentx-quickstart/venv/bin/python \
  /tmp/agentx-quickstart/wheels/ai_dynamo_runtime-*.whl \
  /tmp/agentx-quickstart/wheels/ai_dynamo-*.whl \
  /tmp/agentx-quickstart/wheels/aisimulate-*.whl
uv pip check --python /tmp/agentx-quickstart/venv/bin/python
cd /tmp/agentx-quickstart
printf '%s\n' "$agentx_dynamo_rev" "$agentx_aisim_rev" > source-revisions.txt
sha256sum wheels/*.whl > wheel-sha256.txt
/tmp/agentx-quickstart/venv/bin/python - <<'PYTHON'
from aisimulate.stack import resolve_runner_factory
from aisimulate.config_adapter import resolve_config_adapters
print(resolve_runner_factory("dynamo").capabilities())
print(resolve_config_adapters(["dynamo.router"])["dynamo.router"].name)
PYTHON
```

Dependency resolution remains strict: the Rust core pin, installed AISimulate
Python package and container source wheel must agree. The plugin also checks its
replay API compatibility. Do not bypass these checks with an editable install,
path override or dependency-ignore flag. The full CLI tests and 600/3,600-second
runs must use these same installed artifacts before this source pair is described
as qualified. The checks above verify discovery only; they are not end-to-end
routing acceptance.

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

YAML with `router` automatically selects the installed `dynamo` stack. Explicit
`--stack dynamo` selects the same integration; explicit `--stack engine` retains
the engine choice and rejects the unsupported router configuration. Without a
routing section, the default remains `engine`. A missing plugin, incompatible
core/API version, or unsupported affinity configuration must produce an error;
it must never silently use round-robin.

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

C's installed validation must demonstrate that the native Dynamo policy actually
ran, that conversation/sibling bindings retain the selected worker and DP rank
in each active pool, and that these decisions agree with per-request
`routing_history`. Inspect the router evidence exported by that qualified C
revision. Worker selection or an overlap score alone does not prove cache reuse.

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

The required C qualification covers vLLM/SGLang, aggregated/P-D, session/sibling
affinity, duration, actual worker/DP bindings, and physical cache reuse.
Cache-disabled controls must report zero actual reuse. Run the exact YAML and
two-play subset at both 600 and 3,600 seconds, record completed requests and
admission cutoffs, and check cancellations and unsettled server work. Preparation
requests must be excluded from the measured cohort.

The earlier Python-bridge implementation's counts are historical and have been
removed from this walkthrough; C must supply fresh results for its native adapter
path. Until then, this full example is pending C qualification, not a verified
installation or a hardware-accuracy measurement.

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
