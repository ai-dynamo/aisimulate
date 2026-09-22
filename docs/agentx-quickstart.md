<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Start an AgentX simulation

This walkthrough replays a Weka agentic workload on a simulated disaggregated
deployment: two prefill workers and four decode workers, using eight H200 GPUs
in total. It prepares KV caches from seeded snapshots, then measures the
remaining requests and recycles completed plays to keep 12 client lanes occupied
for a 3,600-second simulated admission window. The simulator runs offline on your
CPU; you do not need to allocate those GPUs or download model weights.

Use the combined source checkout from
[PR #306](https://github.com/ai-dynamo/aisimulate/pull/306), which includes the
continuous profiles from [PR #307](https://github.com/ai-dynamo/aisimulate/pull/307)
and the optional native Dynamo routing adapter. A checkout containing only #307
does not provide the routing integration used below. Build its matching core and
plugin wheels together; this guide does not assume a published plugin release.
Run the commands below from the repository root. This path is experimental and
qualified for functional behavior, not hardware performance accuracy.

The YAML below sets `traffic.load.agentic_profile.duration_seconds: 3600`.
This controls simulated time for issuing profile requests, not CPU wall time.
See [continuous agentic profiles](agentic-profile.md) for the detailed contract.

## 1. Install from source

You need Python 3.11–3.13, `uv`, Rust 1.96.1, and a C/C++ compiler and linker.
On Linux, install the build tools, `pkg-config`, OpenSSL development headers and
CMake. See [source installation](installation.md#use-current-source) for platform
details. The optional adapter is initially qualified on Linux x86-64.

Build and install both wheels from this same checkout:

```bash
mkdir -p /tmp/agentx-quickstart
rustup toolchain install 1.96.1 --profile minimal
uv run --no-project --with 'maturin>=1.12,<2' python scripts/build_dynamo_policy.py \
  --output-dir /tmp/agentx-quickstart/wheels
uv venv --python 3.12 /tmp/agentx-quickstart/venv
uv pip install --python /tmp/agentx-quickstart/venv/bin/python \
  /tmp/agentx-quickstart/wheels/aisimulate-*.whl \
  /tmp/agentx-quickstart/wheels/aisimulate_dynamo_policy-*.whl
uv pip check --python /tmp/agentx-quickstart/venv/bin/python
```

Use a new wheel output directory when rebuilding. The build records wheel hashes,
source revision and the immutable Dynamo dependency in `wheels/manifest.json`.
The loader checks matching Python/native versions, replay API and core source
hash. Mixing wheels built from different core source trees is rejected, even at
the same package version. No local Cargo override or full `ai-dynamo`
installation is needed.

The adapter imports the existing public APIs from already-merged Dynamo revision
`d9eb42db1168131fdae318eef77255637e4d3495`. Dynamo #15149 is not a prerequisite.
Internet access is needed for initial dependencies, trace download and model
metadata. Simulation itself runs offline on your CPU.

For a container built from the same source and paired dependency contract:

```bash
docker build -f python/aisimulate-dynamo-policy/Dockerfile \
  --build-arg AISIMULATE_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t aisimulate-dynamo-policy .
```

Mount the trace/config/output paths when running the container; its entry point
is `aisimulate`. Container construction builds the same two wheels and checks
their installed contract. It does not consume an unrelated published Dynamo image.

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
head -n 12 /tmp/agentx-quickstart/traces.jsonl \
  > /tmp/agentx-quickstart/plays-0000-0011.jsonl
printf '%s  %s\n' \
  df1b8a8561dad5db8711c1fcfbd93872b52dbee383c023ad6363e30a9fccc891 \
  /tmp/agentx-quickstart/plays-0000-0011.jsonl | sha256sum --check
```

The full download is about 569 MB. Each JSONL line is a complete source play;
taking the first 12 lines preserves complete dependency trees, timestamps,
lengths, and prefix hashes. This subset contains 12 plays and 1,560 model
requests, and occupies 11,956,909 bytes. Its SHA-256 is
`df1b8a8561dad5db8711c1fcfbd93872b52dbee383c023ad6363e30a9fccc891`.

The original trace names Claude models. This example projects its workload onto
`Qwen/Qwen3-4B-Instruct-2507`; it does not reproduce Claude performance. The
selected requests require up to 255,672 input-plus-output tokens, so the target
uses a 262,144-token context window.

## 3. Configure the model, GPUs, workers, and traffic

Save the following as `/tmp/agentx-quickstart/agentx.yaml`:

```yaml
traffic:
  source:
    type: trace
    format: weka
    paths: [/tmp/agentx-quickstart/plays-0000-0011.jsonl]
  load:
    type: trace_timestamps
    agentic_lanes: 12
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
```

`router.policy` selects native Dynamo KV-aware worker selection. `affinity.mode`
independently selects the binding key: `session` binds each conversation, while
`sibling_group` binds children of the same parent conversation together. Parents
keep their own keys, and separate plays cannot share an affinity key. Bindings
include the worker and attention-DP rank in each routing pool. The plugin uses
native hard affinity and commits a binding only after the engine accepts dispatch.
The TTL starts when the last active request releases its lease, using simulated
time; supported TTL values are 1 through 31,536,000 seconds, including fractions.

YAML with `router` automatically selects the installed `dynamo-policy` stack.
Explicit `--stack dynamo-policy` selects the same integration; explicit
`--stack dynamo` keeps the existing full Dynamo provider and its own capabilities.
An incompatible explicit stack, missing plugin or invalid routing option fails
with an error. With no routing section and no explicit stack, the engine default
is unchanged. The two separately versioned Dynamo packages need not be installed
together.

The GPU count comes from the worker configuration:

| Role | Workers (`replicas`) | GPUs per worker (`tensor`, with other dimensions set to 1) | Total GPUs |
| --- | --- | --- | --- |
| Prefill | 2 | 2 | 4 |
| Decode | 4 | 1 | 4 |
| Total | 6 | | 8 |

`agentic_lanes: 12` means 12 concurrent play instances, independently of worker
or GPU counts. Here the initial lanes select each of the 12 source plays once.
`seed: 42` makes snapshot selection reproducible for the same input and sampling
version. Requests before each initial snapshot boundary become history; profile
measurement starts with the remaining suffix. When a play and its descendants
reach their client terminal states, its lane takes the next play from the shared
corpus cursor, wrapping after the last source play. Replacement plays start at
turn zero with fresh play, request, conversation, and cache identities.

`duration_seconds: 3600` starts at the preparation barrier. Until that deadline,
lanes can recycle repeatedly through the 12-play corpus. The default idle guards
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

Check `dynamo_policy.native_policy` (`dynamo.SelectionCore`) and
`dynamo_policy.dynamo_revision` to identify the native implementation actually
called. `dynamo_policy.decisions` records request/group, pool, worker, DP rank and
native cache overlap; `physical_kv_events` counts actual engine cache events fed
to the native index. Compare decisions with each request's `routing_history`.
Worker selection alone does not prove physical cache reuse.
Detailed decisions are retained only with `--capture-per-request`; otherwise
`decision_count`, `decisions_by_role`, and physical KV event counts remain
available, with `decisions_captured: false` and an empty `decisions` array.

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
router overlap is not a substitute for cache hits. The 1,560 source requests
include initial snapshot history, and the corpus can be replayed repeatedly as
lanes recycle, so this is not the expected measured request count.

The configuration above was exercised with matching source-built core/plugin
wheels, the pinned 12-play subset, and default AIC timing for both 600-second
and 3,600-second admission windows. One 3,600-second run recorded 579 successful
responses, two canceled requests, 23 started plays, and 93.64% first-admission
prefix reuse. Its native policy recorded 1,422 P/D decisions, including
preparation, and 17,000 physical KV events. Two server requests remained unsettled
after client cancellation, which the report preserved. These are functional
validation observations, not fixed expected counts or hardware accuracy claims;
native stochastic selection can change the results.

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
profiles. Workload/snapshot seeds remain supported; the existing public Dynamo
SelectionCore uses native stochastic selection, so an explicit selector seed is
rejected. Authored DP pins, custom policy classes, online routing, dynamic scaling,
and routing-aware recommendation are not implemented by this optional adapter.

The KV index receives physical simulated engine store/remove events. Cache reuse
is still simulated behavior, not measured GPU performance. The existing
saved-frontier warmup and snapshot sampling differ from the live AgentX harness;
this guide does not claim full recipe or bitwise parity.
