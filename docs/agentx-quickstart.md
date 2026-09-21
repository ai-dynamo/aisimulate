<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Start an AgentX simulation

This walkthrough replays a Weka agentic workload on a simulated disaggregated
deployment: two prefill workers and four decode workers, using eight H200 GPUs
in total. It prepares KV caches from seeded snapshots, then measures the
remaining requests. The simulator runs offline on your CPU; you do not need to
allocate those GPUs or download model weights.

Use a source checkout containing [PR #235](https://github.com/ai-dynamo/aisimulate/pull/235).
Run the commands below from the repository root. This path is experimental and
qualified for functional behavior, not hardware performance accuracy.

To keep lanes occupied for a fixed admission duration, add
`traffic.load.agentic_profile: {duration_seconds: 3600}`. See
[continuous agentic profiles](agentic-profile.md) for idle guards, response
grace periods, and the distinction between admission and observation duration.

## 1. Install from source

You need Python 3.11–3.13, `uv`, Rust/Cargo, and a C/C++ compiler and linker.
See [source installation](installation.md#use-current-source) for platform details.

```bash
uv sync --project python/aisimulate --extra dev
```

This installs the Python environment and builds the native simulator. The
commands below use that environment explicitly. Internet access is needed for
initial dependency installation, the trace download, and model metadata.

## 2. Download a reproducible Weka workload

The input is SemiAnalysis's
[Weka trace dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/tree/8fecd2fc56694469f758f0afbbb6335ad3043740),
original file `traces.jsonl`, revision
`8fecd2fc56694469f758f0afbbb6335ad3043740`. Its upstream dataset card declares
Apache-2.0. Download the card alongside the data to retain its source information.

```bash
mkdir -p /tmp/agentx-quickstart
curl --fail --location --retry 3 \
  https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/resolve/8fecd2fc56694469f758f0afbbb6335ad3043740/traces.jsonl \
  --output /tmp/agentx-quickstart/traces.jsonl
curl --fail --location --retry 3 \
  https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/resolve/8fecd2fc56694469f758f0afbbb6335ad3043740/README.md \
  --output /tmp/agentx-quickstart/UPSTREAM_DATASET_CARD.md
head -n 12 /tmp/agentx-quickstart/traces.jsonl \
  > /tmp/agentx-quickstart/plays-0000-0011.jsonl
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

Save the following as `/tmp/agentx-quickstart/disagg.yaml`:

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

The GPU count comes from the worker configuration:

| Role | Workers (`replicas`) | GPUs per worker (`tensor`, with other dimensions set to 1) | Total GPUs |
| --- | --- | --- | --- |
| Prefill | 2 | 2 | 4 |
| Decode | 4 | 1 | 4 |
| Total | 6 | | 8 |

`agentic_lanes: 12` means 12 concurrent play instances, independently of worker
or GPU counts. Here the initial lanes select each of the 12 source plays once.
`seed: 42` makes snapshot selection reproducible for the same input and sampling
version. Requests before each snapshot boundary become history; only the
remaining suffix is measured.

Default timing uses the AIC timing provider; default KV capacity is derived
from the model and hardware at the selected memory fraction. The 400 GB/s KV
transfer bandwidth is an example assumption, not a measured link speed.

## 4. Run the simulation

```bash
python/aisimulate/.venv/bin/python -m aisimulate predict \
  --stack engine \
  --config /tmp/agentx-quickstart/disagg.yaml \
  --capture-per-request \
  --output-dir /tmp/agentx-quickstart/output \
  --format json
```

For another run, choose a new output directory or add `--overwrite` to replace
the previous output.

With warmup enabled, the run has three stages:

1. **Primer:** feed each live conversation's last historical full input into
   the simulated engine, requesting one output token. This populates native KV
   cache using the same play identity that the measured requests will use.
2. **Warmup:** run ten requests per lane, repeating its primer inputs in
   deterministic order. A lane without history uses its earliest retained
   request. Each warmup produces one output token.
3. **Profile:** after preparation succeeds and both worker pools and their KV
   transfers settle, resume the saved suffix. Preserve the caches, and start
   measurement at this barrier. Profile requests use their original planned
   input and output lengths.

Historical, primer, and warmup requests do not count as measured requests.
Preparation can improve reuse, but routing, capacity, and eviction still affect
actual cache hits. See [warmup inputs, outputs, and barrier semantics](agentic-warmup.md)
for the detailed contract and qualification boundaries.

## 5. Read the results

| File under `/tmp/agentx-quickstart/output` | Contents |
| --- | --- |
| `prediction.json` | Aggregate predictions, snapshot information, preparation/barrier audit, and play outcomes |
| `requests.jsonl` | Measured profile requests, including per-request timing and identity |

Check `agentic_phases` for preparation success and barrier state, and
`agentic_play_outcomes` for completed or incomplete plays. Actual prefix reuse
is recorded by `first_admission_prefix_cache_reused_ratio`; router overlap is
not a substitute for cache hits. The 1,560 source requests include skipped
history, so they are not the expected measured request count. Host execution
time also differs from simulated profile duration.

To compare against a cold snapshot, repeat the command with a separate output
directory and add:

```bash
--set traffic.load.agentic_warmup=false
```

Keep the seed, trace, lanes, and deployment unchanged for this comparison. To
start at turn zero instead, remove both `agentic_snapshot` and
`agentic_warmup` from the YAML. This walkthrough runs a finite suffix; it does
not automatically recycle plays into a continuous benchmark.
