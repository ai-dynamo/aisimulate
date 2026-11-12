<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance-data collector

The collector measures operations or complete forward passes in a pinned runtime
and publishes data with its execution identity. Op-level timing consumes the
operation tables; whole-forward FPM consumes a separate validated Parquet/metadata
pair. Collection success, data coverage, and serving accuracy are distinct gates.

- [Upgrade a runtime](upgrade.md): scope, runtime proof, smoke, full collection, and delivery.
- [Data format and identity](data-format.md): case keys, physical rows, head axes, provenance, and reuse.
- [Review data](reviewing-data.md): Parquet text diffs, content hashes, and row changes.
- [DeepSeek-V4 attention modules](deepseek-v4.md): CSA/HCA measurement boundaries and calibration.
- [MoE sampling](moe-sampling.md): expert quotas and token assignment.
- [FPM self-service](../fpm-self-service/README.md): collect/import whole-forward data.
- [Benchmark guidelines](#benchmark-guidelines): rules every op, FPM, and end-to-end measurement follows.

Read the repository [collector rules](../../../AGENTS.md#required-collector-first-step)
before changing collector code, cases, or the data contract. The executable
[collector README](../../../python/aisim/collector/README.md) documents current
commands; `framework_manifest.yaml` pins runtimes and `op_backend_catalog.yaml`
maps registry operations to storage families.

Run collector commands from `python/aisim/` unless a command explicitly uses
a repository-root path. Keep campaign checkpoints and raw logs in a run-specific
output directory, separate from published data. The packaged data root is
`python/aisim/src/aisim_core/systems/data/`.

## Benchmark guidelines

These rules apply to every measurement AISim stores or compares: operation,
whole-forward (FPM), end-to-end serving, and evaluation runs, whether collected by
a person, a script, or an agent. The PR template's provenance section and
[AGENTS.md](../../../AGENTS.md#benchmark-guidelines) point here. Record the facts
below with the data, never in `docs/`; run logs and results belong under
`benchmarks/evidence/` or the dataset.

Applying the rules is separate from judging the result. Collection quality,
resolved memory, replay coverage, and measured serving accuracy stay separate
claims.

| Topic | Rule |
| --- | --- |
| Effective settings | Record what the runtime reports, not only what was requested: precision, KV dtype, TP/PP/DP/EP/CP/DCP, graph policy and captured sizes, backend, scheduler bounds, and exact runtime version. Keep requested and observed values in separate fields. An unknown value stays null; never fill in a default. |
| Workload | Record token lengths, concurrency, arrival policy, request count, and seed. State whether the shape is synthetic or derived from a trace, and do not compare points whose workloads differ. |
| Warmup and cache policy | Discard warmup and graph-capture iterations and say how many. State the cache state at measurement start (cold, warmed, prefix-cache on or off, and any forced eviction). A result without its cache state is not comparable. |
| Measurement boundaries | Name the timed region: which events start and stop the clock, and what is excluded (setup, model load, host scheduling, detokenization). Report TTFT, TPOT, ITL, and end-to-end latency with at least mean, median, and P90, plus the sample count. Missing data is missing, never zero, and per-device versus total throughput is stated. |
| Rank handling | Say which rank or ranks were timed. Scheduler and resource limits are per attention-DP rank; do not copy TP values to DEP/TEP or sum them silently. Record how per-rank times combine (for example, the slowest rank) and whether ranks ran in lockstep. |
| GPU-time capture | Record GPU type and count, wall seconds, GPU-seconds (wall x GPUs), and the phase split (setup/load, warmup, measure, teardown). Count failed and retried attempts separately and exclude queue wait. Take it from the Slurm or Kubernetes job record when possible, label it `measured` or `declared`, and leave it null when unknown. |
| Provenance | Keep the raw tool output unchanged and hash-linked, with the command, repository and runtime revision, container image, and dataset revision. Mark each value `measured`, `estimated`, or `proxy`, and list exclusions and failures with reasons. |

Before starting a campaign, check whether matching data already exists and
estimate the GPU budget. Before opening a PR that adds or changes measured data,
confirm each row above is answered in the PR's provenance section or the
artifact it links.
