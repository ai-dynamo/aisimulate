<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Replay

Replay predicts how a deployment processes a workload over virtual time. It
combines requests, worker topology, engine scheduling, cache state, and a
forward-pass timing provider into request timelines and serving metrics.
The [performance model](../perf-model/README.md) estimates individual passes;
the [Sweeper](../sweeper/README.md) repeats replay for candidate configurations.

## Runtime boundaries

| Component | Responsibility |
| --- | --- |
| Workload driver | Requests, sessions, dependencies, arrivals and client terminals |
| Replayer | Virtual clock, event queue, worker lifecycle, handoff and reports |
| Engine | Backend scheduling, admission, GPU cache allocation and pass execution |
| Timing provider | Pass duration for the selected model, hardware and execution identity |
| Placement/scaling composition | Select workers and change desired worker counts |

The native runtime executes in process without a network plane or asynchronous
worker tasks. At each meaningful timestamp it settles engine completions,
worker readiness, transfers and admission, then telemetry and scaling. A
logical worker can contain multiple attention-DP rank schedulers; the grouped
pass completes at the slowest rank's latency.

The built-in `engine` stack provides offline execution with round-robin
placement and fixed worker pools. Dynamo supplies its own Router and Planner
composition through neutral contracts. Importing a Dynamo trace is a workload
operation and does not require the Dynamo execution stack.

Native token replay and the analytical AFD/EPD paths have different support and
reporting boundaries. A successful run does not establish silicon accuracy.
Agentic results explicitly retain `functional_only` qualification.

## Read by task

- [Feature support](features.md): backend, topology, workload and composition limits.
- [Workloads](workloads.md): the `traffic` block, trace formats, sessions, stopping rules and
  [agentic lanes, snapshots, warmup and profiles](workloads.md#agentic-load-controls).
- [Engine](engine/README.md): the `engine` block, with pages for
  [workers](engine/workers.md), [KV cache](engine/kv-cache.md),
  [P/D KV transfer](engine/kv-transfer.md),
  [speculative decoding](engine/speculation.md) and
  [analytical AFD/EPD](engine/analytical.md).
- [Agentic quickstart](agentic/quickstart.md): run a reproducible Weka simulation.
- [Dynamo integration](dynamo.md): install matching dependencies and enable Router/Planner.
- [Python API](api/python.md) and [Rust API](api/rust.md): inputs, reports and execution contracts.
- [Extending Replay](extending.md): internal ownership and validation requirements.
- [Adapter ABI](../adapters/README.md): external execution, configuration, output and native composition boundaries.

For throughput, TTFT, ITL, TPOT, completion and goodput definitions, see
[understanding results](../getting-started/understand-results.md).
