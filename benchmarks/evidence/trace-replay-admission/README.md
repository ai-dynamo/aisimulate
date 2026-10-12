<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Trace replay admission: inspection cost and observed peaks

These measurements support replacing the inflated finite-trace peak estimate
(`trace-json-metadata-v1`) and the unknown-peak agentic bound
(`agentic-trace-unqualified-v1`) with `trace-observed-peak-v1`. That model has
three parts:
- a lower bound from one cached streaming pass;
- serial execution until a completed replay of the trace is measured;
- then reservations of 1.5 times the observed peak. See
  [local resources](../../../docs/reference/local-resources.md#finite-trace-replays-lower-bound-and-observed-peak).

All numbers come from one host and are recorded in [results.json](results.json).

| Revision | Commit |
|---|---|
| pre-#396 | `2ea3d7b4` |
| main | `2f0f4cd8` |
| candidate | branch `jthomson04/trace-replay-observed-peaks` |

The change is Python-only, so every revision loads the same release native
extension built from `2f0f4cd8`. The traces are synthetic.
[generate_traces.py](generate_traces.py) regenerates them byte for byte; their
SHA-256 digests are in the JSON.

## Trace inspection cost

`estimate_workload(..., stack="engine")` was timed once per fresh process on one
performance core, with revisions interleaved over three rounds. Each cell is the
median in seconds.

| Trace | pre-#396 | main | candidate |
|---|---:|---:|---:|
| Standard Dynamo, 16k requests (62 MiB) | 0.771 | 1.369 | 0.381 |
| Standard Dynamo, 65k requests (248 MiB) | 3.054 | 5.434 | 1.472 |
| Agentic Dynamo, 16k requests | 0.776 | 0.855 | 0.381 |
| Mooncake, 16k requests | 0.589 | 0.607 | 0.269 |
| Agentic Mooncake, 16k requests | 0.622 | 0.488 | 0.286 |

Three things make the candidate faster:
- Each format has its own scanner that counts hash items with a single equality test.
- The per-event key splitting is gone.
- The second, per-run guard inspection reuses the cached counters.

Validation outcomes are unchanged from main for:
- mixed `agent_context`;
- missing, negative, string or array lengths;
- gzip, multiple files and event wrappers.

One difference is intended. With a budget below the 128-times-file-size check,
main refused before reading later rows. The candidate has no file-size refusal,
so it now reports those rows' validation errors.

## Estimates against measured memory

Standard Dynamo, 16k requests:
- main estimated 62.5 GiB;
- the candidate's lower bound is 29.5 MiB;
- the `predict` replay child peaked at 254 MiB;
- a sweep worker peaked at 221 MiB.

Standard Dynamo, 65k requests:
- main estimated 249 GiB and refused the run on this 125 GiB host;
- the candidate admitted it, and it completed all 65,536 requests in 11.7 s with a 497 MiB peak.

## Prediction

The `predict` run used Qwen3-32B, vLLM 0.24.0, b200 op-level SILICON, TP4 and
concurrency 64 on the 16k standard Dynamo trace. Revisions were interleaved
over three rounds.

| Measure | main | candidate |
|---|---:|---:|
| Median CLI wall time | 7.28 s | 4.95 s |
| Trace-inspection share of py-spy samples | 30.4% | 7.0% |
| Trace inspections | 2 | 1 |

Native replay time and every simulated metric were unchanged. Only the
host-throughput fields derived from wall time differ.

## Recommendation sweeps

Each sweep ran 8 random trials at optimizer parallelism 4, using Qwen3-32B with
fixed timing. Every trial completed and was scored on both revisions.

| Trace | main | candidate |
|---|---|---|
| Standard Dynamo, 16k requests | 75.9 s; 8 serial waves | 12.8 s; waves of 1, 3 and 4 |
| Agentic Mooncake, 16k requests, 32 lanes | 50.3 s; 8 serial waves | 17.3 s; waves of 1, 3 and 4 |

Why main ran serially:
- **Standard Dynamo:** the 62.5 GiB estimate allowed only one worker.
- **Agentic Mooncake:** the peak was unknown.

On the candidate, the first wave measured worker peaks of 221 MiB (Dynamo) and
295 MiB (Agentic Mooncake). Later candidates reserved 542 MiB each, the worker
baseline plus the lower bound, because 1.5 times the observed peak was smaller.
The supervisor's peak for the whole process tree stayed at 1.0–1.3 GiB.

## Limits

- This is one host, synthetic traces, and fixed timing in the sweeps.
- The Dynamo stack was not run.
- The margin covers candidates that need more memory than the measured one, up
  to 1.5 times its peak. Beyond that, live-pressure interruption and the
  scheduler's halving retry apply; they were not exercised here.
