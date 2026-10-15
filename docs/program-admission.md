<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Offline program admission prediction

`aisimulate predict --stack engine` can optionally combine program admission
with the existing native scheduler, prefix cache, eviction and timing model.
This is an experimental behavioral model. It has not been calibrated as a
prediction of measured ThunderAgent end-to-end latency.

The native policy tests cover admission, tool waits, pause/resume ordering,
forced-resume deadlines, cancellation, and complete program retirement.
Those functional checks do not establish measured serving-latency accuracy.

```sh
aisimulate predict --stack engine --config examples/cli/program-admission.yaml
```

The example uses fixed timing for functional reproduction. Replace its timing
with a validated performance model for GPU latency studies. Keep the native
physical KV capacity and the policy's logical program capacity separate.
`capacity_tokens_per_worker` means tokens for one whole replica, not tokens per
GPU multiplied by TP. For example, 1,000 logical blocks of 16 tokens mean
16,000 logical tokens per replica. See the [configuration reference](reference/configuration.md#program-admission)
for every parameter, default and validation constraint.

## Supported behavior

- A program stays assigned to a worker across sequential model/tool turns.
- Its full prompt plus emitted output tokens count against a logical budget.
  Shared prefix blocks are deliberately not deduplicated in that budget.
- New programs queue behind paused programs if the budget cannot admit them.
- Periodic virtual-time decisions pause ACTING programs and mark REASONING
  programs to pause at completion. Committed engine passes are never interrupted.
- Each periodic tick resumes existing paused programs before selecting new
  pauses. A program paused by that tick's capacity phase cannot immediately
  resume in the same tick; request-specific forced-resume deadlines remain
  independent.
- Resume uses the pause threshold, target and hysteresis, the continuation/new/
  acting groups, token-size ordering and remaining worker capacity.
- Request-specific timeouts force resume on the worker with the lowest decayed
  logical load. Forced resume may exceed logical capacity; the engine still
  enforces its physical capacity and can preempt or reject requests.
- End-of-program feedback releases the budget and removes pending deadlines.
  All state is run-owned; inactive history is not retained in the policy.

Public Python configuration is available as `aisimulate.ProgramAdmissionConfig`
and `CorePredictionConfig.program_admission`. Native Rust users can construct
`ProgramAdmissionComposition` with a `ReplayRuntimeInput::Workload` driver.
The regular round-robin composition remains the default.

Set `admission_control: false` to study program affinity alone. Remove the
`program_admission` section for the round-robin control. Compare these controls
before interpreting a combined policy result.

## Output and interpretation

`placement_diagnostics`, also retained in normalized report metadata, records
the resolved configuration, logical budget basis, queue/pause/resume/forced
resume/migration counts, ended programs, total and first-turn/continuation wait,
remaining state and final worker budgets. Native per-request route timestamps
and worker assignments remain the source for individual queue waits.
Incomplete runs retain remaining programs and deadlines; they do not imply
successful cleanup. `ended_programs` includes failed or canceled program
releases; use normal request and graph outcome fields to assess success.

Logical budget occupancy is not physical HBM occupancy. Backend prefix reuse
and committed prefill work remain native-engine measurements. Simulator
`wall_time_ms` is host execution time, not simulated workload completion time.

## Coverage boundaries

The initial version supports static, ready aggregated workers, attention DP=1,
and session or agentic workload drivers with one active request per program.
Overlapping requests within a program are rejected. Disaggregated placement,
scaling, worker startup, snapshot/warmup preparation, single-worker artifact
capture, speculation, host offload and automatic recommendation/search of TA
parameters are unsupported.
Manual parameter studies can call the public predictor repeatedly.

The backend remains FCFS. Router priority adjustments are modeled as dispatch
metadata, but **priority boost and soft demotion have no modeled effect on the
backend queue order**. Their tuning cannot establish a latency benefit with
this version. The report labels this boundary explicitly.

There is no live MDC capacity discovery, router coroutine/network overhead,
HTTP response-tail timing, KV migration transfer cost, or prefix-aware logical
admission alternative. Native engine KV behavior still determines which
cached prefix blocks a new worker can reuse.

## Preparing SWE experiments

Use authored HTTP submission times for program roots and completion-relative
client/tool gaps for continuations. Do not use backend arrival times or
backend-to-backend gaps that already contain measured TA admission wait.
Otherwise the same wait is counted both in input and in the simulated policy.

Fixed observed root timestamps are an open-loop scenario. A smaller sampled
trace changes the offered load and cannot validate full-run latency accuracy.
A closed-loop study also needs the authored client concurrency/lane release
rules, actual program-end semantics and a calibrated backend timing model.
Graph completion is the model's program-end boundary; delayed HTTP end-program
notifications in production are outside this boundary.
The admission estimate uses the trace prompt length; frontend estimation error
and the deployed controller's tick phase need separate calibration.

## Source and computational cost

The modified model is derived from Dynamo's ThunderAgent router and program
state at `65a27b73b1aa35c0655e3eb0fbc50e327e2aa5d9`, Apache-2.0. Source paths,
copyright and immutable URLs are recorded in `THIRD_PARTY_NOTICES.md` and the
implementation headers. It is separate from Dynamo's live Router construction.

Logical budgets are maintained incrementally. Arrival and token processing do
not scan completed program history or expand queued prompts. Per tick,
candidate construction is O(P log W+W), and heap-based worker selection for
resumes is O(R log W). Sorting paused programs has an O(P log squared P) upper
bound because comparison keys consult the ordered pause index. Budget updates
use O(log W) worker-map lookups; deadline edits use O(log Q). Retained
policy state is O(P+Q+W); deadlines are removed on release without tombstones.
Disabled compositions use default empty hooks, with no policy allocations or
program completion queue.
