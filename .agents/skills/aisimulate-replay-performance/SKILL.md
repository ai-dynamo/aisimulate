---
name: aisimulate-replay-performance
description: Design, implement, and review efficient AISimulate engine-only simulation execution. Use for replay features, correctness fixes, and refactors that affect scheduling, state transitions, event processing, or costs that scale with requests, tokens, workers, or resident state. Resolve substantial avoidable overhead before completion. Excludes Dynamo Router, Planner, KVBM, and live event-plane behavior.
---

# AISimulate replay performance

Design new and changed code for efficient execution. Functional correctness and
efficient execution are both completion requirements. Resolve substantial, avoidable
overhead in the same PR before marking the implementation ready for merge.
Assess simulator host cost separately from simulated serving latency. Apply this guide
to AISimulate-owned engine replay; route Dynamo-owned runtime work to its own workflow.

## Design

- Identify the expected workload, operation frequency, input sizes, and state lifetime.
  State how work scales with requests, tokens, events, workers, and resident state.
  Include bursts, long runs, and affected backend configurations where relevant.
- Choose scheduling algorithms, data structures, and ownership to fit those conditions.
  Distinguish the state an operation needs from all stored state. Account for total
  preparation, repeated execution, retained memory, synchronization, and cleanup cost.
- Identify copied state and repeated discovery of information already available to the
  caller. Check whether optional functionality adds cost to configurations that do not
  use it. An event with a small local cost can be expensive at simulation frequency.
- Prefer existing incremental state and borrowed immutable data where appropriate.
  Justify extra caching, concurrency, or ownership complexity with a concrete need and
  expected benefit. Account for reuse, invalidation, contention, and retained memory;
  lower local cost is insufficient if setup or state maintenance consumes the saving.

## Implement

- Keep work proportional to the transition or event being processed. Avoid redundant
  scans, repeated preparation, unnecessary data movement, and bookkeeping without a
  required consumer. Include helper calls and downstream report processing when checking
  total cost. Use the simplest design that meets the workload and correctness needs.
- Preserve request and token totals, completed lifecycles, required report behavior,
  deterministic same-timestamp order, and stable ties. Preserve ownership, capacity,
  eviction, preemption, and handoff semantics. Do not drop, duplicate, or leave work
  incomplete to improve a timing result.
- Preserve transactional state transitions. Establish rollback before the first mutation,
  including mutations in lookups or matching helpers. A failed operation must restore
  all affected state and permit a correct retry. Publish events only at commit and
  preserve commit order; rollback must not leave externally visible events.
- Define mutation isolation when sharing state. Keep validation at public boundaries;
  internal reuse requires established invariants. Preserve diagnostics and error behavior.
  When output or lifecycle behavior can change, read the existing
  [replay-parity guide](../../../python/aisimulate/.agents/skills/aisimulate-replay-parity/SKILL.md).
  Semantic qualification and performance measurement are separate checks.

## Validate and review

- Trace the changed execution paths. Confirm that the workload exercises the changed
  operation, affected configurations, and relevant state sizes. A short run with little
  resident state cannot establish the cost of work that grows with accumulated state.
- Choose proportionate evidence. For a cold or small path, a supported bound on frequency,
  input size, total work, and retained memory can suffice. For likely added hot-path work
  or material cost changes, require repeated matched before/after native
  `aisimulate predict --stack engine` comparisons. Forward-only prediction measurements
  cannot clear scheduling, state management, evidence processing, or other integrated costs.
- Record exact baseline and candidate revisions. Match release builds, dependencies,
  data, workload, and host settings, and verify the loaded native binaries. Use repeated
  paired timings on a representative workload and a small control. Report variation;
  keep profiles separate from timing samples. Measure memory when retention can change.
- Check equivalent work, complete execution, and required behavior before interpreting
  speed differences. Missing data, skipped coverage, and incomplete execution are
  missing evidence. Explain intended changes in work separately from implementation
  overhead; passing behavior checks does not establish efficient execution.
- Use `wall_time_ms` for native preparation, replay execution, and report aggregation.
  It excludes Python startup, trace loading, and output serialization. Distinguish it
  from loop-only timing, full CLI time, and simulated serving latency. A narrower timer
  cannot qualify costs outside its boundary.
- Investigate material costs and correct avoidable overhead before completion. Passing
  tests or documenting a slowdown is not sufficient. Distinguish necessary feature
  cost from implementation waste; explain material necessary costs, measured impact,
  and considered alternatives. There is no universal slowdown limit. Accept an efficient
  implementation with sufficient evidence without demanding further optimization.

Report the cost assessment, evidence and timing scope, behavior checks, and remaining
uncertainty. Distinguish source-level risks, measured regressions, and inconclusive
results. Missing required measurements leave an evidence gap, not performance approval.
In a source-only review, identify the concern and needed comparison without starting
builds, profiles, or benchmarks outside the task's permissions.

## Examples

- [#321](https://github.com/ai-dynamo/aisimulate/pull/321): a correctness change copied
  broad resident state during frequent transitions, increasing CPU cost as state grew.
- [#295](https://github.com/ai-dynamo/aisimulate/pull/295): auxiliary processing around
  predictions added replay cost that a numerical-kernel benchmark would not cover.

These illustrate failure patterns and consequences; their solutions are not requirements.
