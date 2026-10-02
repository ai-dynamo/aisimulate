# TRT-LLM eager execution: proposed measurement contract

Status: deferred at the user’s request. Retain collection evidence; do not
extend the data contract or implement a timing correction. The proposal below
is preserved for a future decision.
The current GPT-OSS op-based prediction still omits a material host execution
boundary. See the [trace report](../python/aisimulate/collector/trtllm/gym-gpt-eager-20260918.md)
for measured evidence and its limits.

## Problem

GPT-OSS-120B, B200 TP2/EP1, 1k/1k, concurrency 4 reproduces client TTFT at
48.86 ms versus a 16.53 ms prediction. The corrected MXFP4/MXFP8 MoE kernel
cost is close to the trace. Eager prefill/mixed execution has substantial CPU
submission gaps and cross-rank waiting; graph decode does not have the same
boundary. Increasing the shared MoE kernel table would charge that delay to
graph decode and erase its existing TPOT alignment.

## Proposed coordinated change

Keep the op-based GPU model and add explicit execution-boundary measurements.
Do not insert an E2E-derived TTFT constant or a model-wide latency multiplier.

1. Extend the collector output contract with an execution-mode identity
   (`cuda_graph` or `eager`) and CPU dispatch timing. Preserve GPU latency as
   a separate measurement. Record actual capture success, runtime source,
   CPU model/affinity, rank membership, warmup policy and raw timing samples.
   The measured runtime version remains part of the identity: rc14 host timing
   must not be published as an rc20 measurement.
   Existing rows without this evidence retain their legacy behavior; they
   must not be relabeled as measured eager execution.
2. Collect both modes at identical serving-selected modules, shapes, dtypes
   and topology. Use phase-correct attention and fused all-reduce/norm
   boundaries. Preserve the overlap within a module and between CPU and GPU;
   a profiler's inflated CPU times are diagnostic evidence, not database rows.
3. Carry the resolved TRT-LLM graph policy from Gym/runner into the native
   performance model. Mixed/prefill selects eager measurements where the
   pinned runtime cannot capture those steps. Decode selects graph data only
   for captured shapes. Graph memory reservation is a separate setting.
4. Compose host submission and GPU work with explicit dependencies. Do not
   add CPU time to the sum of all GPU kernels, or separately add collective
   waiting already caused by a late peer. Validate the composition against
   complete, independent per-rank traces before enabling the new path.
5. Keep missing eager coverage visible. Do not silently report a legacy
   graph lookup as an eager measurement. Preserve the existing path for
   configurations outside the qualified collection domain.

The exact row schema and composition must be frozen after the paired
measurement smoke, before formal collection. Approval would cover the
new execution-mode/CPU-timing contract and its producer/consumer work; it
is not an accuracy claim for an unmeasured composition algorithm.

## Owners and affected surfaces

- Collector: `python/aisimulate/collector/trtllm/`, the network collector for
  fused communication, perf schema/metadata, and collector tests.
- Data loading and native timing: `crates/core/src/perfmodel/perf_database/`,
  `operators/`, and `engine/runtime.rs`; Python lowering and diagnostics.
- Deployment identity: Gym normalization, AISimulate runner/provider config,
  and native engine config. Preserve actual graph capture coverage.
- Documentation and data: measured rows, immutable provenance, paired results,
  missing coverage, and unchanged controls.

## Acceptance checks

- The existing TP2/C4 workload retains its exact 40-request token vectors.
  Repeated unprofiled serving measurements remain independent validation.
- Prefill, mixed, and graph decode are evaluated separately; complete rank
  groups and timing boundaries are checked. CPU affinity is retained.
- Cover neighboring token counts and the affected GPT-OSS topologies, not
  only the diagnostic 886-token step. Select collection points before looking
  at validation errors.
- Replay all 31 GPT-OSS/B200 points. If shared timing or config code changes,
  rerun all 263 points and retain every TTFT/TPOT regression and failure.
- Reject double counting with synthetic dependency tests; test graph/eager
  selection at capture boundaries and explicit missing-data behavior.
- Report remaining queue/frontend error separately from forward error.
  Do not claim the entire 32 ms client gap is a kernel or host-only delta.

## Repository approval boundary

`python/aisimulate/.claude/rules/collector/layer_permissions.md` requires:

> Changing the data contract requires explicit human approval.

Adding execution-mode/CPU-timing fields changes the producer/consumer contract.
The current collection/fix request authorizes diagnostic collection; this
proposal makes the newly discovered contract extension reviewable before its
implementation. No collector policy file is changed.
