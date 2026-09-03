# DEP: Profile-backed vLLM CUDA graph reservation for KV-cache estimation

## Summary

AISimulate will eventually subtract vLLM's rank-local CUDA graph reservation when
estimating KV-cache capacity. The first implementation milestone is deliberately
smaller: it adds a reviewed InfX ingestion workflow, a versioned profile database, and
a safety-gated Python prediction API. It does not change KV-cache capacity.

The future memory integration will resolve a reservation from an explicit override, an
exact record in the profile database, or a validated conservative model. Prediction
will not launch vLLM or require a GPU.

vLLM remains the source of measurement truth. AISimulate owns profile validation,
lookup, fallback modeling, budget integration, and provenance. Other backends and the
feature-disabled path remain unchanged in the first phase.

## Motivation

AISimulate currently computes vLLM KV capacity as:

```text
KV bytes = GPU capacity * gpu_memory_utilization
         - weights - activations - runtime overhead - communication overhead
```

The vLLM backend states that CUDA graphs share this memory limit, but the native memory
breakdown has no CUDA graph term. The existing 2% KV tolerance covers allocator and
profiler variance; it is not a graph-memory estimate.

This caused a concrete capacity error for MiniMax-M2.7 on H200 TP4 with vLLM 0.25.1.
vLLM profiled 99 PIECEWISE and 67 FULL graphs, reserved 13.56 GiB per rank, and exposed
1,926,816 KV tokens. AISimulate exposed 2,420,992 tokens, 25.7% more. At concurrency 256,
the simulator therefore treated KV as available while the server reported 99.7-100%
usage, contributing to a large TTFT underprediction.

The required invariant is: when CUDA graphs are enabled, a capacity estimate must not
silently omit their reservation. Every estimate must expose whether the reservation was
explicit, measured, modeled, disabled, or unavailable.

## Proposal

### Terms and scope

A **CUDA graph reservation** is the per-rank byte quantity vLLM subtracts before it
allocates KV cache. It is the startup profiler's estimated reservation, not the later
incremental CUDA graph pool allocation. A **profile** is an immutable measurement of
that reservation for one execution identity.

Phase 1 covers vLLM. The provider interface may support other backends later, but this
DEP does not change SGLang or TensorRT-LLM memory accounting.

### Milestone 1: database and Python predictor

The first implementation packages these versioned artifacts:

```text
aiconfigurator_core/systems/cuda_graph_profiles/v1/
├── cuda_graph_profiles.parquet
├── cuda_graph_profiles.metadata.json
└── cuda_graph_reservation_model.json
```

The Parquet dataset is the measurement source of truth. The model artifact records its
Parquet checksum, feature schema, component observations, validation metrics, and
applicability gates. The public Python API is
`aiconfigurator_core.sdk.cuda_graph.estimate_cuda_graph_reservation`. An explicit
`aisimulate_core.sdk.cuda_graph` alias is packaged for compatibility.

Milestone 1 resolution order is:

1. CUDA graphs explicitly disabled returns zero.
2. An exact semantic profile returns the largest rank-local reservation.
3. A validated in-domain model returns a calibrated conservative upper bound.
4. Every other request returns `unavailable` with a reason.

Concurrency is retained as provenance only. It is not part of profile identity or any
model feature. Profiles without an immutable model revision remain available with
`unversioned_model` provenance and are never presented as fully pinned identities.

### Future KV-cache integration

`aiconfigurator_core.sdk.memory` will own a CUDA graph reservation provider and remain
the single source of KV-budget math. Replay, Sweeper, the Python SDK, and the Rust
forwarder will pass configuration and consume the returned estimate; they will not
implement independent lookup rules.

Future memory-resolution order is:

1. An existing explicit `num_gpu_blocks` remains authoritative and bypasses estimation.
2. A new `cuda_graph_reserved_bytes` request value overrides profile lookup.
3. An exact profile-key match returns the measured vLLM reservation.
4. A validated model may return a conservative reservation and prediction interval.
5. A miss is reported as `unavailable`; it must not be presented as a zero-byte
   reservation.

The request will support `cuda_graph_policy = off | best_effort | required`:

- `off` preserves current behavior and is the initial rollout default.
- `best_effort` uses a profile or validated model and otherwise returns the legacy
  capacity with an explicit `unavailable` source and warning.
- `required` fails capacity estimation when no reservation is available.

After the validation gates are approved and profile coverage is sufficient,
`best_effort` becomes the vLLM default. `required` is intended for production capacity
checks that must not silently over-admit KV.

### Profile database and InfX ingestion

Profiles will live in a versioned, packaged dataset separate from operation-latency
tables. An exact key must include all inputs that can change graph selection or memory:

- resolved model identity and model-config digest;
- GPU system and rank-local memory capacity;
- vLLM version and build identity;
- TP, PP, attention-DP, MoE-TP, and MoE-EP mapping;
- weight, activation, KV-cache, and MoE dtypes;
- CUDA graph mode and capture-size digest;
- compilation mode/backend and selected attention, MoE, and linear backends;
- FlashInfer autotuning state;
- `max_num_seqs`, `max_num_batched_tokens`, and `max_model_len`;
- attention backend and enabled features such as speculative decoding, LoRA, and
  multimodal execution.

Each record will contain:

- vLLM's `estimated_cuda_graph_bytes` per rank;
- the maximum reservation across ranks, used for a shared scheduler capacity;
- graph counts and largest shape by mode;
- available KV bytes and block/token capacity for cross-checking;
- container, CUDA, PyTorch, source-log, and collector revision provenance;
- collection timestamp, schema version, and content digest.

The initial database is reproduced from reviewed InfX run artifacts. A human-readable
manifest selects approved runs. A generated lock file pins run ID, run attempt, head
SHA, artifact ID and name, and each extracted file's SHA256. The resolver admits only
successful attempts; a running, cancelled, or failed attempt cannot be published.

The parser supports standalone logs and nested multinode archives. Configuration
identity is resolved in this authority order: bundled `config.yaml`, sibling benchmark
JSON, engine configuration log, then artifact name. Incompatible rank identities,
missing provenance, corrupt archives, and duplicate semantic profiles whose reservation
differs by more than 5% fail publication. Publication generates source-mapping,
reconciliation, exclusion, and validation reports.

The producer must run vLLM profiling on the target GPU because vLLM measures real CUDA
allocations. It does not need inference traffic. A successful job publishes a complete
validated record atomically; failed or cancelled jobs publish nothing. Backend, model,
or graph-configuration changes create a new key rather than mutating an old record.
Packaged profiles contain configuration, measurements, and digests only; they must not
contain raw logs, request data, credentials, or host-local paths.

The parser preserves two distinct quantities:

- vLLM's pre-KV `estimated_cuda_graph_bytes`, which is eligible for reservation lookup
  and model training;
- the later `actual_cuda_graph_pool_bytes`, which is diagnostic only.

Explicit graph-disabled runs record a zero reservation. Older runs with only the actual
pool measurement remain useful diagnostics but are ineligible for reservation training.

### Modeled fallback

The V3 fallback mirrors vLLM's estimator structure:

```text
graph bytes = max(shared FULL, shared PIECEWISE)
            + (FULL graph count - 1) * incremental FULL
            + (PIECEWISE graph count - 1) * incremental PIECEWISE
            + encoder graph bytes
```

The profiler supplies four decoder measurements: FULL and PIECEWISE first-capture
bytes, plus FULL and PIECEWISE per-graph bytes. Publication reconstructs the logged
total from those measurements and rejects a mismatch above 5% or 16 MiB. Encoder graph
memory is preserved separately and is not modeled in V3.

Each decoder component uses deterministic inverse-distance interpolation in log space.
Candidates must match model identity, GPU and vLLM family, dtypes, graph and compilation
modes, attention/MoE/linear backends, FlashInfer autotuning, speculative method, and
parallel mode exactly. Numeric coordinates include the mode-specific capture-size
distribution, scheduler limits, speculative token count, rank-local architecture, and
TP/PP/attention-DP/MoE topology. V3 does not extrapolate outside any observed numeric
range and does not claim cross-model generalization.

Leave-one-profile-out validation reports prediction coverage separately from error.
The component model is enabled only with at least 20 independent profiles spanning two
model identities and two GPU families, and metrics satisfying all of these gates:

- holdout prediction coverage at least 80%;
- median MAPE at most 20%;
- p90 APE at most 40%;
- conservative upper-bound coverage at least 95%;
- no underprediction greater than 20%.

If any gate fails, exact lookup stays available and modeled misses return
`unavailable`. The initial reviewed InfX logs contain only INFO-level totals, not the
DEBUG component measurements, so the checked-in V3 model remains disabled until new
profiles are collected.

### Budget, API, and observability

For vLLM's total-memory fraction, the raw budget becomes:

```text
KV bytes = GPU capacity * gpu_memory_utilization
         - weights - activations - runtime overhead - communication overhead
         - CUDA graph reservation
```

The existing KV tolerance is applied after this subtraction. Rewriting
`gpu_memory_utilization` to an "equivalent" value is not used because it hides the
reservation and couples a runtime setting to one profile.

The first milestone exposes only a typed Python request/result API. A later KV-cache
integration will add backward-compatible Python and Rust memory request/result fields,
subtract `reservation_bytes`, and preserve existing serde payloads with defaults.

Logs and reports will expose the lookup key, source, reservation, profile/model version,
raw KV capacity, tolerance-adjusted capacity, and misses. These fields are required for
reproducing a capacity decision and detecting stale coverage.

### Rollout, validation, and alternatives

Rollout has four stages:

1. Add the reviewed InfX ingestion workflow, packaged database, and Python predictor;
   leave KV-cache accounting byte-for-byte unchanged.
2. Add the memory schema, loader, explicit override, provenance, and shadow comparison
   while leaving the policy `off` by default.
3. Seed exact profiles for supported vLLM configurations, train the fallback, and
   compare both against held-out startup profiles and reported KV blocks.
4. Enable `best_effort` by default after owners approve accuracy and coverage gates.

Rollback sets the policy to `off`; no profile deletion or request migration is needed.
Profile/schema upgrades remain additive during the compatibility window.

Rejected alternatives:

- **Fixed GPU-memory percentage:** graph memory varies with model, graph count, shapes,
  backend, and version.
- **Parameter-count formula:** parameter count does not describe captured buffers or
  graph-pool behavior.
- **Pure analytical model:** kernel workspaces and allocator behavior require empirical
  calibration.
- **Online profiling during prediction:** it requires target GPUs, model loading, and
  graph capture, violating offline simulation.
- **Use final actual graph-pool delta:** vLLM sizes KV using its earlier estimated
  reservation, so the later measurement does not reproduce server capacity.

Open decisions are the owner of recurring GPU profile jobs and whether the profile
dataset remains wheel-bundled after it grows past the existing data-size budget.

## Requirements

1. Milestone 1 must not modify KV-cache capacity. The later vLLM integration must
   subtract the resolved CUDA graph reservation before tolerance and token conversion.
2. An exact profile must reproduce vLLM's reported KV block count within one scheduler
   block when the remaining memory inputs match.
3. Profile lookup must require an exact semantic key and must reject ambiguous,
   duplicate, corrupt, or unsupported-schema records.
4. The estimator must report reservation bytes and one of `explicit`, `profile`,
   `modeled`, `disabled`, or `unavailable` for every vLLM result.
5. `required` policy must fail on a profile/model miss; `best_effort` must emit a visible
   warning and `unavailable` provenance when it returns legacy capacity.
6. Modeled results must include a model version, applicability decision, holdout metrics,
   and conservative prediction interval. A model must not answer outside its declared
   domain.
7. Multi-rank profiles must use the largest rank-local reservation when returning one
   shared KV capacity.
8. The implementation must demonstrate that CUDA graph memory is not also included in
   activations or runtime overhead.
9. Milestone 1 adds only a Python predictor. Future Python and Rust memory fields must
   preserve deserialization and behavior of existing callers when the policy is `off`.
10. SGLang, TensorRT-LLM, explicit `num_gpu_blocks`, and CUDA-graph-disabled paths must
    remain unchanged in phase 1.
11. Profile publication must validate required fields, units, non-negative values,
    content digests, and the identity-to-payload match before an atomic write.
12. Milestone 1 tests must cover disabled/exact/modeled/miss resolution, profile
    corruption, external database override, cross-rank selection, packaged artifacts,
    and reviewed numerical regressions. Future integration tests must cover budget
    ordering, miss policies, Rust-Python round-trip, and the MiniMax-M2.7 H200 TP4
    regression.
13. Published profiles must not contain raw logs, request data, credentials, or
    host-local paths.

## Risks

- **Stale or mismatched profiles:** AISimulate may overstate KV capacity. **Mitigation:**
  exact semantic keys, immutable records, build provenance, and `required` miss behavior.
- **Fallback underestimates reservation:** The simulator may recreate the current false
  admission bug. **Mitigation:** conservative bounds, applicability checks, held-out GPU
  validation, and no default promotion before owners approve thresholds.
- **Fallback overestimates reservation:** Feasible deployments may be rejected.
  **Mitigation:** report uncertainty and preserve explicit measured overrides.
- **Double counting:** A future activation model may absorb graph memory. **Mitigation:**
  keep a named breakdown term and require component-level regression tests.
- **Profile-key explosion:** Exact profiles may be expensive to collect and package.
  **Mitigation:** prioritize supported configurations, deduplicate identical capture
  digests, and evaluate an external versioned artifact before the wheel-size limit.
- **vLLM log or estimator changes:** Collection may silently parse the wrong quantity.
  **Mitigation:** version-specific adapters and a validation cross-check against reported
  KV bytes/blocks.
- **Premature public API commitment:** Detailed profile internals may become hard to
  change. **Mitigation:** expose only the scalar override, policy, and stable provenance;
  keep profile schema and model features internal until validated.
- **Artifact disclosure:** Raw logs may contain deployment details unrelated to memory
  profiling. **Mitigation:** package only validated scalar/configuration fields and
  content digests, never raw logs or host-local paths.

## References

- AISimulate native KV budget implementation:
  [memory.py](https://github.com/ai-dynamo/aisimulate/blob/b32333c76cbccd432acfbb9a499ce405a45861a7/python/aisimulate/src/aiconfigurator_core/sdk/memory.py#L315-L442)
- AISimulate vLLM memory defaults and tolerance:
  [vllm_backend.py](https://github.com/ai-dynamo/aisimulate/blob/b32333c76cbccd432acfbb9a499ce405a45861a7/python/aisimulate/src/aiconfigurator_core/sdk/backends/vllm_backend.py#L15-L90)
- AISimulate Rust request and result contract:
  [memory.rs](https://github.com/ai-dynamo/aisimulate/blob/b32333c76cbccd432acfbb9a499ce405a45861a7/crates/core/src/perfmodel/memory.rs#L38-L125)
- Related MiniMax-M2.7 replay work:
  [AISimulate PR #69](https://github.com/ai-dynamo/aisimulate/pull/69)
- vLLM v0.25.1 KV-memory calculation:
  [gpu_worker.py](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu_worker.py#L394-L537)
- vLLM v0.25.1 CUDA graph estimator:
  [gpu_model_runner.py](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu_model_runner.py#L6500-L6644)
- vLLM component-log and reconstruction example:
  [issue #50780](https://github.com/vllm-project/vllm/issues/50780)
- vLLM CUDA graph profiling default:
  [envs.py](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/envs.py#L1902-L1907)
