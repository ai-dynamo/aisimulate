# Simulation performance gate

This advisory check measures the host cost of a complete native AISimulate replay.
It compares a PR head and its merge base on one CPU, using separate release wheels,
locked dependencies, and a revision-local worker for each side. It does not measure
simulated serving throughput or prediction accuracy.

## Measurement

The checked metric is the native report's `wall_time_ms`. Its boundary starts in
`Replayer::run_inner`, includes native replay setup, simulation, and report
aggregation, and stops before Python report normalization. Model construction and
trace loading performed before that boundary are excluded. The controller also
records process-start-to-exit `worker_elapsed_ms`, including imports, setup, JSON
serialization, and output transfer. Only the controller owns this timer; it stops
before controller-side response parsing and validation. Their difference is not
a trace-loading timer.

One equivalence pass captures per-request records and validates completion. Five
measured rounds use normal summary capture. Each case and side starts a fresh
process pinned to the same CPU with one-thread limits and offline model loading.
Base/head order alternates, as does case order. There is no in-process warmup:
prediction caches warm naturally during each complete simulation. No cache state
survives into another case or round.

`EngineReplayRunnerFactory(determinism="canonical_v1")` passes the existing Rust
canonical request-ID policy through the normal `CorePredictionConfig` and runner
path. Ordinary callers retain `random`. Canonical mode is restricted to native
text replay; there is no new CLI flag.

The versioned worker protocol hashes the full controller-supplied case. A worker
checks its local trace hash and reports a fixed model identity for each role:
model, system, backend/version, worker type, parallelism, attention backend,
quantization, KV block size, and the installation-relative packaged data root.
The complete configuration remains in `model_provenance`; additional configuration
metadata does not affect equivalence. Every role requires real op-level timing, fallback denied, SILICON
data, and shared-layer reuse. Missing data is an invalid comparison on either side.

Protocol v2 uses one JSON request on stdin and one JSON response on stdout. Logs
go to stderr. The request contains `protocol_version`, `revision`, `case`, and
`phase` (`availability` or `measure`). The response echoes the version, revision,
phase, `case_id`, and SHA-256 `case_hash`, then adds `status`, elapsed times,
`model_identity`, `model_provenance`, `behavior`, and `coverage`. An unsuccessful
worker returns `status="ERROR"` and an `error` with its type and message. The
controller adds `worker_elapsed_ms` to the saved response. For errors, crashes,
and timeouts it includes the log path and last 4 KiB of stderr; full logs remain
in the artifact. Keep
protocol changes explicit; each revision must adapt its own public APIs.

## Coverage

The 12 cases in `cases.py` cover:

- Dense Qwen3-32B on vLLM, SGLang, and TRT-LLM.
- Qwen3-235B-A22B long prefill on vLLM and long decode on SGLang.
- Four-turn prefix-sharing sessions with constrained KV capacity on vLLM/SGLang.
- DeepSeek-V3.2 with two workers and attention DP=8.
- One prefill and two decode workers on vLLM/SGLang.
- One complete AgentX play on aggregated vLLM and disaggregated SGLang, projected
  onto Qwen3.5-397B-A17B. See `fixtures/README.md` for provenance and attribution.

All use B200, pinned backend versions, and explicit configurations. Synthetic
inputs use the built-in deterministic workload generator: output-token seed
`0xD37A0A7E5EED` and prefix-hash seed `1337`. The input identity explicitly pins
`canonical_v1`, whose worker-selection seed is `0xD1A05EED`. These concurrency
workloads have no random arrival seed. AgentX input is local;
no benchmark downloads traces, model configs, or performance data.
The suite does not cover AFD, encoder overlays, recommendation sweeps, offload,
speculative decoding, or Dynamo router/planner adapters.

The cache equivalence pass also runs the same workload with a larger cache. It
requires nonzero reuse and more committed prefill tokens in the constrained case.
This establishes cache pressure that affects work without instrumenting the timed
runs. P/D equivalence requires a destination activation for every request. Every
AgentX request and the complete play must finish; no virtual-time cutoff is used.

Request/session counts were calibrated by doubling from 256 requests or 64
four-turn sessions until native replay reached two seconds. `WORKLOAD_COUNTS`
contains the frozen values. CI never resizes a case. The real AgentX play is intact.
These counts need qualification on the CI runner before automatic rollout.

## Results

- `PASS`: equivalent work, without a consistent slowdown.
- `PERFORMANCE_REGRESSION`: more than 10% **and** more than 100 ms slower in at
  least four of five rounds. Other positive round counts use an 80% quorum.
- `BEHAVIOR_CHANGED`: successful completion with different simulated results.
  Timing deltas remain visible but are not labeled a confirmed speed regression.
- `INVALID_COMPARISON`: missing data, timeout, incomplete work, bad protocol/input
  identity, malformed results, or inconsistent results within either revision.

All three non-pass results fail the advisory check; it is not a required check.
Counts and identities compare exactly. Floating-point values use `rtol=1e-9` and
`atol=1e-6`. Host timing and host throughput are excluded from behavior
comparison. `contract.py` fixes the compared fields: completion/token and committed
prefill counts, cache reuse, duration, latency distributions/sample counts, and
AgentX trajectory counts/latency and play outcomes. Request comparisons include
identities, lengths, terminal state, simulated times, cache work, admission and
routing results, worker placement, and P/D milestones. Nested records use fixed
fields too. Required fields cannot be missing; additional diagnostics, power,
and provenance do not affect the verdict. The same projection applies to each
revision's equivalence-versus-measurement checks.

Complete per-request records are retained, compared once using this projection,
compressed into separate artifacts, and referenced from the small checkpoints.
An artifact's `sha256` hashes the full records encoded as canonical JSON (sorted
keys and compact separators), not the compressed file or its decompressed bytes.
Model identity changes are
invalid; results from different model configurations must not be timed as peers.

Artifacts include input cases/hashes, raw paired results, separate worker/replay
times, compressed request records, coverage evidence, and subprocess logs. The CI
workflow also retains wheel hashes, dependency manifests, toolchain version, and
model/data Git tree IDs. Checkpoints are written after every paired case. An
interrupted run cannot produce a successful comparison with missing rounds.

## Run locally

Build and install the revisions separately, then run the **base** controller:

```sh
python tools/simulation_perf_gate/run.py \
  --base-python /absolute/base-venv/bin/python \
  --base-worker /absolute/base/python/aisimulate/tools/simulation_perf_gate/worker.py \
  --base-revision BASE_SHA \
  --head-python /absolute/head-venv/bin/python \
  --head-worker /absolute/head/python/aisimulate/tools/simulation_perf_gate/worker.py \
  --head-revision HEAD_SHA \
  --output-dir simulation-perf-results
```

`--case dense-vllm --rounds 1` makes a targeted check without changing the workload.
`--qualification` requires all 12 cases, five rounds, identical revision IDs,
synthetic median replay times of at least two seconds, and at most 900 seconds for
the complete benchmark (including equivalence, imports, and artifact work; builds
are excluded). The default per-process timeout is 120 seconds.

## CI and rollout

`.github/workflows/simulation-performance.yml` is disabled for automatic runs
unless the repository variable `SIMULATION_PERF_ENABLED` is exactly `true`.
Manual dispatch takes `pr_number` and optional `self_compare=true`. The selector
verifies that the trusted `pull-request/<N>` copy matches the current PR head.
Self-comparison builds/installs that head twice and runs three full qualifications.
It does not use an older merge base that lacks the benchmark adapter.

Normal comparisons use the merge base's controller/matrix. Changes to benchmark
Python code or `fixtures/agentx.jsonl` also run the head controller separately.
Documentation and license changes do not add a second benchmark run. Thread
limits are defined locally; the controller does not import the forward gate.
An older base without the adapter produces an explicit **not benchmarked**
summary. Missing head support, or differing or malformed protocol versions,
fail with `INVALID_COMPARISON` before builds.
The workflow has its own concurrency group and does not
cancel the forward-performance workflow.

Land the adapter, fixtures, and protocol before enabling automatic comparisons.
Enable only after three clean same-revision qualifications on the actual CI
runner, plus a detected controlled slowdown and correctly classified behavior
change. Do not remove cases, weaken thresholds, or enable automatic runs to hide
a qualification failure. Hosted qualification remains required even if local
qualification passes.
CPU affinity selects the same CPU for both sides; it does not establish exclusive
CPU ownership. Runner isolation remains a qualification requirement.

See [QUALIFICATION.md](QUALIFICATION.md) for the initial local measurements,
coverage checks, and failure controls. CI runner qualification is still pending.
