# Simulation performance gate

This advisory check measures the host cost of a complete native AISimulate replay.
It compares a PR head and its merge base on one CPU, using separate release wheels,
locked dependencies, and a revision-local worker for each side.

The gate measures runtime. It does not compare simulated latencies, routing,
cache behavior, or per-request results. Correctness and equivalence checks belong
in separate tests. There is no equivalence phase, detailed request capture,
request-record artifact, or large-cache control replay in this gate.

## Measurement

The checked metric is the native report's `wall_time_ms`. Its boundary starts in
`Replayer::run_inner`, includes native replay setup, simulation, and report
aggregation, and stops before Python report normalization. Model construction and
trace loading before that boundary are excluded.

The controller also records process-start-to-exit `worker_elapsed_ms`, including
imports, setup, JSON serialization, and output transfer. Only the controller owns
this timer; it stops before response parsing and validation. The difference
between these timers is not a trace-loading timer.

Each case runs five paired measurement rounds. Every case and side starts a fresh
process pinned to the same CPU with one-thread limits and offline model loading.
Base/head order alternates, as does case order. Prediction caches warm naturally
during each complete simulation; no cache state survives into another round.
There is no build work in the comparison job.

`EngineReplayRunnerFactory(determinism="canonical_v1")` passes the existing Rust
canonical request-ID policy through the normal `CorePredictionConfig` and runner
path. Ordinary callers retain `random`. Canonical mode is restricted to native
text replay; there is no new CLI flag.

Protocol v4 uses one JSON request on stdin and one JSON response on stdout. Logs
go to stderr. The request contains `protocol_version`, `revision`, `case`, and
`phase="measure"`. The response echoes the version, revision, phase, `case_id`, and
SHA-256 `case_hash`, then adds `status`, `wall_time_ms`, `model_identity`,
`model_provenance`, and the native summary `report`. The controller adds
`worker_elapsed_ms`. Protocols v2 and the pre-removal v3 trial are incompatible; their artifacts remain
historical evidence and are not rewritten.

Valid timing requires the expected revision, input hash, complete request/output
token counts, and finite positive elapsed times. AgentX must complete its play.
The controller also requires the expected model roles and comparable model
settings, real op-level timing, fallback denied, SILICON data, and shared-layer
reuse. Missing data, invalid provenance, and non-finite diagnostic values fail
the comparison. Additional finite diagnostics are retained and do not affect the
verdict.

Malformed input and worker exceptions produce `status="ERROR"` with an error type
and message. The controller preserves crash and timeout details, the log path,
and the last 4 KiB of stderr. Full logs remain in the artifact.

## Workloads

The 12 cases in `cases.py` cover:

- Dense Qwen3-32B on vLLM, SGLang, and TRT-LLM.
- Qwen3-235B-A22B long prefill on vLLM and long decode on SGLang.
- Four-turn prefix-sharing sessions with constrained KV capacity on vLLM/SGLang.
- DeepSeek-V3.2 with two workers and attention DP=8.
- One prefill and two decode workers on vLLM/SGLang.
- One complete AgentX play on aggregated vLLM and disaggregated SGLang, projected
  onto Qwen3.5-397B-A17B. See `fixtures/README.md` for attribution.

All use B200, pinned backend versions, and explicit configurations. Synthetic
inputs use the deterministic workload generator: output-token seed
`0xD37A0A7E5EED` and prefix-hash seed `1337`. The input identity pins `canonical_v1`,
whose worker-selection seed is `0xD1A05EED`. These concurrency workloads have no
random arrival seed. Traces, model configs, and performance data are local.
The suite does not cover AFD, encoder overlays, recommendation sweeps, offload,
speculative decoding, or Dynamo router/planner adapters.

`WORKLOAD_COUNTS` fixes each synthetic request count, or session count for the
two four-turn cache cases. Base and head always receive the same count. CI never
resizes cases. Both AgentX cases use the complete play. Count reductions require
a separate CI runner trial; retain a reduction only if the workload still
exercises its intended path and both native medians remain at least two seconds.
The CI trial retained half counts for dense vLLM, dense TRT-LLM, and P/D vLLM.
The other seven synthetic counts remain unchanged. See `QUALIFICATION.md` for
the trial results and normal CI measurements.

## Results

- `PASS`: valid measurements without a consistent slowdown.
- `PERFORMANCE_REGRESSION`: more than 10% **and** more than 100 ms slower in at
  least four of five rounds. Other positive round counts use an 80% quorum.
- `INVALID_COMPARISON`: missing data, timeout, incomplete work, bad protocol/input
  identity, incompatible models, or malformed results.

Both non-pass results fail the advisory check; it is not a required check.
A pass makes no claim about equivalence of simulated behavior.

Artifacts retain input cases/hashes, paired timings, native summaries, model
provenance, subprocess logs, wheel and requirements checksums, source SHAs,
container and build settings, tool versions, and model/data Git tree IDs.
Checkpoints are written after every paired case. An interrupted run cannot pass
with missing rounds.

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
synthetic median replay times of at least two seconds, and at most 900 seconds
for the benchmark, excluding builds. The default per-process timeout is 120 seconds.

## CI and rollout

`.github/workflows/simulation-performance.yml` starts on every update to a
trusted `pull-request/<N>` branch, like the forward-performance workflow. Its
selector uses the complete PR change set. Runtime, model data, dependency, and
benchmark changes select the gate; unrelated and documentation-only changes skip
it. No repository enable variable is required. The selector verifies that the
trusted copy matches the current PR head.

Revision and protocol checks run in the hosted selection job. A two-entry build
matrix builds base and head in parallel with the same container, Python, Rust,
and release settings. Each job uploads its wheel, locked requirements, and build
provenance. The comparison job waits for both builds and downloads artifacts by
exact side and SHA from the current run. It verifies revisions, checksums, and
matching build settings before separate installations. Paired measurements then
run sequentially on one CPU.

Normal comparisons use the merge base's controller and matrix. Changes to
benchmark Python code or `fixtures/agentx.jsonl` also run the head controller.
Documentation and license changes do not add another benchmark run. An older
base without the adapter produces an explicit **not benchmarked** summary.
Missing head support and differing or malformed protocols fail before builds.

Manual dispatch takes `pr_number` and optional `self_compare=true`. Self-comparison
builds the head twice and runs three qualifications. It does not substitute for
normal CI timing evidence. Queue/setup and total times are in the workflow job
records; artifacts record build, installation, and benchmark times.

The workflow has its own concurrency group. CPU affinity selects one CPU for both
sides; it does not establish exclusive CPU ownership. See
[QUALIFICATION.md](QUALIFICATION.md) for historical measurements and current CI
validation evidence.
