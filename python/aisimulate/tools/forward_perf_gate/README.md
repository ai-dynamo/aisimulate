# Forward-prediction performance gate

This tool measures the CPU time for
`InferenceSession.run_static_latency_only`. It compares a pull request head
with its merge base on the same worker. The check is advisory: regressions make
the check red, but repository rules do not require it.

Each revision runs the matrix in one process for the availability pass and one
new process for each measured round. Within that process, cases are grouped by
model and database mode. Before each group, the worker releases the prior
objects, clears the prediction caches, and builds one model, database view,
session, and Rust engine. It initializes each measured phase with one
unrecorded `(batch_size=2, ISL=2048)` query that is outside the benchmark
matrix. It then runs every target point in the group. The first prediction for
each distinct target is the `cold` sample: a new query against a steady-state
engine. Ten unrecorded repeats of that target follow, then 100 repeats produce
the `warm` median. Thus `cold` means a cold query, not a cold process or empty
session. The existing
`benchmark_engine_step.py --cache-mode cold` has a different boundary because
it clears the engine-handle cache before each call.

If the off-matrix priming query fails, the worker records `PRIMING_FAILED` for
only that phase and continues with other phases and groups. A missing-data
priming failure is skipped under the normal data-miss rules. Other priming
failures remain invalid and block the comparison.

The harness treats `clear_caches()` as the complete cache-isolation contract.
It calls the public database eviction interface once per model/database-mode
group and does not know about or manage individual SDK caches.

The worker uses a versioned JSON protocol so each Git revision can adapt its
own internal SDK interface. It accepts either one `case` or an ordered `cases`
array. Run one request with:

```bash
python tools/forward_perf_gate/worker.py --request request.json --pretty
```

The initial matrix uses Qwen3-32B and Qwen3-235B-A22B on B200/vLLM 0.24.0,
with SILICON and EMPIRICAL database modes and the nine points from the existing
prediction regression grid. The isolated worker opts into this pinned raw data
version instead of resolving a moving backend-version alias.

The default comparison requires four of five paired rounds to exceed both a
10% relative threshold and a 2 us absolute threshold. Other round counts use
an 80% quorum. A data miss on the base side has no timing baseline and is
reported as skipped. A working base case that stops working, a malformed
worker response, or an incomplete run remains an invalid, blocking comparison.
The worker treats missing silicon data, unavailable empirical data, missing
system FLOPS, and unavailable SOL models as data misses.
The controller alternates which revision runs first and reverses the case order
on alternating rounds. Raw results are checkpointed after the paired
availability pass and after each paired measured round.

The controller starts each revision's worker from its matching source checkout
and installed wheel. For a local one-point check:

```bash
python tools/forward_perf_gate/run.py \
  --base-python /path/to/base-venv/bin/python \
  --base-worker /path/to/base/tools/forward_perf_gate/worker.py \
  --base-revision BASE_SHA \
  --head-python /path/to/head-venv/bin/python \
  --head-worker /path/to/head/tools/forward_perf_gate/worker.py \
  --head-revision HEAD_SHA \
  --output-dir forward-perf-results \
  --smoke
```

Smoke mode selects one case and defaults to one round, one warmup call, and
three measured calls. Explicit `--rounds`, `--warmup`, or `--iterations`
values override their individual smoke defaults.

Land the worker and protocol before enabling the workflow. This ensures the
merge-base and head revisions both have a revision-local adapter.
