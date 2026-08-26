# Strict-native FPE coverage matrix

The Forward Pass Engine (FPE) matrix answers one narrow question: can the
supported public native estimator build a resolved engine identity and return
positive, finite latency estimates for representative forward-pass shapes?

It does **not** certify the AISimulate CLI, Sweeper, scheduler, Replay,
disaggregated rate matching, deployment validity, or prediction accuracy.
Those behaviors require a smaller end-to-end qualification matrix with their
own evidence.

## Row identity

Each row records the model and architecture, system, backend and version,
the `op_level` forward model, resolved quantization, parallel topology,
role, probe phase, provenance, exact package version and source SHA, and a
machine-readable SDK reproducer.

An identical engine used by multiple roles is compiled once. The row records
all applicable roles:

- `agg` requires prefill, early-decode, late-decode, and mixed-step probes;
- `prefill` requires the prefill probe;
- `decode` requires the early- and late-decode probes.

For disaggregated serving, separate prefill and decode engine coverage is not
evidence that rate matching, queueing, KV behavior, or scheduler behavior works.

## Strict-native policy

The generator uses `aisimulate_core.sdk.EngineHandle.compile`. It never calls
`RustForwardPassPerfModel.best_available`, enables regression fallback, or
uses HYBRID gap filling. Missing native data and unsupported configurations
remain visible.

`PASS` requires the public engine build and the named probe to complete with a
positive finite latency. Provenance is recorded separately; native analytical
operations may report an empirical or SOL source without enabling a whole-model
fallback.

`SDK_UNREPRESENTABLE` is also fail-closed. It means the resolved Task topology
uses a field that the supported public engine builder cannot encode, such as
context parallelism or a large-EP communication backend. The generator does
not silently test a different topology.

Other failures distinguish performance-data gaps, unsupported models,
hardware or framework incompatibility, engine-build failures, and query
failures. Error text is diagnostic evidence, not a stable API.

## Run it

Install the repository package, then generate one op-level FPE shard per system:

```bash
python python/aisimulate/tools/support_matrix/generate_fpe_support_matrix.py \
  --output-dir fpe-support-matrix/b200_sxm \
  --system b200_sxm \
  --forward-model op_level \
  --max-workers 8
```

After all system shards finish, build the split CSV files consumed by the
interactive page:

```bash
python python/aisimulate/tools/support_matrix/build_fpe_support_matrix.py \
  fpe-support-matrix \
  --output-dir python/aisimulate/src/aiconfigurator_core/systems/fpe_support_matrix
```

Use filters and a deterministic topology cap for a focused smoke run:

```bash
python python/aisimulate/tools/support_matrix/generate_fpe_support_matrix.py \
  --output-dir fpe-support-matrix-smoke \
  --model Qwen/Qwen3-8B \
  --system b200_sxm \
  --backend vllm \
  --forward-model op_level \
  --max-topologies-per-role 1 \
  --max-workers 2
```

Each raw shard contains deterministic JSON, CSV, and Markdown coverage
artifacts plus a separate `run_metrics.json`. The rollup produces mode-neutral
rows for the separate FPE Support Matrix and adds real probe counts, topology
counts, native status counts, phase latencies, and source SHA to the detail view.
Wall time, CPU time, peak RSS, and worker count remain separate from the
deterministic coverage artifacts.

## Parallel execution and memory

The runner opens a bounded thread pool for one system/backend/version group at
a time. Native database loading and hot-path queries release the Python GIL,
but Python-backed model compilation and database memory still limit scaling.
Increase `--max-workers` only with measured memory headroom.

The scheduled workflow creates one shard per system/backend pair and runs at
most eight shards concurrently on the repository-specific CPU runner set. Each
shard runs only `forward_model=op_level` with an eight-thread local pool. A
final job combines the raw shards into the split web CSV artifact. This avoids
leaving runners idle when a small system finishes before the largest systems.
Full runs also suppress repeated SDK warnings at the console while preserving
every classified failure and representative error in the matrix artifacts.
Refresh-time claims must name both runner concurrency and per-runner thread
count, plus the source SHA from the measured run.
