# Strict-native FPE coverage matrix

The published [Forward Pass Engine (FPE) matrix](https://ai-dynamo.org/aisimulate/fpe-support-matrix/)
answers one narrow question: can the supported public native estimator build a
resolved engine identity and return positive, finite latency estimates for
representative forward-pass shapes?

It does **not** certify the AISimulate CLI, Sweeper, scheduler, Replay,
disaggregated rate matching, deployment validity, or prediction accuracy.
Those behaviors require a smaller end-to-end qualification matrix with their
own evidence.

## Row identity

Each row records the model and architecture, system, backend and version,
the `op_level` forward model, resolved quantization, parallel topology,
role, probe phase, provenance, exact package version and source SHA, and a
machine-readable SDK reproducer. Attention-backend overrides remain part of
the engine identity and are passed through the supported public builder.

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

A rejected topology is recorded with its actual parallel configuration while
other choices continue to be probed. The public builder's explicit rejection
of mixed tensor/expert parallelism across nodes is `SDK_UNREPRESENTABLE`;
so are its explicit attention-head/TP divisibility and quantized-MoE block
alignment rejections. These do not count as passing coverage. Unexpected build and query errors
remain failures.

Other failures distinguish performance-data gaps, unsupported models,
hardware or framework incompatibility, engine-build failures, and query
failures. Error text is diagnostic evidence, not a stable API.

## Run it

Install a built repository wheel, then generate one op-level FPE shard per
system. The command imports that installed native runtime; it must not prepend
the source-only application package to Python's import path:

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

Nightly CI calls the reusable FPE workflow after confirming that `main` has
changed and building its release artifacts. All FPE shards install the exact
amd64 nightly wheel, verified against the artifact checksums, source commit,
and one recorded wheel hash. A manual run requires the full `expected_sha` and
builds one shared wheel. Neither path rebuilds the native runtime in every
shard.

The workflow discovers one shard per curated system/backend pair and runs at
most 20 shards concurrently on the repository-specific AMD64 CPU runner set.
Other CI runs share this pool, so runner availability can reduce concurrency.
Each shard runs only `forward_model=op_level` with an eight-thread local pool. A
final job validates reports before combining them into the split web CSV
artifact. Qualification requires every discovered shard, exact source and wheel
identity, consistent package version and workload, complete role-appropriate
phases, and no unexpected build or query failure. The small required-probe
manifest additionally requires all four phases of at least one native topology
for each known-good model/system/backend identity. An empty or entirely
unsupported report cannot satisfy that requirement. Classified exploratory
coverage gaps remain visible; they do not certify support.

`fpe-qualification.json` records the accepted source, wheel digest, shard count,
and status counts. Nightly release artifacts do not advance to Artifactory if
qualification fails. The FPE
workflow remains manually dispatchable for an out-of-band refresh. This avoids
leaving runners idle when a small system finishes before the largest systems.
Full runs also suppress repeated SDK warnings at the console while preserving
every classified failure and representative error in the matrix artifacts.
Refresh-time claims must name both runner concurrency and per-runner thread
count, plus the source SHA from the measured run.

## Website publication

GitHub Pages rebuilds after a successful FPE, Nightly, or FPE Release Nightly run
and on public-documentation changes. For main, every deployment selects the retained qualified
FPE artifact with the newest tested source commit in the current main history.
Re-running an older commit cannot displace a newer qualified snapshot. The page
shows the snapshot's source SHA, artifact creation time, and producing CI run.

Pages checks the producing workflow, successful main-branch run, qualification
manifest, row source identities, and complete shard count. It copies only the
indexed CSV data; HTML and scripts always come from main. Pull request previews
use repository data and cannot deploy. If no eligible artifact remains, Pages
fails and preserves the existing website instead of republishing stale committed
data. Run **FPE Support Matrix** on main with the current full main SHA, then
rerun **GitHub Pages** if needed. Web artifacts are retained for 90 days, subject
to the repository's retention policy. Artifacts generated before the qualification
manifest was introduced cannot be published by this path.

Staging waits for the complete matrix. Each shard has a 480-minute timeout;
the 20-shard concurrency limit and runner queues can make the total wait
longer than that per-shard limit. A successful wheel build alone does not make
the nightly available. No release-latency percentile is promised until complete
runs have been measured with this gate enabled.

The final platform-wheel check invokes the package verifier with
`--exercise-engine --exercise-fpe` on each runner. The FPE flag requires the
repository generator and Git checkout; its subprocess runs from an unrelated
temporary directory against the installed wheel. The reduced Docker build
context runs the package/runtime verifier without the repository-only FPE flag.


## Main and release branches

The published matrix's **Branch** selector defaults to `main` and lists the
repository's `release/*` branches. Share a selection using
`?branch=release%2F0.12.0`; the model search (`q`) is preserved when switching.
Each selection loads a separate packaged dataset with its own tested source
SHA, timestamp, and evidence link. Release results never fall back to main's data.

Pages discovers release branches from the fetched `origin` refs. For each
branch, it selects the newest retained qualified artifact with a source SHA in
that branch's history. Eligible producers are successful FPE or Nightly runs on
the selected branch, or the main-hosted release workflow described below with
explicit release and tooling provenance. An old-commit rerun cannot displace a newer tested commit.
A release without retained qualification is labeled **unavailable**; an
expired release artifact also removes its data from the next deployment.
The page shows a **Results not available yet** notice with a **Check again**
button and a link to that branch's coverage runs. It does not promise an
availability date because release refreshes are not automatically scheduled.
Malformed qualification fails the deployment. Main still requires a retained
qualified snapshot before the site can deploy.

**FPE Release Nightly** runs daily at 09:23 UTC from trusted `main`, and can
also be dispatched on `main`. It currently targets `release/0.12.0`. It resolves
the release tip once, builds one wheel from that unmodified checkout, and uses
the release's locked dependencies, curated model inventory, SDK, estimator,
model definitions, and performance tables. Up to 20 system/backend shards run
concurrently with eight probe threads each; the runner pool is shared with
other CI. Every scheduled run refreshes the evidence, even if the release SHA
is unchanged, so retained artifacts do not silently expire.

The probe harness and required-probe manifest come from the workflow's exact
`main` commit. CI records that tooling SHA separately from the tested release
SHA and wheel digest. Before discovery or probing, it verifies the installed
package bytes and import locations against the shared wheel. The release branch
does not need a workflow backport. This job produces qualification evidence;
it does not stage or publish release packages.

After all shards and required probes pass, CI uploads `fpe-support-matrix-web`
for 90 days and triggers Pages. The publisher verifies the trusted producing
workflow, its main-history tooling commit, the artifact's release identity,
and the tested source's membership in release history. It ranks snapshots by
tested source history, then artifact ID, while preserving the existing failed
rerun and expired-artifact protections. An older-source rerun cannot replace a
newer tested source.

Release results come only from GitHub Actions artifacts. No manual ZIP,
committed-result fallback, or main-data fallback is used. Until the first
qualified release run succeeds, the release selector shows **unavailable**.
The page links to the CI run and shows both tested source and probe tooling.

To reproduce a release result, check out the recorded tooling commit at the
workspace root and the recorded release source under `release-source/`.
Download the run's `fpe-release-wheel` artifact into the workspace (preserving
its `fpe-release-wheel/` directory and `fpe-release-shards.json`), install the
release's locked environment and exact wheel as in the workflow, and activate
that environment. The matrix's per-cell command runs the verified release
wrapper with the recorded source, tooling, and branch. Each invocation requires
a fresh `release-probe-harness/` directory; remove only that generated directory
between reproductions. Raw shard reports and the wheel are retained for seven
days; the smaller qualified web dataset is retained for 90 days.

The deployed `data/fpe-support-matrix/branches.json` catalog lists available
and unavailable branches. Main retains the existing data path, and release
data lives under `data/fpe-support-matrix/branches/release/<version>/`.
Repository previews package main's committed snapshot only and require no
GitHub credentials or artifact downloads.

## Legacy AIC snapshot provenance

The Legacy AIC Support Matrix displays a **Historical snapshot** and
**Qualification not recorded**. Its **Latest data change** timestamp is the
commit time of the most recent change to its index or an indexed CSV, with a
link to that data commit. It is not the website build time or evidence of a
complete matrix rerun. No full-matrix generation time or qualification report
was recorded for the retained legacy data.

The Pages builder adds this provenance to the packaged
`data/support-matrix/index.json`. Website-only changes do not refresh the data
date. A build with modified/untracked data, a shallow Git history, or no Git
history leaves the date unavailable. Direct source-tree previews also show the
missing-date state because the committed legacy index contains no provenance.
