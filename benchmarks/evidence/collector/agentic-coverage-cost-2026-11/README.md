# Agentic collection coverage and cost audit

This audit compares the extra **collection GPU-hours**, not allocated GPU-hours,
of broad shared-grid coverage and model-owned long-history coverage. It does not
publish smoke measurements as production performance data.

## Coverage philosophy and evidence

Each model-owned declaration needs three independently identified inputs:

| Input source | Determines | Required evidence |
|---|---|---|
| Model architecture and exact checkpoint configuration | Attention type, query/KV heads, head dimension, windows/compression, expert dimensions and valid context limit | Packaged config and model-profile path at the collector revision; checkpoint/config digest when supplied externally |
| Supported serving recipe | Local geometry after TP/DP/EP, precision, kernel/backend, chunk and verification settings | Effective native runtime logs, launched image/source revision and source audit of resolved settings; CLI settings alone are insufficient |
| Target workload | Useful history, query and correlated batch/prefix anchors | AgentX profiling **and** warmup traces; per-rank forward records when available, with source snapshot and missing-data accounting |

A model's maximum context is a validity ceiling, not a request to collect every
batch/history combination through that ceiling. Declarations reuse existing
model geometry, retain correlated workload tuples and deduplicate through the
shared planner. AgentX seeds checked-in declarations; it is not a runtime
dependency of the collector.

Classify each declared workload point as **observed joint shape**,
**configuration-derived bound**, or **chosen interpolation anchor**. Request
length traces do not establish physical per-rank forward batches. In the absence
of such records, label proposed joint tuples as interpolation choices bounded by
the audited recipe, never as observations. The existing AgentX audit freezes
808 NVIDIA configurations on 2 November 2026; it does not prove every future
serving recipe is covered. Unsupported model architectures remain explicit gaps.

## Comparable inventories

Freeze the #676 base at `69232f77ce94de85e8ca7c40da8c2e484a6bdff4`:

- **B:** that revision's unchanged normal coverage.
- **S:** the previous #664 shared-history policy reconstructed on the same
  planner/model population, using in-memory sweep overrides only.
- **M:** the refactored candidate with model-owned long-history coverage enabled.

B must be exported from a checkout whose **collector tree** equals the pinned
#676 revision; the exporter rejects a changed tree. Export S and M from the
same final candidate so retained non-history MoE/KDA changes are included in
both. The initial B/S-only snapshots in `inventory-summary.json` predate that
integration and are explicitly preliminary. NCCL is not an `OP_PLANS` operation;
count its added sizes separately using its native driver and do not omit its
cost from the series total.

Compute `S − B` and `M − B`, including old anchors newly admitted at larger
batches. Report removed cases as a regression; do not silently turn them into
savings. #676 already declares some near-1M module points, so neither the old
#664 case counts nor the review's roughly 1,500 GPU-hour extrapolation can serve
as the new delta. S is a **policy reconstruction**, not a claim that old and new
collector implementations produce identical inventories or timings.

The eight systems are the non-RTX NVIDIA entries of the auto-collector's
`AISimSystem` enum at `208c494160e9473ad4012dea77c7b5265a3bc1d7`,
`data_collection/job_planner.py`: A100 SXM, L40S, H100 SXM, H200 SXM, B200 SXM,
B300 SXM, GB200 and GB300. This matches the review's six SM90+ systems plus two
older systems. PCIe variants, RTX and XPU are outside this audit.

The CSV plan tables contain case fields and native dispatch, not arbitrary run
labels. Module `model_path` must remain in identity: it selects checkpoint,
architecture and projection/runtime semantics. Deduplicate aliases only when
the planner declares them equivalent; never discard model identity ad hoc.
Runtime memory failures are measured outcomes, not CPU inventory predictions.
MoE token lists and KDA batch/query lists are grouped tasks, not physical cells:
the comparison tool expands those axes before taking differences and retains
the original task as grouping metadata. Adding one token/verification anchor
must not make existing cells appear removed and re-added.

## Sampling and time boundary

Run direct pilots on every target system and framework; do not scale B300 times
by a GPU throughput ratio. Sample the union of both deltas. Separate operation,
phase, dispatched kernel and precision; include model-exact geometries and
expensive shared-only geometries. The tool's work quantiles are an initial
selection, not a runtime model. Add these structural boundary anchors if present:

- Dense prefill: added 24K/32K/40K/64K/128K/256K/512K/1M lengths; B1 and newly
  admitted larger batches, specifically B4 at 262,144.
- Dense/module decode: 262K/512K/786,432/917,504/near-1M history; B1, middle and
  largest declared batches. Preserve past-KV versus total-length conventions.
- Sparse prefill: q16K control and q32K extension, prefix256K and near the model
  ceiling, B1 and largest legal aggregate-query batch; full/reused indexer
  identities remain separate.
- MoE: 16K control and 32K/64K cells within the same loaded group. KDA:
  existing widths2/4/8 versus added3/5/6. NCCL: 512MiB versus 1GiB per actual
  collective/dtype/topology. Report these non-history increments separately.

For each sampled group use three fresh-worker replicates, retaining the normal
case order and compilation reuse. Compare baseline-only and expanded groups to
measure any change in loaded-model/cache-pool setup. Record:

- Group initialization and capacity growth; per-point request/input/history
  preparation; warmup/JIT/autotuning; graph capture; timed-forward loop;
  synchronization, output writing and failure cleanup.
- Forward latency and timing repetition count alongside inclusive point time.
  Forward execution is already included in inclusive time; never add it twice.
- Actual GPU identity, framework/image digest, source revision, dispatch,
  precision, topology, observed clocks and attempted/passed/failed cases.

Exclude scheduler wait, allocation/container launch and idle reservation. Start
the accounting wrapper **inside an existing container**, immediately before the
collector process. Shared initialization is charged only when the extension
adds workers/groups or changes the group's initialization. Per-point cache/history
preparation remains incremental even when the worker was already loaded.
The native driver captures actual `num_runs_executed` and helper phases for
direct calls. For native child-process module calls, use their existing point
clocks and label repetition counts as source-checked requests if the native
log does not expose executed counts. Disable power-adaptive repetitions for
these pilots and record that setting. Phase logs contain nested benchmark
breakdowns, and some module first-point logs also expose shared setup: use
inclusive group/point clocks for totals, not the sum of every printed phase.

Historical B300 anchors in `historical-b300-anchors.json` separate instrumented
worker, native-call and loaded-runner point boundaries. They are not newly timed
candidate anchors and do not establish fleet totals.

## Budget, estimates and acceptance

The approved 12 collection GPU-hours are partitioned into 8 anchor, 2 correctness
and 2 validation GPU-hours. This audit owns 10: **1 anchor GPU-hour and 0.25
validation GPU-hour per system**. The correctness campaign owns the other 2.
Unused quotas require coordinator reassignment before transfer. A file lock
protects one ledger, not independent cluster filesystems: each system receives
its fixed quota and must keep one ledger for every local job using that quota.
Reserve collection timeout plus five seconds of cleanup per participating GPU
before launch. Unfinished reservations remain charged in full. No GPU jobs may
launch before the signed integrated candidate and per-system budget are recorded.

Fit preparation and execution separately using matched native measurements.
Use deterministic held-out anchors (20% where the bucket size permits). Require
held-out per-point error ≤20% and representative grouped-total error ≤15%; add
the largest-residual point and split the stratum if either fails, within the
same budget. After two refinements, report the remaining uncertainty instead
of claiming a measured total. Record first-use compilation separately from
warm-cache work. Do not extrapolate across framework, precision or system.

Report per framework × system × operation: added cases/groups, incremental setup,
preparation/warmup/forward seconds, failure cost, measured/extrapolated fraction,
central and conservative empirical GPU-hour range, and `S − M` savings.
Unavailable or unsupported lanes stay **unmeasured** with a reason; no invented
zero-hour total. GPU-hours are incremental inclusive seconds × participating
GPUs / 3,600. Allocation time is never a cost input.

## Reproduction tools

From `python/aisimulate`, using the supported development Python:

```bash
# Point this to a checkout at the exact pinned #676 collector tree.
base_collector=/path/to/pinned-676/python/aisimulate
python tools/perf_database/collection_cost_inventory.py --backend vllm \
  --collector-root "$base_collector" \
  --system b300_sxm --scenario baseline --output /tmp/cost/B
python tools/perf_database/collection_cost_inventory.py --backend vllm \
  --system b300_sxm --scenario shared --output /tmp/cost/S
python tools/perf_database/collection_cost_inventory.py --backend vllm \
  --system b300_sxm --scenario model --output /tmp/cost/M
python tools/perf_database/collection_cost.py compare \
  --baseline /tmp/cost/B --candidate /tmp/cost/M --output /tmp/cost/delta.json
python tools/perf_database/collection_cost.py sample \
  --difference /tmp/cost/delta.json --output /tmp/cost/anchors.json
```

The model scenario requires the refactor's `include_long_history` planner API.
S and M commands above run from the final candidate checkout, and every summary
records its actual source commit and collector-tree identity. Native sampling
refuses registry operations marked unverified on that SM; report their planned
population separately from executor-eligible totals.
Native pilot specs passed to `collection_cost.py run` contain a unique `run_id`,
`argv` array, `cwd`, participating `gpus` and `timeout_seconds`. Use argument
arrays, not shell interpolation. Supply `--ledger` and the fixed system quota
as `--budget-gpu-hours`. The output contains actual collection duration and
exit/timeout status; inspect native CSVs and phase logs before qualifying results.
