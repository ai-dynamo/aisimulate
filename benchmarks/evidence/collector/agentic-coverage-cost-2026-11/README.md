# Agentic collection coverage and cost audit

This audit compares the extra **collection GPU-hours**, not allocated GPU-hours,
of broad shared-grid coverage and model-owned long-history coverage. It does not
publish smoke measurements as production performance data.

Read the [cost summary](cost-summary.md) for qualified components, exact case
counts and remaining unknown totals. [NCCL components](nccl-cost-components.json)
are shared system costs reported separately from the three framework lanes.
The report snapshot and raw measurements identify their own input revisions;
new evidence requires a fresh report, not relabeling an earlier snapshot.

The [review ledger](review-ledger.md) records fixes, evidence and qualification limits across #664, #665 and #666.

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

### Per-model declaration evidence

The companion [model evidence](model-declaration-evidence.json) records exact
profile/config hashes at candidate `4eaef24ff72a4dce5d4acf2976b746a65de3ca19`,
effective TP/DP/precision/backend values with source-log URLs/page hashes,
profiling/warmup extrema, and the complete declared point lists. Framework
recipes are source evidence; they are not assumed identical to the separately
pinned native collection runtime. Resolve attention-DP before deriving local
heads; global TP/DP flags cannot be substituted for effective attention TP.

| Model family | Geometry/profile evidence | Serving and workload evidence | Declared coverage and classification |
|---|---|---|---|
| Qwen3.5-397B | `Qwen3_5MoeForConditionalGeneration` profile and exact Qwen config; 32 query / 2 KV heads, d256 before effective TP | Native Qwen recipes linked in the companion; maximum same-request history 256,582, including both phases | Model dense geometry through 262K; preserve #666's 95-point B/length calibration frontier. Points are interpolation/calibration choices, not observed per-rank forwards. |
| GLM5.2/5.3 | `GlmMoeDsaForCausalLM` profile and both exact configs; separate full/reused-indexer identity | Effective DSA recipes linked in the companion; maximum history 959,120 | Intermediate 786,432/917,504 history, bounded near-1M batches and B52 × 22-length calibration frontier. Configuration-derived history span; selected joint points are calibration choices. |
| Kimi-K3 | `KimiK3ForConditionalGeneration` profile and exact Kimi config; MLA local heads from actual TP shards, separate KDA widths | Audited active AgentX lane is vLLM; maximum history 950,779 occurs during warmup | New optional MLA points are **withdrawn and deferred**: the generic DeepSeek factory cannot qualify Kimi NoPE/cache/output-gate semantics. This is an uncovered native-adapter gap, not zero-cost completed coverage. Existing #676 coverage and independent KDA widths remain. SGLang standalone-tail and dense 1M fallback gaps also remain. |
| MiniMax-M3.1 | `MiniMaxM3ForCausalLM` profile and bundled M3 geometry, with exact config digest | Effective MSA recipes linked in the companion; maximum history 959,333 | Intermediate history/q32K and legal decode frontier. Core geometry is source-matched; exact M3.1 checkpoint precision policy remains conditional. |

These smaller populations target the selected audited recipes; they do not
qualify all AgentX models. Muse-Spark must not use Muse-Glimmer geometry. Qwen3.8-Flash-Next's QSA
route is not bundled Qwen3.8-Max. DeepSeek-V4.1 needs its own native module and
feature qualification; its short manifest and unqualified speculation/DP/CP
routes are not repaired by generic long-history metadata. These three families
remain explicit support/collection gaps, excluded from a claim of complete
AgentX coverage. Silicon calibration of the Qwen95 or GLM B52 frontier does
not turn those points into observed workload batches.
This campaign withdraws only the new optional Kimi MLA points: the native
NoPE module differs in cache binding, weight processing, forward signature and
output gating. The c4 evidence snapshot retains the proposed points as
**historical withdrawn declarations**, not final M coverage or qualified timing.
Preserve #676 normal coverage and the independent KDA width extension. Report
the Kimi omission separately from savings on equivalent supported workloads.

Canonical attention equivalence does not establish checkpoint setup or memory
equivalence. SGLang initializes full dummy layers before extracting attention;
expert-only quantization can change initialization cost and available KV
capacity. Preserve the pinned normal representative, record its runtime refusals
as gaps, and use a targeted model run for checkpoint-specific qualification.
The NVFP4 B52 recollection is not a BF16 startup/capacity cost anchor.

## Comparable inventories

Freeze the #676 base at `69232f77ce94de85e8ca7c40da8c2e484a6bdff4`:

- **B:** that revision's unchanged normal coverage.
- **S:** the previous #664 shared-history policy reconstructed on the same
  planner/model population, using in-memory sweep overrides only.
- **M:** the refactored candidate with model-owned long-history coverage enabled.

B must be exported from a checkout whose **collector tree** equals the pinned
#676 revision; the exporter rejects a changed tree. Export S and M from the
same final candidate so retained non-history MoE/KDA changes are included in
both. The final `inventory-summary.json` records all 24 system/framework lanes,
B/S/M source commits, collector-tree and tool hashes, task and normalized
inner-cell counts, exact additions/removals, and executor eligibility. B is
pinned676; S/M use final4ea geometry. Runtime validity remains separate. NCCL is not an `OP_PLANS` operation;
count its added sizes separately using its native driver and do not omit its
cost from the series total.

Compute `S − B` and `M − B`, including old anchors newly admitted at larger
batches. Report removed cases as a regression; do not silently turn them into
savings. #676 already declares some near-1M module points, so neither the old
#664 case counts nor the review's roughly 1,500 GPU-hour extrapolation can serve
as the new delta. S is a **policy reconstruction**, not a claim that old and new
collector implementations produce identical inventories or timings.
The shared fixture includes standalone MLA's expanded context lengths and 1M
context budget, plus 262K/512K/1M decode lengths. These are separate from the
MLA/DSA/MSA module grids and must contribute their own added cases.

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

The approved12 collection GPU-hours now contain7 anchor,4.5 validation and
0.5 correctness GPU-hours after coordinator transfers. A100 has no remaining
lease: its unstarted jobs were canceled at the resource cutoff with zero
collection spend. B300 validation is1.875 GPU-hours; H100 validation is0.75;
the other five systems each retain0.375 validation. All seven active systems
retain1 anchor GPU-hour each. Audit allowance is11.5 and correctness0.5.
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
For grouped MoE/KDA anchors the native driver verifies every requested scalar
cell against the generated group, then constructs only valid subset axes.
Irregular KDA samples split into rectangles without introducing extra cells.
Paired baseline/expanded MoE runs must include changed maximum-size autotuning
and workspace allocation, not just two extra timed forwards. A poisoned CUDA
context stops the worker and leaves remaining groups unattempted for a fresh
worker retry; its fail-fast follow-on times are not independent shape samples.
Honor the native normal worker-recycle protocol too: a persisted task followed
by `EXIT_CODE_RESTART` or `WorkerRestartSignal` is completed work, not a failed
kernel. Mark remaining groups unattempted and continue in a fresh worker under
the same budget. Do not disable recycling to make an anchor pass.
NCCL weights each collective by its actual participating 2/4/8-rank group,
never by the entire reserved node for the whole harness.

### Cost report generation

From `python/aisimulate`, generate a report from normalized native samples and
frozen baseline/shared/model inventories:

```bash
audit_root=../../benchmarks/evidence/collector/agentic-coverage-cost-2026-11
unzip "$audit_root/frozen-plan-tables.zip" -d /tmp/pr664-frozen-plans
python -m tools.perf_database.collection_cost_report \
  --samples "$audit_root/measured-samples.jsonl" \
  --source-equivalence "$audit_root/source-equivalence.json" \
  --repository ../.. \
  --source-commit c569abb1cd034c934f47974209515e3507f60b8a \
  --baseline /tmp/pr664-frozen-plans/baseline \
  --shared /tmp/pr664-frozen-plans/shared \
  --model /tmp/pr664-frozen-plans/model \
  --output /tmp/pr664-cost-report.json
```

Each JSONL record identifies system, backend, operation, exact scalar CSV `case`,
`source_commit`, `sample_id`, `group_id`, `measurement_scope`, `status`, inclusive
`collection_seconds` and separate `setup_seconds` when available. Retain
`forward_latency_ms`, actual `num_runs_executed` (or null) and `phase_seconds` as
supporting evidence. Add `scenario` (`baseline`, `shared`, `model`) and `role`
(`fit`, `holdout`). Failed, timed-out and unattempted records never fit timings.

Point estimates require the same checkpoint, geometry, precision, dispatch,
framework, device and source revision. Interpolation requires measured corners
on every workload axis; it never extrapolates. Interpolated components require
at most 20% relative error on explicit held-out points. Independent group
repeat holdouts check forecasts within 15%. Repeated observations bound empirical
ranges; these are not statistical confidence intervals. With one observation,
the range has no measured repeat-variation information.

Inclusive group clocks already contain setup and forwards. For point clocks,
separate setup is charged once per group; forward latency and nested phases are
never added again. Matched native group records carry `paired_group_id`, exact
`group_member_cases` manifests and `participating_gpus` (one for current
single-device op-plan calls; explicit rank count required for NCCL). Their
candidate-minus-baseline difference includes changed initialization and buffer
costs without charging already-paid work twice. Finite total extra GPU-hours
require paired group evidence covering the exact inventory additions/removals.
Otherwise the report exposes supported point components, uncovered counts and
null totals. Deferred or undeclared routes, including native Kimi NoPE history,
remain outside the measured inventory; zero inventory additions do not qualify
those routes. Fleet forecasts remain null while lanes are unqualified.
The immutable SGLang 0.5.21 multi-platform index is preserved verbatim in
[`sglang-0.5.21-index.json`](sglang-0.5.21-index.json). Its payload digest and
registry response digest both equal `b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`.
[`sglang-0.5.21-manifest-proof.json`](sglang-0.5.21-manifest-proof.json)
records the AMD64 `2dbe4c7f…` and ARM64 `1f176b3a…` child mappings.
This proves image-index membership; actual imported squashfs hashes remain
separate run receipts. No registry credentials are retained.

Grace-system scheduler policy requires four reserved GPUs. The audit executes
three concurrent framework workers using exclusive one-GPU steps, records
`SLURM_STEP_GPUS`, `CUDA_VISIBLE_DEVICES` and the actual GPU UUID, and locks
that UUID while a worker runs. The fourth reserved GPU is idle and excluded.
These workers share a Grace host and request eight CPUs each (32 CPUs for the
allocation); their timings are not isolated-host CPU bandwidth measurements.

For full causal dense prefill (`attention_context`, `window_size=0`), sequence
interpolation uses squared `input_len` for execution. When explicitly recorded,
nonoverlapping `preparation_seconds` and `execution_seconds` must sum to the
inclusive point clock; preparation uses ordinary sequence length and execution
uses squared sequence length. Without those clocks, only inclusive time is fit,
with no claimed phase decomposition. Nonlinear fits still require an independent
held-out point within the 20% gate: the measured B300 vLLM 262K/1M endpoints
predict the withheld 512K call within 6%, whereas ordinary sequence interpolation
fails. This does not authorize extrapolation or other native geometries.

An older op anchor can be admitted with `--source-equivalence PATH --repository
REPO`. The JSON `proofs` list names source/target revisions, backend, operation
and identical file SHA256 values (`source_sha256`, `target_sha256`). The report
verifies both immutable Git revisions for the dense implementation, registry,
helper, case-generator and registry-types files, plus vLLM utilities. Admission
is limited to exact scalar native case arguments with the same geometry and
protocol; it is not whole-collector or audit-driver equivalence. Changes to
selection metadata and failed-call handling are outside that proof. The report
retains actual source SHAs and the complete supplied proof, including protocol
qualifications. Historical anchors without observed repetition counts or phase
clocks keep those fields null.

The same proof mechanism admits SGLang `dsa_generation_module` evidence only for
the proof's explicit `model_path`. It additionally verifies the native module,
runtime compatibility/limits and prefill-graph helpers, plus that checkpoint's
cached configuration and quantization configuration. It cannot transfer a GLM
measurement to another checkpoint. All observed status labels, including partial
worker output, remain in the report; only passed observations fit timings.
Physical point anchors may support shared and model-owned deltas alike when
native case semantics match; scenario labels remain binding for paired group
setup comparisons. Full and reused-indexer DSA operations remain separate.
The final ordinary one-pass native measurements are preserved in
[`measured-samples.jsonl`](measured-samples.jsonl), with inclusive worker clocks,
per-point clocks where available, forward latency, native phases and failures.
[`timing-receipts.json`](timing-receipts.json) preserves the underlying native
and wrapper results with original raw-byte hashes.
[`source-equivalence.json`](source-equivalence.json) proves only the listed
operation implementations and dependencies are identical across measured and
report sources; measured SHAs remain unchanged. It is not whole-collector
equivalence. [`pilot-jobs.json`](pilot-jobs.json) records all eight target
submissions; submission is not completion or qualification.

Initial job-shell probes consumed only their first case group; failed startup
imports and timeouts remain charged. Follow-ups fix input forwarding and the
collector import path. TRT MoE normally recycles after its first native mode;
partial groups are excluded from full-group estimates. Across-job cold/warm
cache differences do not establish setup savings. Validation pairs reuse the
same already-populated cache in fresh workers. NCCL, KDA and all other
unmeasured strata retain unknown time subtotals; this snapshot establishes
measured components, not a finite complete fleet total.

[`accounting-snapshot.json`](accounting-snapshot.json) preserves timestamped
per-system anchor/validation ledger events and exact scheduler states. Finished
collection seconds include failed attempts. Open reservations are charged in
full until their receipt arrives; missing ledgers denote work not yet started,
not successful zero-cost coverage. Allocation durations appear only in scheduler
provenance and are never used as collection costs. That historical snapshot predates the final coordinator transfer below.

The S generation fixture unions original #676 points with the guarded broad
extensions. Applying the new KV-index guard to the entire old grid would remove
existing anchors and understate collection cost; the fixture and regression test
retain those baseline points. This is an incremental shared-grid counterfactual,
not a literal replay of old #664 deletions.

The corrected S census uses fixture `a4d70f51c` and final4ea model geometry.
Across all24 lanes and every inventoried operation, both B→S and B→M have
zero removed physical cells. Previous pre-union S removal counts are
superseded; no deletion is priced as reduced collection work.

The earlier407-record snapshot is superseded by the final1018-record file. Observed direct dense
benchmark counts and phases are retained; older52 dense source95 points have
no observed count. For a one-benchmark-per-call direct point, execution is the
sum of the helper’s five disjoint warm-up/capture/timed/cleanup phases and
preparation is the native-call residual; shared worker initialization remains
separate. Grouped MoE/KDA keeps every actual benchmark measurement without
assigning its entire group clock to each token/width.

[`nccl-cost-components.json`](nccl-cost-components.json) records per-system
NCCL1GiB incremental native-call components, actual participating ranks and
latency rows. The native harness is preserved in
[`native-nccl-harness.py`](native-nccl-harness.py), with its actual hash in run
receipts. It calls the existing collector: no timing proxy or NCCL build. Native
requests use40 warm-ups and60 timed iterations for these large messages; the
per-call wall clock includes subprocess startup and those benchmark calls.
Native CSV message_size is elements, so half divides the byte anchor by two.
NCCL is shared per system via TRT, not charged once per framework. Partial
rank groups and missing parent-setup proof retain unknown totals.

The pinned SGLang0.5.21 image lacks `sglang.kernels.ops.kimi_k3` needed by
Blackwell KDA **verify** widths2–8. B200/B300 verify attempts fail strict
completeness with zero rows. This does not establish that KDA context or
generation is unsupported. Hopper verify probes pass. Preserve #676 defaults
and record the missing native verify qualification; do not substitute kernels.

The published NCCL harness adds the required SPDX header only. Its exact executed
bytes are preserved in [the original archive](native-nccl-executed-harness.zip),
with [executed/published hashes](native-nccl-harness-hashes.json). Native run
receipts retain the executed hash; no historical source identity is relabeled.
[`frozen-plan-tables.zip`](frozen-plan-tables.zip) preserves every final B/S/M
planner CSV and lane summary:864 files across24 lanes per scenario. The
embedded plan-manifest records each file’s SHA256, byte size and CSV task/
dispatch-row count plus exact source/environment identities; normalized inner-
cell counts remain in inventory-summary.
[`frozen-plan-tables.manifest.json`](frozen-plan-tables.manifest.json) mirrors
that manifest and records the archive hash. Every ZIP member hash and archive
CRC was verified after creation. B uses pinned676; S is the corrected final5
superset fixture; M retains final4ea geometry. These frozen files require no GPU
work to inspect or reproduce the cost census.

An intermediate coordinator transfer raised B300 validation from0.375 to0.875
GPU-hours. Later A100 cancellation reassigned its unused allowance; the final
per-system quotas and charges are in [the budget summary](final-budget-summary.json).
Audit allowance is11.5 and correctness allowance0.5, totaling12 GPU-hours. The correctness owner confirmed all seven completed jobs have an
administrative allocation upper bound of1711 GPU-seconds (<1800) and will run
no more jobs. That bound permits the transfer; it is not substituted for actual
collection cost in any anchor or estimate. Older timestamped ledger snapshots
retain their contemporaneous quota history.

[ComputeLab image receipts](compute-lab-image-relay.json) record exact reuse of
four previously measured AMD64 squashfs images after scratch-filesystem
whiteout failures blocked fresh imports. CPU job5599744 verified all four
SHA256 values before publishing them. Transfer, CPU validation and allocation
wait are excluded from collection GPU-hours; native runtime identity is still
checked inside each benchmark worker.

Group forecast validation uses independent controlled repeats: two native
training groups predict a third with the same physical member manifest,
verified runtime, rank count and warmed-cache protocol. The 15% gate is
separate from phase accounting. Summing observed point clocks and their residual
setup checks conservation only. Repeated pairs cover each physical addition
once; they are not three copies of campaign work. Cold or unproven-cache pairs
remain evidence but cannot qualify a warm marginal setup forecast.

The [GLM B52 warm-preparation component](warm-history-preparation-fit.md)
provides a reproducible affine-history fit with 17 training and four held-out
points per indexer mode. Maximum held-out errors are 9.54% and 8.45%.
First-use measurements remain separate. This component does not transfer
NVFP4 setup/capacity to BF16 or establish a whole-group/fleet GPU-hour total.
[`fit-inputs-597.zip`](fit-inputs-597.zip) freezes the exact597-record dataset
and source-proof inputs used by the interim report, plus contemporaneous fetched
native/wrapper logs, latency CSVs, image hashes and reused C44 phase logs. Its
manifest hashes every member; the sidecar records the archive and dataset
hashes. Executed normalization producer bytes are archived as audit references
with workspace-specific paths. No model weights or container images are included.
This frozen fit input is distinct from the earlier407-record rendered sample
snapshot and from subsequent live campaign receipts. All ZIP CRC/member hashes
were checked.

**[Attribution correction](fit-inputs-597.disposition.json):** `fit-inputs-597.zip` and its dataset31708… are
historical and superseded by
[`fit-inputs-597-corrected.zip`](fit-inputs-597-corrected.zip), datasetfa257fc68a7f25a6333ba4d58244d259f9eb48d7354c8348fc2b8d3d39396b46.
The planner executes native rows in its own order.44 fresh context samples
previously attached B1/B32 labels by request index; the corrected producer
matches actual native arguments, precision/window and dispatch. Measured clocks
are unchanged, and holdout assignments follow the predeclared physical cases.
The independent audit checked all140 dense identities and unchanged grouped
MoE/KDA/module manifests.
[`normalization-correction-597.json`](normalization-correction-597.json) lists
every affected sample. The old raw archive remains as provenance and must not
be used to fit geometry costs.

The optional audit-driver field `observe_allocations` records native cache
constructor results without changing kernels or replacing returned objects.
vLLM records initial and final-layout cache storage, SGLang records K/V pool
buffers, and TRT records its unique primary pool. Shape, dtype, stride, logical
bytes and unique underlying storage bytes distinguish real allocation floors.
Views are deduplicated by hashed storage identity; tensor objects are never
retained. Exact Q/K/V forward arguments are observed through the native
benchmark closure (TRT records its packed forward argument, not an invented
extra-buffer total). Observation errors remain explicit and invalidate a
feature fit. Small CPU observation overhead belongs to the inclusive call clock.
The driver also records actual framework/Torch/CUDA-build and already-loaded
CUDA-runtime API versions, GPU UUID, image-hash receipt and thread settings.
Unknown host concurrency remains null. This audit instrumentation is separate
from the frozen collector/kernel revision and does not enable long history.

For feature-identification only, `native_passes: 2` repeats the same selected
native rows within one worker and records pass identity. Each shape’s first-in-worker use
and its subsequent use stay separate; three fresh matched groups can then
validate the subsequent-use component independently. Compiled caches may already
exist: neither pass is labeled a guaranteed cold or warm compiler state. Every call remains inside the bounded
collection clock. The default is one pass; repeated sampling adds no coverage
claims or collector behavior changes.

[`a100-cutoff.json`](a100-cutoff.json) records the coordinator cutoff: all14
remaining A100 jobs were pending, canceled with zero elapsed allocation time,
and both collection ledgers were absent. A100 remains unmeasured because no
canonical80GB SXM capacity was available; no other GPU rate is substituted.
The unused1.375-hour lease was transferred to B300 validation (+1) and H100
validation (+0.375), preserving the11.5 audit/0.5 correctness/12 total cap.
Older timestamped quota snapshots retain their historical values.
The corrected597 report replacement is recorded in
[`fit-inputs-597.disposition.json`](fit-inputs-597.disposition.json); the old
wrong-label dataset and report remain explicitly superseded.

Physical GPU UUID is recorded outside software runtime_protocol. A matched
pair can require the same UUID; repeated trials on different GPUs of the same
SKU remain visible as hardware variance rather than different software. A
two-pass worker clock is an experimental observation, not a normal one-pass
collection forecast. Only explicitly scoped pass components may enter a fit.

The conditional dense pilot uses context16K/32K rather than the planned
B4/Q32/d256/64K point. A [preexisting H20 warning](https://github.com/ai-dynamo/aisimulate/blob/e8cbe2dc150f0df309993474842c2e417753873d/python/aisimulate/collector/trtllm/collect_attn.py#L486),
already inherited by #676, associates faults with large packed Q/O offsets.
Its proposed `(B-1) * query_length * Q_heads * head_dim * 2` byte threshold is
explicitly unverified against kernel source, and no committed raw fault trace
was found. It is an empirical warning, not a proven H100/B300 kernel contract.
The omitted point remains planned and unmeasured here; assign neither success
nor zero cost. B1 avoids that particular offset expression, and decode uses
query length1; other memory, context and kernel restrictions still apply.
Tensor-observed point clocks start after the audit pending/progress receipt is
written, so growth of audit JSON cannot masquerade as cache preparation. The
whole bounded worker clock still includes those writes and charges them to the
measurement budget. Native cache/input observations remain inside the point
clock and their small CPU overhead is explicit. The wrapper-minus-point residual
contains startup, planning, audit I/O and teardown; it is not pure initialization
or an independently qualified one-pass forecast. Earlier clocks retain their
original boundary and are not rewritten.

The complete tensor-observed study is frozen in
[`dense-feature-study.zip`](dense-feature-study.zip), with its archive hash in
[`dense-feature-study.manifest.json`](dense-feature-study.manifest.json).
It contains 864 successful point calls from 36 fresh workers across B300/H100,
three frameworks, context/decode, three replicates and two passes, plus 12 smoke
calls kept as diagnostics. Each phase uses two training geometries and a whole
third geometry held out across every replicate. Actual cache storage, input
storage, native phase clocks, repetitions, runtime versions and signed audit
payloads are retained. These two-pass groups are excluded from the ordinary
one-pass fleet report; a separately gated component fit uses subsequent passes.


The completed campaign is frozen in
[`fit-inputs-1018-final.zip`](fit-inputs-1018-final.zip), with per-member hashes
and archive identity in [`fit-inputs-1018-final.manifest.json`](fit-inputs-1018-final.manifest.json).
This final input replaces the407/597 interim datasets. It contains the original
collector source archive, exact selected cases, native results/logs/latencies,
actual image hashes, signed audit tooling and source-equivalence proofs.
The separately archived876-point tensor study is excluded from the ordinary
one-pass input. All seven cluster queues are empty in
[`final-queue-receipt.json`](final-queue-receipt.json); A100 remains unmeasured.

Independent review verified all final1018 physical case identities and phase
partitions. Sparse prefill points use actual call-index/precision/prefix identity,
with native CSV `isl` as query length and `step` as prefix. Sixteen historical
TRT DSA phase partitions now count the separately marked dry-run forward as
execution; inclusive point times and selected cases are unchanged. Native
`bench_*` subphases are never added again to their enclosing `bench`.

[`final-budget-summary.json`](final-budget-summary.json) reports7.263664 measured
audit GPU-hours and7.324775 conservatively charged GPU-hours, including two
unclosed historical reservations totaling220GPU-seconds. Adding correctness's
separate administrative upper bound gives7.800053GPU-hours, below the12-hour
cap. Allocation and idle reservation time are excluded from collection cost.
The final leases remain11.5audit/0.5correctness; no further GPU jobs will launch.

The final NVFP4 GLM control uses B1/B32 at six exact generated histories per
full/skip-indexer variant. All12points passed. Each variant spends about156seconds
inside points plus30–32seconds in worker setup/overhead. History preparation
accounts for147–148seconds per variant; timed-forward benchmark phases are small.
These chosen calibration points are separate from canonical BF16 coverage and
observed per-rank demand. They establish actual collection components without
qualifying a transfer to other checkpoints, frameworks or the whole fleet.
