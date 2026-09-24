<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.3-Flash native operator collection and GB300 profiles

## Implementation contract

Add IndexPool and GLM NoPE sparse-MLA collection and extend native KDA, mHC fusion, dense/MoE and TP collective coverage for both runtimes. Coordinate producer keys with Rust SILICON consumers. Required operations must have measured provenance without analytical fallback or duplicate fused work. Independent prefill and decode whole-forward holdouts must each achieve MAPE <=20% in every required deployment cell.

The required matrix is GB300 × vLLM/SGLang × native FP8/NVIDIA NVFP4 × TP2/TP4, through 131072 context tokens. NVFP4 TP1 is optional after measured memory admission. Text-only scope excludes vision execution, MTP, expert parallelism, offload and cross-node serving.

## Source identities

- FP8: `zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`.
- NVFP4: `nvidia/GLM-5.3-Flash-NVFP4@09b04e5e74bca08ca8549fc736d4cdd8624bfde3`.
- vLLM: `v0.30.0`, source `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- SGLang: `v0.5.20`, source `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`.

These are qualification candidates, not evidence of measured GB300 coverage. Preserve the checkpoint's actual per-module precision; do not relabel all weights as FP8 or NVFP4. TensorRT-LLM serving support is outside this initial matrix.

## Acceptance and data status

Native Ops collection and its Rust measured consumer are implemented on the
shared SOL graph. The observer covers all 45 attention modules, 45 whole FFNs,
backend-specific mHC sites, embedding, final norm, logits and 91 TP all-reduces.
Local compute and synchronous collective intervals are separately accounted;
unobserved or overlapping boundaries fail admission. Existing published-table admission remains eager only. A separate source-bound
FULL decode table consumer is implemented, including an explicit setup marker
and actual capture-size dispatch; no graph table is supplied yet. An experimental SGLang FULL
decode observer now assigns native capture nodes to operation boundaries and
requires an exact API-backed join to CUPTI replay activity. It inserts no CUDA graph nodes;
its separate holdout observes only the complete native replay. This path emits
raw evidence with an implemented raw-to-table exporter and positive calibration
lineage binding. Actual native model qualification, profiling controls and
independent graph accuracy remain pending. Mixed or aggregate-only inputs that lose graph geometry are rejected.

Measured lookup keeps exact physical identity, backend, checkpoint, TP and phase.
Bounded workload interpolation requires complete measured corners with matching
observed kernel signatures and state/graph policy; KDA initial-state and IndexPool
short-context/tail partitions remain separate. Native tile counts are workload
coordinates, not fabricated kernel boundaries. Missing coverage fails in both
SILICON and HYBRID; neither mode silently falls back to SOL. The installed wheel
constructs all eight required model graphs and rejects missing measurements in
all eight SILICON and all eight HYBRID checks.

The graph consumer review fixes pass 27 Rust GLM tests, 17 Rust compiled-spec/serialization
tests and 104 focused Python model, compile, engine-step and policy tests. A rebuilt
public native API check rejects complete schema-20 specifications with zero or two
setup markers, and verifies static/native-total coordinate equivalence through the
inclusive 128K boundary. The preceding graph implementation passed 431 GLM collector
tests (3 skipped); shared repaired-runtime integration passed 113 focused tests.
Graph-node/hook validation includes 40 CPU tests covering CUDA13 ABI
pointer writes, complete kernel/memcpy/memset activity matching, native
enumeration consistency and retention of nondefault dependency metadata. The preceding
graph/runtime change passed 112 focused tests. Historical validation passed
17 Rust GLM tests and 79 Ops contract, evidence, native V2 runtime and shared
retained-lifecycle tests. The fresh installed wheel from commit `855a81d4` verified
all 2,178 hashed RECORD members against installed bytes, 18 collector/evidence
helper imports, all eight model graphs and 16 strict missing-measurement queries
outside the source checkout. Its SHA256 is
`c37e481b27082e7e240ffdc09a5704d99b2a64fdbd1720d25795c402a89c5285`.
Its installed Rust consumer also reproduced both coherent-rank smoke queries below.
The earlier broad collector
run passed 2,362 tests (8 skipped); that count is historical and predates the
latest bounded evidence and native launch fixes. These checks establish software
behavior and evidence integrity, not native GPU performance coverage.

Actual FP8 TP4 eager smoke produced every declared operation on all four ranks
for two SGLang prefill points and, in the subsequent vLLM v4 smoke, two prefill
points and one decode point. Original
postprocessing failures remain preserved; exact pinned mHC source-ownership
normalization permits separate complete reaggregation receipts. The SGLang smoke
exposed overcounting from selecting a different maximum-latency TP rank per
operation. New collection selects one rank per actual forward using its largest
recorded whole-forward interval, with lowest-rank ties, and keeps all that rank's
operation measurements. It records the policy and complete selection evidence;
old per-operation-max tables keep their historical behavior. No residual or
scaling factor is introduced. On the same two instrumented calibration points,
the installed wheel's Rust consumer differs from native DeviceTimer by 3.29% and 2.26%;
these are internal diagnostics, not independent holdout accuracy. Old vLLM smoke
lacks whole-forward GPU intervals and cannot be relabelled with the new policy.
The vLLM v4 native run recorded coherent-rank whole-forward evidence and actual
kernel signatures; its original finalization failed only because `pyarrow` was
absent. Separate offline finalization preserved every original raw, rank-selection
and evidence hash and produced 25 prefill and 13 decode physical rows. Current
strict native evidence validation and the rebuilt Rust consumer pass. Same-calibration
errors are 3.33%/5.01% for prefill and 4.68% for decode, again instrumented diagnostics
with no independent accuracy claim. These frozen v4 observations use stock vLLM
eager execution and do not establish repaired-runtime or native graph performance.
**All eight deployment cells retain data/accuracy status `NOT_EVALUATED`.** No
measured GLM profiles or independent 20% MAPE pass are included. Native framework
qualification and Ops GPU observation are separate evidence streams.

Bounded shards retain every original point and independently identified native
run. Publication requires a complete, disjoint union with immutable point maps.
Repeated physical operator keys use the lowest frozen original point ID, never a
latency-selected owner; all observations remain in the evidence.

The stock vLLM IndexPool helper failed actual GB300 cached-prefill controls at
unaligned prefix 4097 while aligned controls and one-shot references matched.
The measured producer and consumer therefore reject stock vLLM cached prefill
with `prefix % 4 != 0` and `query >= 2`. Original requested points remain
unqualified; they are not removed from acceptance. The [native probe receipt](glm53flash-kpool-native-gb300.json)
preserves the concrete failures. A separately versioned repair candidate requires
its own cache and complete Engine qualification before this restriction can be
relaxed for that exact runtime. A separate source-overlay diagnostic on GB300
passed all 16 uniform/nonuniform-gate, heterogeneous-batch and retained-tail
cache comparisons. That result does not qualify a built runtime or model outputs.
The private versioned-wheel build preserves the actual container binaries and
records the observed PyPI/container Rust executable difference. The FP8 TP4
versioned candidate passed the frozen native Engine suite in GPU job 606669:
20 requests per profile with 32 generated tokens, actual heterogeneous B2/B4
cached-prefill cohorts, complete TP state/token chains and exact output agreement
between stock/reference, candidate/reference and candidate/split. An independent
recheck reproduced the receipt and verified all 60 raw file hashes, plus the
original checkpoint config and revision identity. This is bounded functional
qualification. The same required Engine suite subsequently passed FP8 TP2 and
NVFP4 TP2/TP4; the portable twelve-profile receipt chain is bound by summary
`d43dfdcfabe870cc51983fa41fada4897b4d84d64ac57fafe2236e7753435e67`.
The shared runtime helper admits only that exact built version and source/binary
closure. Ops producer, worker, evidence reader and Rust table gates now share
that exact version; every repaired worker must supply observed source and native
binary hashes, and tables must match the qualified effective source hash.
Frozen plan versions cannot be replaced by raw self-declarations. These gates
and functional qualification do not qualify Ops latency data.
Stock support is not claimed repaired.

The first native vLLM Ops attempt completed request execution but produced no
rank observations because the runtime selected its V2 runner. The observer now
wraps that actual class and joins its separate model and logits completion
phases; its whole-GPU event ends at `compute_logits`, before sampling. An actual
pinned-class CPU probe verified all four entry wrappers. SGLang uses the actual
Engine constructor and initializes the same overlap result queue as its native
loop. Subsequent GPU attempts exposed native initialization warmups entering the
vLLM observer and an SGLang lazy QKV callback nested inside attention. Both have
specific source-bound fixes and new immutable retries; prior failures remain
intact. No completed Ops performance qualification is claimed.

An actual GB300 primitive test established that capture-bound external CUDA
events update on replay, but increased the measured outer window by about 47%.
That result is preserved as a failed timing-equivalence control. The separate
read-only node observer first failed CPU binding against CUDA13; installed
headers and symbols now establish the corrected seven-argument capture-info
and five-argument edge-query ABI. The next GPU probe captured and replayed
successfully, but strict matching rejected different capture and executable
graph/node identities. No positional or bit-field inference admits these nodes.
The corrected CUPTI resource/clone mapping subsequently passed actual GB300
tiny capture/replay qualification in both runtime images (job 609710), with exact
node ownership and no inserted graph nodes. All three following SGLang model
profiles stopped before construction at a strict multiple-loaded-CUDA-runtime
identity guard. A subsequent GPU-visible import probe found two separately loaded
copies with identical bytes; all 41 observed PyTorch CUDA relocations used the
packaged copy. The observer now selects only that proven native provider and
retains both mapped identities. CPU tests and an independent ELF/receipt review
passed; the new full-model graph retry remains unqualified. The frozen graph
v4 payload also predates the current native dispatch-policy snapshot and cannot
be silently promoted into formal evidence. These diagnostics do not establish
native model graph coverage or prediction accuracy. All physical memory operations remain required evidence; their
intervals cannot be omitted from operation accounting.

The new SGLang v5 pilot preserves the original calibration/control/holdout points
and five-plus-ten repetitions. Its traces bind the exact native run, rank,
invocation and sampling role; setup launch/activity coverage is bidirectional.
Its complete capture provenance includes the original driver identity and
verified lazy-projection sources. The frozen bundle passed independent byte and
plan review; actual execution, export and independent error checks remain pending.
The pilot shares a text corpus across disjoint geometry sets, so it does not
establish formal independent-corpus acceptance. The vLLM capture adapter records
actual initialized FULL/PIECEWISE descriptors and eligibility settings before
requests; replay operation measurement and production graph coverage remain pending.

See [the collector contract](../python/aisimulate/collector/README.glm53flash.md),
[the Ops evidence adapter](../python/aisimulate/collector/glm53flash_validation.py)
and [the independent holdout validator](../python/aisimulate/collector/fpm_forward/glm53flash_validation.py)
for native timing boundaries, retained request/state/source receipts and
fail-closed acceptance. See [the shared native graph](glm53flash-native-contract.md)
for operation and precision boundaries.

Formal collection retains at least five warmups and ten observations per point, and separate calibration/holdout token streams and geometry. Exact table self-queries verify integrity, not independent accuracy. Record complete coverage, phase MAPE, WAPE and tail errors. Existing data remains unchanged.

Track [AIC-2000](https://linear.app/nvidia/issue/AIC-2000). SOL is the common prerequisite; FPM and Ops can progress in parallel after their shared execution contract is established.
