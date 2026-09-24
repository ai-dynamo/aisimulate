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
unobserved or overlapping boundaries fail admission. The measured-table contract
currently admits explicit eager execution only. An experimental SGLang FULL
decode observer now assigns native capture nodes to operation boundaries and
requires an exact API-backed join to CUPTI replay activity. It inserts no CUDA graph nodes;
its separate holdout observes only the complete native replay. This path emits
raw evidence, not admitted profiles. Actual native qualification, profiling
controls, complete metadata-node accounting and graph prediction remain pending.

Measured lookup keeps exact physical identity, backend, checkpoint, TP and phase.
Bounded workload interpolation requires complete measured corners with matching
observed kernel signatures and state/graph policy; KDA initial-state and IndexPool
short-context/tail partitions remain separate. Native tile counts are workload
coordinates, not fabricated kernel boundaries. Missing coverage fails in both
SILICON and HYBRID; neither mode silently falls back to SOL. The installed wheel
constructs all eight required model graphs and rejects missing measurements in
all eight SILICON and all eight HYBRID checks.

Current bounded validation passes 273 GLM collector tests and 18 Rust GLM tests.
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
for two SGLang prefill points and one vLLM point in each phase. Original
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
qualification of that deployment; the other three vLLM deployments and latency
acceptance remain separate requirements. Stock support is not claimed repaired.

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
These diagnostics do not establish native model graph coverage or prediction
accuracy. All physical memory operations remain required evidence; their
intervals cannot be omitted from operation accounting.

See [the collector contract](../python/aisimulate/collector/README.glm53flash.md),
[the Ops evidence adapter](../python/aisimulate/collector/glm53flash_validation.py)
and [the independent holdout validator](../python/aisimulate/collector/fpm_forward/glm53flash_validation.py)
for native timing boundaries, retained request/state/source receipts and
fail-closed acceptance. See [the shared native graph](glm53flash-native-contract.md)
for operation and precision boundaries.

Formal collection retains at least five warmups and ten observations per point, and separate calibration/holdout token streams and geometry. Exact table self-queries verify integrity, not independent accuracy. Record complete coverage, phase MAPE, WAPE and tail errors. Existing data remains unchanged.

Track [AIC-2000](https://linear.app/nvidia/issue/AIC-2000). SOL is the common prerequisite; FPM and Ops can progress in parallel after their shared execution contract is established.
