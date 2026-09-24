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
unobserved or overlapping boundaries fail admission. The current observer
admits explicit eager execution only. Production CUDA-graph collection and
replay-bound timing remain unfinished.

Measured lookup keeps exact physical identity, backend, checkpoint, TP and phase.
Bounded workload interpolation requires complete measured corners with matching
observed kernel signatures and state/graph policy; KDA initial-state and IndexPool
short-context/tail partitions remain separate. Native tile counts are workload
coordinates, not fabricated kernel boundaries. Missing coverage fails in both
SILICON and HYBRID; neither mode silently falls back to SOL. The installed wheel
constructs all eight required model graphs and rejects missing measurements in
all eight SILICON and all eight HYBRID checks.

Current focused validation passed 17 Rust GLM tests and 79 Ops contract, evidence,
native V2 runtime and shared retained-lifecycle tests. A wheel built from commit
`5e13be56` passed installation checks outside the source checkout, including
12 collector/evidence helper imports, all eight model graphs and 16 strict
missing-measurement queries. The earlier broad collector
run passed 2,362 tests (8 skipped); that count is historical and predates the
latest bounded evidence and native launch fixes. These checks establish software
behavior and evidence integrity, not native GPU performance coverage.
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
records the observed PyPI/container Rust executable difference. Stock support is
not claimed to be repaired.

The first native vLLM Ops attempt completed request execution but produced no
rank observations because the runtime selected its V2 runner. The observer now
wraps that actual class and joins its separate model and logits completion
phases; its whole-GPU event ends at `compute_logits`, before sampling. An actual
pinned-class CPU probe verified all four entry wrappers. SGLang uses the actual
Engine constructor and initializes the same overlap result queue as its native
loop. New immutable GPU retries are pending; prior failed inputs remain intact.

See [the collector contract](../python/aisimulate/collector/README.glm53flash.md),
[the Ops evidence adapter](../python/aisimulate/collector/glm53flash_validation.py)
and [the independent holdout validator](../python/aisimulate/collector/fpm_forward/glm53flash_validation.py)
for native timing boundaries, retained request/state/source receipts and
fail-closed acceptance. See [the shared native graph](glm53flash-native-contract.md)
for operation and precision boundaries.

Formal collection retains at least five warmups and ten observations per point, and separate calibration/holdout token streams and geometry. Exact table self-queries verify integrity, not independent accuracy. Record complete coverage, phase MAPE, WAPE and tail errors. Existing data remains unchanged.

Track [AIC-2000](https://linear.app/nvidia/issue/AIC-2000). SOL is the common prerequisite; FPM and Ops can progress in parallel after their shared execution contract is established.
