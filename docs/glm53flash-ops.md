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
observed kernel signatures and state/graph policy; KDA chunk and IndexPool
short-context/tail partitions remain separate. Missing coverage fails in both
SILICON and HYBRID; neither mode silently falls back to SOL. The installed wheel
constructs all eight required model graphs and rejects missing measurements in
all eight SILICON and all eight HYBRID checks.

CPU validation passed 2,362 collector tests (8 skipped), 13 Rust GLM tests and
43 focused observer/contract/evidence/runtime tests. Subsequent shared SGLang
context/receipt integration passed 87 focused checks. These checks establish
software behavior and evidence integrity, not native GPU performance coverage.
**All eight deployment cells retain data/accuracy status `NOT_EVALUATED`.** No
measured GLM profiles or independent 20% MAPE pass are included. Native framework
qualification and the first Ops GPU smoke are separate pending evidence.

See [the collector contract](../python/aisimulate/collector/README.glm53flash.md),
[the Ops evidence adapter](../python/aisimulate/collector/glm53flash_validation.py)
and [the independent holdout validator](../python/aisimulate/collector/fpm_forward/glm53flash_validation.py)
for native timing boundaries, retained request/state/source receipts and
fail-closed acceptance. See [the shared native graph](glm53flash-native-contract.md)
for operation and precision boundaries.

Formal collection retains at least five warmups and ten observations per point, and separate calibration/holdout token streams and geometry. Exact table self-queries verify integrity, not independent accuracy. Record complete coverage, phase MAPE, WAPE and tail errors. Existing data remains unchanged.

Track [AIC-2000](https://linear.app/nvidia/issue/AIC-2000). SOL is the common prerequisite; FPM and Ops can progress in parallel after their shared execution contract is established.
