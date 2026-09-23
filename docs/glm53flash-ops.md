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

This draft records the approved implementation boundary. No new GPU measurements, accuracy results, or completed implementation are claimed by this initial checkpoint. Each required deployment cell remains pending qualification, collection and independent validation. Failures and unqualified fake-state observations must remain visible.

Formal collection retains at least five warmups and ten observations per point, and separate calibration/holdout token streams and geometry. Exact table self-queries verify integrity, not independent accuracy. Record complete coverage, phase MAPE, WAPE and tail errors. Existing data remains unchanged.

Track [AIC-2000](https://linear.app/nvidia/issue/AIC-2000). SOL is the common prerequisite; FPM and Ops can progress in parallel after their shared execution contract is established.
