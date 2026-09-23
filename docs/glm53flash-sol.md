<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.3-Flash text architecture and SOL contract

## Implementation contract

Register the original Glm5Next text architecture and both pinned checkpoint configurations. Model 45 layers (34 KDA and 11 NoPE sparse MLA), IndexPool4, native mHC fusion, mixed-precision dense/MoE computation, and actual persistent KV/recurrent/tail state. Python composes operations; Rust owns cost arithmetic. Support pure TP2/TP4 on GB300 with DP/PP/CP/EP=1, text input and speculation disabled.

The required matrix is GB300 × vLLM/SGLang × native FP8/NVIDIA NVFP4 × TP2/TP4, through 131072 context tokens. NVFP4 TP1 is optional after measured memory admission. Text-only scope excludes vision execution, MTP, expert parallelism, offload and cross-node serving.

## Source identities

- FP8: `zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`.
- NVFP4: `nvidia/GLM-5.3-Flash-NVFP4@09b04e5e74bca08ca8549fc736d4cdd8624bfde3`.
- vLLM: `v0.30.0`, source `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- SGLang: `v0.5.20`, source `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`.

These are qualification candidates, not evidence of measured GB300 coverage. Preserve the checkpoint's actual per-module precision; do not relabel all weights as FP8 or NVFP4. TensorRT-LLM serving support is outside this initial matrix.

## Acceptance and data status

The model registration, nested configuration parser, per-layer graph, native Rust SOL and persistent-state accounting are implemented. The public consumer constructs both checkpoints on both supported backends through 128K context. Independent Python/Rust contract checks and both parity suites passed: 419 checks, plus 21 focused Rust arithmetic/serialization checks. Existing parity records are unchanged; four GLM records were appended. See [the native contract](glm53flash-native-contract.md) for exact operation and precision boundaries.

SOL is theoretical cost and payload accounting. Native allocator qualification, FPM data, Ops data and independent error acceptance belong to the companion campaigns. They are not certified by these CPU tests. The initial native GB300 qualification has completed FP8 TP2 and TP4 real-request cases through 128K; it is not prediction accuracy evidence.

Formal collection retains at least five warmups and ten observations per point, and separate calibration/holdout token streams and geometry. Exact table self-queries verify integrity, not independent accuracy. Record complete coverage, phase MAPE, WAPE and tail errors. Existing data remains unchanged.

Track [AIC-1998](https://linear.app/nvidia/issue/AIC-1998). SOL is the common prerequisite; FPM and Ops can progress in parallel after their shared execution contract is established.
