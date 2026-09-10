<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1 Flash text modeling

`deepseek-ai/DeepSeek-V4.1-Flash` uses the pinned checkpoint configuration
`fb2764a5cf321eaa5070ca8f9e892818f477c16d`. The model covers the 40-layer text
backbone with ordinary autoregressive decoding (`nextn=0`). The vision encoder,
three DSpark layers, encoder replay, and host-resident Engram tables require
separate execution contracts.

The SOL graph can be constructed for SGLang, vLLM, and TensorRT-LLM. This is an
analytical model capability; it does not certify that a released version of
those runtimes loads or executes this checkpoint. Initial parallel mappings use
attention DP=1, PP=1, CP=1, and TP dividing eight output groups; attention TP must
match MoE TP times EP. The decomposed shared expert is TP-sharded. Fused shared
expert slots, replicated shared experts, MegaMoE, and alternative collective
implementations require measurements and explicit execution identities.

## Decoder replay

`ModelConfig(decoder_replay=False)` and `compile_engine(..., decoder_replay=False)`
select `full`: every actual extend token traverses all 40 layers. The SGLang-only
`decoder_replay=True` selects `decoder_bounded`, matching the inspected source at
[`1aa0e962b206102b7c439a4a0c4981cfec6e87bc`](https://github.com/sgl-project/sglang/tree/1aa0e962b206102b7c439a4a0c4981cfec6e87bc).
Layers 0–20 process the complete actual extend; layers 21–39 process each
request's last `min(extend_length, 128)` tokens. The absolute sequence endpoint
and shared global KV remain unchanged. A three-token extend after a long cached
prefix still has three late-layer query tokens. The bounded late-layer SWA
window starts at that extend tail. Decode always traverses all layers.

The native mixed path scopes requests before combining their non-attention work
with decode tokens. Telemetry with nonzero prefill variance cannot recover
individual tail lengths and is rejected for bounded replay. Replay never
removes resident weights or changes the cache-capacity inventory.

## Operators and memory

CSA2 has two SWA layers, four Full owners (2, 8, 14, 20), four Reindex layers
(24, 28, 32, 36), and 30 Reuse layers. Full owners alone publish compressed KV;
Reindex layers rebuild selection and Reuse layers share it. Projection, packing,
index scoring/selection, and sparse-attention work remain explicit in the
analytical module. Single-pass mHC uses the system's scalar `fp32_flops` field;
BF16 compressor projections use tensor-core throughput. GB200 and GB300 provide
an explicit nominal FP32 rate. A missing rate is an error rather than a BF16
substitution. HGX B200 and HGX B300 use 75 TFLOPS per GPU, from the
[NVIDIA HGX specification](https://www.nvidia.com/en-us/data-center/hgx/)'s
600 TFLOPS FP32 for each eight-GPU baseboard (accessed September 10, 2026).

The indexer is replicated across attention TP: every rank owns all 32 index
heads, their projections and the full scoring/selection workload. This matches
[SGLang's pinned V4.1 indexer](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py#L203).
The [DeepSeek reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/fb2764a5cf321eaa5070ca8f9e892818f477c16d/inference/model.py#L507)
instead shards index heads and all-reduces their scores before selection.
Those two execution strategies must not be mixed. The TRT-LLM graph currently
uses the same replicated analytical baseline; runtime qualification is pending.

SWA arithmetic counts causal query-key pairs. Its SOL HBM traffic counts unique
window KV rows with ideal reuse across queries: `Q + min(P, W-1)` rows per
prefill request, or `min(S, W)` for decode. Bounded decoder prefill uses `P=0`
for this local window. Actual kernel tiling and cache misses can read more;
this is a lower-bound assumption. Compressed sparse reads retain per-query
traffic because selected positions can differ. Both independent BF16 compressor
matrices are read by ratio-two Full owners.

The V4.1 graph explicitly prices attention output reduction and omits its
redundant pre-MLP dispatch under the required DP=CP=1 topology. Shared
`MoEDispatch` behavior is unchanged: Qwen3.5's SGLang attention-DP path retains
the folded TP reduce-scatter plus DP all-gather, and its unqualified TRT-LLM
path retains the previously documented collective behavior.

Main compressed KV uses 288 bytes per entry (FP4 plus one scale per 16
channels); index KV uses 68 bytes (MXFP4 plus one scale per 32). Three half-rate
owners and one full-rate owner produce a global slope of 890 bytes per token.
Each layer also has an FP8 128-token window, and the ratio-two owners keep FP32
pooling state. Per-sequence memory follows exact publication boundaries.
Batch capacity reserves complete window/state buffers for every scheduler slot
before applying the global slope; this conservatively covers short and odd
sequence lengths. Allocator block padding and runtime-specific cache layouts
remain outside this logical inventory.

Engram's two GPU-resident hash tables include FP8 block scales and TP row
sharding. Their full resident size is independent of tokens accessed. Lookup
traffic assumes uniformly distributed hash ownership; hotspots and cache reuse
need measurement. Replicated projection/gate weights, mHC weights, MoE MXFP4
scales, dispatch workspace, expanded residual buffers, and Engram temporary
buffers are included. Backend activation coefficients remain heuristic, and
runtime allocator measurements are still required to qualify capacity. V4.1
uses the existing MoE coefficient family (SGLang TP4: 13; vLLM/TRT-LLM TP4: 10),
with the mHC and Engram buffers added separately, rather than the dense default.

## Result provenance

SOL returns analytical bounds. HYBRID may combine existing measured/empirical
BF16 GEMM, MoE, and collective data with **SOL** contributions for CSA2, Engram,
single-pass mHC, and 32x32-block FP8 shared-expert projections; those new components are uncalibrated. SILICON fails for missing
V4.1 data and never substitutes V4 attention or mHC tables. EMPIRICAL similarly
requires a V4.1 anchor. Measured support is a separate dependent change.

The independent FPM consumer uses the same model descriptor, resident inventory,
execution profile, and Rust `f64` SOL methods. Its checkpoint/profile/residency
identity and full-model measurements are owned by that dependent implementation.
