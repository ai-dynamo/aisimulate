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

The native mixed path scopes explicitly grouped requests before combining their
non-attention work with decode tokens. For bounded replay, the op-level FPM v1
consumer accepts fresh prefill work only when it describes one prefill request. Its
variance field measures full prompt lengths; equal prompts can have different
cached prefixes or completed chunks, so zero variance cannot establish identical
extend tails. Multiple-prefill aggregates are therefore rejected, including
balanced batches and small total token counts. Explicit static/mixed geometry,
single-prefill telemetry, decode-only work, and Decoder OFF retain their existing
paths. Replay never removes resident weights or changes cache-capacity inventory.

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

Shared and routed expert compute are modeled sequentially, followed by the
post-expert reduction, for every backend and EP configuration. Concurrent
execution requires a qualified runtime contract before overlap can be priced;
the measured SGLang EP1 eager path remains sequential.

SGLang's pinned CUDA FlashMLA layout stores 584 bytes per main/SWA entry
(FP8 NoPE, BF16 RoPE, scales/padding), and its low-ratio index stores 68 bytes.
Three half-rate owners and one full-rate owner give a 1,630-byte global slope.
All 40 layers retain their 128-token window; ratio-two owners retain FP32
pooling state. Reindex/reuse layers share the compressed pools. Per-sequence
capacity follows exact publication boundaries, and batch capacity reserves
complete window/state buffers before applying the slope. The serialized
`sglang_fp8_bf16` layout separates physical storage from attention precision.

The `logical_fp4` layout (288-byte compressed main, 68-byte index, FP8 window,
890-byte global slope) remains the explicit theoretical estimate for vLLM and
TRT-LLM; their runtime storage is unqualified. Both inventories exclude allocator
page padding and spare pages. See [physical KV storage](#persistent-layout-and-ownership)
for full-context scoring and cache read/write accounting.

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

SOL returns analytical bounds. With the default systems database, HYBRID may
combine existing measured/empirical
BF16 GEMM, MoE, and collective data with **SOL** contributions for CSA2, Engram,
single-pass mHC, and 32x32-block FP8 shared-expert projections; those new components are uncalibrated. SILICON fails for missing
V4.1 data and never substitutes V4 attention or mHC tables. EMPIRICAL similarly
requires a V4.1 anchor. The packaged
[GB300 SILICON operator databases](../../../python/aisimulate/src/aisimulate_core/systems/profiles/dsv41/README.md)
provide separate `full` and `decoder_bounded` systems roots for the measured
SGLang runtime. Select the root matching `decoder_replay`; the flag does not
automatically select a table.

The independent FPM consumer uses the same model descriptor, resident inventory,
execution profile, and Rust `f64` SOL methods. Its checkpoint/profile/residency
identity and full-model measurements are owned by that dependent implementation.

## Physical KV storage

These independently expressed analytical formulas use SGLang at immutable
[1aa0e962b206102b7c439a4a0c4981cfec6e87bc](https://github.com/sgl-project/sglang/tree/1aa0e962b206102b7c439a4a0c4981cfec6e87bc).
The upstream sources are Apache-2.0, Copyright SGLang contributors; see
[THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md). No SGLang execution code is included here.

## Index scoring precedes candidate masking

In [`deepseek_v4_backend.py`](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/srt/layers/attention/deepseek_v4_backend.py),
`_low_ratio_index_topk_extend` calls the dense index GEMM over all compressed
keys before publishing/consuming candidate masks. `_low_ratio_index_topk_decode`
likewise calls the paged index GEMM before `two_level_decode_logits` masks its
result. Neither inspected SM100 path gathers candidates before the GEMM.
Thus `candidate_limit=16384` limits eligibility; it cannot cap scoring FLOPs,
key reads, or the materialized score array at contexts 16384/131072. Selection
still respects `index_topk`. No runtime-specific pre-GEMM optimization is assumed
for the other backends' theoretical analytical graph.

For the pinned SGLang SM90 path, FP4 index **storage** is separate from
BF16 score arithmetic. Prefill calls the BF16 `scores` einsum in
`dsv4/dsv41_sparse.py`; decode uses the BF16 `tl.dot` operations in
[`sm90_fp4_indexer.py`](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/kernels/ops/attention/dsv4/sm90_fp4_indexer.py).
SOL therefore uses the system's BF16 tensor-core throughput and two-byte query
elements, retaining the packed 68-byte index-K rows. This also supplies the
roofline needed for FPM interpolation between collected sites on H100/H200.
It does not add an FP4 capability to either system or change measured latencies.
Other SM versions and the theoretical `logical_fp4` contract keep their existing
FP4 throughput requirement; a missing throughput field remains an error.

## Persistent layout and ownership

[`deepseek_v4_memory_pool.py`](https://github.com/sgl-project/sglang/blob/1aa0e962b206102b7c439a4a0c4981cfec6e87bc/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py)
`DeepSeekV4SingleKVPool.get_bytes_per_token/create_buffer` gives
448 FP8 NoPE bytes +128 BF16 RoPE bytes +7 scale bytes +1 scale-padding byte = 584.
Both SWA and compressed main use this layout. `get_dsv4_indexer_bytes_per_token`
forces low-ratio FP4 index storage: 128/2 + 128/32 = 68 bytes. Main FP4 rounding
before FlashMLA storage changes values; it does not allocate a second 288-byte
persistent main record. The exact layout is scoped to 512-wide, 64-RoPE,
128-index, ratio 0/1/2 V4.1; other geometries are rejected. High-ratio and unified
BF16 pool alternatives are outside this contract.

Only four Full layers own compressed main/index pools; Reindex and Reuse share
them. One ratio-one owner and three ratio-two owners yield 652*(1+3/2)=1630 bytes
per token after the windows fill. At 131072 tokens, 40*128*584 window bytes plus
three ratio-two FP32 pair states plus compressed pools total 216662016 bytes
(206.625 MiB). Odd/even publication boundaries are preserved by forward and
inverse capacity APIs. This is physical **payload**, not allocator consumption:
576-byte page rounding, spare pages, fragmentation and other workspaces are not
included. Runtime measurements are still needed to qualify total capacity.

## Ideal read/write traffic and compatibility

Sparse attention reads 584 bytes per selected main row; SWA reads 584 per unique
window row with the existing ideal reuse assumption. A fused SWA norm/RoPE/store
reads the BF16 projection output and writes one physical row per token. The
projection already accounts for producing its BF16 output. Full owners publish
one compressed main/index record per completed group; no extra persistent
FP4 main record or extra fused main store is counted. Low-ratio incomplete-group
or padding writes in fallback kernels and intermediate traffic are not measured;
these remain SOL lower bounds, not an exact kernel traffic trace.
SM90 prefill's temporary BF16 K and per-head score materializations are likewise
outside this ideal traffic model; SM90 decode unpacks K within its score kernel.

Other backends retain the explicitly unqualified `logical_fp4` inventory and
traffic, avoiding a guessed physical cache precision. The operator serializes
`kv_cache_layout` independently of `fmha_quant_mode`. Legacy JSON without this
field defaults to the theoretical layout; fresh SGLang graphs always name the
physical layout.


## Execution identity

Whole-forward data carries four additional strings in the exact model/backend/topology
identity: `model_config_sha256`, `execution_profile`, `engram_residency`, and
`input_modality`. The config hash uses canonical JSON after the same quantization
normalization used by the SDK. V4.1 uses `hbm_tp_sharded` and `text`; its profile
is `full` or `decoder_bounded`. Changing the config or profile cannot borrow
another cell.

The shared model retains the entire resident-weight inventory even when a replay
profile executes fewer decoder tokens. FPM interpolation uses the original
stage-aware SOL graph. Vision and speculative decoding remain outside this
campaign's measurement contract.

When a native FPM table labels FMHA by its cache precision, select that table
with `fpm_fmha_dtype: "fp8"` in native engine/replay JSON, or
`ModelConfig(fpm_fmha_quant_mode=FMHAQuantMode.fp8, forward_model="fpm")`.
This option requires `forward_model="fpm"` and changes only the exact FPM cell
selector. The checkpoint's analytical attention graph, interpolation SOL
anchors, and memory inventory remain unchanged. `activation_dtype` retains its
existing arithmetic-override meaning; it is not a substitute for this selector.
Keep table-selector precision separate from any activation-arithmetic override.

`--fpm-decoder-replay` describes true bounded decoder execution. The current
vLLM route rejects it because the verified preview executes the full backbone.
Prefix-cache/SWA tail recomputation does not establish true decoder replay.
Replay OFF data must never be relabeled as ON data.

For Decoder ON, the FPM v1 telemetry API rejects an iteration with multiple
prefill requests and fresh prefill tokens. Its prompt-length variance cannot
prove equal current extends when requests have different cached prefixes or
completed chunks. Single-prefill and decode-only telemetry remain supported;
explicit homogeneous static inputs retain their separate table-query path.
This admission limit also applies to whole-forward FPM engines before lookup.
Historical reports retain their original predictor identities and coverage;
their supported counts do not describe this stricter current admission rule.

## Collection boundary

Whole-forward collection uses the text backbone, pure TP4, native checkpoint
precision, GPU-resident Engram, eager execution, and DSpark disabled. Token
history affects Engram, so cached-prefill/decode qualification requires real
model-computed KV and a reproducible tokenizer-generated corpus. Producers
must earn `real_kv` through execution; fake fallback, skipped warmup, and legacy
warm provenance are rejected for these phases.

Keep the checkpoint/config hash, image/instrumentation revisions, corpus hash,
resolved runtime settings, and native artifacts with the data. The experimental
GB200 calibration is not admitted for serving prediction because ordinary
serving validation found a substantial mismatch. Loading a checkpoint, importing
instrumentation, or passing a SOL check alone does not qualify timing.

The [self-service guide](../fpm-self-service/implementation.md) owns campaign,
transport, validation, and publication steps. Experimental evidence is kept
outside user guides in [accuracy evidence](../../../benchmarks/evidence/accuracy/).
