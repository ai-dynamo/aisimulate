<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GLM-5.3-Flash native text contract

The `GLM53FLASH` model family resolves the two immutable configurations listed
in `sdk/glm53flash.py`. `Glm53FlashConfig.from_text_config()` validates their
45-layer schedule and supplies the same geometry to the model, FPM planner,
and Ops collector. The full checkpoint configuration remains cached; the
prediction graph contains text autoregressive inference only. Vision and the
checkpoint's one MTP layer are not executed. Context coverage for the initial
measurement campaign is at most 131072 tokens; the checkpoint's advertised
maximum context is retained as metadata, not a measured-coverage claim.

Construct production models through
`RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig(...))`.
The initial topology is TP 1/2/4, MoE TP equal to attention TP,
DP=PP=CP=EP=1, `nextn=0`, and full decoder execution. GB300 TP2/TP4 are the
formal targets; TP1 acceptance requires measured allocator admission.
TensorRT-LLM fails explicitly. No new estimator constructor is introduced.

The appended native variants are `Glm53Attention`, `Glm53Mhc`, and
`Glm53Router` (diagnostic), `Glm53Ffn`, and `Glm53Primitive`. Their serialized bodies include `backend` and
`checkpoint_format`; the display name is not physical geometry. Attention
also identifies `is_context`, `layer_kind`, TP-local heads and dimensions,
replicated indexer geometry, projection dtype and cache dtype. Rust owns
all latency/SOL arithmetic, including the fractional workload coordinates
used by whole-model FPM interpolation. The existing generic GEMM, MoE and
NCCL operators supply analytical children within these boundaries.

- Attention includes local projections, KDA gates/convolution/recurrence/output
  norm, or NoPE sparse MLA and its IndexPool. Block input norm and output
  collective are outside this boundary. `index_topk=2048` selects 512 pools
  of four tokens, with the unfinished tail retained. Short-context index score
  work is skipped by the pinned vLLM implementation; SGLang skips it only
  for short prefill and also omits index query/head-gate projections there. KDA uses BF16 projections
  and FP32 recurrent state. vLLM also materializes sparse MLA projections in
  BF16; SGLang retains FP8 main sparse projections for the native FP8 checkpoint.
- mHC requires explicit `tp_size` and `is_context` in its native identity: TP1/2/4 cannot
  borrow one another's measured dispatch. Its replicated local SOL work is
  unchanged by this identity field. Missing TP or phase metadata is rejected.
  mHC `pre` includes input RMSNorm. vLLM emits one pre, 89 fused post/pre,
  and one post, plus expand/contract. SGLang emits 90 pre and 90 post, plus
  expand/contract. A collector must include SGLang's fallback RMSNorm if its
  native pre reports that normalization was not fused.
- Production FFN is the complete local native MLP, including router/gate,
  sigmoid/top-k, routed/shared experts and SwiGLU clamp. Its output collective
  is outside the boundary. `Glm53Ffn` records the routing/clamp/precision
  contract explicitly; generic MoE measurements cannot satisfy it. Rust
  analytical children supply SOL only and are excluded from the measured
  identity. The diagnostic `Glm53Router` means FP32 GateLinear alone and is
  nested inside FFN, never emitted as an additional production measured op.
- NVFP4 dense and routed FFNs use W4A4 group16. Shared experts and all
  attention remain BF16. The FP8 checkpoint quantizes shared experts.

`glm53_cache_bytes(json_attention_body, sequence_length)` is the internal
Rust payload-accounting FFI. It accounts one FP32 recurrence plus three BF16
convolution histories per KDA layer and active sequence, replicated FP8
latent cache per sparse layer, one 132-byte index entry per completed pool,
and two four-slot BF16 tail buffers. Capacity reserves fixed state for every
active scheduler slot before distributing the token budget. Weight inventory
includes all routed experts, FP8 block scales or NVFP4 group scales, replicated
mHC/router parameters and BF16 exceptions. Weight/cache values are tensor
payload estimates; allocator pages, CUDA graphs, prefix-state snapshots,
workspace and fragmentation require native runtime admission evidence.

The SOL implementation expresses mathematical work and payload traffic,
not kernel launch counts or a claim of measured performance. KDA prefill
uses a tensor-core lower bound for delta-rule recurrence, while decode uses
FP32 scalar throughput. mHC and the router require an explicit GPU FP32 rate.
Until the Ops data consumer is installed, explicit SILICON requests for the
new operators fail; HYBRID retains `Source::Sol`. No Kimi/GDN/Mamba table is
borrowed. GPU collection, accuracy gates and measured readiness are independent
acceptance steps and are not certified by the CPU model tests.

Configuration and execution-source licenses and immutable revisions are
recorded in the canonical root `THIRD_PARTY_NOTICES.md`; the Python package
contains an identical notice and the configuration's complete MIT license.

`Glm53Primitive` owns the remaining production boundaries: local `embedding`,
`final_norm`, whole `logits`, and separately observed `allreduce` calls. Its
physical key includes backend, checkpoint, TP, phase, role, token selection,
output dtype and collective type. Analytical children are excluded from measured
identity and run only through the Rust SOL view. Strict measured mode cannot
borrow generic embedding, GEMM, elementwise or collective data.

Logits select one final scheduled token per request for both full and cached
prefill, as well as decode. The native processor includes a BF16 local LM head
and BF16 vocab all-gather; SGLang additionally casts output to FP32. Its SOL child
token count is therefore batch size, independently of scheduled query length.
The other primitives consume all scheduled tokens. Primitive names are phase
independent; `is_context` remains part of the key. Local embedding, attention and
FFN observations exclude their separately witnessed output collective intervals.
