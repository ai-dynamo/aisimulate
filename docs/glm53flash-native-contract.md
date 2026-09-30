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

Each boundary also carries a `measured` list of generic operators. SOL mode
never reads it: op-level SOL and the FPM roofline keep the GLM formulas and
are bit-identical to the SOL-only graph. SILICON and HYBRID price the
boundary as the sum of its measured children against the generic tables,
like Kimi-K3 and DeepSeek-V4:

| Boundary | Measured generic composition |
|---|---|
| KDA attention | BF16 `Gemm` projections (vLLM one fused q/k/v/b/f_a/g_a; SGLang six), `f_b`/`g_b` and `o_proj` `Gemm`, `Kda` conv and delta-rule kernels, analytic gated-norm `Elementwise` |
| Sparse MLA attention | none; the one GLM table `glm53_attention_module_perf.parquet` |
| mHC `pre`/`post`/`fused_post_pre` | generic `Mhc` module rows at scale 0.5 per site (one row covers a layer's two sites), plus the analytic input RMSNorm for `pre` |
| mHC `expand`/`contract` | analytic `Elementwise` |
| FFN | dense/shared `Gemm` in checkpoint precision, analytic SwiGLU `Elementwise`, BF16 router `Gemm` and routed `Moe` |
| All-reduce | `CustomAllReduce` |
| Embedding, final norm, logits | `Embedding`, `Elementwise`, BF16 `Gemm` plus NCCL vocab gather (and SGLang FP32 cast) |

KDA `kernel_source` names per backend and phase are listed in
`sdk/models/glm53flash.py::KDA_KERNELS`. The router is priced as a BF16 GEMM
because no generic FP32 GEMM table exists; pure-TP MoE dispatch and combine
are the explicit all-reduces, so no `MoEDispatch` op is emitted.

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
  contract explicitly. Rust analytical children supply SOL only; the
  measured list above supplies SILICON/HYBRID. The diagnostic `Glm53Router`
  means FP32 GateLinear alone and is nested inside the SOL children, never
  emitted as a production or measured op.
- Routed-expert SOL keeps compute and activation traffic on the actual
  `tokens x top-8` assignments, but reads each distinct expert's local TP
  shard once per forward. The distinct count is the uniform-routing
  expectation `E * (1 - (1 - k/E)^T)` (E=288, k=8), capped at E and at
  `T * k`; batch 32 reads about 171 experts rather than 256. The same term
  is used by op-level SOL and by the FPM roofline. Skewed routing touches
  fewer experts, so this is an expected-uniform bound, not a worst-case
  minimum.
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

SILICON fails closed. Construction requires every consumed table
(`gemm_perf`, `moe_perf`, `kda_perf`, `mhc_module_perf`,
`custom_allreduce_perf` and the GLM attention table) to be measured on the
exact requested runtime: the primary file of that system/backend/version or
a donor declared in its `reuse.yaml`. Earlier-version siblings and
cross-backend fill do not satisfy readiness, and the GLM attention table is
read from the exact primary only. At query time a generic child that answers
from its analytical fallback, for example a KDA kernel miss, is an error in
SILICON. Strict evaluation also sets `enable_shared_layer=False`, so no
earlier-version rows are merged into the exact tables. HYBRID keeps each
generic child's labelled fallback and falls back to the GLM SOL
(`Source::Sol`) when a required table is absent. EMPIRICAL is unsupported.
Existing Kimi-K3 KDA rows (12/24/48/96 heads) never cover GLM's 16/32/64-head
shards.

The pinned runtimes are data coordinates `vllm/0.30.0+glm53tail.eb4704514fdf`
and `sglang/0.5.20` under `data/gb300/<family>/<backend>/<version>/`. The GLM
attention table uses the DeepSeek-V4.1 module schema (`component=attention`,
canonical sorted `geometry` JSON of the `Glm53Attention` body without
`name`/`measured`, `batch_size`, `prefix`, `x` = new tokens for prefill or
absolute KV length for decode, local-compute `latency` in ms, and complete
provenance). The reader interpolates utilization over batch, prefix and `x`
and holds the boundary utilization outside the measured range. GPU
collection and accuracy gates are independent acceptance steps and are not
certified by the CPU model tests.

Configuration and execution-source licenses and immutable revisions are
recorded in the canonical root `THIRD_PARTY_NOTICES.md`; the Python package
contains an identical notice and the configuration's complete MIT license.

`Glm53Primitive` owns the remaining production boundaries: local `embedding`,
`final_norm`, whole `logits`, and separately observed `allreduce` calls. Its
physical key includes backend, checkpoint, TP, phase, role, token selection,
output dtype and collective type. Analytical children run only through the
Rust SOL view; SILICON/HYBRID use the measured generic embedding, GEMM,
elementwise and collective operators listed above.

Logits select one final scheduled token per request for both full and cached
prefill, as well as decode. The native processor includes a BF16 local LM head
and BF16 vocab all-gather; SGLang additionally casts output to FP32. Its SOL child
token count is therefore batch size, independently of scheduled query length.
The other primitives consume all scheduled tokens. Primitive names are phase
independent; `is_context` remains part of the key. Local embedding, attention and
FFN observations exclude their separately witnessed output collective intervals.
