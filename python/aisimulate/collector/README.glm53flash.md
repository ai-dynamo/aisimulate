# GLM-5.3-Flash native operation measurements

This collector observes the framework's loaded text decoder while the serving
scheduler executes actual requests. It never constructs attention metadata,
fills synthetic KDA state, changes a native dispatch choice, or moves a collective
outside its native call. MTP, multimodal, EP and DP execution are outside this
initial contract. TP2 and TP4 are required; NVFP4 TP1 is optional after admission.

`glm53flash_contract.py` obtains the complete operation identities from the
production graph. The parquet key is `(component, geometry, batch_size, prefix,
x)`. Geometry is canonical sorted JSON with the display name removed, including
backend and checkpoint format. Context uses query length `x` and actual cached
`prefix`; decode uses past-KV length `x` and `prefix=0`. Token-only components use
total tokens `x`, `batch_size=1`, `prefix=0`. Latency is milliseconds; integer
columns are physical INT64 constrained to uint32. The Rust reader only admits
exact measured keys, with no interpolation or analytical fallback.

`glm53flash_native_hooks.py` binds all 45 loaded decoder layers to
`NativeOperationObserver`. The serving adapter must pass native request IDs,
prefix execution receipts and actual phase coordinates to `begin(NativeWorkload)`
before the real forward and call `end()` after it. `close()` restores every
original callable. These first hooks admit eager observations only; graphs must
use capture-bound events and replay evidence. Eager data cannot be labelled as
production graph data. Missing native invocations fail complete-graph coverage.

Attention includes local projections, KDA or pooled-index NoPE sparse MLA, cache
writes, and the local output projection. Separately observed synchronous
same-stream collective intervals are excluded; overlap or fused communication
must receive a separate qualified contract. SGLang's hoisted latent projection
is a disjoint part of its attention measurement. KDA conv/recurrent state and
the sparse indexer's incomplete pool tail must come from the real request's
native prefix execution.

mHC is measured at native boundaries, including the output RMSNorm: vLLM has one
pre, 89 fused post/pre and one post interval, plus expand/contract; SGLang has 90
pre and 90 post intervals plus expand/contract. If SGLang leaves RMSNorm outside
its pre call, the current hook rejects that path rather than undercounting it.
Router measures the FP32 projection only; scoring and top-k belong to MoE.
Ordinary GEMM, expert compute, activation and collective measurements retain
their existing operation schemas and require separate native dispatch evidence.

Raw rank JSONL records retain every layer occurrence, request/history identity,
sample, invocation and excluded collective. Publication first requires complete
graph occurrences on every rank, then takes the median of per-invocation rank
maxima. Distinct checkpoint formats, runtimes, graph modes, seed policies or
state modes never silently collapse onto the same physical key. Failed attempts
remain separate evidence. A collector success is not accuracy acceptance; the
formal Ops gate is phase/cell MAPE <=20% against independent whole-forward truth.

## Upstream integration sources

These are original observation adapters calling upstream implementations, not
copies of the model or kernels. The bindings and architectural interpretation
are based on these immutable sources, under Apache-2.0:

- vLLM, Copyright contributors to the vLLM project, revision
  `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (v0.30.0):
  [model.py](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/glm5next/nvidia/model.py),
  `attention.py`, `kda.py` in the same directory, and
  `vllm/distributed/parallel_state.py`.
- SGLang, Copyright 2023-2024 SGLang Team and SGLang contributors, revision
  `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` (v0.5.20):
  [glm5_next.py](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/models/glm5_next.py),
  `python/sglang/srt/models/deepseek_v2.py`, and
  `python/sglang/srt/layers/communicator_mhc.py`.

Checkpoint identities come from `config.json` and model cards at
`zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`
and `nvidia/GLM-5.3-Flash-NVFP4@09b04e5e74bca08ca8549fc736d4cdd8624bfde3`.
Their config/license attribution is maintained with the production model's
packaged configuration files. See the canonical root `THIRD_PARTY_NOTICES.md`.
