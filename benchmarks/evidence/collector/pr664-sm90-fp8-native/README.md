# H100 FP8 head-512 native Attention boundary audit

Both the collector and native vLLM `Attention.forward` reproduce the same FA4
scale-stride failure for **n4/KV1/head512**, batch 1, BF16 query and FP8 KV,
with 4,096-token prefill or 4,096-token persisted decode history. This closes
the missing native-layer control; these shapes remain failed coverage with
no published performance rows and no scale/stride workaround.

Job **28298** completed its diagnostic harness on one H100 80GB HBM3,
`pool0-0002`, AWS IAD Slurm. All four attempts reached the same native error:
`Expected strides[leading_dim] == 1, but got 0.` The harness catches failures
and saves them; scheduler exit 0 does not mean these attention cases passed.

| Layer path | Phase | Whole Python process | Result |
|---|---|---:|---|
| Collector MockAttentionLayer | Prefill | 32.020 s | Native FA4 scale-stride failure |
| Collector MockAttentionLayer | Decode | 23.019 s | Native FA4 scale-stride failure |
| Native Attention.forward | Prefill | 23.183 s | Same native failure |
| Native Attention.forward | Decode | 23.052 s | Same native failure |

Total **101.273 seconds = 0.028131 GPU-hours** includes Python startup,
framework imports, harness setup, collector initialization and attempted
forward execution. Shell timestamps run inside the container immediately
before and after Python. Container startup, scheduler/allocation time,
source extraction, image hashing, GPU receipts and gaps between processes
are outside this measured boundary. The first process also pays cold setup;
these four attempts are diagnostic failure costs, not a successful-call or
steady-state latency estimate.

The source is signed collector commit
`c569abb1cd034c934f47974209515e3507f60b8a`. The existing image reports vLLM
0.31.0; its exact squashfs SHA256 is
`8cb6197c0eb7434f7f9e28f22aaf60eedfbddac32add0aa07476bc9a9e5f8803`.
The collector and native implementation file hashes are in each result;
`summary.json` also hashes the installed native Attention and forward-context
sources extracted read-only from that same image.

The native control constructs a real `Attention` under the same VllmConfig,
lets its own selector choose the backend and implementation, moves its
registered scale buffers to CUDA, and calls its actual `forward` through
`set_forward_context`, cache update and unified attention operations. The
harness shares collector-generated inputs, cache layout and native-builder
metadata to isolate the layer boundary. The selected implementation has
`supports_quant_query_input=False`, so both paths pass BF16 Q to FA4.
This is bounded layer evidence; full-model scheduling, checkpoint-loaded
scales and end-to-end serving remain outside its scope.

`result.json` records every implementation field, metadata field, native
buffer and FA4 entry argument. Prefill and decode metadata match exactly
between paths. Implementation and entry arguments differ only in the last
floating-point bit of softmax scale from `512**-0.5` versus
`1/(512**0.5)` (relative difference below 1e-15). Query/cache shapes, strides,
dtypes, descales and selected FA version match. Native registered K/V scales
and collector scales are FP32 scalars equal to 1. Their native expansion to
`[1,1]` has strides `[0,0]`; FA4's CuTE conversion requests `leading_dim=1`
and rejects that zero stride. Q descale is absent on this selected path.
The source audit lists direct backend layer reads (`_q_scale`, `_k_scale`,
`_v_scale`, `impl`); the `impl` read belongs to the metadata builder's registered
sliding-window scan. This global-attention case uses the same recorded
metadata in both paths.

The pinned native call chain is `Attention.forward` →
`unified_attention_with_output` → `FlashAttentionImpl.forward` →
`_FA4_DENSE_ATTENTION_KERNEL` → `flash_attn_varlen_func` →
`cute/interface.py:1436` → `to_cute_tensor(..., leading_dim=1)` →
`mark_layout_dynamic`. See installed source hashes and full traces in
`28298/native-{0,1}/result.json` and the adjacent logs. The native Attention
entry point is
[vLLM attention.py](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/attention/attention.py),
and native descale expansion is in
[FlashAttentionImpl.forward](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/flash_attn.py).

The follow-up is to qualify an upstream fix for singleton descale strides
against these exact prefill/decode reproducers and retain the earlier n16/KV2
positive controls. Until then, keep this n4/KV1 shape classified as a native
failure. A collector-only stride change would require separate serving-parity
proof; this evidence introduces no such change.

The earlier job **28297** failed before Python because the image lacks
`/usr/bin/time`; it is preserved as a setup failure. Job 28298 replaces that
utility with shell timestamps. Reproduce with its `audit.sbatch` and source
archive from the recorded signed commit. Remote root:
`/lustre/fsw/portfolios/coreai/projects/coreai_comparch_lights-out-inf/users/simonec/results/pr664-fp8-native-20261103/{28297,28298}`.

`executed-artifacts.zip` preserves every original fetched artifact byte.
Readable copies normalize whitespace and format the harness; the manifest
records executed and published hashes. `summary.json` derives comparisons
and timings from these raw artifacts. No diagnostic row enters the package.
