# DeepSeek-V4.1 scoring and physical KV payload

These independently expressed analytical formulas use SGLang at immutable
[1aa0e962b206102b7c439a4a0c4981cfec6e87bc](https://github.com/sgl-project/sglang/tree/1aa0e962b206102b7c439a4a0c4981cfec6e87bc).
The upstream sources are Apache-2.0, Copyright SGLang contributors; see
`THIRD_PARTY_NOTICES.md`. No SGLang execution code is included here.

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

Other backends retain the explicitly unqualified `logical_fp4` inventory and
traffic, avoiding a guessed physical cache precision. The operator serializes
`kv_cache_layout` independently of `fmha_quant_mode`. Legacy JSON without this
field defaults to the theoretical layout; fresh SGLang graphs always name the
physical layout. Engine wire schema 19 rejects schema 18 bincode before decoding
the new positional field. Model observations and calibration data are unchanged.
