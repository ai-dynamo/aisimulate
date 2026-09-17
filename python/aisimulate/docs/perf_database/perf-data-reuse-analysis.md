# Shareability report

Classifies every `(system, op_file, kernel_source)` triple in the perf database into one of two tiers:

- **`shared`** — named kernel_source. The engine resolver inherits these rows from sibling backend/version directories (cross-version and cross-backend) when shared-layer reuse is enabled.
- **`shared_fallback`** — `kernel_source = default`. Framework-implicit, low-fidelity. Inherited alongside `shared` rows because shared-layer modes already accept coarser fallbacks.

Rows with a blank/`<unknown>` kernel_source are skipped during the scan (the current corpus has none).

## Headline numbers

- Total rows scanned: **13,685,285**
- Within-framework cross-version dedup-able rows: **1,737** (~0.0%)
- Tier distribution (groups / rows):
  - `shared`: 1026 groups · 13,381,644 rows
  - `shared_fallback`: 52 groups · 303,641 rows


## `computescale_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `sglang` | shared | sglang | sglang:1 | 0 / 1 | 0 | — | — | — |
| b200_sxm | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| b300_sxm | `sglang` | shared | sglang | sglang:1 | 0 / 1 | 0 | — | — | — |
| b300_sxm | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| gb200 | `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | shared | vllm | vllm:1625 | 0 / 1625 | 0 | — | — | — |
| gb200 | `sglang` | shared | sglang | sglang:1 | 0 / 1 | 0 | — | — | — |
| gb200 | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| gb300 | `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | shared | vllm | vllm:1626 | 0 / 1626 | 0 | — | — | — |
| gb300 | `sglang` | shared | sglang | sglang:1 | 0 / 1 | 0 | — | — | — |
| gb300 | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| h100_sxm | `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | shared | vllm | vllm:1628 | 0 / 1628 | 0 | — | — | — |
| h100_sxm | `sglang` | shared | sglang | sglang:3 | 0 / 3 | 0 | — | — | — |
| h100_sxm | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| h200_sxm | `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | shared | vllm | vllm:1628 | 0 / 1628 | 0 | — | — | — |
| h200_sxm | `sglang` | shared | sglang | sglang:6 | 0 / 6 | 0 | — | — | — |
| h200_sxm | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| l40s | `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | shared | vllm | vllm:1575 | 0 / 1575 | 0 | — | — | — |
| l40s | `sglang` | shared | sglang | sglang:18 | 0 / 18 | 0 | — | — | — |
| l40s | `torch_ops` | shared | trtllm | trtllm:1607 | 0 / 1607 | 0 | — | — | — |
| rtx_pro_6000_server | `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | shared | vllm | vllm:1611 | 0 / 1611 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang` | shared | sglang | sglang:10 | 0 / 10 | 0 | — | — | — |
| rtx_pro_6000_server | `torch_ops` | shared | trtllm | trtllm:1613 | 0 / 1613 | 0 | — | — | — |

## `context_attention_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `flash_attention` | shared | sglang | sglang:5627 | 0 / 5627 | 0 | — | — | — |
| a100_sxm | `torch_flow` | shared | trtllm | trtllm:4864 | 0 / 4864 | 0 | — | — | — |
| a100_sxm | `vllm_flash_attn` | shared | vllm | vllm:5457 | 0 / 5049 | 0 | — | — | — |
| b200_sxm | `flashinfer` | shared | sglang | sglang:2584 | 0 / 2584 | 0 | — | — | — |
| b200_sxm | `torch_flow` | shared | trtllm | trtllm:68535 | 0 / 68535 | 0 | — | — | — |
| b200_sxm | `torch_flow_flashinfer` | shared | trtllm | trtllm:2312 | 0 / 2312 | 0 | — | — | — |
| b200_sxm | `triton` | shared | sglang | sglang:9486 | 0 / 9486 | 0 | — | — | — |
| b200_sxm | `trtllm_mha` | shared | sglang | sglang:32093 | 0 / 32093 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer` | shared | vllm | vllm:35360 | 0 / 35360 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer_flashinfertrtllmapidecode` | shared | vllm | vllm:3900 | 0 / 3900 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer_trtllmprefill` | shared | vllm | vllm:47304 | 0 / 47304 | 0 | — | — | — |
| b200_sxm | `vllm_triton_attn` | shared | vllm | vllm:3264 | 0 / 3264 | 0 | — | — | — |
| b300_sxm | `flashinfer` | shared | sglang | sglang:2584 | 0 / 2584 | 0 | — | — | — |
| b300_sxm | `torch_flow` | shared | trtllm | trtllm:67106 | 0 / 67106 | 0 | — | — | — |
| b300_sxm | `torch_flow_flashinfer` | shared | trtllm | trtllm:2312 | 0 / 2312 | 0 | — | — | — |
| b300_sxm | `triton` | shared | sglang | sglang:9486 | 0 / 9486 | 0 | — | — | — |
| b300_sxm | `trtllm_mha` | shared | sglang | sglang:32097 | 0 / 32097 | 0 | — | — | — |
| b300_sxm | `vllm_flashinfer_flashinfertrtllmapidecode` | shared | vllm | vllm:3900 | 0 / 3900 | 0 | — | — | — |
| b300_sxm | `vllm_flashinfer_trtllmprefill` | shared | vllm | vllm:47304 | 0 / 47304 | 0 | — | — | — |
| b300_sxm | `vllm_triton_attn` | shared | vllm | vllm:1632 | 0 / 1632 | 0 | — | — | — |
| b60 | `vllm_flash_attn` | shared | vllm | vllm:34118 | 0 / 17930 | 1 | — | — | — |
| gb200 | `flashinfer` | shared | sglang | sglang:2584 | 0 / 2584 | 0 | — | — | — |
| gb200 | `torch_flow` | shared | trtllm | trtllm:68535 | 0 / 68535 | 0 | — | — | — |
| gb200 | `torch_flow_flashinfer` | shared | trtllm | trtllm:2312 | 0 / 2312 | 0 | — | — | — |
| gb200 | `triton` | shared | sglang | sglang:9486 | 0 / 9486 | 0 | — | — | — |
| gb200 | `trtllm_mha` | shared | sglang | sglang:31138 | 0 / 31138 | 0 | — | — | — |
| gb200 | `vllm_flashinfer_trtllmdecode` | shared | vllm | vllm:3740 | 0 / 3740 | 0 | — | — | — |
| gb200 | `vllm_flashinfer_trtllmprefill` | shared | vllm | vllm:45356 | 0 / 45356 | 0 | — | — | — |
| gb200 | `vllm_triton_attn` | shared | vllm | vllm:1632 | 0 / 1632 | 0 | — | — | — |
| gb300 | `flashinfer` | shared | sglang | sglang:2584 | 0 / 2584 | 0 | — | — | — |
| gb300 | `torch_flow` | shared | trtllm | trtllm:68533 | 0 / 68533 | 0 | — | — | — |
| gb300 | `torch_flow_flashinfer` | shared | trtllm | trtllm:2312 | 0 / 2312 | 0 | — | — | — |
| gb300 | `triton` | shared | sglang | sglang:9486 | 0 / 9486 | 0 | — | — | — |
| gb300 | `trtllm_mha` | shared | sglang | sglang:31144 | 0 / 31144 | 0 | — | — | — |
| gb300 | `vllm_flashinfer_trtllmdecode` | shared | vllm | vllm:3740 | 0 / 3740 | 0 | — | — | — |
| gb300 | `vllm_flashinfer_trtllmprefill` | shared | vllm | vllm:45356 | 0 / 45356 | 0 | — | — | — |
| gb300 | `vllm_triton_attn` | shared | vllm | vllm:1632 | 0 / 1632 | 0 | — | — | — |
| h100_sxm | `fa3` | shared | sglang | sglang:31622 | 0 / 31622 | 0 | — | — | — |
| h100_sxm | `torch_flow` | shared | trtllm | trtllm:75840 | 0 / 75840 | 0 | — | — | — |
| h100_sxm | `triton` | shared | sglang | sglang:3468 | 0 / 3468 | 0 | — | — | — |
| h100_sxm | `vllm_flash_attn_fa3` | shared | vllm | vllm:50252 | 0 / 50252 | 0 | — | — | — |
| h100_sxm | `vllm_flash_attn_fa4` | shared | vllm | vllm:578 | 0 / 578 | 0 | — | — | — |
| h200_sxm | `fa3` | shared | sglang | sglang:32574 | 0 / 32574 | 0 | — | — | — |
| h200_sxm | `torch_flow` | shared | trtllm | trtllm:77265 | 0 / 77265 | 0 | — | — | — |
| h200_sxm | `triton` | shared | sglang | sglang:3468 | 0 / 3468 | 0 | — | — | — |
| h200_sxm | `vllm_flash_attn_fa3` | shared | vllm | vllm:51204 | 0 / 51204 | 0 | — | — | — |
| h200_sxm | `vllm_flash_attn_fa4` | shared | vllm | vllm:578 | 0 / 578 | 0 | — | — | — |
| l40s | `flashinfer` | shared | sglang | sglang:26236 | 0 / 26236 | 0 | — | — | — |
| l40s | `torch_flow` | shared | trtllm | trtllm:75768 | 0 / 75768 | 0 | — | — | — |
| l40s | `triton` | shared | sglang | sglang:9792 | 0 / 9792 | 0 | — | — | — |
| l40s | `vllm_flash_attn_fa2` | shared | vllm | vllm:25092 | 0 / 25092 | 0 | — | — | — |
| l40s | `vllm_flashinfer_fidecode` | shared | vllm | vllm:1744 | 0 / 1744 | 0 | — | — | — |
| l40s | `vllm_flashinfer_fiprefill` | shared | vllm | vllm:21690 | 0 / 21690 | 0 | — | — | — |
| l40s | `vllm_triton_attn` | shared | vllm | vllm:1632 | 0 / 1632 | 0 | — | — | — |
| rtx_pro_6000_server | `flashinfer` | shared | sglang | sglang:23410 | 0 / 23410 | 0 | — | — | — |
| rtx_pro_6000_server | `torch_flow` | shared | trtllm | trtllm:71929 | 0 / 71929 | 0 | — | — | — |
| rtx_pro_6000_server | `triton` | shared | sglang | sglang:16116 | 0 / 16116 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_flash_attn_fa2` | shared | vllm | vllm:25126 | 0 / 25126 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_flashinfer_fidecode` | shared | vllm | vllm:1728 | 0 / 1728 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_flashinfer_fiprefill` | shared | vllm | vllm:21677 | 0 / 21677 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_triton_attn` | shared | vllm | vllm:1632 | 0 / 1632 | 0 | — | — | — |

## `context_mla_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `default` | shared_fallback | trtllm | trtllm:2436 | 0 / 2436 | 0 | — | — | — |
| b200_sxm | `default` | shared_fallback | trtllm | trtllm:1760 | 0 / 1760 | 0 | — | — | — |
| b200_sxm | `trtllm_mla` | shared | sglang | sglang:1760 | 0 / 1760 | 0 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:1760 | 0 / 1760 | 0 | — | — | — |
| b300_sxm | `trtllm_mla` | shared | sglang | sglang:3520 | 0 / 1760 | 0 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:1760 | 0 / 1760 | 0 | — | — | — |
| gb200 | `trtllm_mla` | shared | sglang | sglang:3520 | 0 / 1760 | 0 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:1760 | 0 / 1760 | 0 | — | — | — |
| gb300 | `trtllm_mla` | shared | sglang | sglang:3520 | 0 / 1760 | 0 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:1760 | 0 / 1760 | 0 | — | — | — |
| h100_sxm | `flash_attention` | shared | sglang | sglang:3520 | 0 / 1760 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:1760 | 0 / 1760 | 0 | — | — | — |
| h200_sxm | `flash_attention` | shared | sglang | sglang:1760 | 0 / 1760 | 0 | — | — | — |
| l40s | `default` | shared_fallback | trtllm | trtllm:2436 | 0 / 2436 | 0 | — | — | — |
| l40s | `triton` | shared | sglang | sglang:880 | 0 / 880 | 0 | — | — | — |
| rtx_pro_6000_server | `default` | shared_fallback | trtllm | trtllm:1760 | 0 / 1760 | 0 | — | — | — |
| rtx_pro_6000_server | `triton` | shared | sglang | sglang:1760 | 0 / 880 | 0 | — | — | — |

## `custom_allreduce_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:138 | 0 / 69 | 0 | — | — | — |
| a100_sxm | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:138 | 0 / 69 | 0 | — | — | — |
| a100_sxm | `TRTLLM` | shared | trtllm | trtllm:69 | 0 / 69 | 0 | — | — | — |
| a100_sxm | `vLLM_custom_eager` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| a100_sxm | `vLLM_custom_graph` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| b200_sxm | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:207 | 0 / 69 | 0 | — | — | — |
| b200_sxm | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:207 | 0 / 69 | 0 | — | — | — |
| b200_sxm | `TRTLLM` | shared | trtllm | trtllm:207 | 0 / 69 | 0 | — | — | — |
| b200_sxm | `vLLM_custom_eager` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| b200_sxm | `vLLM_custom_graph` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| b300_sxm | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:207 | 0 / 69 | 0 | — | — | — |
| b300_sxm | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:207 | 0 / 69 | 0 | — | — | — |
| b300_sxm | `TRTLLM` | shared | trtllm | trtllm:138 | 0 / 69 | 1 | — | — | — |
| b300_sxm | `vLLM_custom_eager` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| b300_sxm | `vLLM_custom_graph` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| b60 | `vLLM_custom_eager` | shared | vllm | vllm:207 | 0 / 69 | 23 | — | — | — |
| gb200 | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:230 | 0 / 92 | 0 | — | — | — |
| gb200 | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:230 | 0 / 92 | 0 | — | — | — |
| gb200 | `TRTLLM` | shared | trtllm | trtllm:92 | 0 / 46 | 0 | — | — | — |
| gb200 | `vLLM_custom_eager` | shared | vllm | vllm:138 | 0 / 46 | 0 | — | — | — |
| gb200 | `vLLM_custom_graph` | shared | vllm | vllm:138 | 0 / 46 | 0 | — | — | — |
| gb300 | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:230 | 0 / 92 | 0 | — | — | — |
| gb300 | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:230 | 0 / 92 | 0 | — | — | — |
| gb300 | `TRTLLM` | shared | trtllm | trtllm:92 | 0 / 46 | 0 | — | — | — |
| gb300 | `TRTLLM_MNNVL_oneshot` | shared | trtllm | trtllm:19 | 0 / 19 | 0 | — | — | — |
| gb300 | `TRTLLM_MNNVL_twoshot` | shared | trtllm | trtllm:27 | 0 / 27 | 0 | — | — | — |
| gb300 | `vLLM_custom_eager` | shared | vllm | vllm:230 | 0 / 92 | 0 | — | — | — |
| gb300 | `vLLM_custom_graph` | shared | vllm | vllm:230 | 0 / 92 | 0 | — | — | — |
| h100_sxm | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:207 | 0 / 69 | 0 | — | — | — |
| h100_sxm | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:207 | 0 / 69 | 0 | — | — | — |
| h100_sxm | `TRTLLM` | shared | trtllm | trtllm:276 | 0 / 69 | 0 | — | — | — |
| h100_sxm | `vLLM_custom_eager` | shared | vllm | vllm:207 | 0 / 69 | 0 | — | — | — |
| h100_sxm | `vLLM_custom_graph` | shared | vllm | vllm:207 | 0 / 69 | 1 | — | — | — |
| h200_sxm | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:207 | 0 / 69 | 0 | — | — | — |
| h200_sxm | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:207 | 0 / 69 | 0 | — | — | — |
| h200_sxm | `TRTLLM` | shared | trtllm | trtllm:276 | 0 / 69 | 0 | — | — | — |
| h200_sxm | `vLLM_custom_eager` | shared | vllm | vllm:276 | 0 / 69 | 0 | — | — | — |
| h200_sxm | `vLLM_custom_graph` | shared | vllm | vllm:276 | 0 / 69 | 1 | — | — | — |
| l40s | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:207 | 0 / 69 | 0 | — | — | — |
| l40s | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:207 | 0 / 69 | 1 | — | — | — |
| l40s | `TRTLLM` | shared | trtllm | trtllm:276 | 0 / 69 | 0 | — | — | — |
| l40s | `vLLM_custom_eager` | shared | vllm | vllm:138 | 0 / 69 | 0 | — | — | — |
| l40s | `vLLM_custom_graph` | shared | vllm | vllm:138 | 0 / 69 | 0 | — | — | — |
| rtx_pro_6000_server | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:92 | 0 / 69 | 0 | — | — | — |
| rtx_pro_6000_server | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:92 | 0 / 69 | 0 | — | — | — |
| rtx_pro_6000_server | `TRTLLM` | shared | trtllm | trtllm:92 | 0 / 69 | 0 | — | — | — |
| rtx_pro_6000_server | `vLLM_custom_eager` | shared | vllm | vllm:92 | 0 / 69 | 0 | — | — | — |
| rtx_pro_6000_server | `vLLM_custom_graph` | shared | vllm | vllm:92 | 0 / 69 | 0 | — | — | — |

## `dsa_context_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `default` | shared_fallback | trtllm | trtllm:14640 | 0 / 14640 | 0 | — | — | — |
| b200_sxm | `sglang_dsa_dense_mha_trtllm_ragged` | shared | sglang | sglang:35080 | 0 / 35080 | 0 | — | — | — |
| b200_sxm | `sglang_dsa_indexer_flashmla_sparse` | shared | sglang | sglang:17096 | 0 / 17096 | 0 | — | — | — |
| b200_sxm | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:17104 | 0 / 17104 | 0 | — | — | — |
| b200_sxm | `sglang_dsa_skip_indexer_flashmla_sparse` | shared | sglang | sglang:12963 | 0 / 12963 | 0 | — | — | — |
| b200_sxm | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:12960 | 0 / 12960 | 0 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:14640 | 0 / 14640 | 0 | — | — | — |
| b300_sxm | `sglang_dsa_dense_mha_trtllm_ragged` | shared | sglang | sglang:27896 | 0 / 20940 | 0 | — | — | — |
| b300_sxm | `sglang_dsa_indexer_flashmla_sparse` | shared | sglang | sglang:16918 | 0 / 16918 | 0 | — | — | — |
| b300_sxm | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:17096 | 0 / 17096 | 0 | — | — | — |
| b300_sxm | `sglang_dsa_skip_indexer_flashmla_sparse` | shared | sglang | sglang:6392 | 0 / 6392 | 0 | — | — | — |
| b300_sxm | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:6480 | 0 / 6480 | 0 | — | — | — |
| gb200 | `FLASHINFER_MLA_SPARSE` | shared | vllm | vllm:10248 | 0 / 10248 | 0 | — | — | — |
| gb200 | `FLASHMLA_SPARSE` | shared | vllm | vllm:4391 | 0 / 4391 | 0 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:14640 | 0 / 14640 | 0 | — | — | — |
| gb200 | `sglang_dsa_dense_mha_trtllm_ragged` | shared | sglang | sglang:46371 | 0 / 21048 | 0 | — | — | — |
| gb200 | `sglang_dsa_indexer_flashmla_sparse` | shared | sglang | sglang:31196 | 0 / 16984 | 0 | — | — | — |
| gb200 | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:25613 | 0 / 16984 | 0 | — | — | — |
| gb200 | `sglang_dsa_skip_indexer_flashmla_sparse` | shared | sglang | sglang:8272 | 0 / 6368 | 0 | — | — | — |
| gb200 | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:7878 | 0 / 6368 | 0 | — | — | — |
| gb300 | `FLASHINFER_MLA_SPARSE` | shared | vllm | vllm:10248 | 0 / 10248 | 0 | — | — | — |
| gb300 | `FLASHMLA_SPARSE` | shared | vllm | vllm:4392 | 0 / 4392 | 0 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:14640 | 0 / 14640 | 0 | — | — | — |
| gb300 | `sglang_dsa_dense_mha_trtllm_ragged` | shared | sglang | sglang:28064 | 0 / 21048 | 0 | — | — | — |
| gb300 | `sglang_dsa_indexer_flashmla_sparse` | shared | sglang | sglang:16984 | 0 / 16984 | 0 | — | — | — |
| gb300 | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:16984 | 0 / 16984 | 0 | — | — | — |
| gb300 | `sglang_dsa_skip_indexer_flashmla_sparse` | shared | sglang | sglang:6368 | 0 / 6368 | 0 | — | — | — |
| gb300 | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:6368 | 0 / 6368 | 0 | — | — | — |
| h100_sxm | `FLASHMLA_SPARSE` | shared | vllm | vllm:14668 | 0 / 14668 | 0 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:11654 | 0 / 7770 | 1 | — | — | — |
| h100_sxm | `sglang_dsa_dense_mha_fa3` | shared | sglang | sglang:50621 | 0 / 31203 | 0 | — | — | — |
| h100_sxm | `sglang_dsa_indexer_flashmla_kv` | shared | sglang | sglang:27320 | 0 / 13660 | 0 | — | — | — |
| h100_sxm | `sglang_dsa_indexer_flashmla_sparse` | shared | sglang | sglang:27312 | 0 / 13619 | 0 | — | — | — |
| h100_sxm | `sglang_dsa_skip_indexer_flashmla_kv` | shared | sglang | sglang:11037 | 0 / 11037 | 0 | — | — | — |
| h100_sxm | `sglang_dsa_skip_indexer_flashmla_sparse` | shared | sglang | sglang:9512 | 0 / 9512 | 0 | — | — | — |
| h200_sxm | `FLASHMLA_SPARSE` | shared | vllm | vllm:14816 | 0 / 14816 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:11712 | 0 / 7808 | 10 | — | — | — |
| h200_sxm | `sglang_dsa_dense_mha_fa3` | shared | sglang | sglang:32280 | 0 / 32280 | 0 | — | — | — |
| h200_sxm | `sglang_dsa_indexer_flashmla_kv` | shared | sglang | sglang:13660 | 0 / 13660 | 0 | — | — | — |
| h200_sxm | `sglang_dsa_indexer_flashmla_sparse` | shared | sglang | sglang:13660 | 0 / 13660 | 0 | — | — | — |
| h200_sxm | `sglang_dsa_skip_indexer_flashmla_kv` | shared | sglang | sglang:11125 | 0 / 11125 | 0 | — | — | — |
| h200_sxm | `sglang_dsa_skip_indexer_flashmla_sparse` | shared | sglang | sglang:11111 | 0 / 11111 | 0 | — | — | — |

## `dsa_generation_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `default` | shared_fallback | trtllm | trtllm:11040 | 0 / 11040 | 0 | — | — | — |
| b200_sxm | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:3600 | 0 / 3600 | 0 | — | — | — |
| b200_sxm | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:2448 | 0 / 2448 | 0 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:11037 | 0 / 11037 | 0 | — | — | — |
| b300_sxm | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:3600 | 0 / 3600 | 0 | — | — | — |
| b300_sxm | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:1224 | 0 / 1224 | 0 | — | — | — |
| gb200 | `FLASHINFER_MLA_SPARSE` | shared | vllm | vllm:7728 | 0 / 7728 | 0 | — | — | — |
| gb200 | `FLASHMLA_SPARSE` | shared | vllm | vllm:3311 | 0 / 3311 | 0 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:11036 | 0 / 11036 | 0 | — | — | — |
| gb200 | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:3600 | 0 / 3600 | 0 | — | — | — |
| gb200 | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:1224 | 0 / 1224 | 0 | — | — | — |
| gb300 | `FLASHINFER_MLA_SPARSE` | shared | vllm | vllm:7727 | 0 / 7727 | 0 | — | — | — |
| gb300 | `FLASHMLA_SPARSE` | shared | vllm | vllm:3312 | 0 / 3312 | 0 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:11025 | 0 / 11025 | 0 | — | — | — |
| gb300 | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:3600 | 0 / 3600 | 0 | — | — | — |
| gb300 | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:1224 | 0 / 1224 | 0 | — | — | — |
| h100_sxm | `FLASHMLA_SPARSE` | shared | vllm | vllm:11679 | 0 / 11679 | 0 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:8832 | 0 / 5888 | 2 | — | — | — |
| h100_sxm | `sglang_dsa_indexer_fa3` | shared | sglang | sglang:3568 | 0 / 1784 | 0 | — | — | — |
| h100_sxm | `sglang_dsa_indexer_flashmla_kv` | shared | sglang | sglang:3568 | 0 / 1784 | 0 | — | — | — |
| h100_sxm | `sglang_dsa_skip_indexer_fa3` | shared | sglang | sglang:1224 | 0 / 1224 | 0 | — | — | — |
| h100_sxm | `sglang_dsa_skip_indexer_flashmla_kv` | shared | sglang | sglang:1224 | 0 / 1224 | 0 | — | — | — |
| h200_sxm | `FLASHMLA_SPARSE` | shared | vllm | vllm:11776 | 0 / 11776 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:8832 | 0 / 5888 | 7 | — | — | — |
| h200_sxm | `sglang_dsa_indexer_fa3` | shared | sglang | sglang:1784 | 0 / 1784 | 0 | — | — | — |
| h200_sxm | `sglang_dsa_indexer_flashmla_kv` | shared | sglang | sglang:1784 | 0 / 1784 | 0 | — | — | — |
| h200_sxm | `sglang_dsa_skip_indexer_fa3` | shared | sglang | sglang:1224 | 0 / 1224 | 0 | — | — | — |
| h200_sxm | `sglang_dsa_skip_indexer_flashmla_kv` | shared | sglang | sglang:1224 | 0 / 1224 | 0 | — | — | — |

## `dsv4_csa_context_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:5587 | 0 / 5587 | 0 | — | — | — |
| b200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:5840 | 0 / 5840 | 0 | — | — | — |
| b200_sxm | `compressed_flashmla` | shared | sglang | sglang:43804 | 0 / 43804 | 0 | — | — | — |
| b300_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:5600 | 0 / 5600 | 0 | — | — | — |
| b300_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:5840 | 0 / 5840 | 0 | — | — | — |
| b300_sxm | `compressed_flashmla` | shared | sglang | sglang:89689 | 0 / 45449 | 0 | — | — | — |
| gb200 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:5584 | 0 / 5584 | 0 | — | — | — |
| gb200 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:5840 | 0 / 5840 | 0 | — | — | — |
| gb200 | `compressed_flashmla` | shared | sglang | sglang:53376 | 0 / 42610 | 0 | — | — | — |
| gb300 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:5600 | 0 / 5600 | 0 | — | — | — |
| gb300 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:5704 | 0 / 5704 | 0 | — | — | — |
| gb300 | `compressed_flashmla` | shared | sglang | sglang:94650 | 0 / 45424 | 0 | — | — | — |
| h100_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:5776 | 0 / 5776 | 0 | — | — | — |
| h100_sxm | `compressed_flashmla` | shared | sglang | sglang:52902 | 0 / 40384 | 0 | — | — | — |
| h200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:5840 | 0 / 5840 | 0 | — | — | — |
| h200_sxm | `compressed_flashmla` | shared | sglang | sglang:40592 | 0 / 40592 | 0 | — | — | — |

## `dsv4_csa_generation_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:1544 | 0 / 1544 | 0 | — | — | — |
| b200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1544 | 0 / 1544 | 0 | — | — | — |
| b200_sxm | `compressed_flashmla` | shared | sglang | sglang:3200 | 0 / 3200 | 0 | — | — | — |
| b300_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:1544 | 0 / 1544 | 0 | — | — | — |
| b300_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1544 | 0 / 1544 | 0 | — | — | — |
| b300_sxm | `compressed_flashmla` | shared | sglang | sglang:3200 | 0 / 3200 | 0 | — | — | — |
| gb200 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:1544 | 0 / 1544 | 0 | — | — | — |
| gb200 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1544 | 0 / 1544 | 0 | — | — | — |
| gb200 | `compressed_flashmla` | shared | sglang | sglang:3200 | 0 / 3200 | 0 | — | — | — |
| gb300 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:1544 | 0 / 1544 | 0 | — | — | — |
| gb300 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1048 | 0 / 1048 | 0 | — | — | — |
| gb300 | `compressed_flashmla` | shared | sglang | sglang:3200 | 0 / 3200 | 0 | — | — | — |
| h100_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1544 | 0 / 1544 | 0 | — | — | — |
| h100_sxm | `compressed_flashmla` | shared | sglang | sglang:3200 | 0 / 3200 | 0 | — | — | — |
| h200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1544 | 0 / 1544 | 0 | — | — | — |
| h200_sxm | `compressed_flashmla` | shared | sglang | sglang:3200 | 0 / 3200 | 0 | — | — | — |
| rtx_pro_6000_server | `compressed_flashmla` | shared | sglang | sglang:909 | 0 / 868 | 0 | — | — | — |

## `dsv4_csa_topk_calib_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `topk_transform_v1` | shared | sglang | sglang:4598 | 0 / 4598 | 0 | — | — | — |
| b200_sxm | `topk_transform_v2` | shared | sglang | sglang:244 | 0 / 244 | 0 | — | — | — |
| b300_sxm | `topk_transform_v1` | shared | sglang | sglang:4598 | 0 / 4598 | 0 | — | — | — |
| b300_sxm | `topk_transform_v2` | shared | sglang | sglang:244 | 0 / 244 | 0 | — | — | — |
| gb200 | `topk_transform_v1` | shared | sglang | sglang:4598 | 0 / 4598 | 0 | — | — | — |
| gb200 | `topk_transform_v2` | shared | sglang | sglang:244 | 0 / 244 | 0 | — | — | — |
| gb300 | `topk_transform_v1` | shared | sglang | sglang:4598 | 0 / 4598 | 0 | — | — | — |
| gb300 | `topk_transform_v2` | shared | sglang | sglang:244 | 0 / 244 | 0 | — | — | — |
| h100_sxm | `topk_transform_v1` | shared | sglang | sglang:3686 | 0 / 3686 | 0 | — | — | — |
| h100_sxm | `topk_transform_v2` | shared | sglang | sglang:244 | 0 / 244 | 0 | — | — | — |
| h200_sxm | `topk_transform_v1` | shared | sglang | sglang:3686 | 0 / 3686 | 0 | — | — | — |
| h200_sxm | `topk_transform_v2` | shared | sglang | sglang:244 | 0 / 244 | 0 | — | — | — |

## `dsv4_hca_attn_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:3918 | 0 / 3918 | 0 | — | — | — |
| b300_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:3918 | 0 / 3918 | 0 | — | — | — |
| gb200 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1959 | 0 / 1959 | 0 | — | — | — |
| gb300 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1959 | 0 / 1959 | 0 | — | — | — |
| h100_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1959 | 0 / 1959 | 0 | — | — | — |
| h200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1959 | 0 / 1959 | 0 | — | — | — |

## `dsv4_hca_context_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:5589 | 0 / 5589 | 0 | — | — | — |
| b200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:5840 | 0 / 5840 | 0 | — | — | — |
| b200_sxm | `compressed_flashmla` | shared | sglang | sglang:47456 | 0 / 47456 | 0 | — | — | — |
| b300_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:5600 | 0 / 5600 | 0 | — | — | — |
| b300_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:5840 | 0 / 5840 | 0 | — | — | — |
| b300_sxm | `compressed_flashmla` | shared | sglang | sglang:61527 | 0 / 47550 | 0 | — | — | — |
| gb200 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:5584 | 0 / 5584 | 0 | — | — | — |
| gb200 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:5840 | 0 / 5840 | 0 | — | — | — |
| gb200 | `compressed_flashmla` | shared | sglang | sglang:47456 | 0 / 47456 | 0 | — | — | — |
| gb300 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:5600 | 0 / 5600 | 0 | — | — | — |
| gb300 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:5840 | 0 / 5840 | 0 | — | — | — |
| gb300 | `compressed_flashmla` | shared | sglang | sglang:54924 | 0 / 47552 | 0 | — | — | — |
| h100_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:5793 | 0 / 5793 | 0 | — | — | — |
| h100_sxm | `compressed_flashmla` | shared | sglang | sglang:40383 | 0 / 40383 | 0 | — | — | — |
| h200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:5840 | 0 / 5840 | 0 | — | — | — |
| h200_sxm | `compressed_flashmla` | shared | sglang | sglang:40592 | 0 / 40592 | 0 | — | — | — |
| rtx_pro_6000_server | `compressed_flashmla` | shared | sglang | sglang:40384 | 0 / 20192 | 0 | — | — | — |

## `dsv4_hca_generation_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:1544 | 0 / 1544 | 0 | — | — | — |
| b200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1544 | 0 / 1544 | 0 | — | — | — |
| b200_sxm | `compressed_flashmla` | shared | sglang | sglang:3200 | 0 / 3200 | 0 | — | — | — |
| b300_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:1544 | 0 / 1544 | 0 | — | — | — |
| b300_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1544 | 0 / 1544 | 0 | — | — | — |
| b300_sxm | `compressed_flashmla` | shared | sglang | sglang:3200 | 0 / 3200 | 0 | — | — | — |
| gb200 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:1544 | 0 / 1544 | 0 | — | — | — |
| gb200 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1544 | 0 / 1544 | 0 | — | — | — |
| gb200 | `compressed_flashmla` | shared | sglang | sglang:3200 | 0 / 3200 | 0 | — | — | — |
| gb300 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:1544 | 0 / 1544 | 0 | — | — | — |
| gb300 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1544 | 0 / 1544 | 0 | — | — | — |
| gb300 | `compressed_flashmla` | shared | sglang | sglang:3200 | 0 / 3200 | 0 | — | — | — |
| h100_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1544 | 0 / 1544 | 0 | — | — | — |
| h100_sxm | `compressed_flashmla` | shared | sglang | sglang:3200 | 0 / 3200 | 0 | — | — | — |
| h200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:1544 | 0 / 1544 | 0 | — | — | — |
| h200_sxm | `compressed_flashmla` | shared | sglang | sglang:3200 | 0 / 3200 | 0 | — | — | — |
| rtx_pro_6000_server | `compressed_flashmla` | shared | sglang | sglang:1600 | 0 / 1600 | 0 | — | — | — |

## `dsv4_megamoe_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `deepgemm_megamoe` | shared | sglang | sglang:64 | 0 / 64 | 0 | — | — | — |
| gb200 | `deepgemm_megamoe` | shared | sglang | sglang:256 | 0 / 256 | 0 | — | — | — |
| gb300 | `deepgemm_megamoe` | shared | sglang | sglang:456 | 0 / 456 | 0 | — | — | — |

## `dsv4_paged_mqa_logits_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `deep_gemm.fp8_paged_mqa_logits` | shared | sglang | sglang:2181 | 0 / 2181 | 0 | — | — | — |
| b200_sxm | `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | shared | vllm | vllm:3918 | 0 / 3918 | 0 | — | — | — |
| b300_sxm | `deep_gemm.fp8_paged_mqa_logits` | shared | sglang | sglang:4362 | 0 / 2181 | 0 | — | — | — |
| b300_sxm | `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | shared | vllm | vllm:3918 | 0 / 3918 | 0 | — | — | — |
| gb200 | `deep_gemm.fp8_paged_mqa_logits` | shared | sglang | sglang:4261 | 0 / 2181 | 0 | — | — | — |
| gb200 | `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | shared | vllm | vllm:1959 | 0 / 1959 | 0 | — | — | — |
| gb300 | `deep_gemm.fp8_paged_mqa_logits` | shared | sglang | sglang:4362 | 0 / 2181 | 0 | — | — | — |
| gb300 | `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | shared | vllm | vllm:1959 | 0 / 1959 | 0 | — | — | — |
| h100_sxm | `deep_gemm.fp8_paged_mqa_logits` | shared | sglang | sglang:2055 | 0 / 2055 | 0 | — | — | — |
| h100_sxm | `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | shared | vllm | vllm:1959 | 0 / 1959 | 0 | — | — | — |
| h200_sxm | `deep_gemm.fp8_paged_mqa_logits` | shared | sglang | sglang:2055 | 0 / 2055 | 0 | — | — | — |
| h200_sxm | `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | shared | vllm | vllm:1959 | 0 / 1959 | 0 | — | — | — |

## `encoder_attention_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `flash_attention_v4` | shared | sglang | sglang:7679 | 0 / 7679 | 0 | — | — | — |
| b200_sxm | `torch_flow` | shared | trtllm | trtllm:7679 | 0 / 7679 | 0 | — | — | — |
| b200_sxm | `vllm_vit_flash_attn_fa4` | shared | vllm | vllm:7679 | 0 / 7679 | 0 | — | — | — |
| b300_sxm | `torch_flow` | shared | trtllm | trtllm:7679 | 0 / 7679 | 0 | — | — | — |
| b300_sxm | `triton` | shared | sglang | sglang:7679 | 0 / 7679 | 0 | — | — | — |
| b300_sxm | `vllm_vit_flash_attn_fa4` | shared | vllm | vllm:7679 | 0 / 7679 | 0 | — | — | — |
| gb200 | `flash_attention_v4` | shared | sglang | sglang:7679 | 0 / 7679 | 0 | — | — | — |
| gb200 | `torch_flow` | shared | trtllm | trtllm:7679 | 0 / 7679 | 0 | — | — | — |
| gb200 | `vllm_vit_flash_attn_fa4` | shared | vllm | vllm:7679 | 0 / 7679 | 0 | — | — | — |
| gb300 | `torch_flow` | shared | trtllm | trtllm:7679 | 0 / 7679 | 0 | — | — | — |
| gb300 | `triton` | shared | sglang | sglang:7679 | 0 / 7679 | 0 | — | — | — |
| gb300 | `vllm_vit_flash_attn_fa4` | shared | vllm | vllm:7679 | 0 / 7679 | 0 | — | — | — |
| h100_sxm | `flash_attention_v3` | shared | sglang | sglang:7679 | 0 / 7679 | 0 | — | — | — |
| h100_sxm | `torch_flow` | shared | trtllm | trtllm:7679 | 0 / 7679 | 0 | — | — | — |
| h100_sxm | `vllm_vit_flash_attn_fa3` | shared | vllm | vllm:7679 | 0 / 7679 | 0 | — | — | — |
| h200_sxm | `flash_attention_v3` | shared | sglang | sglang:7679 | 0 / 7679 | 0 | — | — | — |
| h200_sxm | `torch_flow` | shared | trtllm | trtllm:7679 | 0 / 7679 | 0 | — | — | — |
| h200_sxm | `vllm_vit_flash_attn_fa3` | shared | vllm | vllm:7679 | 0 / 7679 | 0 | — | — | — |
| l40s | `torch_flow` | shared | trtllm | trtllm:7679 | 0 / 7679 | 0 | — | — | — |
| l40s | `triton` | shared | sglang | sglang:7679 | 0 / 7679 | 0 | — | — | — |
| l40s | `vllm_vit_flash_attn_fa2` | shared | vllm | vllm:7679 | 0 / 7679 | 0 | — | — | — |
| rtx_pro_6000_server | `torch_flow` | shared | trtllm | trtllm:7679 | 0 / 7679 | 0 | — | — | — |
| rtx_pro_6000_server | `triton` | shared | sglang | sglang:7679 | 0 / 7679 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_vit_flash_attn_fa2` | shared | vllm | vllm:7679 | 0 / 7679 | 0 | — | — | — |

## `gdn_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `causal_conv1d_fn` | shared | sglang | sglang:824 | 0 / 824 | 0 | — | — | — |
| a100_sxm | `causal_conv1d_update` | shared | sglang | sglang:88 | 0 / 88 | 0 | — | — | — |
| a100_sxm | `fused_recurrent_gated_delta_rule` | shared | sglang | sglang:86 | 0 / 86 | 0 | — | — | — |
| b200_sxm | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:4529, trtllm:854, vllm:5264 | 4559 / 4704 | 1 | 7.1 | 137.3 | 277.4 |
| b200_sxm | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:462, trtllm:88, vllm:515 | 461 / 462 | 0 | 11.5 | 74.9 | 130.8 |
| b200_sxm | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:4529, trtllm:854 | 824 / 4559 | 0 | 18.7 | 41.1 | 45.6 |
| b200_sxm | `chunk_gated_delta_rule_flashinfer` | shared | vllm | vllm:5264 | 0 / 4704 | 0 | — | — | — |
| b200_sxm | `flashinfer_gated_delta_rule_decode` | shared | sglang | sglang:52 | 0 / 52 | 0 | — | — | — |
| b200_sxm | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| b200_sxm | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:457, vllm:509 | 457 / 457 | 0 | 0.5 | 4.5 | 9.3 |
| b300_sxm | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:4529, trtllm:837, vllm:5264 | 4567 / 4704 | 0 | 8.1 | 138.5 | 277.1 |
| b300_sxm | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:462, trtllm:88, vllm:515 | 461 / 462 | 0 | 10.2 | 70.9 | 129.4 |
| b300_sxm | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:4529, trtllm:836 | 799 / 4566 | 0 | 31.2 | 46.0 | 50.6 |
| b300_sxm | `chunk_gated_delta_rule_flashinfer` | shared | vllm | vllm:5264 | 0 / 4704 | 1 | — | — | — |
| b300_sxm | `flashinfer_gated_delta_rule_decode` | shared | sglang | sglang:52 | 0 / 52 | 0 | — | — | — |
| b300_sxm | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| b300_sxm | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:457, vllm:509 | 457 / 457 | 0 | 1.2 | 6.3 | 9.3 |
| gb200 | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:4529, trtllm:854, vllm:4704 | 4559 / 4704 | 0 | 3.6 | 137.3 | 276.0 |
| gb200 | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:462, trtllm:88, vllm:461 | 461 / 462 | 0 | 10.6 | 112.0 | 137.5 |
| gb200 | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:4529, trtllm:854 | 824 / 4559 | 0 | 22.8 | 54.0 | 68.0 |
| gb200 | `chunk_gated_delta_rule_flashinfer` | shared | vllm | vllm:4704 | 0 / 4704 | 0 | — | — | — |
| gb200 | `flashinfer_gated_delta_rule_decode` | shared | sglang | sglang:52 | 0 / 52 | 0 | — | — | — |
| gb200 | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| gb200 | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:457, vllm:457 | 457 / 457 | 0 | 1.0 | 4.5 | 6.6 |
| gb300 | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:4529, trtllm:837, vllm:4704 | 4567 / 4704 | 0 | 5.9 | 140.4 | 275.9 |
| gb300 | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:462, trtllm:88, vllm:461 | 461 / 462 | 0 | 10.5 | 141.3 | 182.3 |
| gb300 | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:4529, trtllm:836 | 799 / 4566 | 0 | 20.3 | 59.3 | 67.4 |
| gb300 | `chunk_gated_delta_rule_flashinfer` | shared | vllm | vllm:4704 | 0 / 4704 | 0 | — | — | — |
| gb300 | `flashinfer_gated_delta_rule_decode` | shared | sglang | sglang:52 | 0 / 52 | 0 | — | — | — |
| gb300 | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| gb300 | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:457, vllm:457 | 457 / 457 | 0 | 1.4 | 8.0 | 21.2 |
| h100_sxm | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:4004, trtllm:816, vllm:4141 | 4004 / 4141 | 0 | 4.2 | 135.9 | 279.2 |
| h100_sxm | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:407, trtllm:88, vllm:407 | 407 / 407 | 0 | 14.6 | 101.5 | 136.6 |
| h100_sxm | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:4004, trtllm:806 | 806 / 4004 | 0 | 42.2 | 59.4 | 69.5 |
| h100_sxm | `chunk_gated_delta_rule_flashinfer` | shared | vllm | vllm:4141 | 0 / 4141 | 0 | — | — | — |
| h100_sxm | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| h100_sxm | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:405, vllm:405 | 405 / 405 | 0 | 1.3 | 4.6 | 7.0 |
| h200_sxm | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:4004, trtllm:798, vllm:4144 | 4017 / 4144 | 0 | 5.1 | 194.5 | 279.1 |
| h200_sxm | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:407, trtllm:88, vllm:407 | 407 / 407 | 0 | 14.6 | 99.3 | 133.2 |
| h200_sxm | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:4004, trtllm:797 | 785 / 4016 | 0 | 44.5 | 130.0 | 145.6 |
| h200_sxm | `chunk_gated_delta_rule_flashinfer` | shared | vllm | vllm:4144 | 0 / 4144 | 0 | — | — | — |
| h200_sxm | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| h200_sxm | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:405, vllm:405 | 405 / 405 | 0 | 2.1 | 71.0 | 104.6 |
| l40s | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:4004, trtllm:786, vllm:3950 | 3956 / 4100 | 0 | 2.4 | 147.1 | 276.1 |
| l40s | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:407, trtllm:88, vllm:407 | 407 / 407 | 0 | 14.1 | 137.2 | 168.2 |
| l40s | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:4004, trtllm:786 | 786 / 4004 | 0 | 21.5 | 104.5 | 118.2 |
| l40s | `chunk_gated_delta_rule_triton` | shared | vllm | vllm:3935 | 0 / 3935 | 0 | — | — | — |
| l40s | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| l40s | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:405, vllm:405 | 405 / 405 | 0 | 0.4 | 4.3 | 11.7 |
| rtx_pro_6000_server | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:4004, trtllm:824, vllm:3952 | 3965 / 4102 | 0 | 3.1 | 137.7 | 272.9 |
| rtx_pro_6000_server | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:407, trtllm:88, vllm:407 | 407 / 407 | 0 | 9.2 | 105.0 | 139.6 |
| rtx_pro_6000_server | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:4004, trtllm:824 | 824 / 4004 | 0 | 17.7 | 84.7 | 97.0 |
| rtx_pro_6000_server | `chunk_gated_delta_rule_triton` | shared | vllm | vllm:3935 | 0 / 3935 | 0 | — | — | — |
| rtx_pro_6000_server | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| rtx_pro_6000_server | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:405, vllm:405 | 405 / 405 | 0 | 0.3 | 8.2 | 23.2 |

## `gemm_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `sglang` | shared | sglang | sglang:35742 | 0 / 35742 | 0 | — | — | — |
| a100_sxm | `torch_flow` | shared | trtllm | trtllm:9240 | 0 / 9240 | 0 | — | — | — |
| a100_sxm | `trt_flow_/smooth_quant_gemm_L96/PLUGIN_V2_SmoothQuantGemm_0` | shared | trtllm | trtllm:6048 | 0 / 6048 | 0 | — | — | — |
| a100_sxm | `trt_flow_/weight_only_quant_matmul_L257/PLUGIN_V2_WeightOnlyQuantMatmul_0` | shared | trtllm | trtllm:12096 | 0 / 12096 | 0 | — | — | — |
| a100_sxm | `vllm_default` | shared | vllm | vllm:9240 | 0 / 9240 | 0 | — | — | — |
| b200_sxm | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:73408 | 0 / 37518 | 22 | — | — | — |
| b200_sxm | `CutlassFp8BlockScaledMMKernel` | shared | vllm | vllm:10138 | 0 / 5328 | 0 | — | — | — |
| b200_sxm | `DeepGemmFp8BlockScaledMMKernel` | shared | vllm | vllm:62086 | 0 / 31080 | 0 | — | — | — |
| b200_sxm | `FlashInferCuteDslNvFp4LinearKernel` | shared | vllm | vllm:73408 | 0 / 37518 | 145 | — | — | — |
| b200_sxm | `deepgemm` | shared | trtllm | trtllm:29268 | 0 / 29268 | 0 | — | — | — |
| b200_sxm | `sglang_deepgemm_gemm_nt_f8f8bf16` | shared | sglang | sglang:59052 | 0 / 29526 | 15 | — | — | — |
| b200_sxm | `sglang_flashinfer_cutedsl_nvfp4` | shared | sglang | sglang:59052 | 0 / 29526 | 26 | — | — | — |
| b200_sxm | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:71558 | 0 / 35816 | 85 | — | — | — |
| b200_sxm | `sglang_torch_linear` | shared | sglang | sglang:71632 | 0 / 35890 | 86 | — | — | — |
| b200_sxm | `torch.nn.functional.linear` | shared | vllm | vllm:73408 | 0 / 37518 | 41 | — | — | — |
| b200_sxm | `torch_flow` | shared | trtllm | trtllm:100888 | 0 / 100888 | 0 | — | — | — |
| b300_sxm | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:73408 | 0 / 37518 | 13 | — | — | — |
| b300_sxm | `CutlassFp8BlockScaledMMKernel` | shared | vllm | vllm:10138 | 0 / 5328 | 0 | — | — | — |
| b300_sxm | `DeepGemmFp8BlockScaledMMKernel` | shared | vllm | vllm:62086 | 0 / 31080 | 0 | — | — | — |
| b300_sxm | `FlashInferCuteDslNvFp4LinearKernel` | shared | vllm | vllm:73408 | 0 / 37518 | 134 | — | — | — |
| b300_sxm | `deepgemm` | shared | trtllm | trtllm:29268 | 0 / 29268 | 0 | — | — | — |
| b300_sxm | `sglang_deepgemm_gemm_nt_f8f8bf16` | shared | sglang | sglang:59052 | 0 / 29526 | 10 | — | — | — |
| b300_sxm | `sglang_flashinfer_cutedsl_nvfp4` | shared | sglang | sglang:59052 | 0 / 29526 | 30 | — | — | — |
| b300_sxm | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:71706 | 0 / 35964 | 49 | — | — | — |
| b300_sxm | `sglang_torch_linear` | shared | sglang | sglang:71854 | 0 / 36112 | 104 | — | — | — |
| b300_sxm | `torch.nn.functional.linear` | shared | vllm | vllm:73408 | 0 / 37518 | 120 | — | — | — |
| b300_sxm | `torch_flow` | shared | trtllm | trtllm:100888 | 0 / 100888 | 0 | — | — | — |
| b60 | `vllm_default` | shared | vllm | vllm:40530 | 0 / 22176 | 1 | — | — | — |
| gb200 | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:71632 | 0 / 35890 | 12 | — | — | — |
| gb200 | `CutlassFp8BlockScaledMMKernel` | shared | vllm | vllm:9546 | 0 / 4810 | 0 | — | — | — |
| gb200 | `DeepGemmFp8BlockScaledMMKernel` | shared | vllm | vllm:62012 | 0 / 31006 | 3 | — | — | — |
| gb200 | `FlashInferCuteDslNvFp4LinearKernel` | shared | vllm | vllm:71632 | 0 / 35890 | 67 | — | — | — |
| gb200 | `deepgemm` | shared | trtllm | trtllm:29268 | 0 / 29268 | 0 | — | — | — |
| gb200 | `sglang_deepgemm_gemm_nt_f8f8bf16` | shared | sglang | sglang:59052 | 0 / 29526 | 14 | — | — | — |
| gb200 | `sglang_flashinfer_cutedsl_nvfp4` | shared | sglang | sglang:59052 | 0 / 29526 | 21 | — | — | — |
| gb200 | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:71558 | 0 / 35816 | 43 | — | — | — |
| gb200 | `sglang_torch_linear` | shared | sglang | sglang:71632 | 0 / 35890 | 61 | — | — | — |
| gb200 | `torch.nn.functional.linear` | shared | vllm | vllm:71632 | 0 / 35890 | 100 | — | — | — |
| gb200 | `torch_flow` | shared | trtllm | trtllm:100888 | 0 / 100888 | 0 | — | — | — |
| gb300 | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:71632 | 0 / 35890 | 18 | — | — | — |
| gb300 | `CutlassFp8BlockScaledMMKernel` | shared | vllm | vllm:9546 | 0 / 4810 | 0 | — | — | — |
| gb300 | `DeepGemmFp8BlockScaledMMKernel` | shared | vllm | vllm:62012 | 0 / 31006 | 2 | — | — | — |
| gb300 | `FlashInferCuteDslNvFp4LinearKernel` | shared | vllm | vllm:71632 | 0 / 35890 | 64 | — | — | — |
| gb300 | `deepgemm` | shared | trtllm | trtllm:29268 | 0 / 29268 | 0 | — | — | — |
| gb300 | `sglang_deepgemm_gemm_nt_f8f8bf16` | shared | sglang | sglang:59052 | 0 / 29526 | 4 | — | — | — |
| gb300 | `sglang_flashinfer_cutedsl_nvfp4` | shared | sglang | sglang:59052 | 0 / 29526 | 46 | — | — | — |
| gb300 | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:71558 | 0 / 35816 | 69 | — | — | — |
| gb300 | `sglang_torch_linear` | shared | sglang | sglang:71632 | 0 / 35890 | 85 | — | — | — |
| gb300 | `torch.nn.functional.linear` | shared | vllm | vllm:71632 | 0 / 35890 | 70 | — | — | — |
| gb300 | `torch_flow` | shared | trtllm | trtllm:100888 | 0 / 100888 | 0 | — | — | — |
| h100_sxm | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:35742 | 0 / 35742 | 0 | — | — | — |
| h100_sxm | `CutlassFp8BlockScaledMMKernel` | shared | vllm | vllm:4736 | 0 / 4736 | 0 | — | — | — |
| h100_sxm | `DeepGemmFp8BlockScaledMMKernel` | shared | vllm | vllm:23850 | 0 / 23850 | 0 | — | — | — |
| h100_sxm | `FlashInferFp8BlockScaledMMKernel` | shared | vllm | vllm:7118 | 0 / 7118 | 0 | — | — | — |
| h100_sxm | `sglang` | shared | sglang | sglang:35280 | 0 / 35280 | 0 | — | — | — |
| h100_sxm | `sglang_deepgemm_gemm_nt_f8f8bf16` | shared | sglang | sglang:29526 | 0 / 29526 | 0 | — | — | — |
| h100_sxm | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:35742 | 0 / 35742 | 0 | — | — | — |
| h100_sxm | `sglang_torch_linear` | shared | sglang | sglang:35742 | 0 / 35742 | 0 | — | — | — |
| h100_sxm | `torch.nn.functional.linear` | shared | vllm | vllm:35742 | 0 / 35742 | 0 | — | — | — |
| h100_sxm | `torch_flow` | shared | trtllm | trtllm:100668 | 0 / 100668 | 0 | — | — | — |
| h200_sxm | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:35742 | 0 / 35742 | 0 | — | — | — |
| h200_sxm | `CutlassFp8BlockScaledMMKernel` | shared | vllm | vllm:4736 | 0 / 4736 | 0 | — | — | — |
| h200_sxm | `DeepGemmFp8BlockScaledMMKernel` | shared | vllm | vllm:23882 | 0 / 23882 | 0 | — | — | — |
| h200_sxm | `FlashInferFp8BlockScaledMMKernel` | shared | vllm | vllm:7123 | 0 / 7123 | 0 | — | — | — |
| h200_sxm | `sglang_deepgemm_gemm_nt_f8f8bf16` | shared | sglang | sglang:29526 | 0 / 29526 | 0 | — | — | — |
| h200_sxm | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:35742 | 0 / 35742 | 0 | — | — | — |
| h200_sxm | `sglang_torch_linear` | shared | sglang | sglang:35742 | 0 / 35742 | 0 | — | — | — |
| h200_sxm | `torch.nn.functional.linear` | shared | vllm | vllm:35742 | 0 / 35742 | 0 | — | — | — |
| h200_sxm | `torch_flow` | shared | trtllm | trtllm:100668 | 0 / 100668 | 0 | — | — | — |
| l40s | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:35718 | 0 / 35718 | 0 | — | — | — |
| l40s | `TritonFp8BlockScaledMMKernel` | shared | vllm | vllm:35676 | 0 / 35676 | 0 | — | — | — |
| l40s | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:35742 | 0 / 35742 | 0 | — | — | — |
| l40s | `sglang_torch_linear` | shared | sglang | sglang:35742 | 0 / 35742 | 0 | — | — | — |
| l40s | `torch.nn.functional.linear` | shared | vllm | vllm:35672 | 0 / 35672 | 0 | — | — | — |
| l40s | `torch_flow` | shared | trtllm | trtllm:100668 | 0 / 100668 | 0 | — | — | — |
| rtx_pro_6000_server | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:35742 | 0 / 35742 | 0 | — | — | — |
| rtx_pro_6000_server | `CutlassFp8BlockScaledMMKernel` | shared | vllm | vllm:518 | 0 / 518 | 0 | — | — | — |
| rtx_pro_6000_server | `FlashInferCutlassNvFp4LinearKernel` | shared | vllm | vllm:35742 | 0 / 35742 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_flashinfer_cutlass_nvfp4` | shared | sglang | sglang:29526 | 0 / 29526 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:35742 | 0 / 35742 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_torch_linear` | shared | sglang | sglang:35742 | 0 / 35742 | 0 | — | — | — |
| rtx_pro_6000_server | `torch.nn.functional.linear` | shared | vllm | vllm:35742 | 0 / 35742 | 0 | — | — | — |
| rtx_pro_6000_server | `torch_flow` | shared | trtllm | trtllm:130156 | 0 / 130156 | 0 | — | — | — |

## `generation_attention_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `flash_attention` | shared | sglang | sglang:5093 | 0 / 5093 | 0 | — | — | — |
| a100_sxm | `torch_flow` | shared | trtllm | trtllm:5026 | 0 / 5026 | 0 | — | — | — |
| a100_sxm | `vllm_flash_attn` | shared | vllm | vllm:5431 | 0 / 5431 | 0 | — | — | — |
| b200_sxm | `flashinfer` | shared | sglang | sglang:3570 | 0 / 3570 | 0 | — | — | — |
| b200_sxm | `torch_flow` | shared | trtllm | trtllm:48314 | 0 / 48314 | 0 | — | — | — |
| b200_sxm | `torch_flow_flashinfer` | shared | trtllm | trtllm:3540 | 0 / 3540 | 0 | — | — | — |
| b200_sxm | `triton` | shared | sglang | sglang:8620 | 0 / 8620 | 0 | — | — | — |
| b200_sxm | `trtllm_mha` | shared | sglang | sglang:37643 | 0 / 37643 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer` | shared | vllm | vllm:36240 | 0 / 36240 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer_flashinfertrtllmapidecode` | shared | vllm | vllm:63180 | 0 / 63180 | 0 | — | — | — |
| b200_sxm | `vllm_triton_attn` | shared | vllm | vllm:4456 | 0 / 4456 | 0 | — | — | — |
| b300_sxm | `flashinfer` | shared | sglang | sglang:3570 | 0 / 3570 | 0 | — | — | — |
| b300_sxm | `torch_flow` | shared | trtllm | trtllm:46980 | 0 / 46980 | 0 | — | — | — |
| b300_sxm | `torch_flow_flashinfer` | shared | trtllm | trtllm:3540 | 0 / 3540 | 0 | — | — | — |
| b300_sxm | `triton` | shared | sglang | sglang:8620 | 0 / 8620 | 0 | — | — | — |
| b300_sxm | `trtllm_mha` | shared | sglang | sglang:37490 | 0 / 37490 | 0 | — | — | — |
| b300_sxm | `vllm_flashinfer_flashinfertrtllmapidecode` | shared | vllm | vllm:63180 | 0 / 63180 | 0 | — | — | — |
| b300_sxm | `vllm_triton_attn` | shared | vllm | vllm:2228 | 0 / 2228 | 0 | — | — | — |
| b60 | `vllm_flash_attn` | shared | vllm | vllm:57209 | 0 / 30948 | 3 | — | — | — |
| gb200 | `flashinfer` | shared | sglang | sglang:3570 | 0 / 3570 | 0 | — | — | — |
| gb200 | `torch_flow` | shared | trtllm | trtllm:48316 | 0 / 48316 | 0 | — | — | — |
| gb200 | `torch_flow_flashinfer` | shared | trtllm | trtllm:3540 | 0 / 3540 | 0 | — | — | — |
| gb200 | `triton` | shared | sglang | sglang:8620 | 0 / 8620 | 0 | — | — | — |
| gb200 | `trtllm_mha` | shared | sglang | sglang:36082 | 0 / 36082 | 0 | — | — | — |
| gb200 | `vllm_flashinfer_trtllmdecode` | shared | vllm | vllm:59844 | 0 / 59844 | 0 | — | — | — |
| gb200 | `vllm_triton_attn` | shared | vllm | vllm:2228 | 0 / 2228 | 0 | — | — | — |
| gb300 | `flashinfer` | shared | sglang | sglang:3570 | 0 / 3570 | 0 | — | — | — |
| gb300 | `torch_flow` | shared | trtllm | trtllm:48316 | 0 / 48316 | 0 | — | — | — |
| gb300 | `torch_flow_flashinfer` | shared | trtllm | trtllm:3540 | 0 / 3540 | 0 | — | — | — |
| gb300 | `triton` | shared | sglang | sglang:8620 | 0 / 8620 | 0 | — | — | — |
| gb300 | `trtllm_mha` | shared | sglang | sglang:36078 | 0 / 36078 | 0 | — | — | — |
| gb300 | `vllm_flashinfer_trtllmdecode` | shared | vllm | vllm:59844 | 0 / 59844 | 0 | — | — | — |
| gb300 | `vllm_triton_attn` | shared | vllm | vllm:2228 | 0 / 2228 | 0 | — | — | — |
| h100_sxm | `fa3` | shared | sglang | sglang:37176 | 0 / 37176 | 0 | — | — | — |
| h100_sxm | `torch_flow` | shared | trtllm | trtllm:52033 | 0 / 52033 | 0 | — | — | — |
| h100_sxm | `triton` | shared | sglang | sglang:3292 | 0 / 3292 | 0 | — | — | — |
| h100_sxm | `vllm_flash_attn_fa3` | shared | vllm | vllm:61366 | 0 / 61366 | 0 | — | — | — |
| h100_sxm | `vllm_flash_attn_fa4` | shared | vllm | vllm:800 | 0 / 800 | 0 | — | — | — |
| h200_sxm | `fa3` | shared | sglang | sglang:38510 | 0 / 38510 | 0 | — | — | — |
| h200_sxm | `torch_flow` | shared | trtllm | trtllm:53367 | 0 / 53367 | 0 | — | — | — |
| h200_sxm | `triton` | shared | sglang | sglang:3292 | 0 / 3292 | 0 | — | — | — |
| h200_sxm | `vllm_flash_attn_fa3` | shared | vllm | vllm:62700 | 0 / 62700 | 0 | — | — | — |
| h200_sxm | `vllm_flash_attn_fa4` | shared | vllm | vllm:800 | 0 / 800 | 0 | — | — | — |
| l40s | `flashinfer` | shared | sglang | sglang:29158 | 0 / 29158 | 0 | — | — | — |
| l40s | `torch_flow` | shared | trtllm | trtllm:52714 | 0 / 52714 | 0 | — | — | — |
| l40s | `triton` | shared | sglang | sglang:9384 | 0 / 9384 | 0 | — | — | — |
| l40s | `vllm_flash_attn_fa2` | shared | vllm | vllm:30679 | 0 / 30679 | 0 | — | — | — |
| l40s | `vllm_flashinfer_fidecode` | shared | vllm | vllm:27616 | 0 / 27616 | 0 | — | — | — |
| l40s | `vllm_triton_attn` | shared | vllm | vllm:2228 | 0 / 2228 | 0 | — | — | — |
| rtx_pro_6000_server | `flashinfer` | shared | sglang | sglang:24694 | 0 / 24694 | 0 | — | — | — |
| rtx_pro_6000_server | `torch_flow` | shared | trtllm | trtllm:51690 | 0 / 51690 | 0 | — | — | — |
| rtx_pro_6000_server | `triton` | shared | sglang | sglang:15188 | 0 / 15188 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_flash_attn_fa2` | shared | vllm | vllm:30677 | 0 / 30677 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_flashinfer_fidecode` | shared | vllm | vllm:27396 | 0 / 27396 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_triton_attn` | shared | vllm | vllm:2228 | 0 / 2228 | 0 | — | — | — |

## `generation_mla_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `default` | shared_fallback | trtllm | trtllm:5068 | 0 / 5068 | 0 | — | — | — |
| b200_sxm | `default` | shared_fallback | trtllm | trtllm:2896 | 0 / 2896 | 0 | — | — | — |
| b200_sxm | `trtllm_mla` | shared | sglang | sglang:2896 | 0 / 2896 | 0 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:2896 | 0 / 2896 | 0 | — | — | — |
| b300_sxm | `trtllm_mla` | shared | sglang | sglang:5792 | 0 / 2896 | 0 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:2896 | 0 / 2896 | 0 | — | — | — |
| gb200 | `trtllm_mla` | shared | sglang | sglang:5792 | 0 / 2896 | 0 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:2896 | 0 / 2896 | 0 | — | — | — |
| gb300 | `trtllm_mla` | shared | sglang | sglang:5792 | 0 / 2896 | 0 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:2896 | 0 / 2896 | 0 | — | — | — |
| h100_sxm | `flash_attention` | shared | sglang | sglang:5792 | 0 / 2896 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:2896 | 0 / 2896 | 0 | — | — | — |
| h200_sxm | `flash_attention` | shared | sglang | sglang:2896 | 0 / 2896 | 0 | — | — | — |
| l40s | `default` | shared_fallback | trtllm | trtllm:5068 | 0 / 5068 | 0 | — | — | — |
| l40s | `triton` | shared | sglang | sglang:1162 | 0 / 1162 | 0 | — | — | — |
| rtx_pro_6000_server | `default` | shared_fallback | trtllm | trtllm:2896 | 0 / 2896 | 0 | — | — | — |
| rtx_pro_6000_server | `triton` | shared | sglang | sglang:2400 | 0 / 1200 | 0 | — | — | — |

## `glm5_dsa_attn_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `flash_mla_sparse_fwd` | shared | sglang | sglang:3332 | 0 / 3332 | 0 | — | — | — |
| b300_sxm | `flash_mla_sparse_fwd` | shared | sglang | sglang:3577 | 0 / 3332 | 0 | — | — | — |
| gb200 | `flash_mla_sparse_fwd` | shared | sglang | sglang:3332 | 0 / 3332 | 0 | — | — | — |
| gb300 | `flash_mla_sparse_fwd` | shared | sglang | sglang:3550 | 0 / 3332 | 0 | — | — | — |
| h100_sxm | `flash_mla_sparse_fwd` | shared | sglang | sglang:3025 | 0 / 3025 | 0 | — | — | — |
| h200_sxm | `flash_mla_sparse_fwd` | shared | sglang | sglang:3025 | 0 / 3025 | 0 | — | — | — |

## `glm5_mqa_logits_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `deep_gemm.fp8_mqa_logits` | shared | sglang | sglang:3332 | 0 / 3332 | 0 | — | — | — |
| b300_sxm | `deep_gemm.fp8_mqa_logits` | shared | sglang | sglang:3332 | 0 / 3332 | 0 | — | — | — |
| gb200 | `deep_gemm.fp8_mqa_logits` | shared | sglang | sglang:3332 | 0 / 3332 | 0 | — | — | — |
| gb300 | `deep_gemm.fp8_mqa_logits` | shared | sglang | sglang:3332 | 0 / 3332 | 0 | — | — | — |
| h100_sxm | `deep_gemm.fp8_mqa_logits` | shared | sglang | sglang:3025 | 0 / 3025 | 0 | — | — | — |
| h200_sxm | `deep_gemm.fp8_mqa_logits` | shared | sglang | sglang:3025 | 0 / 3025 | 0 | — | — | — |

## `glm5_topk_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `fast_topk_transform_fused` | shared | sglang | sglang:4220 | 0 / 4220 | 0 | — | — | — |
| b300_sxm | `fast_topk_transform_fused` | shared | sglang | sglang:4220 | 0 / 4220 | 0 | — | — | — |
| gb200 | `fast_topk_transform_fused` | shared | sglang | sglang:4220 | 0 / 4220 | 0 | — | — | — |
| gb300 | `fast_topk_transform_fused` | shared | sglang | sglang:4220 | 0 / 4220 | 0 | — | — | — |
| h100_sxm | `fast_topk_transform_fused` | shared | sglang | sglang:3606 | 0 / 3606 | 0 | — | — | — |
| h200_sxm | `fast_topk_transform_fused` | shared | sglang | sglang:3606 | 0 / 3606 | 0 | — | — | — |

## `kda_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `causal_conv1d_fn_qkv3` | shared | sglang, vllm | sglang:396, vllm:428 | 396 / 428 | 0 | 180.7 | 192.3 | 194.2 |
| b200_sxm | `causal_conv1d_update` | shared | sglang, vllm | sglang:33, vllm:152 | 33 / 152 | 0 | 14.1 | 32.4 | 33.4 |
| b200_sxm | `chunk_kda` | shared | sglang | sglang:396 | 0 / 396 | 0 | — | — | — |
| b200_sxm | `flashkda_fwd` | shared | vllm | vllm:428 | 0 / 428 | 0 | — | — | — |
| b200_sxm | `fused_kda_decode` | shared | vllm | vllm:44 | 0 / 44 | 0 | — | — | — |
| b200_sxm | `fused_kda_decode_mtp_dspark` | shared | sglang | sglang:107 | 0 / 107 | 0 | — | — | — |
| b200_sxm | `fused_recurrent_kda` | shared | vllm | vllm:108 | 0 / 108 | 0 | — | — | — |
| b200_sxm | `fused_recurrent_kda_packed_decode` | shared | sglang, vllm | sglang:33, vllm:43 | 32 / 44 | 0 | 13.5 | 68.4 | 68.9 |
| b200_sxm | `kda_fused_decode` | shared | sglang | sglang:11 | 0 / 11 | 0 | — | — | — |
| b300_sxm | `causal_conv1d_fn_qkv3` | shared | sglang, vllm | sglang:396, vllm:428 | 396 / 428 | 0 | 184.0 | 193.6 | 199.4 |
| b300_sxm | `causal_conv1d_update` | shared | sglang, vllm | sglang:33, vllm:152 | 33 / 152 | 0 | 16.4 | 37.4 | 52.4 |
| b300_sxm | `chunk_kda` | shared | sglang | sglang:396 | 0 / 396 | 0 | — | — | — |
| b300_sxm | `flashkda_fwd` | shared | vllm | vllm:428 | 0 / 428 | 0 | — | — | — |
| b300_sxm | `fused_kda_decode` | shared | vllm | vllm:44 | 0 / 44 | 0 | — | — | — |
| b300_sxm | `fused_kda_decode_mtp_dspark` | shared | sglang | sglang:107 | 0 / 107 | 0 | — | — | — |
| b300_sxm | `fused_recurrent_kda` | shared | vllm | vllm:108 | 0 / 108 | 0 | — | — | — |
| b300_sxm | `fused_recurrent_kda_packed_decode` | shared | sglang, vllm | sglang:33, vllm:43 | 32 / 44 | 0 | 12.7 | 62.7 | 69.6 |
| b300_sxm | `kda_fused_decode` | shared | sglang | sglang:11 | 0 / 11 | 0 | — | — | — |
| gb200 | `causal_conv1d_fn_qkv3` | shared | sglang, vllm | sglang:396, vllm:428 | 396 / 428 | 0 | 193.3 | 197.1 | 197.6 |
| gb200 | `causal_conv1d_update` | shared | sglang, vllm | sglang:33, vllm:152 | 33 / 152 | 0 | 13.1 | 27.5 | 28.5 |
| gb200 | `chunk_kda` | shared | sglang | sglang:396 | 0 / 396 | 0 | — | — | — |
| gb200 | `flashkda_fwd` | shared | vllm | vllm:428 | 0 / 428 | 0 | — | — | — |
| gb200 | `fused_kda_decode` | shared | vllm | vllm:44 | 0 / 44 | 0 | — | — | — |
| gb200 | `fused_kda_decode_mtp_dspark` | shared | sglang | sglang:107 | 0 / 107 | 0 | — | — | — |
| gb200 | `fused_recurrent_kda` | shared | vllm | vllm:108 | 0 / 108 | 0 | — | — | — |
| gb200 | `fused_recurrent_kda_packed_decode` | shared | sglang, vllm | sglang:33, vllm:43 | 32 / 44 | 0 | 6.7 | 69.5 | 77.5 |
| gb200 | `kda_fused_decode` | shared | sglang | sglang:11 | 0 / 11 | 0 | — | — | — |
| gb300 | `causal_conv1d_fn_qkv3` | shared | sglang, vllm | sglang:396, vllm:848 | 396 / 448 | 0 | 24.8 | 196.3 | 197.7 |
| gb300 | `causal_conv1d_update` | shared | sglang, vllm | sglang:33, vllm:304 | 33 / 152 | 0 | 18.5 | 34.4 | 39.6 |
| gb300 | `chunk_kda` | shared | sglang | sglang:396 | 0 / 396 | 0 | — | — | — |
| gb300 | `flashkda_fwd` | shared | vllm | vllm:847 | 0 / 447 | 0 | — | — | — |
| gb300 | `fused_kda_decode` | shared | vllm | vllm:44 | 0 / 44 | 0 | — | — | — |
| gb300 | `fused_kda_decode_mtp_dspark` | shared | sglang | sglang:107 | 0 / 107 | 0 | — | — | — |
| gb300 | `fused_recurrent_kda` | shared | vllm | vllm:216 | 0 / 108 | 0 | — | — | — |
| gb300 | `fused_recurrent_kda_packed_decode` | shared | sglang, vllm | sglang:33, vllm:86 | 32 / 44 | 1 | 11.8 | 61.4 | 65.1 |
| gb300 | `kda_fused_decode` | shared | sglang | sglang:11 | 0 / 11 | 0 | — | — | — |
| h100_sxm | `causal_conv1d_fn_qkv3` | shared | sglang, vllm | sglang:396, vllm:428 | 396 / 428 | 0 | 181.9 | 193.1 | 194.3 |
| h100_sxm | `causal_conv1d_update` | shared | sglang, vllm | sglang:141, vllm:152 | 141 / 152 | 0 | 19.2 | 60.3 | 75.9 |
| h100_sxm | `chunk_kda` | shared | sglang | sglang:396 | 0 / 396 | 0 | — | — | — |
| h100_sxm | `flashkda_fwd` | shared | vllm | vllm:428 | 0 / 428 | 0 | — | — | — |
| h100_sxm | `fused_kda_decode` | shared | vllm | vllm:44 | 0 / 44 | 0 | — | — | — |
| h100_sxm | `fused_recurrent_kda` | shared | vllm | vllm:108 | 0 / 108 | 0 | — | — | — |
| h100_sxm | `fused_recurrent_kda_packed_decode` | shared | sglang, vllm | sglang:33, vllm:43 | 32 / 44 | 0 | 16.7 | 66.6 | 72.7 |
| h100_sxm | `fused_sigmoid_gating_delta_rule_update` | shared | sglang | sglang:108 | 0 / 108 | 0 | — | — | — |
| h100_sxm | `kda_fused_decode` | shared | sglang | sglang:11 | 0 / 11 | 0 | — | — | — |
| h200_sxm | `causal_conv1d_fn_qkv3` | shared | sglang, vllm | sglang:396, vllm:428 | 396 / 428 | 0 | 189.5 | 196.1 | 196.6 |
| h200_sxm | `causal_conv1d_update` | shared | sglang, vllm | sglang:141, vllm:152 | 141 / 152 | 0 | 21.0 | 70.4 | 79.3 |
| h200_sxm | `chunk_kda` | shared | sglang | sglang:396 | 0 / 396 | 0 | — | — | — |
| h200_sxm | `flashkda_fwd` | shared | vllm | vllm:428 | 0 / 428 | 0 | — | — | — |
| h200_sxm | `fused_kda_decode` | shared | vllm | vllm:44 | 0 / 44 | 0 | — | — | — |
| h200_sxm | `fused_recurrent_kda` | shared | vllm | vllm:108 | 0 / 108 | 0 | — | — | — |
| h200_sxm | `fused_recurrent_kda_packed_decode` | shared | sglang, vllm | sglang:33, vllm:43 | 32 / 44 | 0 | 19.4 | 49.9 | 52.6 |
| h200_sxm | `fused_sigmoid_gating_delta_rule_update` | shared | sglang | sglang:108 | 0 / 108 | 0 | — | — | — |
| h200_sxm | `kda_fused_decode` | shared | sglang | sglang:11 | 0 / 11 | 0 | — | — | — |
| l40s | `causal_conv1d_fn_qkv3` | shared | sglang, vllm | sglang:396, vllm:428 | 396 / 428 | 0 | 190.5 | 196.8 | 198.0 |
| l40s | `causal_conv1d_update` | shared | sglang, vllm | sglang:141, vllm:152 | 141 / 152 | 0 | 17.0 | 53.6 | 102.9 |
| l40s | `chunk_kda` | shared | sglang | sglang:396 | 0 / 396 | 0 | — | — | — |
| l40s | `chunk_kda_with_fused_gate` | shared | vllm | vllm:414 | 0 / 414 | 0 | — | — | — |
| l40s | `fused_recurrent_kda` | shared | vllm | vllm:108 | 0 / 108 | 0 | — | — | — |
| l40s | `fused_recurrent_kda_packed_decode` | shared | sglang, vllm | sglang:33, vllm:43 | 32 / 44 | 0 | 8.1 | 41.9 | 46.4 |
| l40s | `fused_sigmoid_gating_delta_rule_update` | shared | sglang | sglang:108 | 0 / 108 | 0 | — | — | — |
| rtx_pro_6000_server | `causal_conv1d_fn_qkv3` | shared | sglang, vllm | sglang:396, vllm:428 | 396 / 428 | 0 | 184.6 | 193.6 | 194.2 |
| rtx_pro_6000_server | `causal_conv1d_update` | shared | sglang, vllm | sglang:141, vllm:152 | 141 / 152 | 0 | 27.3 | 68.8 | 82.5 |
| rtx_pro_6000_server | `chunk_kda` | shared | sglang | sglang:396 | 0 / 396 | 0 | — | — | — |
| rtx_pro_6000_server | `flashkda_fwd` | shared | vllm | vllm:428 | 0 / 428 | 0 | — | — | — |
| rtx_pro_6000_server | `fused_kda_decode` | shared | vllm | vllm:44 | 0 / 44 | 0 | — | — | — |
| rtx_pro_6000_server | `fused_recurrent_kda` | shared | vllm | vllm:108 | 0 / 108 | 0 | — | — | — |
| rtx_pro_6000_server | `fused_recurrent_kda_packed_decode` | shared | sglang, vllm | sglang:33, vllm:43 | 32 / 44 | 0 | 12.2 | 80.8 | 89.2 |
| rtx_pro_6000_server | `fused_sigmoid_gating_delta_rule_update` | shared | sglang | sglang:108 | 0 / 108 | 0 | — | — | — |
| rtx_pro_6000_server | `kda_fused_decode` | shared | sglang | sglang:11 | 0 / 11 | 0 | — | — | — |

## `mamba2_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `causal_conv1d_fn` | shared | trtllm | trtllm:619 | 0 / 619 | 0 | — | — | — |
| b200_sxm | `causal_conv1d_update` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |
| b300_sxm | `causal_conv1d_fn` | shared | trtllm | trtllm:586 | 0 / 586 | 0 | — | — | — |
| b300_sxm | `causal_conv1d_update` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |
| gb200 | `causal_conv1d_fn` | shared | trtllm | trtllm:624 | 0 / 624 | 0 | — | — | — |
| gb200 | `causal_conv1d_update` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |
| gb300 | `causal_conv1d_fn` | shared | trtllm | trtllm:586 | 0 / 586 | 0 | — | — | — |
| gb300 | `causal_conv1d_update` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |
| h100_sxm | `causal_conv1d_fn` | shared | trtllm | trtllm:592 | 0 / 592 | 0 | — | — | — |
| h100_sxm | `causal_conv1d_update` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |
| h200_sxm | `causal_conv1d_fn` | shared | trtllm | trtllm:619 | 0 / 619 | 0 | — | — | — |
| h200_sxm | `causal_conv1d_update` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |
| l40s | `causal_conv1d_fn` | shared | trtllm | trtllm:559 | 0 / 559 | 0 | — | — | — |
| l40s | `causal_conv1d_update` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |
| rtx_pro_6000_server | `causal_conv1d_fn` | shared | trtllm | trtllm:598 | 0 / 598 | 0 | — | — | — |
| rtx_pro_6000_server | `causal_conv1d_update` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |

## `mhc_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `sglang_tilelang_mhc_post` | shared | sglang | sglang:70 | 0 / 70 | 0 | — | — | — |
| b200_sxm | `sglang_tilelang_mhc_pre` | shared | sglang | sglang:70 | 0 / 70 | 0 | — | — | — |
| b200_sxm | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:70 | 0 / 70 | 0 | — | — | — |
| b200_sxm | `trtllm_mhc_pre_dg_nosplit` | shared | trtllm | trtllm:27 | 0 / 27 | 0 | — | — | — |
| b200_sxm | `trtllm_mhc_pre_dg_splitk` | shared | trtllm | trtllm:26 | 0 / 26 | 0 | — | — | — |
| b200_sxm | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:16 | 0 / 16 | 0 | — | — | — |
| b200_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:70 | 0 / 70 | 0 | — | — | — |
| b200_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:70 | 0 / 70 | 0 | — | — | — |
| b300_sxm | `sglang_tilelang_mhc_post` | shared | sglang | sglang:70 | 0 / 70 | 0 | — | — | — |
| b300_sxm | `sglang_tilelang_mhc_pre` | shared | sglang | sglang:70 | 0 / 70 | 0 | — | — | — |
| b300_sxm | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:70 | 0 / 70 | 0 | — | — | — |
| b300_sxm | `trtllm_mhc_pre_dg_nosplit` | shared | trtllm | trtllm:28 | 0 / 28 | 0 | — | — | — |
| b300_sxm | `trtllm_mhc_pre_dg_splitk` | shared | trtllm | trtllm:26 | 0 / 26 | 0 | — | — | — |
| b300_sxm | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:16 | 0 / 16 | 0 | — | — | — |
| b300_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:70 | 0 / 70 | 0 | — | — | — |
| b300_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:70 | 0 / 70 | 0 | — | — | — |
| gb200 | `sglang_tilelang_mhc_post` | shared | sglang | sglang:70 | 0 / 70 | 0 | — | — | — |
| gb200 | `sglang_tilelang_mhc_pre` | shared | sglang | sglang:70 | 0 / 70 | 0 | — | — | — |
| gb200 | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:70 | 0 / 70 | 0 | — | — | — |
| gb200 | `trtllm_mhc_pre_dg_nosplit` | shared | trtllm | trtllm:26 | 0 / 26 | 0 | — | — | — |
| gb200 | `trtllm_mhc_pre_dg_splitk` | shared | trtllm | trtllm:27 | 0 / 27 | 0 | — | — | — |
| gb200 | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:16 | 0 / 16 | 0 | — | — | — |
| gb200 | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:70 | 0 / 70 | 0 | — | — | — |
| gb200 | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:70 | 0 / 70 | 0 | — | — | — |
| gb300 | `sglang_tilelang_mhc_post` | shared | sglang | sglang:70 | 0 / 70 | 0 | — | — | — |
| gb300 | `sglang_tilelang_mhc_pre` | shared | sglang | sglang:70 | 0 / 70 | 0 | — | — | — |
| gb300 | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:70 | 0 / 70 | 0 | — | — | — |
| gb300 | `trtllm_mhc_pre_dg_nosplit` | shared | trtllm | trtllm:28 | 0 / 28 | 0 | — | — | — |
| gb300 | `trtllm_mhc_pre_dg_splitk` | shared | trtllm | trtllm:26 | 0 / 26 | 0 | — | — | — |
| gb300 | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:16 | 0 / 16 | 0 | — | — | — |
| gb300 | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:70 | 0 / 70 | 0 | — | — | — |
| gb300 | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:70 | 0 / 70 | 0 | — | — | — |
| h100_sxm | `sglang_tilelang_mhc_post` | shared | sglang | sglang:68 | 0 / 68 | 0 | — | — | — |
| h100_sxm | `sglang_tilelang_mhc_pre` | shared | sglang | sglang:70 | 0 / 70 | 0 | — | — | — |
| h100_sxm | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:69 | 0 / 69 | 0 | — | — | — |
| h100_sxm | `trtllm_mhc_pre_dg_nosplit` | shared | trtllm | trtllm:24 | 0 / 24 | 0 | — | — | — |
| h100_sxm | `trtllm_mhc_pre_dg_splitk` | shared | trtllm | trtllm:27 | 0 / 27 | 0 | — | — | — |
| h100_sxm | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:16 | 0 / 16 | 0 | — | — | — |
| h100_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| h100_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:70 | 0 / 70 | 0 | — | — | — |
| h200_sxm | `sglang_tilelang_mhc_post` | shared | sglang | sglang:70 | 0 / 70 | 0 | — | — | — |
| h200_sxm | `sglang_tilelang_mhc_pre` | shared | sglang | sglang:70 | 0 / 70 | 0 | — | — | — |
| h200_sxm | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:70 | 0 / 70 | 0 | — | — | — |
| h200_sxm | `trtllm_mhc_pre_dg_nosplit` | shared | trtllm | trtllm:26 | 0 / 26 | 0 | — | — | — |
| h200_sxm | `trtllm_mhc_pre_dg_splitk` | shared | trtllm | trtllm:24 | 0 / 24 | 0 | — | — | — |
| h200_sxm | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:19 | 0 / 19 | 0 | — | — | — |
| h200_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:70 | 0 / 70 | 0 | — | — | — |
| h200_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:70 | 0 / 70 | 0 | — | — | — |
| l40s | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:67 | 0 / 67 | 0 | — | — | — |
| l40s | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:65 | 0 / 65 | 0 | — | — | — |
| l40s | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:67 | 0 / 67 | 0 | — | — | — |
| l40s | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_tilelang_mhc_post` | shared | sglang | sglang:67 | 0 / 67 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_torch_mhc_pre` | shared | sglang | sglang:67 | 0 / 67 | 0 | — | — | — |
| rtx_pro_6000_server | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:69 | 0 / 69 | 0 | — | — | — |
| rtx_pro_6000_server | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:67 | 0 / 67 | 0 | — | — | — |

## `mla_bmm_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `default` | shared_fallback | sglang, trtllm | sglang:424, trtllm:424 | 224 / 224 | 0 | 1.4 | 18.4 | 115.9 |
| b200_sxm | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:636 | 0 / 336 | 0 | — | — | — |
| b200_sxm | `sglang_torch_bmm` | shared | sglang | sglang:636 | 0 / 336 | 0 | — | — | — |
| b200_sxm | `trtllm_bmm_out` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| b200_sxm | `trtllm_bmm_out_dequant_bf16` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| b200_sxm | `vllm_torch_bmm` | shared | vllm | vllm:636 | 0 / 336 | 0 | — | — | — |
| b300_sxm | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| b300_sxm | `sglang_torch_bmm` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| b300_sxm | `trtllm_bmm_out` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| b300_sxm | `trtllm_bmm_out_dequant_bf16` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| b300_sxm | `vllm_torch_bmm` | shared | vllm | vllm:636 | 0 / 336 | 0 | — | — | — |
| gb200 | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| gb200 | `sglang_torch_bmm` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| gb200 | `trtllm_bmm_out` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| gb200 | `trtllm_bmm_out_dequant_bf16` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| gb200 | `vllm_torch_bmm` | shared | vllm | vllm:636 | 0 / 336 | 0 | — | — | — |
| gb300 | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| gb300 | `sglang_torch_bmm` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| gb300 | `trtllm_bmm_out` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| gb300 | `trtllm_bmm_out_dequant_bf16` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| gb300 | `vllm_torch_bmm` | shared | vllm | vllm:1072 | 0 / 336 | 2 | — | — | — |
| h100_sxm | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| h100_sxm | `sglang_torch_bmm` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| h100_sxm | `trtllm_bmm_out` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| h100_sxm | `trtllm_fp8_block_scaling_bmm_out` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| h100_sxm | `vllm_torch_bmm` | shared | vllm | vllm:636 | 0 / 336 | 0 | — | — | — |
| h200_sxm | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:424 | 0 / 224 | 0 | — | — | — |
| h200_sxm | `sglang_torch_bmm` | shared | sglang | sglang:424 | 0 / 224 | 0 | — | — | — |
| h200_sxm | `trtllm_bmm_out` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| h200_sxm | `trtllm_fp8_block_scaling_bmm_out` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| h200_sxm | `vllm_torch_bmm` | shared | vllm | vllm:636 | 0 / 336 | 0 | — | — | — |
| l40s | `default` | shared_fallback | trtllm | trtllm:848 | 0 / 448 | 0 | — | — | — |
| l40s | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:424 | 0 / 224 | 0 | — | — | — |
| l40s | `sglang_torch_bmm` | shared | sglang | sglang:424 | 0 / 224 | 0 | — | — | — |
| l40s | `trtllm_bmm_out` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| l40s | `trtllm_fp8_block_scaling_bmm_out` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_torch_bmm` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| rtx_pro_6000_server | `trtllm_bmm_out` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |
| rtx_pro_6000_server | `trtllm_fp8_block_scaling_bmm_out` | shared | trtllm | trtllm:424 | 0 / 224 | 0 | — | — | — |

## `mla_context_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `FLASHINFER_MLA` | shared | vllm | vllm:855 | 0 / 855 | 0 | — | — | — |
| b200_sxm | `FLASH_ATTN` | shared | vllm | vllm:8136 | 0 / 8136 | 0 | — | — | — |
| b200_sxm | `TRTLLM_RAGGED` | shared | vllm | vllm:4068 | 0 / 4068 | 0 | — | — | — |
| b200_sxm | `default` | shared_fallback | trtllm | trtllm:5856 | 0 / 5856 | 0 | — | — | — |
| b300_sxm | `FLASHINFER_MLA` | shared | vllm | vllm:855 | 0 / 855 | 0 | — | — | — |
| b300_sxm | `FLASH_ATTN` | shared | vllm | vllm:8136 | 0 / 8136 | 0 | — | — | — |
| b300_sxm | `TRTLLM_RAGGED` | shared | vllm | vllm:4068 | 0 / 4068 | 0 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:5856 | 0 / 5856 | 0 | — | — | — |
| gb200 | `FLASHINFER_MLA` | shared | vllm | vllm:648 | 0 / 648 | 0 | — | — | — |
| gb200 | `FLASH_ATTN` | shared | vllm | vllm:5424 | 0 / 5424 | 0 | — | — | — |
| gb200 | `TRTLLM_RAGGED` | shared | vllm | vllm:2712 | 0 / 2712 | 0 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:5856 | 0 / 5856 | 0 | — | — | — |
| gb300 | `FLASHINFER_MLA` | shared | vllm | vllm:2264 | 0 / 1620 | 0 | — | — | — |
| gb300 | `FLASH_ATTN` | shared | vllm | vllm:18925 | 0 / 13560 | 0 | — | — | — |
| gb300 | `TRTLLM_RAGGED` | shared | vllm | vllm:9455 | 0 / 6780 | 0 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:5856 | 0 / 5856 | 0 | — | — | — |
| h100_sxm | `FLASHMLA` | shared | vllm | vllm:702 | 0 / 702 | 0 | — | — | — |
| h100_sxm | `FLASH_ATTN` | shared | vllm | vllm:2176 | 0 / 2176 | 0 | — | — | — |
| h100_sxm | `FLASH_ATTN_MLA` | shared | vllm | vllm:1008 | 0 / 1008 | 0 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:3866 | 0 / 3866 | 0 | — | — | — |
| h200_sxm | `FLASHMLA` | shared | vllm | vllm:702 | 0 / 702 | 0 | — | — | — |
| h200_sxm | `FLASH_ATTN` | shared | vllm | vllm:2176 | 0 / 2176 | 0 | — | — | — |
| h200_sxm | `FLASH_ATTN_MLA` | shared | vllm | vllm:1008 | 0 / 1008 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:3869 | 0 / 3869 | 0 | — | — | — |
| l40s | `FLASH_ATTN` | shared | vllm | vllm:1790 | 0 / 1790 | 0 | — | — | — |
| l40s | `TRITON_MLA` | shared | vllm | vllm:144 | 0 / 144 | 0 | — | — | — |
| rtx_pro_6000_server | `FLASH_ATTN` | shared | vllm | vllm:3616 | 0 / 3616 | 0 | — | — | — |
| rtx_pro_6000_server | `TRITON_MLA` | shared | vllm | vllm:162 | 0 / 162 | 0 | — | — | — |
| rtx_pro_6000_server | `default` | shared_fallback | trtllm | trtllm:5828 | 0 / 5828 | 0 | — | — | — |

## `mla_generation_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `FLASHINFER_MLA` | shared | vllm | vllm:11400 | 0 / 11400 | 0 | — | — | — |
| b200_sxm | `default` | shared_fallback | trtllm | trtllm:8831 | 0 / 8831 | 0 | — | — | — |
| b300_sxm | `FLASHINFER_MLA` | shared | vllm | vllm:11400 | 0 / 11400 | 0 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:8832 | 0 / 8832 | 0 | — | — | — |
| gb200 | `FLASHINFER_MLA` | shared | vllm | vllm:8832 | 0 / 8832 | 0 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:8832 | 0 / 8832 | 0 | — | — | — |
| gb300 | `FLASHINFER_MLA` | shared | vllm | vllm:30812 | 0 / 22080 | 0 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:8832 | 0 / 8832 | 0 | — | — | — |
| h100_sxm | `FLASHMLA` | shared | vllm | vllm:2944 | 0 / 2944 | 0 | — | — | — |
| h100_sxm | `FLASH_ATTN_MLA` | shared | vllm | vllm:2896 | 0 / 2896 | 0 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:5888 | 0 / 5888 | 0 | — | — | — |
| h200_sxm | `FLASHMLA` | shared | vllm | vllm:2944 | 0 / 2944 | 0 | — | — | — |
| h200_sxm | `FLASH_ATTN_MLA` | shared | vllm | vllm:2944 | 0 / 2944 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:5888 | 0 / 5888 | 0 | — | — | — |
| l40s | `TRITON_MLA` | shared | vllm | vllm:2656 | 0 / 2656 | 0 | — | — | — |
| rtx_pro_6000_server | `TRITON_MLA` | shared | vllm | vllm:2990 | 0 / 2990 | 0 | — | — | — |
| rtx_pro_6000_server | `default` | shared_fallback | trtllm | trtllm:8832 | 0 / 8832 | 0 | — | — | — |

## `moe_a2a_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `deepep` | shared | trtllm | trtllm:704 | 0 / 704 | 0 | — | — | — |
| b300_sxm | `deepep` | shared | trtllm | trtllm:704 | 0 / 704 | 0 | — | — | — |
| gb200 | `deepep` | shared | trtllm, vllm | trtllm:704, vllm:324 | 0 / 1028 | 0 | — | — | — |
| gb300 | `deepep` | shared | trtllm, vllm | trtllm:704, vllm:324 | 0 / 1028 | 0 | — | — | — |
| h100_sxm | `deepep` | shared | trtllm, vllm | trtllm:564, vllm:648 | 0 / 1212 | 0 | — | — | — |
| h200_sxm | `deepep` | shared | trtllm, vllm | trtllm:564, vllm:324 | 0 / 888 | 0 | — | — | — |

## `moe_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `moe_torch_flow` | shared | trtllm | trtllm:8940 | 0 / 8940 | 0 | — | — | — |
| a100_sxm | `sglang_fused_moe_triton` | shared | sglang | sglang:36531 | 0 / 33453 | 0 | — | — | — |
| a100_sxm | `sglang_marlin_moe` | shared | sglang | sglang:19035 | 0 / 16686 | 0 | — | — | — |
| a100_sxm | `vllm_fused_moe` | shared | vllm | vllm:6360 | 0 / 6360 | 0 | — | — | — |
| b200_sxm | `deepgemm` | shared | trtllm | trtllm:38799 | 0 / 38799 | 0 | — | — | — |
| b200_sxm | `moe_torch_flow` | shared | trtllm | trtllm:10692 | 0 / 6480 | 5 | — | — | — |
| b200_sxm | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:114048 | 0 / 114048 | 0 | — | — | — |
| b200_sxm | `moe_torch_flow_min_latency` | shared | trtllm | trtllm:36045 | 0 / 36045 | 0 | — | — | — |
| b200_sxm | `moe_torch_flow_nongated` | shared | trtllm | trtllm:22923 | 0 / 22923 | 0 | — | — | — |
| b200_sxm | `sglang_flashinfer_trtllm_moe` | shared | sglang | sglang:121662 | 0 / 121662 | 0 | — | — | — |
| b200_sxm | `sglang_fused_moe_triton` | shared | sglang | sglang:36936 | 0 / 36936 | 0 | — | — | — |
| b200_sxm | `sglang_mxfp4_flashinfer_trtllm_moe` | shared | sglang | sglang:5184 | 0 / 5184 | 0 | — | — | — |
| b200_sxm | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| b200_sxm | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| b200_sxm | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmodular` | shared | vllm | vllm:3402 | 0 / 3402 | 0 | — | — | — |
| b200_sxm | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic` | shared | vllm | vllm:14737 | 0 / 14737 | 0 | — | — | — |
| b200_sxm | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer_trtllm_moe_fp4` | shared | vllm | vllm:10287 | 0 / 9234 | 0 | — | — | — |
| b200_sxm | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmodular` | shared | vllm | vllm:4374 | 0 / 4374 | 0 | — | — | — |
| b200_sxm | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:13527 | 0 / 12474 | 0 | — | — | — |
| b200_sxm | `vllm_fused_moe` | shared | vllm | vllm:83407 | 0 / 80329 | 0 | — | — | — |
| b200_sxm | `vllm_gptossmxfp4moe_flashinfer_trtllm_mxfp4_bf16_trtllmmxfp4expertsmonolithic` | shared | vllm | vllm:1944 | 0 / 1944 | 0 | — | — | — |
| b200_sxm | `vllm_marlin_int4_moe` | shared | vllm | vllm:10209 | 0 / 9237 | 0 | — | — | — |
| b200_sxm | `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:1863 | 0 / 1863 | 0 | — | — | — |
| b200_sxm | `vllm_modeloptfp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:12269 | 0 / 12269 | 0 | — | — | — |
| b200_sxm | `vllm_mxfp4_moe` | shared | vllm | vllm:1138 | 0 / 1138 | 0 | — | — | — |
| b200_sxm | `vllm_mxfp4moe_flashinfer_trtllm_mxfp4_mxfp8_trtllmmxfp4expertsmodular` | shared | vllm | vllm:2187 | 0 / 2187 | 0 | — | — | — |
| b200_sxm | `vllm_unquantizedfusedmoe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:1782 | 0 / 1782 | 0 | — | — | — |
| b200_sxm | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16experts` | shared | vllm | vllm:13296 | 0 / 13296 | 0 | — | — | — |
| b200_sxm | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16expertsmonolithic` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| b200_sxm | `vllm_unquantizedfusedmoe_triton_tritonexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| b300_sxm | `deepgemm` | shared | trtllm | trtllm:38799 | 0 / 38799 | 0 | — | — | — |
| b300_sxm | `moe_torch_flow` | shared | trtllm | trtllm:12472 | 0 / 7371 | 9 | — | — | — |
| b300_sxm | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:114048 | 0 / 114048 | 0 | — | — | — |
| b300_sxm | `moe_torch_flow_nongated` | shared | trtllm | trtllm:22923 | 0 / 22923 | 0 | — | — | — |
| b300_sxm | `sglang_flashinfer_trtllm_moe` | shared | sglang | sglang:198607 | 0 / 118584 | 0 | — | — | — |
| b300_sxm | `sglang_fused_moe_triton` | shared | sglang | sglang:61582 | 0 / 36936 | 0 | — | — | — |
| b300_sxm | `sglang_mxfp4_flashinfer_trtllm_moe` | shared | sglang | sglang:9720 | 0 / 5184 | 0 | — | — | — |
| b300_sxm | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| b300_sxm | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| b300_sxm | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmodular` | shared | vllm | vllm:3402 | 0 / 3402 | 0 | — | — | — |
| b300_sxm | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic` | shared | vllm | vllm:14789 | 0 / 14789 | 0 | — | — | — |
| b300_sxm | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| b300_sxm | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmodular` | shared | vllm | vllm:4374 | 0 / 4374 | 0 | — | — | — |
| b300_sxm | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:13527 | 0 / 12474 | 1 | — | — | — |
| b300_sxm | `vllm_gptossmxfp4moe_flashinfer_trtllm_mxfp4_bf16_trtllmmxfp4expertsmonolithic` | shared | vllm | vllm:1944 | 0 / 1944 | 0 | — | — | — |
| b300_sxm | `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:1863 | 0 / 1863 | 0 | — | — | — |
| b300_sxm | `vllm_modeloptfp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:12555 | 0 / 12555 | 0 | — | — | — |
| b300_sxm | `vllm_mxfp4moe_flashinfer_trtllm_mxfp4_mxfp8_trtllmmxfp4expertsmodular` | shared | vllm | vllm:2187 | 0 / 2187 | 0 | — | — | — |
| b300_sxm | `vllm_unquantizedfusedmoe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:1782 | 0 / 1782 | 0 | — | — | — |
| b300_sxm | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16experts` | shared | vllm | vllm:13504 | 0 / 13504 | 0 | — | — | — |
| b300_sxm | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16expertsmonolithic` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| b300_sxm | `vllm_unquantizedfusedmoe_triton_tritonexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| b60 | `vllm_xpu_moe` | shared | vllm | vllm:1836 | 0 / 1404 | 0 | — | — | — |
| b60 | `vllm_xpu_moe_mxfp4` | shared | vllm | vllm:2592 | 0 / 1296 | 0 | — | — | — |
| gb200 | `deepgemm` | shared | trtllm | trtllm:38799 | 0 / 38799 | 0 | — | — | — |
| gb200 | `moe_torch_flow` | shared | trtllm | trtllm:12312 | 0 / 7371 | 4 | — | — | — |
| gb200 | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:114048 | 0 / 114048 | 0 | — | — | — |
| gb200 | `moe_torch_flow_min_latency` | shared | trtllm | trtllm:36045 | 0 / 36045 | 0 | — | — | — |
| gb200 | `moe_torch_flow_nongated` | shared | trtllm | trtllm:22923 | 0 / 22923 | 0 | — | — | — |
| gb200 | `sglang_flashinfer_trtllm_moe` | shared | sglang | sglang:156213 | 0 / 121659 | 0 | — | — | — |
| gb200 | `sglang_fused_moe_triton` | shared | sglang | sglang:50625 | 0 / 36936 | 0 | — | — | — |
| gb200 | `sglang_mxfp4_flashinfer_trtllm_moe` | shared | sglang | sglang:7533 | 0 / 5184 | 0 | — | — | — |
| gb200 | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| gb200 | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| gb200 | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmodular` | shared | vllm | vllm:2754 | 0 / 2754 | 0 | — | — | — |
| gb200 | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic` | shared | vllm | vllm:15075 | 0 / 15075 | 0 | — | — | — |
| gb200 | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| gb200 | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmodular` | shared | vllm | vllm:3807 | 0 / 3807 | 0 | — | — | — |
| gb200 | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:12474 | 0 / 12474 | 0 | — | — | — |
| gb200 | `vllm_gptossmxfp4moe_flashinfer_trtllm_mxfp4_bf16_trtllmmxfp4expertsmonolithic` | shared | vllm | vllm:1944 | 0 / 1944 | 0 | — | — | — |
| gb200 | `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| gb200 | `vllm_modeloptfp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:11138 | 0 / 11138 | 0 | — | — | — |
| gb200 | `vllm_mxfp4moe_flashinfer_trtllm_mxfp4_mxfp8_trtllmmxfp4expertsmodular` | shared | vllm | vllm:2187 | 0 / 2187 | 0 | — | — | — |
| gb200 | `vllm_unquantizedfusedmoe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:1944 | 0 / 1944 | 0 | — | — | — |
| gb200 | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16experts` | shared | vllm | vllm:12320 | 0 / 12320 | 0 | — | — | — |
| gb200 | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16expertsmonolithic` | shared | vllm | vllm:1026 | 0 / 1026 | 0 | — | — | — |
| gb300 | `deepgemm` | shared | trtllm | trtllm:38799 | 0 / 38799 | 0 | — | — | — |
| gb300 | `moe_torch_flow` | shared | trtllm | trtllm:12474 | 0 / 7371 | 6 | — | — | — |
| gb300 | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:114048 | 0 / 114048 | 0 | — | — | — |
| gb300 | `moe_torch_flow_nongated` | shared | trtllm | trtllm:22923 | 0 / 22923 | 0 | — | — | — |
| gb300 | `sglang_flashinfer_trtllm_moe` | shared | sglang | sglang:159345 | 0 / 121659 | 0 | — | — | — |
| gb300 | `sglang_fused_moe_triton` | shared | sglang | sglang:50627 | 0 / 36936 | 0 | — | — | — |
| gb300 | `sglang_mxfp4_flashinfer_trtllm_moe` | shared | sglang | sglang:7533 | 0 / 5184 | 0 | — | — | — |
| gb300 | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| gb300 | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| gb300 | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmodular` | shared | vllm | vllm:648 | 0 / 648 | 0 | — | — | — |
| gb300 | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic` | shared | vllm | vllm:13996 | 0 / 13996 | 0 | — | — | — |
| gb300 | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| gb300 | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmodular` | shared | vllm | vllm:3807 | 0 / 3807 | 0 | — | — | — |
| gb300 | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:12474 | 0 / 12474 | 0 | — | — | — |
| gb300 | `vllm_gptossmxfp4moe_flashinfer_trtllm_mxfp4_bf16_trtllmmxfp4expertsmonolithic` | shared | vllm | vllm:1944 | 0 / 1944 | 0 | — | — | — |
| gb300 | `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| gb300 | `vllm_modeloptfp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:11295 | 0 / 11295 | 0 | — | — | — |
| gb300 | `vllm_mxfp4moe_flashinfer_trtllm_mxfp4_mxfp8_trtllmmxfp4expertsmodular` | shared | vllm | vllm:2187 | 0 / 2187 | 0 | — | — | — |
| gb300 | `vllm_unquantizedfusedmoe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:1944 | 0 / 1944 | 0 | — | — | — |
| gb300 | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16experts` | shared | vllm | vllm:12267 | 0 / 12267 | 0 | — | — | — |
| gb300 | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16expertsmonolithic` | shared | vllm | vllm:1026 | 0 / 1026 | 0 | — | — | — |
| h100_sxm | `moe_torch_flow` | shared | trtllm | trtllm:1539 | 0 / 1053 | 1 | — | — | — |
| h100_sxm | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:185328 | 0 / 129357 | 16 | — | — | — |
| h100_sxm | `moe_torch_flow_nongated` | shared | trtllm | trtllm:32643 | 0 / 24462 | 9 | — | — | — |
| h100_sxm | `sglang_flashinfer_cutlass_moe` | shared | sglang | sglang:8586 | 0 / 8586 | 0 | — | — | — |
| h100_sxm | `sglang_fused_moe_triton` | shared | sglang | sglang:157464 | 0 / 112509 | 1 | — | — | — |
| h100_sxm | `sglang_marlin_moe` | shared | sglang | sglang:22113 | 0 / 17415 | 0 | — | — | — |
| h100_sxm | `sglang_marlin_moe_situ_as_silu` | shared | sglang | sglang:541 | 0 / 541 | 0 | — | — | — |
| h100_sxm | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| h100_sxm | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| h100_sxm | `vllm_fp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:9234 | 0 / 9234 | 0 | — | — | — |
| h100_sxm | `vllm_fp8moe_triton_tritonexperts` | shared | vllm | vllm:5988 | 0 / 5988 | 0 | — | — | — |
| h100_sxm | `vllm_gptossmxfp4moe_triton_oaitritonmxfp4expertsmonolithic` | shared | vllm | vllm:1944 | 0 / 1944 | 0 | — | — | — |
| h100_sxm | `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:14418 | 0 / 14418 | 0 | — | — | — |
| h100_sxm | `vllm_unquantizedfusedmoe_triton_tritonexperts` | shared | vllm | vllm:16253 | 0 / 16253 | 0 | — | — | — |
| h200_sxm | `moe_torch_flow` | shared | trtllm | trtllm:2268 | 0 / 1134 | 1 | — | — | — |
| h200_sxm | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:176580 | 0 / 129357 | 12 | — | — | — |
| h200_sxm | `moe_torch_flow_nongated` | shared | trtllm | trtllm:32724 | 0 / 24543 | 14 | — | — | — |
| h200_sxm | `sglang_flashinfer_cutlass_moe` | shared | sglang | sglang:8667 | 0 / 8667 | 0 | — | — | — |
| h200_sxm | `sglang_fused_moe_triton` | shared | sglang | sglang:157464 | 0 / 112509 | 1 | — | — | — |
| h200_sxm | `sglang_marlin_moe` | shared | sglang | sglang:22113 | 0 / 17415 | 0 | — | — | — |
| h200_sxm | `sglang_marlin_moe_situ_as_silu` | shared | sglang | sglang:488 | 0 / 488 | 0 | — | — | — |
| h200_sxm | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| h200_sxm | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| h200_sxm | `vllm_fp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:9234 | 0 / 9234 | 0 | — | — | — |
| h200_sxm | `vllm_fp8moe_triton_tritonexperts` | shared | vllm | vllm:5994 | 0 / 5994 | 0 | — | — | — |
| h200_sxm | `vllm_fused_moe` | shared | vllm | vllm:36288 | 0 / 33210 | 0 | — | — | — |
| h200_sxm | `vllm_gptossmxfp4moe_triton_oaitritonmxfp4expertsmonolithic` | shared | vllm | vllm:1944 | 0 / 1944 | 0 | — | — | — |
| h200_sxm | `vllm_marlin_int4_moe` | shared | vllm | vllm:10206 | 0 / 9234 | 0 | — | — | — |
| h200_sxm | `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:14418 | 0 / 14418 | 0 | — | — | — |
| h200_sxm | `vllm_mxfp4_moe` | shared | vllm | vllm:1701 | 0 / 1296 | 0 | — | — | — |
| h200_sxm | `vllm_unquantizedfusedmoe_triton_tritonexperts` | shared | vllm | vllm:16281 | 0 / 16281 | 0 | — | — | — |
| l40s | `moe_torch_flow` | shared | trtllm | trtllm:1960 | 0 / 1960 | 0 | — | — | — |
| l40s | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:77077 | 0 / 72092 | 7 | — | — | — |
| l40s | `moe_torch_flow_nongated` | shared | trtllm | trtllm:14372 | 0 / 14372 | 0 | — | — | — |
| l40s | `sglang_fused_moe_triton` | shared | sglang | sglang:54402 | 0 / 54402 | 0 | — | — | — |
| l40s | `sglang_fused_moe_triton_situ_as_silu` | shared | sglang | sglang:2997 | 0 / 2997 | 0 | — | — | — |
| l40s | `sglang_marlin_moe` | shared | sglang | sglang:3078 | 0 / 3078 | 0 | — | — | — |
| l40s | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:1026 | 0 / 1026 | 0 | — | — | — |
| l40s | `vllm_fp8moe_triton_tritonexperts` | shared | vllm | vllm:15118 | 0 / 15118 | 0 | — | — | — |
| l40s | `vllm_gptossmxfp4moe_marlin_marlinexperts` | shared | vllm | vllm:1944 | 0 / 1944 | 0 | — | — | — |
| l40s | `vllm_modeloptfp8moe_triton_tritonexperts` | shared | vllm | vllm:14174 | 0 / 14174 | 0 | — | — | — |
| l40s | `vllm_unquantizedfusedmoe_triton_tritonexperts` | shared | vllm | vllm:16055 | 0 / 16055 | 0 | — | — | — |
| rtx_pro_6000_server | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:119394 | 0 / 119394 | 0 | — | — | — |
| rtx_pro_6000_server | `moe_torch_flow_nongated` | shared | trtllm | trtllm:22921 | 0 / 22921 | 0 | — | — | — |
| rtx_pro_6000_server | `moe_torch_flow_triton_fp8_block` | shared | trtllm | trtllm:32238 | 0 / 32238 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_flashinfer_cutlass_moe` | shared | sglang | sglang:35478 | 0 / 35478 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_fused_moe_triton` | shared | sglang | sglang:77355 | 0 / 77355 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_marlin_moe` | shared | sglang | sglang:4456 | 0 / 4456 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_marlin_moe_situ_as_silu` | shared | sglang | sglang:551 | 0 / 551 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:15822 | 0 / 15822 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_gptossmxfp4moe_marlin_marlinexperts` | shared | vllm | vllm:1944 | 0 / 1944 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:13284 | 0 / 13284 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_unquantizedfusedmoe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:15903 | 0 / 15903 | 0 | — | — | — |

## `msa_context_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `MiniMaxM3SparseMSAImpl` | shared | vllm | vllm:14110 | 0 / 14110 | 0 | — | — | — |
| b200_sxm | `msa_fmha_sm100` | shared | trtllm | trtllm:5070 | 0 / 5070 | 0 | — | — | — |
| b200_sxm | `sglang_minimax_prefill_triton_sparse` | shared | sglang | sglang:26933 | 0 / 26933 | 0 | — | — | — |
| b300_sxm | `MiniMaxM3SparseMSAImpl` | shared | vllm | vllm:14137 | 0 / 14137 | 0 | — | — | — |
| b300_sxm | `msa_fmha_sm100` | shared | trtllm | trtllm:5070 | 0 / 5070 | 0 | — | — | — |
| b300_sxm | `sglang_minimax_prefill_triton_sparse` | shared | sglang | sglang:26850 | 0 / 26850 | 0 | — | — | — |
| gb200 | `MiniMaxM3SparseMSAImpl` | shared | vllm | vllm:11712 | 0 / 11712 | 0 | — | — | — |
| gb200 | `msa_fmha_sm100` | shared | trtllm | trtllm:5070 | 0 / 5070 | 0 | — | — | — |
| gb200 | `sglang_minimax_prefill_triton_sparse` | shared | sglang | sglang:26608 | 0 / 26608 | 0 | — | — | — |
| gb300 | `MiniMaxM3SparseMSAImpl` | shared | vllm | vllm:11712 | 0 / 11712 | 0 | — | — | — |
| gb300 | `msa_fmha_sm100` | shared | trtllm | trtllm:5070 | 0 / 5070 | 0 | — | — | — |
| gb300 | `sglang_minimax_prefill_triton_sparse` | shared | sglang | sglang:27242 | 0 / 27242 | 0 | — | — | — |
| h100_sxm | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:7808 | 0 / 7808 | 0 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:3382 | 0 / 3382 | 0 | — | — | — |
| h100_sxm | `sglang_minimax_prefill_triton_sparse` | shared | sglang | sglang:26348 | 0 / 26348 | 0 | — | — | — |
| h200_sxm | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:7808 | 0 / 7808 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:3388 | 0 / 3388 | 0 | — | — | — |
| h200_sxm | `sglang_minimax_prefill_triton_sparse` | shared | sglang | sglang:26845 | 0 / 26845 | 0 | — | — | — |
| l40s | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:7808 | 0 / 7808 | 0 | — | — | — |
| l40s | `default` | shared_fallback | trtllm | trtllm:3313 | 0 / 3313 | 0 | — | — | — |
| rtx_pro_6000_server | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:7808 | 0 / 7808 | 0 | — | — | — |

## `msa_generation_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `msa_fmha_sm100` | shared | trtllm | trtllm:3864 | 0 / 3864 | 0 | — | — | — |
| b200_sxm | `sglang_minimax_decode_triton_sparse_topk_radix` | shared | sglang | sglang:119 | 0 / 119 | 0 | — | — | — |
| b200_sxm | `sglang_minimax_decode_triton_sparse_topk_split` | shared | sglang | sglang:1358 | 0 / 1358 | 0 | — | — | — |
| b300_sxm | `msa_fmha_sm100` | shared | trtllm | trtllm:3864 | 0 / 3864 | 0 | — | — | — |
| b300_sxm | `sglang_minimax_decode_triton_sparse_topk_radix` | shared | sglang | sglang:120 | 0 / 120 | 0 | — | — | — |
| b300_sxm | `sglang_minimax_decode_triton_sparse_topk_split` | shared | sglang | sglang:1360 | 0 / 1360 | 0 | — | — | — |
| gb200 | `MiniMaxM3SparseMSAImpl` | shared | vllm | vllm:8832 | 0 / 8832 | 0 | — | — | — |
| gb200 | `msa_fmha_sm100` | shared | trtllm | trtllm:3864 | 0 / 3864 | 0 | — | — | — |
| gb200 | `sglang_minimax_decode_triton_sparse_topk_radix` | shared | sglang | sglang:119 | 0 / 119 | 0 | — | — | — |
| gb200 | `sglang_minimax_decode_triton_sparse_topk_split` | shared | sglang | sglang:1358 | 0 / 1358 | 0 | — | — | — |
| gb300 | `MiniMaxM3SparseMSAImpl` | shared | vllm | vllm:8832 | 0 / 8832 | 0 | — | — | — |
| gb300 | `msa_fmha_sm100` | shared | trtllm | trtllm:3864 | 0 / 3864 | 0 | — | — | — |
| gb300 | `sglang_minimax_decode_triton_sparse_topk_radix` | shared | sglang | sglang:120 | 0 / 120 | 0 | — | — | — |
| gb300 | `sglang_minimax_decode_triton_sparse_topk_split` | shared | sglang | sglang:1360 | 0 / 1360 | 0 | — | — | — |
| h100_sxm | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:5870 | 0 / 5870 | 0 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:2548 | 0 / 2548 | 0 | — | — | — |
| h100_sxm | `sglang_minimax_decode_triton_sparse_topk_radix` | shared | sglang | sglang:240 | 0 / 240 | 0 | — | — | — |
| h100_sxm | `sglang_minimax_decode_triton_sparse_topk_split` | shared | sglang | sglang:1227 | 0 / 1227 | 0 | — | — | — |
| h200_sxm | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:5882 | 0 / 5882 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:2556 | 0 / 2556 | 0 | — | — | — |
| h200_sxm | `sglang_minimax_decode_triton_sparse_topk_radix` | shared | sglang | sglang:246 | 0 / 246 | 0 | — | — | — |
| h200_sxm | `sglang_minimax_decode_triton_sparse_topk_split` | shared | sglang | sglang:1231 | 0 / 1231 | 0 | — | — | — |
| l40s | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:5850 | 0 / 5850 | 0 | — | — | — |
| l40s | `default` | shared_fallback | trtllm | trtllm:2508 | 0 / 2508 | 0 | — | — | — |
| rtx_pro_6000_server | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:5882 | 0 / 5882 | 0 | — | — | — |

## `nccl_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b300_sxm | `NCCL` | shared | trtllm | trtllm:1512 | 0 / 126 | 0 | — | — | — |

## `scale_matrix_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `sglang` | shared | sglang | sglang:1628 | 0 / 1628 | 0 | — | — | — |
| b200_sxm | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| b300_sxm | `sglang` | shared | sglang | sglang:1628 | 0 / 1628 | 0 | — | — | — |
| b300_sxm | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| gb200 | `sglang` | shared | sglang | sglang:1628 | 0 / 1628 | 0 | — | — | — |
| gb200 | `static_scaled_fp8_quant` | shared | vllm | vllm:1628 | 0 / 1628 | 0 | — | — | — |
| gb200 | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| gb300 | `sglang` | shared | sglang | sglang:1628 | 0 / 1628 | 0 | — | — | — |
| gb300 | `static_scaled_fp8_quant` | shared | vllm | vllm:1628 | 0 / 1628 | 0 | — | — | — |
| gb300 | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| h100_sxm | `sglang` | shared | sglang | sglang:1628 | 0 / 1628 | 0 | — | — | — |
| h100_sxm | `static_scaled_fp8_quant` | shared | vllm | vllm:1628 | 0 / 1628 | 0 | — | — | — |
| h100_sxm | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| h200_sxm | `sglang` | shared | sglang | sglang:1628 | 0 / 1628 | 0 | — | — | — |
| h200_sxm | `static_scaled_fp8_quant` | shared | vllm | vllm:1628 | 0 / 1628 | 0 | — | — | — |
| h200_sxm | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| l40s | `sglang` | shared | sglang | sglang:1628 | 0 / 1628 | 0 | — | — | — |
| l40s | `static_scaled_fp8_quant` | shared | vllm | vllm:1628 | 0 / 1628 | 0 | — | — | — |
| l40s | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang` | shared | sglang | sglang:1628 | 0 / 1628 | 0 | — | — | — |
| rtx_pro_6000_server | `static_scaled_fp8_quant` | shared | vllm | vllm:1628 | 0 / 1628 | 0 | — | — | — |
| rtx_pro_6000_server | `torch_ops` | shared | trtllm | trtllm:1628 | 0 / 1628 | 0 | — | — | — |

## `trtllm_alltoall_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| gb200 | `NVLinkOneSided` | shared | trtllm | trtllm:296 | 0 / 148 | 0 | — | — | — |
| gb200 | `NVLinkTwoSided` | shared | trtllm | trtllm:1800 | 0 / 540 | 0 | — | — | — |

## `wideep_context_mla_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `trtllm_mla` | shared | sglang | sglang:500 | 0 / 500 | 0 | — | — | — |
| b300_sxm | `trtllm_mla` | shared | sglang | sglang:1125 | 0 / 625 | 0 | — | — | — |
| gb200 | `trtllm_mla` | shared | sglang | sglang:625 | 0 / 625 | 0 | — | — | — |
| gb300 | `trtllm_mla` | shared | sglang | sglang:625 | 0 / 625 | 0 | — | — | — |
| h100_sxm | `fa3` | shared | sglang | sglang:1552 | 0 / 1052 | 0 | — | — | — |
| h100_sxm | `flashinfer` | shared | sglang | sglang:1577 | 0 / 1077 | 0 | — | — | — |
| h200_sxm | `fa3` | shared | sglang | sglang:960 | 0 / 960 | 0 | — | — | — |
| h200_sxm | `flashinfer` | shared | sglang | sglang:960 | 0 / 960 | 0 | — | — | — |

## `wideep_context_mlp_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| h100_sxm | `deepseek_v3` | shared | sglang | sglang:18 | 0 / 18 | 0 | — | — | — |
| h200_sxm | `deepseek_v3` | shared | sglang | sglang:18 | 0 / 18 | 0 | — | — | — |

## `wideep_context_moe_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `deepepmoe` | shared | sglang | sglang:426 | 0 / 426 | 0 | — | — | — |
| b300_sxm | `deepepmoe` | shared | sglang | sglang:426 | 0 / 426 | 0 | — | — | — |
| gb200 | `deepepmoe` | shared | sglang | sglang:426 | 0 / 426 | 0 | — | — | — |
| gb300 | `deepepmoe` | shared | sglang | sglang:426 | 0 / 426 | 0 | — | — | — |
| h100_sxm | `deepepmoe` | shared | sglang | sglang:892 | 0 / 461 | 0 | — | — | — |
| h200_sxm | `deepepmoe` | shared | sglang | sglang:954 | 0 / 492 | 0 | — | — | — |

## `wideep_deepep_ll_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `deepep` | shared | sglang | sglang:176 | 0 / 176 | 0 | — | — | — |
| b300_sxm | `deepep` | shared | sglang | sglang:176 | 0 / 176 | 0 | — | — | — |
| gb200 | `deepep` | shared | sglang | sglang:176 | 0 / 176 | 0 | — | — | — |
| gb300 | `deepep` | shared | sglang | sglang:176 | 0 / 176 | 0 | — | — | — |
| h100_sxm | `deepep` | shared | sglang | sglang:271 | 0 / 271 | 0 | — | — | — |
| h200_sxm | `deepep` | shared | sglang | sglang:271 | 0 / 271 | 0 | — | — | — |

## `wideep_deepep_normal_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `deepep` | shared | sglang | sglang:2484 | 0 / 2484 | 0 | — | — | — |
| b300_sxm | `deepep` | shared | sglang | sglang:2484 | 0 / 2484 | 0 | — | — | — |
| gb200 | `deepep` | shared | sglang | sglang:2622 | 0 / 2622 | 0 | — | — | — |
| gb300 | `deepep` | shared | sglang | sglang:2622 | 0 / 2622 | 0 | — | — | — |
| h100_sxm | `deepep` | shared | sglang | sglang:2921 | 0 / 2898 | 0 | — | — | — |
| h200_sxm | `deepep` | shared | sglang | sglang:2921 | 0 / 2898 | 0 | — | — | — |

## `wideep_generation_mla_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `trtllm_mla` | shared | sglang | sglang:528 | 0 / 528 | 0 | — | — | — |
| b300_sxm | `trtllm_mla` | shared | sglang | sglang:1188 | 0 / 660 | 20 | — | — | — |
| gb200 | `trtllm_mla` | shared | sglang | sglang:660 | 0 / 660 | 0 | — | — | — |
| gb300 | `trtllm_mla` | shared | sglang | sglang:660 | 0 / 660 | 0 | — | — | — |
| h100_sxm | `fa3` | shared | sglang | sglang:1680 | 0 / 1152 | 9 | — | — | — |
| h100_sxm | `flashinfer` | shared | sglang | sglang:1732 | 0 / 1204 | 5 | — | — | — |
| h200_sxm | `fa3` | shared | sglang | sglang:1056 | 0 / 1056 | 0 | — | — | — |
| h200_sxm | `flashinfer` | shared | sglang | sglang:1056 | 0 / 1056 | 0 | — | — | — |

## `wideep_generation_mlp_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| h100_sxm | `deepseek_v3` | shared | sglang | sglang:18 | 0 / 18 | 0 | — | — | — |
| h200_sxm | `deepseek_v3` | shared | sglang | sglang:18 | 0 / 18 | 0 | — | — | — |

## `wideep_generation_moe_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `deepepmoe` | shared | sglang | sglang:336 | 0 / 336 | 0 | — | — | — |
| b300_sxm | `deepepmoe` | shared | sglang | sglang:336 | 0 / 336 | 0 | — | — | — |
| gb200 | `deepepmoe` | shared | sglang | sglang:336 | 0 / 336 | 0 | — | — | — |
| gb300 | `deepepmoe` | shared | sglang | sglang:336 | 0 / 336 | 0 | — | — | — |
| h100_sxm | `deepepmoe` | shared | sglang | sglang:719 | 0 / 384 | 0 | — | — | — |
| h200_sxm | `deepepmoe` | shared | sglang | sglang:720 | 0 / 384 | 0 | — | — | — |

## `wideep_moe_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `wideep_compute_cutlass` | shared | trtllm | trtllm:4158 | 0 / 4158 | 0 | — | — | — |
| b300_sxm | `wideep_compute_cutlass` | shared | trtllm | trtllm:4158 | 0 / 4158 | 0 | — | — | — |
| gb200 | `wideep_compute_cutlass` | shared | trtllm | trtllm:4158 | 0 / 4158 | 0 | — | — | — |
| gb300 | `wideep_compute_cutlass` | shared | trtllm | trtllm:4158 | 0 / 4158 | 0 | — | — | — |
| h100_sxm | `wideep_compute_cutlass` | shared | trtllm | trtllm:4158 | 0 / 4158 | 0 | — | — | — |
| h200_sxm | `wideep_compute_cutlass` | shared | trtllm | trtllm:4158 | 0 / 4158 | 0 | — | — | — |
| rtx_pro_6000_server | `wideep_compute_cutlass` | shared | trtllm | trtllm:1350 | 0 / 1273 | 0 | — | — | — |

## Appendix: all kernel sources

Each row is one distinct `kernel_source` value seen in the corpus, with the union of frameworks, op files, and systems it appears in. Tier is determined by the kernel_source name alone, so a single kernel_source has one tier across the whole corpus.


### `shared` (165 kernel sources)

| kernel_source | frameworks | op files | systems | rows |
|---|---|---|---|---|
| `causal_conv1d_fn` | sglang, trtllm, vllm | gdn_perf.parquet, mamba2_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 82,468 |
| `causal_conv1d_fn_qkv3` | sglang, vllm | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 7,012 |
| `causal_conv1d_update` | sglang, trtllm, vllm | gdn_perf.parquet, kda_perf.parquet, mamba2_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 10,440 |
| `chunk_gated_delta_rule` | sglang, trtllm | gdn_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 40,725 |
| `chunk_gated_delta_rule_flashinfer` | vllm | gdn_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 28,221 |
| `chunk_gated_delta_rule_triton` | vllm | gdn_perf.parquet | l40s, rtx_pro_6000_server | 7,870 |
| `chunk_kda` | sglang | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 3,168 |
| `chunk_kda_with_fused_gate` | vllm | kda_perf.parquet | l40s | 414 |
| `compressed_flashmla` | sglang | dsv4_csa_context_module_perf.parquet, dsv4_csa_generation_module_perf.parquet, dsv4_hca_context_module_perf.parquet, dsv4_hca_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 748,644 |
| `CutlassFp8BlockScaledMMKernel` | vllm | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 49,358 |
| `CutlassFP8ScaledMMLinearKernel` | vllm | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 433,024 |
| `deep_gemm.fp8_mqa_logits` | sglang | glm5_mqa_logits_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 19,378 |
| `deep_gemm.fp8_paged_mqa_logits` | sglang | dsv4_paged_mqa_logits_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 19,276 |
| `deepep` | sglang, trtllm, vllm | moe_a2a_perf.parquet, wideep_deepep_ll_perf.parquet, wideep_deepep_normal_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 22,864 |
| `deepepmoe` | sglang | wideep_context_moe_perf.parquet, wideep_generation_moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 6,333 |
| `deepgemm` | trtllm | gemm_perf.parquet, moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 272,268 |
| `deepgemm_megamoe` | sglang | dsv4_megamoe_module_perf.parquet | b200_sxm, gb200, gb300 | 776 |
| `DeepGemmFp8BlockScaledMMKernel` | vllm | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 295,928 |
| `deepseek_v3` | sglang | wideep_context_mlp_perf.parquet, wideep_generation_mlp_perf.parquet | h100_sxm, h200_sxm | 72 |
| `DeepseekV4TrtllmAttention` | trtllm | dsv4_csa_context_module_perf.parquet, dsv4_csa_generation_module_perf.parquet, dsv4_hca_context_module_perf.parquet, dsv4_hca_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 57,096 |
| `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | vllm | computescale_perf.parquet | gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 9,693 |
| `fa3` | sglang | context_attention_perf.parquet, generation_attention_perf.parquet, wideep_context_mla_perf.parquet, wideep_generation_mla_perf.parquet | h100_sxm, h200_sxm | 145,130 |
| `fast_topk_transform_fused` | sglang | glm5_topk_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 24,092 |
| `flash_attention` | sglang | context_attention_perf.parquet, context_mla_perf.parquet, generation_attention_perf.parquet, generation_mla_perf.parquet | a100_sxm, h100_sxm, h200_sxm | 24,688 |
| `flash_attention_v3` | sglang | encoder_attention_perf.parquet | h100_sxm, h200_sxm | 15,358 |
| `flash_attention_v4` | sglang | encoder_attention_perf.parquet | b200_sxm, gb200 | 15,358 |
| `FLASH_ATTN` | vllm | mla_context_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 50,379 |
| `FLASH_ATTN_MLA` | vllm | mla_context_module_perf.parquet, mla_generation_module_perf.parquet | h100_sxm, h200_sxm | 7,856 |
| `flash_mla_sparse_fwd` | sglang | glm5_dsa_attn_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 19,841 |
| `flashinfer` | sglang | context_attention_perf.parquet, generation_attention_perf.parquet, wideep_context_mla_perf.parquet, wideep_generation_mla_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 133,439 |
| `flashinfer_gated_delta_rule_decode` | sglang | gdn_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 208 |
| `FLASHINFER_MLA` | vllm | mla_context_module_perf.parquet, mla_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 67,066 |
| `FLASHINFER_MLA_SPARSE` | vllm | dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet | gb200, gb300 | 35,951 |
| `FlashInferCuteDslNvFp4LinearKernel` | vllm | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 290,080 |
| `FlashInferCutlassNvFp4LinearKernel` | vllm | gemm_perf.parquet | rtx_pro_6000_server | 35,742 |
| `FlashInferFp8BlockScaledMMKernel` | vllm | gemm_perf.parquet | h100_sxm, h200_sxm | 14,241 |
| `flashkda_fwd` | vllm | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 3,415 |
| `FLASHMLA` | vllm | mla_context_module_perf.parquet, mla_generation_module_perf.parquet | h100_sxm, h200_sxm | 7,292 |
| `FLASHMLA_SPARSE` | vllm | dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet | gb200, gb300, h100_sxm, h200_sxm | 68,345 |
| `FLASHMLA_SPARSE_DSV4` | vllm | dsv4_csa_context_module_perf.parquet, dsv4_csa_generation_module_perf.parquet, dsv4_hca_attn_module_perf.parquet, dsv4_hca_context_module_perf.parquet, dsv4_hca_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 103,537 |
| `fused_kda_decode` | vllm | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 308 |
| `fused_kda_decode_mtp_dspark` | sglang | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 428 |
| `fused_recurrent_gated_delta_rule` | sglang, trtllm | gdn_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 774 |
| `fused_recurrent_gated_delta_rule_packed_decode` | sglang, vllm | gdn_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 7,000 |
| `fused_recurrent_kda` | vllm | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 972 |
| `fused_recurrent_kda_packed_decode` | sglang, vllm | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 651 |
| `fused_sigmoid_gating_delta_rule_update` | sglang | kda_perf.parquet | h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 432 |
| `kda_fused_decode` | sglang | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 77 |
| `MiniMaxM3SparseMSAImpl` | vllm | msa_context_module_perf.parquet, msa_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 69,335 |
| `MiniMaxM3SparseTritonImpl` | vllm | msa_context_module_perf.parquet, msa_generation_module_perf.parquet | h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 54,716 |
| `moe_torch_flow` | trtllm | moe_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s | 62,657 |
| `moe_torch_flow_cutlass` | trtllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,014,571 |
| `moe_torch_flow_min_latency` | trtllm | moe_perf.parquet | b200_sxm, gb200 | 72,090 |
| `moe_torch_flow_nongated` | trtllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 194,352 |
| `moe_torch_flow_triton_fp8_block` | trtllm | moe_perf.parquet | rtx_pro_6000_server | 32,238 |
| `msa_fmha_sm100` | trtllm | msa_context_module_perf.parquet, msa_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 35,736 |
| `NCCL` | trtllm | nccl_perf.parquet | b300_sxm | 1,512 |
| `NVLinkOneSided` | trtllm | trtllm_alltoall_perf.parquet | gb200 | 296 |
| `NVLinkTwoSided` | trtllm | trtllm_alltoall_perf.parquet | gb200 | 1,800 |
| `sglang` | sglang | computescale_perf.parquet, gemm_perf.parquet, scale_matrix_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 84,087 |
| `SGLang_CustomAllReduce_eager` | sglang | custom_allreduce_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,725 |
| `SGLang_CustomAllReduce_graph` | sglang | custom_allreduce_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,725 |
| `sglang_deepgemm_gemm_nt_f8f8bf16` | sglang | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 295,260 |
| `sglang_dsa_dense_mha_fa3` | sglang | dsa_context_module_perf.parquet | h100_sxm, h200_sxm | 82,901 |
| `sglang_dsa_dense_mha_trtllm_ragged` | sglang | dsa_context_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 137,411 |
| `sglang_dsa_indexer_fa3` | sglang | dsa_generation_module_perf.parquet | h100_sxm, h200_sxm | 5,352 |
| `sglang_dsa_indexer_flashmla_kv` | sglang | dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet | h100_sxm, h200_sxm | 46,332 |
| `sglang_dsa_indexer_flashmla_sparse` | sglang | dsa_context_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 123,166 |
| `sglang_dsa_indexer_trtllm` | sglang | dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 91,197 |
| `sglang_dsa_skip_indexer_fa3` | sglang | dsa_generation_module_perf.parquet | h100_sxm, h200_sxm | 2,448 |
| `sglang_dsa_skip_indexer_flashmla_kv` | sglang | dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet | h100_sxm, h200_sxm | 24,610 |
| `sglang_dsa_skip_indexer_flashmla_sparse` | sglang | dsa_context_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 54,618 |
| `sglang_dsa_skip_indexer_trtllm` | sglang | dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 39,806 |
| `sglang_flashinfer_cutedsl_nvfp4` | sglang | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 236,208 |
| `sglang_flashinfer_cutlass_moe` | sglang | moe_perf.parquet | h100_sxm, h200_sxm, rtx_pro_6000_server | 52,731 |
| `sglang_flashinfer_cutlass_nvfp4` | sglang | gemm_perf.parquet | rtx_pro_6000_server | 29,526 |
| `sglang_flashinfer_trtllm_moe` | sglang | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 635,827 |
| `sglang_fused_moe_triton` | sglang | moe_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 682,986 |
| `sglang_fused_moe_triton_situ_as_silu` | sglang | moe_perf.parquet | l40s | 2,997 |
| `sglang_marlin_moe` | sglang | moe_perf.parquet | a100_sxm, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 70,795 |
| `sglang_marlin_moe_situ_as_silu` | sglang | moe_perf.parquet | h100_sxm, h200_sxm, rtx_pro_6000_server | 1,580 |
| `sglang_minimax_decode_triton_sparse_topk_radix` | sglang | msa_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 964 |
| `sglang_minimax_decode_triton_sparse_topk_split` | sglang | msa_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 7,894 |
| `sglang_minimax_prefill_triton_sparse` | sglang | msa_context_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 160,826 |
| `sglang_mxfp4_flashinfer_trtllm_moe` | sglang | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 29,970 |
| `sglang_sgl_kernel_bmm_fp8` | sglang | mla_bmm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 5,724 |
| `sglang_sgl_kernel_fp8_scaled_mm` | sglang | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 429,348 |
| `sglang_tilelang_mhc_post` | sglang | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 485 |
| `sglang_tilelang_mhc_pre` | sglang | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 420 |
| `sglang_torch_bmm` | sglang | mla_bmm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 5,724 |
| `sglang_torch_linear` | sglang | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 429,718 |
| `sglang_torch_mhc_pre` | sglang | mhc_module_perf.parquet | rtx_pro_6000_server | 67 |
| `static_scaled_fp8_quant` | vllm | scale_matrix_perf.parquet | gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 9,768 |
| `topk_transform_v1` | sglang | dsv4_csa_topk_calib_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 25,764 |
| `topk_transform_v2` | sglang | dsv4_csa_topk_calib_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 1,464 |
| `torch.nn.functional.linear` | vllm | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 432,978 |
| `torch_flow` | trtllm | context_attention_perf.parquet, encoder_attention_perf.parquet, gemm_perf.parquet, generation_attention_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,891,515 |
| `torch_flow_flashinfer` | trtllm | context_attention_perf.parquet, generation_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 23,408 |
| `torch_ops` | trtllm | computescale_perf.parquet, scale_matrix_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 26,012 |
| `triton` | sglang | context_attention_perf.parquet, context_mla_perf.parquet, encoder_attention_perf.parquet, generation_attention_perf.parquet, generation_mla_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 173,342 |
| `TRITON_MLA` | vllm | mla_context_module_perf.parquet, mla_generation_module_perf.parquet | l40s, rtx_pro_6000_server | 5,952 |
| `TritonFp8BlockScaledMMKernel` | vllm | gemm_perf.parquet | l40s | 35,676 |
| `trt_flow_/smooth_quant_gemm_L96/PLUGIN_V2_SmoothQuantGemm_0` | trtllm | gemm_perf.parquet | a100_sxm | 6,048 |
| `trt_flow_/weight_only_quant_matmul_L257/PLUGIN_V2_WeightOnlyQuantMatmul_0` | trtllm | gemm_perf.parquet | a100_sxm | 12,096 |
| `TRTLLM` | trtllm | custom_allreduce_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,518 |
| `trtllm_bmm_out` | trtllm | mla_bmm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 3,392 |
| `trtllm_bmm_out_dequant_bf16` | trtllm | mla_bmm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 1,696 |
| `trtllm_fp8_block_scaling_bmm_out` | trtllm | mla_bmm_perf.parquet | h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,696 |
| `trtllm_mha` | sglang | context_attention_perf.parquet, generation_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 273,765 |
| `trtllm_mhc_post_mapping` | trtllm | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 555 |
| `trtllm_mhc_pre_dg_nosplit` | trtllm | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 159 |
| `trtllm_mhc_pre_dg_splitk` | trtllm | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 156 |
| `trtllm_mhc_pre_fma` | trtllm | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 231 |
| `trtllm_mla` | sglang | context_mla_perf.parquet, generation_mla_perf.parquet, wideep_context_mla_perf.parquet, wideep_generation_mla_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 38,503 |
| `TRTLLM_MNNVL_oneshot` | trtllm | custom_allreduce_perf.parquet | gb300 | 19 |
| `TRTLLM_MNNVL_twoshot` | trtllm | custom_allreduce_perf.parquet | gb300 | 27 |
| `TRTLLM_RAGGED` | vllm | mla_context_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 20,303 |
| `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | vllm | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s | 486 |
| `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | vllm | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s | 489 |
| `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | vllm | dsv4_paged_mqa_logits_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 15,672 |
| `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 5,832 |
| `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutlass_flashinferexperts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, rtx_pro_6000_server | 19,710 |
| `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmodular` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 10,206 |
| `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 58,597 |
| `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 8,397 |
| `vLLM_custom_eager` | vllm | custom_allreduce_perf.parquet | a100_sxm, b200_sxm, b300_sxm, b60, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,495 |
| `vLLM_custom_graph` | vllm | custom_allreduce_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,288 |
| `vllm_default` | vllm | gemm_perf.parquet | a100_sxm, b60 | 49,770 |
| `vllm_flash_attn` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | a100_sxm, b60 | 102,215 |
| `vllm_flash_attn_fa2` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | l40s, rtx_pro_6000_server | 111,574 |
| `vllm_flash_attn_fa3` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | h100_sxm, h200_sxm | 225,522 |
| `vllm_flash_attn_fa4` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | h100_sxm, h200_sxm | 2,756 |
| `vllm_flashinfer` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | b200_sxm | 71,600 |
| `vllm_flashinfer_fidecode` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | l40s, rtx_pro_6000_server | 58,484 |
| `vllm_flashinfer_fiprefill` | vllm | context_attention_perf.parquet | l40s, rtx_pro_6000_server | 43,367 |
| `vllm_flashinfer_flashinfertrtllmapidecode` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | b200_sxm, b300_sxm | 134,160 |
| `vllm_flashinfer_trtllm_moe_fp4` | vllm | moe_perf.parquet | b200_sxm | 10,287 |
| `vllm_flashinfer_trtllmdecode` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | gb200, gb300 | 127,168 |
| `vllm_flashinfer_trtllmprefill` | vllm | context_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 185,320 |
| `vllm_fp8moe_flashinfer_cutlass_flashinferexperts` | vllm | moe_perf.parquet | h100_sxm, h200_sxm | 18,468 |
| `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmodular` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 16,362 |
| `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 52,002 |
| `vllm_fp8moe_triton_tritonexperts` | vllm | moe_perf.parquet | h100_sxm, h200_sxm, l40s | 27,100 |
| `vllm_fused_moe` | vllm | moe_perf.parquet | a100_sxm, b200_sxm, h200_sxm | 126,055 |
| `vllm_gptossmxfp4moe_flashinfer_trtllm_mxfp4_bf16_trtllmmxfp4expertsmonolithic` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 7,776 |
| `vllm_gptossmxfp4moe_marlin_marlinexperts` | vllm | moe_perf.parquet | l40s, rtx_pro_6000_server | 3,888 |
| `vllm_gptossmxfp4moe_triton_oaitritonmxfp4expertsmonolithic` | vllm | moe_perf.parquet | h100_sxm, h200_sxm | 3,888 |
| `vllm_marlin_int4_moe` | vllm | moe_perf.parquet | b200_sxm, h200_sxm | 20,415 |
| `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 47,790 |
| `vllm_modeloptfp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 47,257 |
| `vllm_modeloptfp8moe_triton_tritonexperts` | vllm | moe_perf.parquet | l40s | 14,174 |
| `vllm_mxfp4_moe` | vllm | moe_perf.parquet | b200_sxm, h200_sxm | 2,839 |
| `vllm_mxfp4moe_flashinfer_trtllm_mxfp4_mxfp8_trtllmmxfp4expertsmodular` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 8,748 |
| `vllm_torch_bmm` | vllm | mla_bmm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 4,252 |
| `vllm_triton_attn` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, l40s, rtx_pro_6000_server | 27,020 |
| `vllm_unquantizedfusedmoe_flashinfer_cutlass_flashinferexperts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, rtx_pro_6000_server | 23,355 |
| `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16experts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 51,387 |
| `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16expertsmonolithic` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 4,158 |
| `vllm_unquantizedfusedmoe_triton_tritonexperts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, h100_sxm, h200_sxm, l40s | 50,695 |
| `vllm_vit_flash_attn_fa2` | vllm | encoder_attention_perf.parquet | l40s, rtx_pro_6000_server | 15,358 |
| `vllm_vit_flash_attn_fa3` | vllm | encoder_attention_perf.parquet | h100_sxm, h200_sxm | 15,358 |
| `vllm_vit_flash_attn_fa4` | vllm | encoder_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 30,716 |
| `vllm_xpu_moe` | vllm | moe_perf.parquet | b60 | 1,836 |
| `vllm_xpu_moe_mxfp4` | vllm | moe_perf.parquet | b60 | 2,592 |
| `wideep_compute_cutlass` | trtllm | wideep_moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 26,298 |

### `shared_fallback` (1 kernel sources)

| kernel_source | frameworks | op files | systems | rows |
|---|---|---|---|---|
| `default` | sglang, trtllm | context_mla_perf.parquet, dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet, generation_mla_perf.parquet, mla_bmm_perf.parquet, mla_context_module_perf.parquet, mla_generation_module_perf.parquet, msa_context_module_perf.parquet, msa_generation_module_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 303,641 |
