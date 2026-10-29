# Shareability report

Classifies every `(system, op_file, kernel_source)` triple in the perf database into one of two tiers:

- **`shared`** — named kernel_source. The engine resolver inherits these rows from sibling backend/version directories (cross-version and cross-backend) when shared-layer reuse is enabled.
- **`shared_fallback`** — `kernel_source = default`. Framework-implicit, low-fidelity. Inherited alongside `shared` rows because shared-layer modes already accept coarser fallbacks.

Rows with a blank/`<unknown>` kernel_source are skipped during the scan (the current corpus has none).

## Headline numbers

- Total rows scanned: **31,755,054**
- Within-framework cross-version dedup-able rows: **47,131** (~0.1%)
- Tier distribution (groups / rows):
  - `shared`: 1299 groups · 31,156,118 rows
  - `shared_fallback`: 54 groups · 598,936 rows


## `computescale_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | shared | vllm | vllm:3242 | 0 / 1623 | 1 | — | — | — |
| b200_sxm | `sglang` | shared | sglang | sglang:2 | 0 / 1 | 0 | — | — | — |
| b200_sxm | `torch_ops` | shared | trtllm | trtllm:3256 | 0 / 3256 | 0 | — | — | — |
| b300_sxm | `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | shared | vllm | vllm:3255 | 0 / 1628 | 10 | — | — | — |
| b300_sxm | `sglang` | shared | sglang | sglang:2 | 0 / 1 | 0 | — | — | — |
| b300_sxm | `torch_ops` | shared | trtllm | trtllm:3256 | 0 / 1628 | 0 | — | — | — |
| gb200 | `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | shared | vllm | vllm:3253 | 0 / 1628 | 6 | — | — | — |
| gb200 | `sglang` | shared | sglang | sglang:5 | 0 / 4 | 0 | — | — | — |
| gb200 | `torch_ops` | shared | trtllm | trtllm:3256 | 0 / 1628 | 0 | — | — | — |
| gb300 | `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | shared | vllm | vllm:4880 | 0 / 1628 | 22 | — | — | — |
| gb300 | `sglang` | shared | sglang | sglang:16 | 0 / 10 | 0 | — | — | — |
| gb300 | `torch_ops` | shared | trtllm | trtllm:3256 | 0 / 1628 | 1 | — | — | — |
| h100_sxm | `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | shared | vllm | vllm:3256 | 0 / 1628 | 15 | — | — | — |
| h100_sxm | `sglang` | shared | sglang | sglang:11 | 0 / 8 | 0 | — | — | — |
| h100_sxm | `torch_ops` | shared | trtllm | trtllm:3256 | 0 / 1628 | 0 | — | — | — |
| h200_sxm | `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | shared | vllm | vllm:3256 | 0 / 1628 | 10 | — | — | — |
| h200_sxm | `sglang` | shared | sglang | sglang:17 | 0 / 11 | 0 | — | — | — |
| h200_sxm | `torch_ops` | shared | trtllm | trtllm:3256 | 0 / 1628 | 0 | — | — | — |
| l40s | `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | shared | vllm | vllm:3155 | 0 / 1584 | 182 | — | — | — |
| l40s | `sglang` | shared | sglang | sglang:35 | 0 / 18 | 1 | — | — | — |
| l40s | `torch_ops` | shared | trtllm | trtllm:3234 | 0 / 1627 | 1 | — | — | — |
| rtx_pro_6000_server | `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | shared | vllm | vllm:3216 | 0 / 1615 | 7 | — | — | — |
| rtx_pro_6000_server | `sglang` | shared | sglang | sglang:20 | 0 / 10 | 0 | — | — | — |
| rtx_pro_6000_server | `torch_ops` | shared | trtllm | trtllm:3237 | 0 / 1625 | 0 | — | — | — |

## `context_attention_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `flash_attention` | shared | sglang | sglang:5627 | 0 / 5627 | 0 | — | — | — |
| a100_sxm | `torch_flow` | shared | trtllm | trtllm:4864 | 0 / 4864 | 0 | — | — | — |
| a100_sxm | `vllm_flash_attn` | shared | vllm | vllm:5457 | 0 / 5049 | 0 | — | — | — |
| b200_sxm | `flashinfer` | shared | sglang | sglang:5168 | 0 / 2584 | 4 | — | — | — |
| b200_sxm | `torch_flow` | shared | trtllm | trtllm:139563 | 0 / 139563 | 0 | — | — | — |
| b200_sxm | `torch_flow_flashinfer` | shared | trtllm | trtllm:5100 | 0 / 5100 | 0 | — | — | — |
| b200_sxm | `triton` | shared | sglang | sglang:18972 | 0 / 9486 | 15 | — | — | — |
| b200_sxm | `trtllm_mha` | shared | sglang | sglang:65618 | 0 / 33526 | 32 | — | — | — |
| b200_sxm | `vllm_flash_attn_fa2` | shared | vllm | vllm:1628 | 0 / 1628 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer` | shared | vllm | vllm:35360 | 0 / 35360 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer_flashinfertrtllmapidecode` | shared | vllm | vllm:3900 | 0 / 3900 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer_flashinfertrtllmapidecode_trtllm_gen` | shared | vllm | vllm:3847 | 0 / 3847 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer_trtllmdecode` | shared | vllm | vllm:3756 | 0 / 3756 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer_trtllmprefill` | shared | vllm | vllm:139508 | 0 / 47304 | 243 | — | — | — |
| b200_sxm | `vllm_triton_attn` | shared | vllm | vllm:7222 | 0 / 3961 | 6 | — | — | — |
| b300_sxm | `flashinfer` | shared | sglang | sglang:5168 | 0 / 2584 | 5 | — | — | — |
| b300_sxm | `torch_flow` | shared | trtllm | trtllm:138136 | 0 / 71030 | 53 | — | — | — |
| b300_sxm | `torch_flow_flashinfer` | shared | trtllm | trtllm:5100 | 0 / 2788 | 6 | — | — | — |
| b300_sxm | `triton` | shared | sglang | sglang:18972 | 0 / 9486 | 14 | — | — | — |
| b300_sxm | `trtllm_mha` | shared | sglang | sglang:65619 | 0 / 33526 | 30 | — | — | — |
| b300_sxm | `vllm_flash_attn_fa2` | shared | vllm | vllm:1626 | 0 / 1626 | 0 | — | — | — |
| b300_sxm | `vllm_flashinfer_flashinfertrtllmapidecode` | shared | vllm | vllm:3900 | 0 / 3900 | 0 | — | — | — |
| b300_sxm | `vllm_flashinfer_flashinfertrtllmapidecode_trtllm_gen` | shared | vllm | vllm:3847 | 0 / 3847 | 0 | — | — | — |
| b300_sxm | `vllm_flashinfer_trtllmdecode` | shared | vllm | vllm:3668 | 0 / 3668 | 0 | — | — | — |
| b300_sxm | `vllm_flashinfer_trtllmprefill` | shared | vllm | vllm:138440 | 0 / 47304 | 303 | — | — | — |
| b300_sxm | `vllm_triton_attn` | shared | vllm | vllm:5588 | 0 / 2329 | 15 | — | — | — |
| b60 | `vllm_flash_attn` | shared | vllm | vllm:31688 | 0 / 17930 | 1 | — | — | — |
| gb200 | `flashinfer` | shared | sglang | sglang:5168 | 0 / 2584 | 5 | — | — | — |
| gb200 | `torch_flow` | shared | trtllm | trtllm:139562 | 0 / 71027 | 53 | — | — | — |
| gb200 | `torch_flow_flashinfer` | shared | trtllm | trtllm:5100 | 0 / 2788 | 2 | — | — | — |
| gb200 | `triton` | shared | sglang | sglang:18972 | 0 / 9486 | 20 | — | — | — |
| gb200 | `trtllm_mha` | shared | sglang | sglang:64661 | 0 / 33525 | 28 | — | — | — |
| gb200 | `vllm_flash_attn_fa2` | shared | vllm | vllm:1630 | 0 / 1630 | 0 | — | — | — |
| gb200 | `vllm_flashinfer_flashinfertrtllmapidecode` | shared | vllm | vllm:3900 | 0 / 3900 | 0 | — | — | — |
| gb200 | `vllm_flashinfer_flashinfertrtllmapidecode_trtllm_gen` | shared | vllm | vllm:3847 | 0 / 3847 | 0 | — | — | — |
| gb200 | `vllm_flashinfer_trtllmdecode` | shared | vllm | vllm:3740 | 0 / 3740 | 0 | — | — | — |
| gb200 | `vllm_flashinfer_trtllmprefill` | shared | vllm | vllm:139320 | 0 / 47304 | 232 | — | — | — |
| gb200 | `vllm_triton_attn` | shared | vllm | vllm:5591 | 0 / 2329 | 6 | — | — | — |
| gb300 | `flashinfer` | shared | sglang | sglang:5168 | 0 / 2584 | 3 | — | — | — |
| gb300 | `torch_flow` | shared | trtllm | trtllm:139563 | 0 / 71030 | 49 | — | — | — |
| gb300 | `torch_flow_flashinfer` | shared | trtllm | trtllm:5100 | 0 / 2788 | 5 | — | — | — |
| gb300 | `triton` | shared | sglang | sglang:18972 | 0 / 9486 | 4 | — | — | — |
| gb300 | `trtllm_mha` | shared | sglang | sglang:64667 | 0 / 33525 | 18 | — | — | — |
| gb300 | `vllm_flash_attn_fa2` | shared | vllm | vllm:1629 | 0 / 1629 | 0 | — | — | — |
| gb300 | `vllm_flashinfer_flashinfertrtllmapidecode` | shared | vllm | vllm:3900 | 0 / 3900 | 0 | — | — | — |
| gb300 | `vllm_flashinfer_flashinfertrtllmapidecode_trtllm_gen` | shared | vllm | vllm:3847 | 0 / 3847 | 0 | — | — | — |
| gb300 | `vllm_flashinfer_trtllmdecode` | shared | vllm | vllm:3740 | 0 / 3740 | 0 | — | — | — |
| gb300 | `vllm_flashinfer_trtllmprefill` | shared | vllm | vllm:139320 | 0 / 47304 | 301 | — | — | — |
| gb300 | `vllm_triton_attn` | shared | vllm | vllm:5587 | 0 / 2329 | 5 | — | — | — |
| h100_sxm | `fa3` | shared | sglang | sglang:66100 | 0 / 34478 | 28 | — | — | — |
| h100_sxm | `torch_flow` | shared | trtllm | trtllm:154782 | 0 / 79050 | 57 | — | — | — |
| h100_sxm | `triton` | shared | sglang | sglang:7650 | 0 / 4182 | 0 | — | — | — |
| h100_sxm | `vllm_flash_attn_fa3` | shared | vllm | vllm:156400 | 0 / 53074 | 93 | — | — | — |
| h100_sxm | `vllm_flash_attn_fa4` | shared | vllm | vllm:1972 | 0 / 697 | 3 | — | — | — |
| h100_sxm | `vllm_triton_attn` | shared | vllm | vllm:697 | 0 / 697 | 0 | — | — | — |
| h200_sxm | `fa3` | shared | sglang | sglang:67052 | 0 / 34478 | 31 | — | — | — |
| h200_sxm | `torch_flow` | shared | trtllm | trtllm:156199 | 0 / 79038 | 69 | — | — | — |
| h200_sxm | `triton` | shared | sglang | sglang:7650 | 0 / 4182 | 1 | — | — | — |
| h200_sxm | `vllm_flash_attn_fa3` | shared | vllm | vllm:157352 | 0 / 53074 | 320 | — | — | — |
| h200_sxm | `vllm_flash_attn_fa4` | shared | vllm | vllm:1972 | 0 / 697 | 3 | — | — | — |
| h200_sxm | `vllm_triton_attn` | shared | vllm | vllm:697 | 0 / 697 | 0 | — | — | — |
| l40s | `flashinfer` | shared | sglang | sglang:52506 | 0 / 29092 | 477 | — | — | — |
| l40s | `torch_flow` | shared | trtllm | trtllm:154712 | 0 / 78973 | 1671 | — | — | — |
| l40s | `triton` | shared | sglang | sglang:26622 | 0 / 16830 | 39 | — | — | — |
| l40s | `vllm_flash_attn_fa2` | shared | vllm | vllm:51624 | 0 / 26537 | 1065 | — | — | — |
| l40s | `vllm_flashinfer_fidecode` | shared | vllm | vllm:3690 | 0 / 1946 | 90 | — | — | — |
| l40s | `vllm_flashinfer_fiprefill` | shared | vllm | vllm:45334 | 0 / 23644 | 294 | — | — | — |
| l40s | `vllm_triton_attn` | shared | vllm | vllm:3961 | 0 / 2329 | 35 | — | — | — |
| rtx_pro_6000_server | `flashinfer` | shared | sglang | sglang:49692 | 0 / 26282 | 50 | — | — | — |
| rtx_pro_6000_server | `torch_flow` | shared | trtllm | trtllm:147105 | 0 / 75176 | 179 | — | — | — |
| rtx_pro_6000_server | `triton` | shared | sglang | sglang:32946 | 0 / 16830 | 15 | — | — | — |
| rtx_pro_6000_server | `vllm_flash_attn_fa2` | shared | vllm | vllm:51662 | 0 / 26537 | 71 | — | — | — |
| rtx_pro_6000_server | `vllm_flashinfer_fidecode` | shared | vllm | vllm:1781 | 0 / 1781 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_flashinfer_fiprefill` | shared | vllm | vllm:45312 | 0 / 23635 | 29 | — | — | — |
| rtx_pro_6000_server | `vllm_flashinfer_flashinfertrtllmapidecode_xqa` | shared | vllm | vllm:1807 | 0 / 1807 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_triton_attn` | shared | vllm | vllm:3961 | 0 / 2329 | 1 | — | — | — |

## `context_mla_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `default` | shared_fallback | trtllm | trtllm:2436 | 0 / 2436 | 0 | — | — | — |
| b200_sxm | `default` | shared_fallback | trtllm | trtllm:6160 | 0 / 4620 | 1 | — | — | — |
| b200_sxm | `trtllm_mla` | shared | sglang | sglang:6160 | 0 / 3080 | 4 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:4840 | 0 / 3080 | 3 | — | — | — |
| b300_sxm | `trtllm_mla` | shared | sglang | sglang:6160 | 0 / 3080 | 7 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:4840 | 0 / 3080 | 4 | — | — | — |
| gb200 | `trtllm_mla` | shared | sglang | sglang:6600 | 0 / 3960 | 0 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:4840 | 0 / 3080 | 4 | — | — | — |
| gb300 | `trtllm_mla` | shared | sglang | sglang:6600 | 0 / 3960 | 3 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:4840 | 0 / 3080 | 3 | — | — | — |
| h100_sxm | `flash_attention` | shared | sglang | sglang:6600 | 0 / 3080 | 1 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:6160 | 0 / 4620 | 4 | — | — | — |
| h200_sxm | `flash_attention` | shared | sglang | sglang:6160 | 0 / 3080 | 2 | — | — | — |
| l40s | `default` | shared_fallback | trtllm | trtllm:2436 | 0 / 2436 | 0 | — | — | — |
| l40s | `triton` | shared | sglang | sglang:2420 | 0 / 1540 | 0 | — | — | — |
| rtx_pro_6000_server | `default` | shared_fallback | trtllm | trtllm:4840 | 0 / 3080 | 4 | — | — | — |
| rtx_pro_6000_server | `triton` | shared | sglang | sglang:3300 | 0 / 1540 | 2 | — | — | — |

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
| b200_sxm | `TRTLLM` | shared | trtllm | trtllm:207 | 0 / 138 | 0 | — | — | — |
| b200_sxm | `vLLM_custom_eager` | shared | vllm | vllm:138 | 0 / 69 | 0 | — | — | — |
| b200_sxm | `vLLM_custom_graph` | shared | vllm | vllm:138 | 0 / 69 | 0 | — | — | — |
| b300_sxm | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:276 | 0 / 69 | 0 | — | — | — |
| b300_sxm | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:276 | 0 / 69 | 0 | — | — | — |
| b300_sxm | `TRTLLM` | shared | trtllm | trtllm:138 | 0 / 69 | 1 | — | — | — |
| b300_sxm | `vLLM_custom_eager` | shared | vllm | vllm:207 | 0 / 69 | 0 | — | — | — |
| b300_sxm | `vLLM_custom_graph` | shared | vllm | vllm:207 | 0 / 69 | 0 | — | — | — |
| b60 | `vLLM_custom_eager` | shared | vllm | vllm:115 | 0 / 69 | 0 | — | — | — |
| b60 | `vLLM_custom_graph` | shared | vllm | vllm:46 | 0 / 46 | 0 | — | — | — |
| gb200 | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:276 | 0 / 92 | 0 | — | — | — |
| gb200 | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:276 | 0 / 92 | 0 | — | — | — |
| gb200 | `TRTLLM` | shared | trtllm | trtllm:92 | 0 / 46 | 0 | — | — | — |
| gb200 | `vLLM_custom_eager` | shared | vllm | vllm:184 | 0 / 46 | 0 | — | — | — |
| gb200 | `vLLM_custom_graph` | shared | vllm | vllm:184 | 0 / 46 | 0 | — | — | — |
| gb300 | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:276 | 0 / 92 | 0 | — | — | — |
| gb300 | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:276 | 0 / 92 | 0 | — | — | — |
| gb300 | `TRTLLM` | shared | trtllm | trtllm:92 | 0 / 46 | 0 | — | — | — |
| gb300 | `TRTLLM_MNNVL_oneshot` | shared | trtllm | trtllm:19 | 0 / 19 | 0 | — | — | — |
| gb300 | `TRTLLM_MNNVL_twoshot` | shared | trtllm | trtllm:27 | 0 / 27 | 0 | — | — | — |
| gb300 | `vLLM_custom_eager` | shared | vllm | vllm:276 | 0 / 92 | 0 | — | — | — |
| gb300 | `vLLM_custom_graph` | shared | vllm | vllm:276 | 0 / 92 | 0 | — | — | — |
| h100_sxm | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:276 | 0 / 69 | 0 | — | — | — |
| h100_sxm | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:276 | 0 / 69 | 0 | — | — | — |
| h100_sxm | `TRTLLM` | shared | trtllm | trtllm:276 | 0 / 69 | 0 | — | — | — |
| h100_sxm | `vLLM_custom_eager` | shared | vllm | vllm:276 | 0 / 69 | 0 | — | — | — |
| h100_sxm | `vLLM_custom_graph` | shared | vllm | vllm:276 | 0 / 69 | 1 | — | — | — |
| h200_sxm | `SGLang_CustomAllReduce_eager` | shared | sglang | sglang:276 | 0 / 69 | 0 | — | — | — |
| h200_sxm | `SGLang_CustomAllReduce_graph` | shared | sglang | sglang:276 | 0 / 69 | 0 | — | — | — |
| h200_sxm | `TRTLLM` | shared | trtllm | trtllm:276 | 0 / 69 | 0 | — | — | — |
| h200_sxm | `vLLM_custom_eager` | shared | vllm | vllm:345 | 0 / 69 | 0 | — | — | — |
| h200_sxm | `vLLM_custom_graph` | shared | vllm | vllm:345 | 0 / 69 | 1 | — | — | — |
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
| b200_sxm | `FLASHINFER_MLA_SPARSE` | shared | vllm | vllm:254766 | 0 / 159801 | 0 | — | — | — |
| b200_sxm | `FLASHMLA_SPARSE` | shared | vllm | vllm:138371 | 0 / 87864 | 4 | — | — | — |
| b200_sxm | `default` | shared_fallback | trtllm | trtllm:29274 | 0 / 29274 | 0 | — | — | — |
| b200_sxm | `sglang_dsa_dense_mha_trtllm_ragged` | shared | sglang | sglang:69992 | 0 / 35080 | 14 | — | — | — |
| b200_sxm | `sglang_dsa_indexer_flashmla_sparse` | shared | sglang | sglang:34946 | 0 / 29790 | 3 | — | — | — |
| b200_sxm | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:34716 | 0 / 29702 | 1 | — | — | — |
| b200_sxm | `sglang_dsa_skip_indexer_flashmla_sparse` | shared | sglang | sglang:25657 | 0 / 12963 | 5 | — | — | — |
| b200_sxm | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:25654 | 0 / 12960 | 3 | — | — | — |
| b300_sxm | `FLASHINFER_MLA_SPARSE` | shared | vllm | vllm:254692 | 0 / 159780 | 5 | — | — | — |
| b300_sxm | `FLASHMLA_SPARSE` | shared | vllm | vllm:138360 | 0 / 87864 | 4 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:29277 | 0 / 14640 | 19 | — | — | — |
| b300_sxm | `sglang_dsa_dense_mha_trtllm_ragged` | shared | sglang | sglang:62869 | 0 / 34972 | 1 | — | — | — |
| b300_sxm | `sglang_dsa_indexer_flashmla_sparse` | shared | sglang | sglang:34605 | 0 / 29449 | 0 | — | — | — |
| b300_sxm | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:34946 | 0 / 29790 | 2 | — | — | — |
| b300_sxm | `sglang_dsa_skip_indexer_flashmla_sparse` | shared | sglang | sglang:19086 | 0 / 19086 | 0 | — | — | — |
| b300_sxm | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:19174 | 0 / 19174 | 0 | — | — | — |
| gb200 | `FLASHINFER_MLA_SPARSE` | shared | vllm | vllm:254692 | 0 / 159780 | 2 | — | — | — |
| gb200 | `FLASHMLA_SPARSE` | shared | vllm | vllm:138360 | 0 / 87864 | 7 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:29276 | 0 / 14640 | 18 | — | — | — |
| gb200 | `sglang_dsa_dense_mha_trtllm_ragged` | shared | sglang | sglang:81451 | 0 / 35080 | 0 | — | — | — |
| gb200 | `sglang_dsa_indexer_flashmla_sparse` | shared | sglang | sglang:49046 | 0 / 29678 | 0 | — | — | — |
| gb200 | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:43463 | 0 / 29678 | 2 | — | — | — |
| gb200 | `sglang_dsa_skip_indexer_flashmla_sparse` | shared | sglang | sglang:20966 | 0 / 19062 | 0 | — | — | — |
| gb200 | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:20572 | 0 / 19062 | 0 | — | — | — |
| gb300 | `FLASHINFER_MLA_SPARSE` | shared | vllm | vllm:254692 | 0 / 159780 | 2 | — | — | — |
| gb300 | `FLASHMLA_SPARSE` | shared | vllm | vllm:138360 | 0 / 87864 | 3 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:29275 | 0 / 14640 | 19 | — | — | — |
| gb300 | `sglang_dsa_dense_mha_trtllm_ragged` | shared | sglang | sglang:63144 | 0 / 35080 | 1 | — | — | — |
| gb300 | `sglang_dsa_indexer_flashmla_sparse` | shared | sglang | sglang:34834 | 0 / 29678 | 0 | — | — | — |
| gb300 | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:34834 | 0 / 29678 | 1 | — | — | — |
| gb300 | `sglang_dsa_skip_indexer_flashmla_sparse` | shared | sglang | sglang:19062 | 0 / 19062 | 0 | — | — | — |
| gb300 | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:19062 | 0 / 19062 | 0 | — | — | — |
| h100_sxm | `FLASHMLA_SPARSE` | shared | vllm | vllm:227400 | 0 / 143058 | 15 | — | — | — |
| h100_sxm | `FLASH_ATTN_MLA_SPARSE` | shared | vllm | vllm:194364 | 0 / 128988 | 0 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:19444 | 0 / 7790 | 1 | — | — | — |
| h100_sxm | `sglang_dsa_dense_mha_fa3` | shared | sglang | sglang:80601 | 0 / 31203 | 5 | — | — | — |
| h100_sxm | `sglang_dsa_indexer_flashmla_kv` | shared | sglang | sglang:42717 | 0 / 24685 | 3 | — | — | — |
| h100_sxm | `sglang_dsa_indexer_flashmla_sparse` | shared | sglang | sglang:41017 | 0 / 23131 | 1 | — | — | — |
| h100_sxm | `sglang_dsa_skip_indexer_flashmla_kv` | shared | sglang | sglang:22083 | 0 / 11047 | 11 | — | — | — |
| h100_sxm | `sglang_dsa_skip_indexer_flashmla_sparse` | shared | sglang | sglang:19024 | 0 / 9512 | 2 | — | — | — |
| h200_sxm | `FLASHMLA_SPARSE` | shared | vllm | vllm:227700 | 0 / 143244 | 12 | — | — | — |
| h200_sxm | `FLASH_ATTN_MLA_SPARSE` | shared | vllm | vllm:194364 | 0 / 128988 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:19520 | 0 / 7808 | 10 | — | — | — |
| h200_sxm | `sglang_dsa_dense_mha_fa3` | shared | sglang | sglang:64510 | 0 / 32280 | 4 | — | — | — |
| h200_sxm | `sglang_dsa_indexer_flashmla_kv` | shared | sglang | sglang:29171 | 0 / 24799 | 3 | — | — | — |
| h200_sxm | `sglang_dsa_indexer_flashmla_sparse` | shared | sglang | sglang:29157 | 0 / 24785 | 2 | — | — | — |
| h200_sxm | `sglang_dsa_skip_indexer_flashmla_kv` | shared | sglang | sglang:22146 | 0 / 11126 | 4 | — | — | — |
| h200_sxm | `sglang_dsa_skip_indexer_flashmla_sparse` | shared | sglang | sglang:22216 | 0 / 11112 | 2 | — | — | — |
| rtx_pro_6000_server | `FLASHINFER_MLA_SPARSE_SM120` | shared | vllm | vllm:107980 | 0 / 71660 | 0 | — | — | — |

## `dsa_generation_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `FLASHINFER_MLA_SPARSE` | shared | vllm | vllm:40222 | 0 / 26706 | 0 | — | — | — |
| b200_sxm | `FLASHMLA_SPARSE` | shared | vllm | vllm:19894 | 0 / 13248 | 0 | — | — | — |
| b200_sxm | `default` | shared_fallback | trtllm | trtllm:22077 | 0 / 22077 | 0 | — | — | — |
| b200_sxm | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:7256 | 0 / 6072 | 2 | — | — | — |
| b200_sxm | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:4912 | 0 / 2464 | 6 | — | — | — |
| b300_sxm | `FLASHINFER_MLA_SPARSE` | shared | vllm | vllm:40584 | 0 / 26916 | 0 | — | — | — |
| b300_sxm | `FLASHMLA_SPARSE` | shared | vllm | vllm:19872 | 0 / 13248 | 0 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:22070 | 0 / 11039 | 120 | — | — | — |
| b300_sxm | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:7256 | 0 / 6072 | 3 | — | — | — |
| b300_sxm | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:3688 | 0 / 3688 | 0 | — | — | — |
| gb200 | `FLASHINFER_MLA_SPARSE` | shared | vllm | vllm:40584 | 0 / 26916 | 0 | — | — | — |
| gb200 | `FLASHMLA_SPARSE` | shared | vllm | vllm:19872 | 0 / 13248 | 0 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:22070 | 0 / 11040 | 60 | — | — | — |
| gb200 | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:7256 | 0 / 6072 | 2 | — | — | — |
| gb200 | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:3688 | 0 / 3688 | 0 | — | — | — |
| gb300 | `FLASHINFER_MLA_SPARSE` | shared | vllm | vllm:40584 | 0 / 26916 | 0 | — | — | — |
| gb300 | `FLASHMLA_SPARSE` | shared | vllm | vllm:19872 | 0 / 13248 | 0 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:22062 | 0 / 11040 | 78 | — | — | — |
| gb300 | `sglang_dsa_indexer_trtllm` | shared | sglang | sglang:7256 | 0 / 6072 | 3 | — | — | — |
| gb300 | `sglang_dsa_skip_indexer_trtllm` | shared | sglang | sglang:3688 | 0 / 3688 | 0 | — | — | — |
| h100_sxm | `FLASHMLA_SPARSE` | shared | vllm | vllm:46782 | 0 / 31083 | 0 | — | — | — |
| h100_sxm | `FLASH_ATTN_MLA_SPARSE` | shared | vllm | vllm:19548 | 0 / 13032 | 0 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:14720 | 0 / 5888 | 10 | — | — | — |
| h100_sxm | `sglang_dsa_indexer_fa3` | shared | sglang | sglang:5396 | 0 / 3020 | 0 | — | — | — |
| h100_sxm | `sglang_dsa_indexer_flashmla_kv` | shared | sglang | sglang:5396 | 0 / 3020 | 1 | — | — | — |
| h100_sxm | `sglang_dsa_skip_indexer_fa3` | shared | sglang | sglang:2456 | 0 / 1232 | 8 | — | — | — |
| h100_sxm | `sglang_dsa_skip_indexer_flashmla_kv` | shared | sglang | sglang:2456 | 0 / 1232 | 11 | — | — | — |
| h200_sxm | `FLASHMLA_SPARSE` | shared | vllm | vllm:46998 | 0 / 31227 | 0 | — | — | — |
| h200_sxm | `FLASH_ATTN_MLA_SPARSE` | shared | vllm | vllm:19872 | 0 / 13248 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:14720 | 0 / 5888 | 16 | — | — | — |
| h200_sxm | `sglang_dsa_indexer_fa3` | shared | sglang | sglang:3612 | 0 / 3020 | 3 | — | — | — |
| h200_sxm | `sglang_dsa_indexer_flashmla_kv` | shared | sglang | sglang:3612 | 0 / 3020 | 0 | — | — | — |
| h200_sxm | `sglang_dsa_skip_indexer_fa3` | shared | sglang | sglang:2456 | 0 / 1232 | 7 | — | — | — |
| h200_sxm | `sglang_dsa_skip_indexer_flashmla_kv` | shared | sglang | sglang:2456 | 0 / 1232 | 1 | — | — | — |
| rtx_pro_6000_server | `FLASHINFER_MLA_SPARSE_SM120` | shared | vllm | vllm:11040 | 0 / 7360 | 0 | — | — | — |

## `dsv41_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `sglang.srt.layers.engram.Engram.apply_gate+sglang.srt.layers.engram.EngramEmbedding._owned_rows+sglang.srt.layers.linear.ReplicatedLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | shared | sglang | sglang:60 | 0 / 60 | 0 | — | — | — |
| b200_sxm | `sglang.srt.layers.linear.MergedColumnParallelLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | shared | sglang | sglang:30 | 0 / 30 | 0 | — | — | — |
| b200_sxm | `sglang.srt.layers.linear.RowParallelLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | shared | sglang | sglang:30 | 0 / 30 | 0 | — | — | — |
| b200_sxm | `sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer._hc_mix_and_combine+sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_post` | shared | sglang | sglang:15 | 0 / 15 | 0 | — | — | — |
| b200_sxm | `sglang.srt.models.deepseek_v4.MQALayer.forward` | shared | sglang | sglang:2260 | 0 / 2260 | 0 | — | — | — |
| gb200 | `sglang.srt.layers.engram.Engram.apply_gate+sglang.srt.layers.engram.EngramEmbedding._owned_rows+sglang.srt.layers.linear.ReplicatedLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | shared | sglang | sglang:60 | 0 / 60 | 0 | — | — | — |
| gb200 | `sglang.srt.layers.linear.MergedColumnParallelLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | shared | sglang | sglang:30 | 0 / 30 | 0 | — | — | — |
| gb200 | `sglang.srt.layers.linear.RowParallelLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | shared | sglang | sglang:30 | 0 / 30 | 0 | — | — | — |
| gb200 | `sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer._hc_mix_and_combine+sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_post` | shared | sglang | sglang:15 | 0 / 15 | 0 | — | — | — |
| gb200 | `sglang.srt.models.deepseek_v4.MQALayer.forward` | shared | sglang | sglang:2260 | 0 / 2260 | 0 | — | — | — |
| h100_sxm | `sglang.srt.layers.engram.Engram.apply_gate+sglang.srt.layers.engram.EngramEmbedding._owned_rows+sglang.srt.layers.linear.ReplicatedLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | shared | sglang | sglang:60 | 0 / 60 | 0 | — | — | — |
| h100_sxm | `sglang.srt.layers.linear.MergedColumnParallelLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | shared | sglang | sglang:30 | 0 / 30 | 0 | — | — | — |
| h100_sxm | `sglang.srt.layers.linear.RowParallelLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | shared | sglang | sglang:30 | 0 / 30 | 0 | — | — | — |
| h100_sxm | `sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer._hc_mix_and_combine+sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_post` | shared | sglang | sglang:15 | 0 / 15 | 0 | — | — | — |
| h100_sxm | `sglang.srt.models.deepseek_v4.MQALayer.forward` | shared | sglang | sglang:2260 | 0 / 2260 | 0 | — | — | — |
| h200_sxm | `sglang.srt.layers.engram.Engram.apply_gate+sglang.srt.layers.engram.EngramEmbedding._owned_rows+sglang.srt.layers.linear.ReplicatedLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | shared | sglang | sglang:60 | 0 / 60 | 0 | — | — | — |
| h200_sxm | `sglang.srt.layers.linear.MergedColumnParallelLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | shared | sglang | sglang:30 | 0 / 30 | 0 | — | — | — |
| h200_sxm | `sglang.srt.layers.linear.RowParallelLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | shared | sglang | sglang:30 | 0 / 30 | 0 | — | — | — |
| h200_sxm | `sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer._hc_mix_and_combine+sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_post` | shared | sglang | sglang:15 | 0 / 15 | 0 | — | — | — |
| h200_sxm | `sglang.srt.models.deepseek_v4.MQALayer.forward` | shared | sglang | sglang:2260 | 0 / 2260 | 0 | — | — | — |

## `dsv4_csa_context_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:11421 | 0 / 11421 | 0 | — | — | — |
| b200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:17510 | 0 / 8760 | 39 | — | — | — |
| b200_sxm | `compressed_flashmla` | shared | sglang | sglang:89024 | 0 / 67120 | 12 | — | — | — |
| b300_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:11440 | 0 / 11440 | 0 | — | — | — |
| b300_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:17373 | 0 / 8624 | 35 | — | — | — |
| b300_sxm | `compressed_flashmla` | shared | sglang | sglang:137145 | 0 / 70160 | 6 | — | — | — |
| gb200 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:11419 | 0 / 11419 | 0 | — | — | — |
| gb200 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:17507 | 0 / 8760 | 27 | — | — | — |
| gb200 | `compressed_flashmla` | shared | sglang | sglang:100528 | 0 / 68354 | 11 | — | — | — |
| gb300 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:11440 | 0 / 11440 | 0 | — | — | — |
| gb300 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:17371 | 0 / 8624 | 28 | — | — | — |
| gb300 | `compressed_flashmla` | shared | sglang | sglang:142106 | 0 / 70152 | 8 | — | — | — |
| h100_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:5739 | 0 / 5739 | 0 | — | — | — |
| h100_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:17325 | 0 / 5790 | 47 | — | — | — |
| h100_sxm | `compressed_flashmla` | shared | sglang | sglang:92982 | 0 / 40384 | 9 | — | — | — |
| h200_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:5840 | 0 / 5840 | 0 | — | — | — |
| h200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:17510 | 0 / 5840 | 62 | — | — | — |
| h200_sxm | `compressed_flashmla` | shared | sglang | sglang:80976 | 0 / 40592 | 11 | — | — | — |
| rtx_pro_6000_server | `FLASHINFER_MLA_SPARSE_DSV4` | shared | vllm | vllm:5780 | 0 / 5780 | 0 | — | — | — |

## `dsv4_csa_generation_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:3088 | 0 / 3088 | 0 | — | — | — |
| b200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:4632 | 0 / 2316 | 17 | — | — | — |
| b200_sxm | `compressed_flashmla` | shared | sglang | sglang:6400 | 0 / 4800 | 4 | — | — | — |
| b300_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:3088 | 0 / 3088 | 0 | — | — | — |
| b300_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:4136 | 0 / 1820 | 22 | — | — | — |
| b300_sxm | `compressed_flashmla` | shared | sglang | sglang:6400 | 0 / 4800 | 1 | — | — | — |
| gb200 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:3088 | 0 / 3088 | 0 | — | — | — |
| gb200 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:4632 | 0 / 2316 | 13 | — | — | — |
| gb200 | `compressed_flashmla` | shared | sglang | sglang:6400 | 0 / 4800 | 2 | — | — | — |
| gb300 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:3088 | 0 / 3088 | 0 | — | — | — |
| gb300 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:4136 | 0 / 1820 | 17 | — | — | — |
| gb300 | `compressed_flashmla` | shared | sglang | sglang:6400 | 0 / 4800 | 3 | — | — | — |
| h100_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:1544 | 0 / 1544 | 0 | — | — | — |
| h100_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:4632 | 0 / 1544 | 43 | — | — | — |
| h100_sxm | `compressed_flashmla` | shared | sglang | sglang:6396 | 0 / 3200 | 4 | — | — | — |
| h200_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:1544 | 0 / 1544 | 0 | — | — | — |
| h200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:4632 | 0 / 1544 | 91 | — | — | — |
| h200_sxm | `compressed_flashmla` | shared | sglang | sglang:6400 | 0 / 3200 | 10 | — | — | — |
| rtx_pro_6000_server | `FLASHINFER_MLA_SPARSE_DSV4` | shared | vllm | vllm:1544 | 0 / 1544 | 0 | — | — | — |
| rtx_pro_6000_server | `compressed_flashmla` | shared | sglang | sglang:909 | 0 / 868 | 0 | — | — | — |

## `dsv4_csa_topk_calib_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `topk_transform_v1` | shared | sglang | sglang:7156 | 0 / 4598 | 0 | — | — | — |
| b200_sxm | `topk_transform_v2` | shared | sglang | sglang:384 | 0 / 244 | 0 | — | — | — |
| b300_sxm | `topk_transform_v1` | shared | sglang | sglang:7156 | 0 / 4598 | 0 | — | — | — |
| b300_sxm | `topk_transform_v2` | shared | sglang | sglang:384 | 0 / 244 | 0 | — | — | — |
| gb200 | `topk_transform_v1` | shared | sglang | sglang:7156 | 0 / 4598 | 0 | — | — | — |
| gb200 | `topk_transform_v2` | shared | sglang | sglang:384 | 0 / 244 | 0 | — | — | — |
| gb300 | `topk_transform_v1` | shared | sglang | sglang:7156 | 0 / 4598 | 0 | — | — | — |
| gb300 | `topk_transform_v2` | shared | sglang | sglang:384 | 0 / 244 | 0 | — | — | — |
| h100_sxm | `topk_transform_v1` | shared | sglang | sglang:5750 | 0 / 3686 | 0 | — | — | — |
| h100_sxm | `topk_transform_v2` | shared | sglang | sglang:384 | 0 / 244 | 0 | — | — | — |
| h200_sxm | `topk_transform_v1` | shared | sglang | sglang:5750 | 0 / 3686 | 0 | — | — | — |
| h200_sxm | `topk_transform_v2` | shared | sglang | sglang:384 | 0 / 244 | 0 | — | — | — |

## `dsv4_hca_context_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:11429 | 0 / 11429 | 0 | — | — | — |
| b200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:17520 | 0 / 8760 | 32 | — | — | — |
| b200_sxm | `compressed_flashmla` | shared | sglang | sglang:95152 | 0 / 71424 | 3 | — | — | — |
| b300_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:11440 | 0 / 11440 | 0 | — | — | — |
| b300_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:17520 | 0 / 8760 | 51 | — | — | — |
| b300_sxm | `compressed_flashmla` | shared | sglang | sglang:109223 | 0 / 71470 | 11 | — | — | — |
| gb200 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:11424 | 0 / 11424 | 0 | — | — | — |
| gb200 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:17520 | 0 / 8760 | 19 | — | — | — |
| gb200 | `compressed_flashmla` | shared | sglang | sglang:95152 | 0 / 71424 | 9 | — | — | — |
| gb300 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:11440 | 0 / 11440 | 0 | — | — | — |
| gb300 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:17520 | 0 / 8760 | 37 | — | — | — |
| gb300 | `compressed_flashmla` | shared | sglang | sglang:102620 | 0 / 71472 | 11 | — | — | — |
| h100_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:5793 | 0 / 5793 | 0 | — | — | — |
| h100_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:17379 | 0 / 5793 | 65 | — | — | — |
| h100_sxm | `compressed_flashmla` | shared | sglang | sglang:79855 | 0 / 40624 | 14 | — | — | — |
| h200_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:5840 | 0 / 5840 | 0 | — | — | — |
| h200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:17520 | 0 / 5840 | 118 | — | — | — |
| h200_sxm | `compressed_flashmla` | shared | sglang | sglang:81376 | 0 / 40784 | 8 | — | — | — |
| rtx_pro_6000_server | `FLASHINFER_MLA_SPARSE_DSV4` | shared | vllm | vllm:5810 | 0 / 5810 | 0 | — | — | — |
| rtx_pro_6000_server | `compressed_flashmla` | shared | sglang | sglang:79856 | 0 / 40048 | 0 | — | — | — |

## `dsv4_hca_generation_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:3088 | 0 / 3088 | 0 | — | — | — |
| b200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:4632 | 0 / 2316 | 17 | — | — | — |
| b200_sxm | `compressed_flashmla` | shared | sglang | sglang:6400 | 0 / 4800 | 10 | — | — | — |
| b300_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:3088 | 0 / 3088 | 0 | — | — | — |
| b300_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:4632 | 0 / 2316 | 24 | — | — | — |
| b300_sxm | `compressed_flashmla` | shared | sglang | sglang:6400 | 0 / 4800 | 4 | — | — | — |
| gb200 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:3088 | 0 / 3088 | 0 | — | — | — |
| gb200 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:4632 | 0 / 2316 | 12 | — | — | — |
| gb200 | `compressed_flashmla` | shared | sglang | sglang:6400 | 0 / 4800 | 11 | — | — | — |
| gb300 | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:3088 | 0 / 3088 | 0 | — | — | — |
| gb300 | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:4632 | 0 / 2316 | 29 | — | — | — |
| gb300 | `compressed_flashmla` | shared | sglang | sglang:6400 | 0 / 4800 | 14 | — | — | — |
| h100_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:1544 | 0 / 1544 | 0 | — | — | — |
| h100_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:4632 | 0 / 1544 | 56 | — | — | — |
| h100_sxm | `compressed_flashmla` | shared | sglang | sglang:6400 | 0 / 3200 | 5 | — | — | — |
| h200_sxm | `DeepseekV4TrtllmAttention` | shared | trtllm | trtllm:1544 | 0 / 1544 | 0 | — | — | — |
| h200_sxm | `FLASHMLA_SPARSE_DSV4` | shared | vllm | vllm:4632 | 0 / 1544 | 123 | — | — | — |
| h200_sxm | `compressed_flashmla` | shared | sglang | sglang:6400 | 0 / 3200 | 2 | — | — | — |
| rtx_pro_6000_server | `FLASHINFER_MLA_SPARSE_DSV4` | shared | vllm | vllm:1544 | 0 / 1544 | 0 | — | — | — |
| rtx_pro_6000_server | `compressed_flashmla` | shared | sglang | sglang:4800 | 0 / 3200 | 0 | — | — | — |

## `dsv4_megamoe_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `deepgemm_megamoe` | shared | sglang | sglang:64 | 0 / 64 | 0 | — | — | — |
| gb200 | `deepgemm_megamoe` | shared | sglang | sglang:256 | 0 / 256 | 0 | — | — | — |
| gb300 | `deepgemm_megamoe` | shared | sglang | sglang:456 | 0 / 456 | 0 | — | — | — |

## `dsv4_paged_mqa_logits_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `deep_gemm.fp8_paged_mqa_logits` | shared | sglang | sglang:6971 | 0 / 4790 | 0 | — | — | — |
| b200_sxm | `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | shared | vllm | vllm:9795 | 0 / 3918 | 718 | — | — | — |
| b300_sxm | `deep_gemm.fp8_paged_mqa_logits` | shared | sglang | sglang:9152 | 0 / 4790 | 0 | — | — | — |
| b300_sxm | `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | shared | vllm | vllm:9795 | 0 / 3918 | 879 | — | — | — |
| gb200 | `deep_gemm.fp8_paged_mqa_logits` | shared | sglang | sglang:9051 | 0 / 4790 | 0 | — | — | — |
| gb200 | `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | shared | vllm | vllm:9795 | 0 / 3918 | 398 | — | — | — |
| gb300 | `deep_gemm.fp8_paged_mqa_logits` | shared | sglang | sglang:9152 | 0 / 4790 | 0 | — | — | — |
| gb300 | `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | shared | vllm | vllm:9795 | 0 / 3918 | 471 | — | — | — |
| h100_sxm | `deep_gemm.fp8_paged_mqa_logits` | shared | sglang | sglang:6165 | 0 / 4110 | 0 | — | — | — |
| h100_sxm | `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | shared | vllm | vllm:9795 | 0 / 3918 | 351 | — | — | — |
| h200_sxm | `deep_gemm.fp8_paged_mqa_logits` | shared | sglang | sglang:6165 | 0 / 4110 | 0 | — | — | — |
| h200_sxm | `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | shared | vllm | vllm:9795 | 0 / 3918 | 432 | — | — | — |
| rtx_pro_6000_server | `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | shared | vllm | vllm:3918 | 0 / 3918 | 0 | — | — | — |

## `encoder_attention_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `flash_attention_v4` | shared | sglang | sglang:15358 | 0 / 7679 | 12 | — | — | — |
| b200_sxm | `torch_flow` | shared | trtllm | trtllm:15358 | 0 / 15358 | 0 | — | — | — |
| b200_sxm | `vllm_vit_flash_attn_fa4` | shared | vllm | vllm:23037 | 0 / 7679 | 62 | — | — | — |
| b300_sxm | `flash_attention_v4` | shared | sglang | sglang:7679 | 0 / 7679 | 0 | — | — | — |
| b300_sxm | `torch_flow` | shared | trtllm | trtllm:15358 | 0 / 7679 | 21 | — | — | — |
| b300_sxm | `triton` | shared | sglang | sglang:7679 | 0 / 7679 | 0 | — | — | — |
| b300_sxm | `vllm_vit_flash_attn_fa4` | shared | vllm | vllm:23037 | 0 / 7679 | 53 | — | — | — |
| gb200 | `flash_attention_v4` | shared | sglang | sglang:15358 | 0 / 7679 | 14 | — | — | — |
| gb200 | `torch_flow` | shared | trtllm | trtllm:15358 | 0 / 7679 | 14 | — | — | — |
| gb200 | `vllm_vit_flash_attn_fa4` | shared | vllm | vllm:23037 | 0 / 7679 | 49 | — | — | — |
| gb300 | `flash_attention_v4` | shared | sglang | sglang:7679 | 0 / 7679 | 0 | — | — | — |
| gb300 | `torch_flow` | shared | trtllm | trtllm:15358 | 0 / 7679 | 17 | — | — | — |
| gb300 | `triton` | shared | sglang | sglang:7679 | 0 / 7679 | 0 | — | — | — |
| gb300 | `vllm_vit_flash_attn_fa4` | shared | vllm | vllm:23037 | 0 / 7679 | 45 | — | — | — |
| h100_sxm | `flash_attention_v3` | shared | sglang | sglang:15358 | 0 / 7679 | 3 | — | — | — |
| h100_sxm | `torch_flow` | shared | trtllm | trtllm:15358 | 0 / 7679 | 16 | — | — | — |
| h100_sxm | `vllm_vit_flash_attn_fa3` | shared | vllm | vllm:23037 | 0 / 7679 | 21 | — | — | — |
| h200_sxm | `flash_attention_v3` | shared | sglang | sglang:16458 | 0 / 8779 | 6 | — | — | — |
| h200_sxm | `torch_flow` | shared | trtllm | trtllm:15358 | 0 / 7679 | 16 | — | — | — |
| h200_sxm | `vllm_vit_flash_attn_fa3` | shared | vllm | vllm:24137 | 0 / 8779 | 60 | — | — | — |
| l40s | `torch_flow` | shared | trtllm | trtllm:15358 | 0 / 7679 | 96 | — | — | — |
| l40s | `triton` | shared | sglang | sglang:15358 | 0 / 7679 | 121 | — | — | — |
| l40s | `vllm_vit_flash_attn_fa2` | shared | vllm | vllm:15358 | 0 / 7679 | 228 | — | — | — |
| rtx_pro_6000_server | `torch_flow` | shared | trtllm | trtllm:15358 | 0 / 7679 | 27 | — | — | — |
| rtx_pro_6000_server | `triton` | shared | sglang | sglang:15358 | 0 / 7679 | 15 | — | — | — |
| rtx_pro_6000_server | `vllm_vit_flash_attn_fa2` | shared | vllm | vllm:15358 | 0 / 7679 | 16 | — | — | — |

## `gdn_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `causal_conv1d_fn` | shared | sglang | sglang:824 | 0 / 824 | 0 | — | — | — |
| a100_sxm | `causal_conv1d_update` | shared | sglang | sglang:88 | 0 / 88 | 0 | — | — | — |
| a100_sxm | `fused_recurrent_gated_delta_rule` | shared | sglang | sglang:86 | 0 / 86 | 0 | — | — | — |
| b200_sxm | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:9058, trtllm:5470, vllm:14112 | 4616 / 5558 | 38 | 89.8 | 208.1 | 277.9 |
| b200_sxm | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:462, trtllm:550, vllm:1383 | 462 / 550 | 2 | 65.6 | 101.6 | 161.9 |
| b200_sxm | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:9058, trtllm:854 | 0 / 5383 | 1 | — | — | — |
| b200_sxm | `chunk_gated_delta_rule_flashinfer` | shared | trtllm, vllm | trtllm:4616, vllm:14112 | 4616 / 4704 | 0 | 9.9 | 23.1 | 30.6 |
| b200_sxm | `flashinfer_gated_delta_rule_decode` | shared | sglang | sglang:52 | 0 / 52 | 0 | — | — | — |
| b200_sxm | `fused_qkvzba_causal_conv1d_update_contiguous` | shared | sglang | sglang:462 | 0 / 462 | 0 | — | — | — |
| b200_sxm | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| b200_sxm | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:914, vllm:1371 | 457 / 457 | 7 | 0.4 | 3.4 | 9.3 |
| b200_sxm | `fused_sigmoid_gating_delta_rule_update_flashinfer` | shared | trtllm | trtllm:462 | 0 / 462 | 0 | — | — | — |
| b300_sxm | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:9058, trtllm:5487, vllm:14112 | 4650 / 4704 | 18 | 92.8 | 208.2 | 278.3 |
| b300_sxm | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:462, trtllm:550, vllm:1383 | 462 / 462 | 0 | 62.9 | 98.4 | 160.3 |
| b300_sxm | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:9058, trtllm:836 | 799 / 4566 | 0 | 39.6 | 56.6 | 61.1 |
| b300_sxm | `chunk_gated_delta_rule_flashinfer` | shared | trtllm, vllm | trtllm:4650, vllm:14112 | 4650 / 4704 | 1 | 8.4 | 22.2 | 30.1 |
| b300_sxm | `flashinfer_gated_delta_rule_decode` | shared | sglang | sglang:52 | 0 / 52 | 0 | — | — | — |
| b300_sxm | `fused_qkvzba_causal_conv1d_update_contiguous` | shared | sglang | sglang:462 | 0 / 462 | 0 | — | — | — |
| b300_sxm | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| b300_sxm | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:914, vllm:1371 | 457 / 457 | 2 | 0.3 | 3.6 | 5.3 |
| b300_sxm | `fused_sigmoid_gating_delta_rule_update_flashinfer` | shared | trtllm | trtllm:462 | 0 / 462 | 0 | — | — | — |
| gb200 | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:9058, trtllm:5470, vllm:14112 | 4616 / 4704 | 11 | 97.8 | 210.7 | 278.2 |
| gb200 | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:462, trtllm:550, vllm:1383 | 462 / 462 | 2 | 83.7 | 135.7 | 163.8 |
| gb200 | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:9058, trtllm:854 | 824 / 4559 | 1 | 13.5 | 49.1 | 70.1 |
| gb200 | `chunk_gated_delta_rule_flashinfer` | shared | trtllm, vllm | trtllm:4616, vllm:14112 | 4616 / 4704 | 1 | 11.4 | 27.0 | 33.3 |
| gb200 | `flashinfer_gated_delta_rule_decode` | shared | sglang | sglang:52 | 0 / 52 | 0 | — | — | — |
| gb200 | `fused_qkvzba_causal_conv1d_update_contiguous` | shared | sglang | sglang:462 | 0 / 462 | 0 | — | — | — |
| gb200 | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| gb200 | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:914, vllm:1371 | 457 / 457 | 0 | 2.7 | 4.7 | 5.9 |
| gb200 | `fused_sigmoid_gating_delta_rule_update_flashinfer` | shared | trtllm | trtllm:462 | 0 / 462 | 0 | — | — | — |
| gb300 | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:9058, trtllm:5487, vllm:14112 | 4650 / 4704 | 12 | 101.6 | 208.0 | 277.4 |
| gb300 | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:462, trtllm:550, vllm:1383 | 462 / 462 | 1 | 82.1 | 153.6 | 198.4 |
| gb300 | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:9058, trtllm:836 | 799 / 4566 | 0 | 17.6 | 41.9 | 55.4 |
| gb300 | `chunk_gated_delta_rule_flashinfer` | shared | trtllm, vllm | trtllm:4650, vllm:14112 | 4650 / 4704 | 1 | 10.2 | 25.7 | 32.5 |
| gb300 | `flashinfer_gated_delta_rule_decode` | shared | sglang | sglang:52 | 0 / 52 | 0 | — | — | — |
| gb300 | `fused_qkvzba_causal_conv1d_update_contiguous` | shared | sglang | sglang:462 | 0 / 462 | 0 | — | — | — |
| gb300 | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| gb300 | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:914, vllm:1371 | 457 / 457 | 2 | 0.4 | 3.6 | 5.3 |
| gb300 | `fused_sigmoid_gating_delta_rule_update_flashinfer` | shared | trtllm | trtllm:462 | 0 / 462 | 0 | — | — | — |
| h100_sxm | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:8533, trtllm:5325, vllm:13507 | 4529 / 4694 | 5 | 92.8 | 220.5 | 281.1 |
| h100_sxm | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:407, trtllm:550, vllm:1329 | 461 / 462 | 1 | 79.4 | 115.8 | 160.1 |
| h100_sxm | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:4004, trtllm:806 | 806 / 4004 | 0 | 42.2 | 59.4 | 69.5 |
| h100_sxm | `chunk_gated_delta_rule_flashinfer` | shared | trtllm, vllm | trtllm:4509, vllm:13507 | 4498 / 4694 | 0 | 18.5 | 65.1 | 83.8 |
| h100_sxm | `flashinfer_chunk_gated_delta_rule` | shared | sglang | sglang:4529 | 0 / 4529 | 0 | — | — | — |
| h100_sxm | `fused_qkvzba_causal_conv1d_update_contiguous` | shared | sglang | sglang:462 | 0 / 462 | 0 | — | — | — |
| h100_sxm | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| h100_sxm | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:862, vllm:1319 | 457 / 457 | 0 | 1.0 | 3.0 | 5.7 |
| h100_sxm | `fused_sigmoid_gating_delta_rule_update_flashinfer` | shared | trtllm | trtllm:462 | 0 / 462 | 0 | — | — | — |
| h200_sxm | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:8533, trtllm:5400, vllm:13550 | 4602 / 4703 | 14 | 93.7 | 220.8 | 281.1 |
| h200_sxm | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:407, trtllm:550, vllm:1329 | 461 / 462 | 2 | 82.2 | 116.6 | 151.5 |
| h200_sxm | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:4004, trtllm:797 | 785 / 4016 | 0 | 44.5 | 130.0 | 145.6 |
| h200_sxm | `chunk_gated_delta_rule_flashinfer` | shared | trtllm, vllm | trtllm:4584, vllm:13550 | 4584 / 4703 | 1 | 18.1 | 60.7 | 67.5 |
| h200_sxm | `flashinfer_chunk_gated_delta_rule` | shared | sglang | sglang:4529 | 0 / 4529 | 0 | — | — | — |
| h200_sxm | `fused_qkvzba_causal_conv1d_update_contiguous` | shared | sglang | sglang:462 | 0 / 462 | 0 | — | — | — |
| h200_sxm | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| h200_sxm | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:862, vllm:1319 | 457 / 457 | 5 | 0.7 | 3.1 | 6.6 |
| h200_sxm | `fused_sigmoid_gating_delta_rule_update_flashinfer` | shared | trtllm | trtllm:462 | 0 / 462 | 0 | — | — | — |
| l40s | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:8533, trtllm:5186, vllm:8364 | 4511 / 4638 | 74 | 100.5 | 197.6 | 275.8 |
| l40s | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:407, trtllm:550, vllm:868 | 461 / 462 | 3 | 109.5 | 151.3 | 181.4 |
| l40s | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:8533, trtllm:5186 | 4400 / 4529 | 2 | 7.5 | 33.9 | 65.3 |
| l40s | `chunk_gated_delta_rule_triton` | shared | vllm | vllm:8330 | 0 / 4395 | 0 | — | — | — |
| l40s | `fused_qkvzba_causal_conv1d_update_contiguous` | shared | sglang | sglang:462 | 0 / 462 | 0 | — | — | — |
| l40s | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| l40s | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:862, vllm:862 | 457 / 457 | 3 | 0.4 | 3.0 | 24.3 |
| l40s | `fused_sigmoid_gating_delta_rule_update_triton` | shared | trtllm | trtllm:462 | 0 / 462 | 0 | — | — | — |
| rtx_pro_6000_server | `causal_conv1d_fn` | shared | sglang, trtllm, vllm | sglang:8533, trtllm:5353, vllm:8653 | 4529 / 4701 | 29 | 86.9 | 199.3 | 274.0 |
| rtx_pro_6000_server | `causal_conv1d_update` | shared | sglang, trtllm, vllm | sglang:407, trtllm:550, vllm:868 | 461 / 462 | 1 | 70.8 | 94.0 | 146.9 |
| rtx_pro_6000_server | `chunk_gated_delta_rule` | shared | sglang, trtllm | sglang:8533, trtllm:824 | 824 / 4529 | 4 | 21.2 | 87.6 | 99.6 |
| rtx_pro_6000_server | `chunk_gated_delta_rule_flashinfer` | shared | trtllm, vllm | trtllm:4529, vllm:4701 | 4529 / 4701 | 0 | 21.8 | 53.8 | 103.1 |
| rtx_pro_6000_server | `chunk_gated_delta_rule_triton` | shared | vllm | vllm:3935 | 0 / 3935 | 0 | — | — | — |
| rtx_pro_6000_server | `fused_qkvzba_causal_conv1d_update_contiguous` | shared | sglang | sglang:462 | 0 / 462 | 0 | — | — | — |
| rtx_pro_6000_server | `fused_recurrent_gated_delta_rule` | shared | trtllm | trtllm:86 | 0 / 86 | 0 | — | — | — |
| rtx_pro_6000_server | `fused_recurrent_gated_delta_rule_packed_decode` | shared | sglang, vllm | sglang:862, vllm:862 | 457 / 457 | 0 | 0.3 | 5.7 | 24.9 |
| rtx_pro_6000_server | `fused_sigmoid_gating_delta_rule_update_triton` | shared | trtllm | trtllm:462 | 0 / 462 | 0 | — | — | — |

## `gemm_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `sglang` | shared | sglang | sglang:35742 | 0 / 35742 | 0 | — | — | — |
| a100_sxm | `torch_flow` | shared | trtllm | trtllm:9240 | 0 / 9240 | 0 | — | — | — |
| a100_sxm | `trt_flow_/smooth_quant_gemm_L96/PLUGIN_V2_SmoothQuantGemm_0` | shared | trtllm | trtllm:6048 | 0 / 6048 | 0 | — | — | — |
| a100_sxm | `trt_flow_/weight_only_quant_matmul_L257/PLUGIN_V2_WeightOnlyQuantMatmul_0` | shared | trtllm | trtllm:12096 | 0 / 12096 | 0 | — | — | — |
| a100_sxm | `vllm_default` | shared | vllm | vllm:9240 | 0 / 9240 | 0 | — | — | — |
| b200_sxm | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:146668 | 0 / 37518 | 222 | — | — | — |
| b200_sxm | `CutlassFp8BlockScaledMMKernel` | shared | vllm | vllm:15466 | 0 / 5328 | 14 | — | — | — |
| b200_sxm | `DeepGemmFp8BlockScaledMMKernel` | shared | vllm | vllm:93166 | 0 / 31080 | 38 | — | — | — |
| b200_sxm | `FlashInferCuteDslNvFp4LinearKernel` | shared | vllm | vllm:146668 | 0 / 37518 | 695 | — | — | — |
| b200_sxm | `deepgemm` | shared | trtllm | trtllm:58610 | 0 / 58610 | 0 | — | — | — |
| b200_sxm | `sglang.srt.layers.vocab_parallel_embedding.ParallelLMHead.quant_method.apply/sglang.srt.layers.quantization.unquant.UnquantizedEmbeddingMethod` | shared | sglang | sglang:30 | 0 / 30 | 0 | — | — | — |
| b200_sxm | `sglang.srt.models.deepseek_v2.MoEGate.forward` | shared | sglang | sglang:15 | 0 / 15 | 0 | — | — | — |
| b200_sxm | `sglang_deepgemm_gemm_nt_f8f8bf16` | shared | sglang | sglang:88652 | 0 / 29600 | 36 | — | — | — |
| b200_sxm | `sglang_flashinfer_cutedsl_nvfp4` | shared | sglang | sglang:88652 | 0 / 29600 | 70 | — | — | — |
| b200_sxm | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:108410 | 0 / 36852 | 175 | — | — | — |
| b200_sxm | `sglang_torch_linear` | shared | sglang | sglang:109150 | 0 / 37518 | 190 | — | — | — |
| b200_sxm | `torch.nn.functional.linear` | shared | vllm | vllm:146668 | 0 / 37518 | 489 | — | — | — |
| b200_sxm | `torch_flow` | shared | trtllm | trtllm:204292 | 0 / 204292 | 0 | — | — | — |
| b300_sxm | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:183298 | 0 / 38258 | 298 | — | — | — |
| b300_sxm | `CutlassFp8BlockScaledMMKernel` | shared | vllm | vllm:20572 | 0 / 5624 | 28 | — | — | — |
| b300_sxm | `DeepGemmFp8BlockScaledMMKernel` | shared | vllm | vllm:124542 | 0 / 31450 | 213 | — | — | — |
| b300_sxm | `FlashInferCuteDslNvFp4LinearKernel` | shared | vllm | vllm:183298 | 0 / 38258 | 929 | — | — | — |
| b300_sxm | `deepgemm` | shared | trtllm | trtllm:58610 | 0 / 29342 | 23 | — | — | — |
| b300_sxm | `sglang_cutedsl_tgv_bf16` | shared | sglang | sglang:1317 | 0 / 1317 | 0 | — | — | — |
| b300_sxm | `sglang_deepgemm_gemm_nt_f8f8bf16` | shared | sglang | sglang:120028 | 0 / 31450 | 64 | — | — | — |
| b300_sxm | `sglang_flashinfer_bf16_direct` | shared | sglang | sglang:8 | 0 / 8 | 0 | — | — | — |
| b300_sxm | `sglang_flashinfer_bf16_splitk` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| b300_sxm | `sglang_flashinfer_cutedsl_nvfp4` | shared | sglang | sglang:125282 | 0 / 36704 | 122 | — | — | — |
| b300_sxm | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:108558 | 0 / 36852 | 73 | — | — | — |
| b300_sxm | `sglang_torch_linear` | shared | sglang | sglang:144642 | 0 / 38215 | 330 | — | — | — |
| b300_sxm | `sglang_triton_w8a8_block_fp8` | shared | sglang | sglang:1998 | 0 / 1998 | 0 | — | — | — |
| b300_sxm | `torch.nn.functional.linear` | shared | vllm | vllm:183298 | 0 / 38258 | 848 | — | — | — |
| b300_sxm | `torch_flow` | shared | trtllm | trtllm:204292 | 0 / 103404 | 72 | — | — | — |
| b60 | `vllm_default` | shared | vllm | vllm:61698 | 0 / 23058 | 1 | — | — | — |
| gb200 | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:183298 | 0 / 38258 | 357 | — | — | — |
| gb200 | `CutlassFp8BlockScaledMMKernel` | shared | vllm | vllm:20572 | 0 / 5624 | 46 | — | — | — |
| gb200 | `DeepGemmFp8BlockScaledMMKernel` | shared | vllm | vllm:124542 | 0 / 31450 | 173 | — | — | — |
| gb200 | `FlashInferCuteDslNvFp4LinearKernel` | shared | vllm | vllm:183298 | 0 / 38258 | 812 | — | — | — |
| gb200 | `deepgemm` | shared | trtllm | trtllm:58610 | 0 / 29342 | 18 | — | — | — |
| gb200 | `sglang.srt.layers.vocab_parallel_embedding.ParallelLMHead.quant_method.apply/sglang.srt.layers.quantization.unquant.UnquantizedEmbeddingMethod` | shared | sglang | sglang:30 | 0 / 30 | 0 | — | — | — |
| gb200 | `sglang.srt.models.deepseek_v2.MoEGate.forward` | shared | sglang | sglang:15 | 0 / 15 | 0 | — | — | — |
| gb200 | `sglang_cutedsl_tgv_bf16` | shared | sglang | sglang:1317 | 0 / 1317 | 0 | — | — | — |
| gb200 | `sglang_deepgemm_gemm_nt_f8f8bf16` | shared | sglang | sglang:120028 | 0 / 31450 | 78 | — | — | — |
| gb200 | `sglang_flashinfer_bf16_direct` | shared | sglang | sglang:8 | 0 / 8 | 0 | — | — | — |
| gb200 | `sglang_flashinfer_bf16_splitk` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| gb200 | `sglang_flashinfer_cutedsl_nvfp4` | shared | sglang | sglang:125282 | 0 / 36704 | 134 | — | — | — |
| gb200 | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:108410 | 0 / 36852 | 56 | — | — | — |
| gb200 | `sglang_torch_linear` | shared | sglang | sglang:144420 | 0 / 38215 | 347 | — | — | — |
| gb200 | `sglang_triton_w8a8_block_fp8` | shared | sglang | sglang:1998 | 0 / 1998 | 0 | — | — | — |
| gb200 | `torch.nn.functional.linear` | shared | vllm | vllm:183298 | 0 / 38258 | 850 | — | — | — |
| gb200 | `torch_flow` | shared | trtllm | trtllm:204292 | 0 / 103404 | 108 | — | — | — |
| gb300 | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:183150 | 0 / 38184 | 437 | — | — | — |
| gb300 | `CutlassFp8BlockScaledMMKernel` | shared | vllm | vllm:20572 | 0 / 5624 | 41 | — | — | — |
| gb300 | `DeepGemmFp8BlockScaledMMKernel` | shared | vllm | vllm:124542 | 0 / 31450 | 118 | — | — | — |
| gb300 | `FlashInferCuteDslNvFp4LinearKernel` | shared | vllm | vllm:183150 | 0 / 38184 | 723 | — | — | — |
| gb300 | `deepgemm` | shared | trtllm | trtllm:58610 | 0 / 29342 | 16 | — | — | — |
| gb300 | `sglang_cutedsl_tgv_bf16` | shared | sglang | sglang:1295 | 0 / 1295 | 0 | — | — | — |
| gb300 | `sglang_deepgemm_gemm_nt_f8f8bf16` | shared | sglang | sglang:120028 | 0 / 31450 | 67 | — | — | — |
| gb300 | `sglang_flashinfer_bf16_direct` | shared | sglang | sglang:8 | 0 / 8 | 0 | — | — | — |
| gb300 | `sglang_flashinfer_bf16_splitk` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| gb300 | `sglang_flashinfer_cutedsl_nvfp4` | shared | sglang | sglang:125134 | 0 / 36556 | 145 | — | — | — |
| gb300 | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:144892 | 0 / 37518 | 126 | — | — | — |
| gb300 | `sglang_torch_linear` | shared | sglang | sglang:144294 | 0 / 38163 | 428 | — | — | — |
| gb300 | `sglang_triton_w8a8_block_fp8` | shared | sglang | sglang:1850 | 0 / 1850 | 0 | — | — | — |
| gb300 | `torch.nn.functional.linear` | shared | vllm | vllm:183150 | 0 / 38184 | 888 | — | — | — |
| gb300 | `torch_flow` | shared | trtllm | trtllm:204292 | 0 / 103404 | 97 | — | — | — |
| h100_sxm | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:147407 | 0 / 38258 | 97 | — | — | — |
| h100_sxm | `CutlassFp8BlockScaledMMKernel` | shared | vllm | vllm:15762 | 0 / 5624 | 25 | — | — | — |
| h100_sxm | `DeepGemmFp8BlockScaledMMKernel` | shared | vllm | vllm:72018 | 0 / 24225 | 90 | — | — | — |
| h100_sxm | `FlashInferFp8BlockScaledMMKernel` | shared | vllm | vllm:21484 | 0 / 7225 | 60 | — | — | — |
| h100_sxm | `sglang` | shared | sglang | sglang:35280 | 0 / 35280 | 0 | — | — | — |
| h100_sxm | `sglang.srt.layers.vocab_parallel_embedding.ParallelLMHead.quant_method.apply/sglang.srt.layers.quantization.unquant.UnquantizedEmbeddingMethod` | shared | sglang | sglang:30 | 0 / 30 | 0 | — | — | — |
| h100_sxm | `sglang.srt.models.deepseek_v2.MoEGate.forward` | shared | sglang | sglang:15 | 0 / 15 | 0 | — | — | — |
| h100_sxm | `sglang_deepgemm_gemm_nt_f8f8bf16` | shared | sglang | sglang:90501 | 0 / 31450 | 43 | — | — | — |
| h100_sxm | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:72594 | 0 / 36852 | 8 | — | — | — |
| h100_sxm | `sglang_torch_linear` | shared | sglang | sglang:109878 | 0 / 38258 | 42 | — | — | — |
| h100_sxm | `sglang_triton_w8a8_block_fp8` | shared | sglang | sglang:1998 | 0 / 1998 | 0 | — | — | — |
| h100_sxm | `torch.nn.functional.linear` | shared | vllm | vllm:147408 | 0 / 38258 | 147 | — | — | — |
| h100_sxm | `torch_flow` | shared | trtllm | trtllm:203852 | 0 / 103184 | 30 | — | — | — |
| h200_sxm | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:147400 | 0 / 38258 | 193 | — | — | — |
| h200_sxm | `CutlassFp8BlockScaledMMKernel` | shared | vllm | vllm:15762 | 0 / 5624 | 44 | — | — | — |
| h200_sxm | `DeepGemmFp8BlockScaledMMKernel` | shared | vllm | vllm:72048 | 0 / 24225 | 128 | — | — | — |
| h200_sxm | `FlashInferFp8BlockScaledMMKernel` | shared | vllm | vllm:21488 | 0 / 7225 | 81 | — | — | — |
| h200_sxm | `sglang.srt.layers.vocab_parallel_embedding.ParallelLMHead.quant_method.apply/sglang.srt.layers.quantization.unquant.UnquantizedEmbeddingMethod` | shared | sglang | sglang:30 | 0 / 30 | 0 | — | — | — |
| h200_sxm | `sglang.srt.models.deepseek_v2.MoEGate.forward` | shared | sglang | sglang:15 | 0 / 15 | 0 | — | — | — |
| h200_sxm | `sglang_deepgemm_gemm_nt_f8f8bf16` | shared | sglang | sglang:90502 | 0 / 31450 | 36 | — | — | — |
| h200_sxm | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:72594 | 0 / 36852 | 4 | — | — | — |
| h200_sxm | `sglang_torch_linear` | shared | sglang | sglang:109890 | 0 / 38258 | 53 | — | — | — |
| h200_sxm | `sglang_triton_w8a8_block_fp8` | shared | sglang | sglang:1998 | 0 / 1998 | 0 | — | — | — |
| h200_sxm | `torch.nn.functional.linear` | shared | vllm | vllm:147408 | 0 / 38258 | 266 | — | — | — |
| h200_sxm | `torch_flow` | shared | trtllm | trtllm:203852 | 0 / 103184 | 49 | — | — | — |
| l40s | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:73217 | 0 / 37516 | 34 | — | — | — |
| l40s | `MarlinFP8ScaledMMLinearKernel` | shared | vllm | vllm:37518 | 0 / 37518 | 0 | — | — | — |
| l40s | `TritonFp8BlockScaledMMKernel` | shared | vllm | vllm:35676 | 0 / 35676 | 0 | — | — | — |
| l40s | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:72594 | 0 / 36852 | 469 | — | — | — |
| l40s | `sglang_torch_linear` | shared | sglang | sglang:73260 | 0 / 37518 | 371 | — | — | — |
| l40s | `torch.nn.functional.linear` | shared | vllm | vllm:73168 | 0 / 37505 | 1491 | — | — | — |
| l40s | `torch_flow` | shared | trtllm | trtllm:203964 | 0 / 103304 | 917 | — | — | — |
| rtx_pro_6000_server | `CutlassFP8ScaledMMLinearKernel` | shared | vllm | vllm:73260 | 0 / 37518 | 39 | — | — | — |
| rtx_pro_6000_server | `CutlassFp8BlockScaledMMKernel` | shared | vllm | vllm:1036 | 0 / 518 | 0 | — | — | — |
| rtx_pro_6000_server | `DeepGemmFp8BlockScaledMMKernel` | shared | vllm | vllm:31080 | 0 / 31080 | 0 | — | — | — |
| rtx_pro_6000_server | `FlashInferCutlassNvFp4LinearKernel` | shared | vllm | vllm:73260 | 0 / 37518 | 123 | — | — | — |
| rtx_pro_6000_server | `sglang_flashinfer_cutlass_nvfp4` | shared | sglang | sglang:59126 | 0 / 29600 | 77 | — | — | — |
| rtx_pro_6000_server | `sglang_sgl_kernel_fp8_scaled_mm` | shared | sglang | sglang:72594 | 0 / 36852 | 113 | — | — | — |
| rtx_pro_6000_server | `sglang_torch_linear` | shared | sglang | sglang:73260 | 0 / 37518 | 148 | — | — | — |
| rtx_pro_6000_server | `torch.nn.functional.linear` | shared | vllm | vllm:73260 | 0 / 37518 | 103 | — | — | — |
| rtx_pro_6000_server | `torch_flow` | shared | trtllm | trtllm:263014 | 0 / 132866 | 100 | — | — | — |

## `generation_attention_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `flash_attention` | shared | sglang | sglang:5093 | 0 / 5093 | 0 | — | — | — |
| a100_sxm | `torch_flow` | shared | trtllm | trtllm:5026 | 0 / 5026 | 0 | — | — | — |
| a100_sxm | `vllm_flash_attn` | shared | vllm | vllm:5431 | 0 / 5431 | 0 | — | — | — |
| b200_sxm | `flashinfer` | shared | sglang | sglang:7140 | 0 / 3570 | 11 | — | — | — |
| b200_sxm | `torch_flow` | shared | trtllm | trtllm:98310 | 0 / 98310 | 0 | — | — | — |
| b200_sxm | `torch_flow_flashinfer` | shared | trtllm | trtllm:3540 | 0 / 3540 | 0 | — | — | — |
| b200_sxm | `triton` | shared | sglang | sglang:17240 | 0 / 8620 | 19 | — | — | — |
| b200_sxm | `trtllm_mha` | shared | sglang | sglang:76995 | 0 / 39358 | 78 | — | — | — |
| b200_sxm | `vllm_flash_attn_fa2` | shared | vllm | vllm:2224 | 0 / 2224 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer` | shared | vllm | vllm:36240 | 0 / 36240 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer_flashinfertrtllmapidecode` | shared | vllm | vllm:63180 | 0 / 63180 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer_flashinfertrtllmapidecode_trtllm_gen` | shared | vllm | vllm:62252 | 0 / 62252 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer_trtllmdecode` | shared | vllm | vllm:60916 | 0 / 60916 | 0 | — | — | — |
| b200_sxm | `vllm_triton_attn` | shared | vllm | vllm:9835 | 0 / 5384 | 30 | — | — | — |
| b300_sxm | `flashinfer` | shared | sglang | sglang:7140 | 0 / 3570 | 16 | — | — | — |
| b300_sxm | `torch_flow` | shared | trtllm | trtllm:96976 | 0 / 49996 | 75 | — | — | — |
| b300_sxm | `torch_flow_flashinfer` | shared | trtllm | trtllm:3540 | 0 / 3540 | 0 | — | — | — |
| b300_sxm | `triton` | shared | sglang | sglang:17240 | 0 / 8620 | 18 | — | — | — |
| b300_sxm | `trtllm_mha` | shared | sglang | sglang:76840 | 0 / 39360 | 68 | — | — | — |
| b300_sxm | `vllm_flash_attn_fa2` | shared | vllm | vllm:2227 | 0 / 2227 | 0 | — | — | — |
| b300_sxm | `vllm_flashinfer_flashinfertrtllmapidecode` | shared | vllm | vllm:63180 | 0 / 63180 | 0 | — | — | — |
| b300_sxm | `vllm_flashinfer_flashinfertrtllmapidecode_trtllm_gen` | shared | vllm | vllm:62252 | 0 / 62252 | 0 | — | — | — |
| b300_sxm | `vllm_flashinfer_trtllmdecode` | shared | vllm | vllm:58510 | 0 / 58510 | 0 | — | — | — |
| b300_sxm | `vllm_triton_attn` | shared | vllm | vllm:7600 | 0 / 3156 | 14 | — | — | — |
| b60 | `vllm_flash_attn` | shared | vllm | vllm:54675 | 0 / 31096 | 0 | — | — | — |
| gb200 | `flashinfer` | shared | sglang | sglang:7140 | 0 / 3570 | 4 | — | — | — |
| gb200 | `torch_flow` | shared | trtllm | trtllm:98312 | 0 / 49996 | 123 | — | — | — |
| gb200 | `torch_flow_flashinfer` | shared | trtllm | trtllm:7688 | 0 / 4148 | 7 | — | — | — |
| gb200 | `triton` | shared | sglang | sglang:17240 | 0 / 8620 | 10 | — | — | — |
| gb200 | `trtllm_mha` | shared | sglang | sglang:75434 | 0 / 39360 | 54 | — | — | — |
| gb200 | `vllm_flash_attn_fa2` | shared | vllm | vllm:2225 | 0 / 2225 | 0 | — | — | — |
| gb200 | `vllm_flashinfer_flashinfertrtllmapidecode` | shared | vllm | vllm:63180 | 0 / 63180 | 0 | — | — | — |
| gb200 | `vllm_flashinfer_flashinfertrtllmapidecode_trtllm_gen` | shared | vllm | vllm:62252 | 0 / 62252 | 0 | — | — | — |
| gb200 | `vllm_flashinfer_trtllmdecode` | shared | vllm | vllm:59844 | 0 / 59844 | 0 | — | — | — |
| gb200 | `vllm_triton_attn` | shared | vllm | vllm:7601 | 0 / 3156 | 15 | — | — | — |
| gb300 | `flashinfer` | shared | sglang | sglang:7140 | 0 / 3570 | 4 | — | — | — |
| gb300 | `torch_flow` | shared | trtllm | trtllm:98312 | 0 / 49996 | 88 | — | — | — |
| gb300 | `torch_flow_flashinfer` | shared | trtllm | trtllm:7688 | 0 / 4148 | 12 | — | — | — |
| gb300 | `triton` | shared | sglang | sglang:17240 | 0 / 8620 | 8 | — | — | — |
| gb300 | `trtllm_mha` | shared | sglang | sglang:75428 | 0 / 39354 | 58 | — | — | — |
| gb300 | `vllm_flash_attn_fa2` | shared | vllm | vllm:2226 | 0 / 2226 | 0 | — | — | — |
| gb300 | `vllm_flashinfer_flashinfertrtllmapidecode` | shared | vllm | vllm:63180 | 0 / 63180 | 0 | — | — | — |
| gb300 | `vllm_flashinfer_flashinfertrtllmapidecode_trtllm_gen` | shared | vllm | vllm:62252 | 0 / 62252 | 0 | — | — | — |
| gb300 | `vllm_flashinfer_trtllmdecode` | shared | vllm | vllm:59844 | 0 / 59844 | 0 | — | — | — |
| gb300 | `vllm_triton_attn` | shared | vllm | vllm:7603 | 0 / 3156 | 11 | — | — | — |
| h100_sxm | `fa3` | shared | sglang | sglang:78420 | 0 / 41244 | 20 | — | — | — |
| h100_sxm | `torch_flow` | shared | trtllm | trtllm:107080 | 0 / 55047 | 170 | — | — | — |
| h100_sxm | `triton` | shared | sglang | sglang:7120 | 0 / 3828 | 3 | — | — | — |
| h100_sxm | `vllm_flash_attn_fa3` | shared | vllm | vllm:192926 | 0 / 65780 | 416 | — | — | — |
| h100_sxm | `vllm_flash_attn_fa4` | shared | vllm | vllm:2656 | 0 / 928 | 1 | — | — | — |
| h100_sxm | `vllm_triton_attn` | shared | vllm | vllm:928 | 0 / 928 | 0 | — | — | — |
| h200_sxm | `fa3` | shared | sglang | sglang:79754 | 0 / 41244 | 29 | — | — | — |
| h200_sxm | `torch_flow` | shared | trtllm | trtllm:108414 | 0 / 55047 | 219 | — | — | — |
| h200_sxm | `triton` | shared | sglang | sglang:7120 | 0 / 3828 | 7 | — | — | — |
| h200_sxm | `vllm_flash_attn_fa3` | shared | vllm | vllm:194260 | 0 / 65780 | 1415 | — | — | — |
| h200_sxm | `vllm_flash_attn_fa4` | shared | vllm | vllm:2656 | 0 / 928 | 5 | — | — | — |
| h200_sxm | `vllm_triton_attn` | shared | vllm | vllm:928 | 0 / 928 | 0 | — | — | — |
| l40s | `flashinfer` | shared | sglang | sglang:57836 | 0 / 33226 | 624 | — | — | — |
| l40s | `torch_flow` | shared | trtllm | trtllm:108442 | 0 / 55728 | 1314 | — | — | — |
| l40s | `triton` | shared | sglang | sglang:25108 | 0 / 15724 | 290 | — | — | — |
| l40s | `vllm_flash_attn_fa2` | shared | vllm | vllm:63551 | 0 / 32887 | 1907 | — | — | — |
| l40s | `vllm_flashinfer_fidecode` | shared | vllm | vllm:58928 | 0 / 31312 | 1428 | — | — | — |
| l40s | `vllm_triton_attn` | shared | vllm | vllm:5384 | 0 / 3156 | 122 | — | — | — |
| rtx_pro_6000_server | `flashinfer` | shared | sglang | sglang:53808 | 0 / 29114 | 77 | — | — | — |
| rtx_pro_6000_server | `torch_flow` | shared | trtllm | trtllm:106394 | 0 / 54704 | 160 | — | — | — |
| rtx_pro_6000_server | `triton` | shared | sglang | sglang:30912 | 0 / 15724 | 45 | — | — | — |
| rtx_pro_6000_server | `vllm_flash_attn_fa2` | shared | vllm | vllm:63543 | 0 / 32888 | 81 | — | — | — |
| rtx_pro_6000_server | `vllm_flashinfer_fidecode` | shared | vllm | vllm:28324 | 0 / 28324 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_flashinfer_flashinfertrtllmapidecode_xqa` | shared | vllm | vllm:29214 | 0 / 29214 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_triton_attn` | shared | vllm | vllm:5376 | 0 / 3153 | 8 | — | — | — |

## `generation_mla_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `default` | shared_fallback | trtllm | trtllm:5068 | 0 / 5068 | 0 | — | — | — |
| b200_sxm | `default` | shared_fallback | trtllm | trtllm:7602 | 0 / 7602 | 0 | — | — | — |
| b200_sxm | `trtllm_mla` | shared | sglang | sglang:7602 | 0 / 6154 | 2 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:7602 | 0 / 4706 | 12 | — | — | — |
| b300_sxm | `trtllm_mla` | shared | sglang | sglang:10498 | 0 / 6154 | 1 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:7602 | 0 / 4706 | 8 | — | — | — |
| gb200 | `trtllm_mla` | shared | sglang | sglang:10498 | 0 / 6154 | 4 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:7602 | 0 / 4706 | 9 | — | — | — |
| gb300 | `trtllm_mla` | shared | sglang | sglang:10498 | 0 / 6154 | 2 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:7964 | 0 / 5068 | 16 | — | — | — |
| h100_sxm | `flash_attention` | shared | sglang | sglang:10860 | 0 / 5068 | 4 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:7964 | 0 / 5068 | 12 | — | — | — |
| h200_sxm | `flash_attention` | shared | sglang | sglang:7964 | 0 / 5068 | 1 | — | — | — |
| l40s | `default` | shared_fallback | trtllm | trtllm:5068 | 0 / 5068 | 0 | — | — | — |
| l40s | `triton` | shared | sglang | sglang:3470 | 0 / 2308 | 10 | — | — | — |
| rtx_pro_6000_server | `default` | shared_fallback | trtllm | trtllm:7964 | 0 / 5068 | 5 | — | — | — |
| rtx_pro_6000_server | `triton` | shared | sglang | sglang:4928 | 0 / 2528 | 5 | — | — | — |

## `glm53_attention_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b300_sxm | `sglang0.5.20|DeepseekV2AttentionMLA|IndexerKPool|qkv_a=Fp8LinearMethod|q_b=Fp8LinearMethod|kv_b=UnquantizedLinearMethod|o=Fp8LinearMethod|attn_backend=dsa|method=MLA` | shared | sglang | sglang:926 | 0 / 926 | 0 | — | — | — |
| b300_sxm | `sglang0.5.20|DeepseekV2AttentionMLA|IndexerKPool|qkv_a=UnquantizedLinearMethod|q_b=UnquantizedLinearMethod|kv_b=UnquantizedLinearMethod|o=UnquantizedLinearMethod|attn_backend=dsa|method=MLA` | shared | sglang | sglang:926 | 0 / 926 | 0 | — | — | — |
| b300_sxm | `vllm0.31.0|Glm5NextMLAAttention|SparseAttnIndexerKpool|projections=UnquantizedLinearMethod|mla_impl=FlashInferMLASparseImpl` | shared | vllm | vllm:2315 | 0 / 2315 | 0 | — | — | — |
| gb200 | `sglang0.5.20|DeepseekV2AttentionMLA|IndexerKPool|qkv_a=Fp8LinearMethod|q_b=Fp8LinearMethod|kv_b=UnquantizedLinearMethod|o=Fp8LinearMethod|attn_backend=dsa|method=MLA` | shared | sglang | sglang:463 | 0 / 463 | 0 | — | — | — |
| gb200 | `sglang0.5.20|DeepseekV2AttentionMLA|IndexerKPool|qkv_a=UnquantizedLinearMethod|q_b=UnquantizedLinearMethod|kv_b=UnquantizedLinearMethod|o=UnquantizedLinearMethod|attn_backend=dsa|method=MLA` | shared | sglang | sglang:925 | 0 / 925 | 0 | — | — | — |
| gb200 | `vllm0.31.0|Glm5NextMLAAttention|SparseAttnIndexerKpool|projections=UnquantizedLinearMethod|mla_impl=FlashInferMLASparseImpl` | shared | vllm | vllm:1389 | 0 / 1389 | 0 | — | — | — |
| gb300 | `sglang0.5.20|DeepseekV2AttentionMLA|IndexerKPool|qkv_a=Fp8LinearMethod|q_b=Fp8LinearMethod|kv_b=UnquantizedLinearMethod|o=Fp8LinearMethod|attn_backend=dsa|method=MLA` | shared | sglang | sglang:926 | 0 / 926 | 0 | — | — | — |
| gb300 | `sglang0.5.20|DeepseekV2AttentionMLA|IndexerKPool|qkv_a=UnquantizedLinearMethod|q_b=UnquantizedLinearMethod|kv_b=UnquantizedLinearMethod|o=UnquantizedLinearMethod|attn_backend=dsa|method=MLA` | shared | sglang | sglang:1366 | 0 / 1366 | 0 | — | — | — |
| gb300 | `vllm0.31.0|Glm5NextMLAAttention|SparseAttnIndexerKpool|projections=UnquantizedLinearMethod|mla_impl=FlashInferMLASparseImpl` | shared | vllm | vllm:2315 | 0 / 2315 | 0 | — | — | — |
| h100_sxm | `sglang0.5.20|DeepseekV2AttentionMLA|IndexerKPool|qkv_a=Fp8LinearMethod|q_b=Fp8LinearMethod|kv_b=UnquantizedLinearMethod|o=Fp8LinearMethod|attn_backend=dsa|method=MLA` | shared | sglang | sglang:433 | 0 / 433 | 0 | — | — | — |
| h100_sxm | `vllm0.31.0|Glm5NextMLAAttention|SparseAttnIndexerKpool|projections=UnquantizedLinearMethod|mla_impl=FlashInferMLASparseSM90Impl` | shared | vllm | vllm:449 | 0 / 449 | 0 | — | — | — |
| h200_sxm | `sglang0.5.20|DeepseekV2AttentionMLA|IndexerKPool|qkv_a=Fp8LinearMethod|q_b=Fp8LinearMethod|kv_b=UnquantizedLinearMethod|o=Fp8LinearMethod|attn_backend=dsa|method=MLA` | shared | sglang | sglang:911 | 0 / 911 | 0 | — | — | — |
| h200_sxm | `vllm0.31.0|Glm5NextMLAAttention|SparseAttnIndexerKpool|projections=UnquantizedLinearMethod|mla_impl=FlashInferMLASparseSM90Impl` | shared | vllm | vllm:925 | 0 / 925 | 0 | — | — | — |

## `glm5_dsa_attn_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `flash_mla_sparse_fwd` | shared | sglang | sglang:6665 | 0 / 3333 | 0 | — | — | — |
| b300_sxm | `flash_mla_sparse_fwd` | shared | sglang | sglang:6910 | 0 / 3333 | 0 | — | — | — |
| gb200 | `flash_mla_sparse_fwd` | shared | sglang | sglang:6665 | 0 / 3333 | 0 | — | — | — |
| gb300 | `flash_mla_sparse_fwd` | shared | sglang | sglang:6883 | 0 / 3333 | 0 | — | — | — |
| h100_sxm | `flash_mla_sparse_fwd` | shared | sglang | sglang:6051 | 0 / 3026 | 0 | — | — | — |
| h200_sxm | `flash_mla_sparse_fwd` | shared | sglang | sglang:6051 | 0 / 3026 | 0 | — | — | — |

## `glm5_mqa_logits_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `deep_gemm.fp8_mqa_logits` | shared | sglang | sglang:6665 | 0 / 3333 | 0 | — | — | — |
| b300_sxm | `deep_gemm.fp8_mqa_logits` | shared | sglang | sglang:6665 | 0 / 3333 | 0 | — | — | — |
| gb200 | `deep_gemm.fp8_mqa_logits` | shared | sglang | sglang:6665 | 0 / 3333 | 0 | — | — | — |
| gb300 | `deep_gemm.fp8_mqa_logits` | shared | sglang | sglang:6665 | 0 / 3333 | 0 | — | — | — |
| h100_sxm | `deep_gemm.fp8_mqa_logits` | shared | sglang | sglang:6051 | 0 / 3026 | 0 | — | — | — |
| h200_sxm | `deep_gemm.fp8_mqa_logits` | shared | sglang | sglang:6051 | 0 / 3026 | 0 | — | — | — |

## `glm5_topk_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `fast_topk_transform_fused` | shared | sglang | sglang:8442 | 0 / 4222 | 0 | — | — | — |
| b300_sxm | `fast_topk_transform_fused` | shared | sglang | sglang:8442 | 0 / 4222 | 0 | — | — | — |
| gb200 | `fast_topk_transform_fused` | shared | sglang | sglang:8442 | 0 / 4222 | 0 | — | — | — |
| gb300 | `fast_topk_transform_fused` | shared | sglang | sglang:8442 | 0 / 4222 | 0 | — | — | — |
| h100_sxm | `fast_topk_transform_fused` | shared | sglang | sglang:7214 | 0 / 3608 | 0 | — | — | — |
| h200_sxm | `fast_topk_transform_fused` | shared | sglang | sglang:7214 | 0 / 3608 | 0 | — | — | — |

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
| b300_sxm | `causal_conv1d_fn` | shared | sglang, vllm | sglang:448, vllm:420 | 420 / 448 | 0 | 19.6 | 122.7 | 142.1 |
| b300_sxm | `causal_conv1d_fn_qkv3` | shared | sglang, vllm | sglang:396, vllm:428 | 396 / 428 | 0 | 184.0 | 193.6 | 199.4 |
| b300_sxm | `causal_conv1d_update` | shared | sglang, vllm | sglang:77, vllm:224 | 77 / 224 | 0 | 6.0 | 52.5 | 57.5 |
| b300_sxm | `chunk_kda` | shared | sglang | sglang:824 | 0 / 824 | 0 | — | — | — |
| b300_sxm | `flashkda_fwd` | shared | vllm | vllm:847 | 0 / 847 | 0 | — | — | — |
| b300_sxm | `fused_kda_decode` | shared | vllm | vllm:44 | 0 / 44 | 0 | — | — | — |
| b300_sxm | `fused_kda_decode_mtp_dspark` | shared | sglang | sglang:107 | 0 / 107 | 0 | — | — | — |
| b300_sxm | `fused_recurrent_kda` | shared | vllm | vllm:179 | 0 / 179 | 0 | — | — | — |
| b300_sxm | `fused_recurrent_kda_packed_decode` | shared | sglang, vllm | sglang:33, vllm:43 | 32 / 44 | 0 | 12.7 | 62.7 | 69.6 |
| b300_sxm | `fused_sigmoid_gating_delta_rule_update` | shared | sglang | sglang:44 | 0 / 44 | 0 | — | — | — |
| b300_sxm | `kda_fused_decode` | shared | sglang | sglang:11 | 0 / 11 | 0 | — | — | — |
| gb200 | `causal_conv1d_fn` | shared | sglang, vllm | sglang:447, vllm:419 | 419 / 447 | 0 | 30.3 | 129.5 | 139.1 |
| gb200 | `causal_conv1d_fn_qkv3` | shared | sglang, vllm | sglang:396, vllm:428 | 396 / 428 | 0 | 193.3 | 197.1 | 197.6 |
| gb200 | `causal_conv1d_update` | shared | sglang, vllm | sglang:77, vllm:224 | 77 / 224 | 0 | 3.6 | 49.0 | 54.0 |
| gb200 | `chunk_kda` | shared | sglang | sglang:824 | 0 / 824 | 0 | — | — | — |
| gb200 | `flashkda_fwd` | shared | vllm | vllm:844 | 0 / 844 | 0 | — | — | — |
| gb200 | `fused_kda_decode` | shared | vllm | vllm:44 | 0 / 44 | 0 | — | — | — |
| gb200 | `fused_kda_decode_mtp_dspark` | shared | sglang | sglang:107 | 0 / 107 | 0 | — | — | — |
| gb200 | `fused_recurrent_kda` | shared | vllm | vllm:179 | 0 / 179 | 0 | — | — | — |
| gb200 | `fused_recurrent_kda_packed_decode` | shared | sglang, vllm | sglang:33, vllm:43 | 32 / 44 | 0 | 6.7 | 69.5 | 77.5 |
| gb200 | `fused_sigmoid_gating_delta_rule_update` | shared | sglang | sglang:44 | 0 / 44 | 0 | — | — | — |
| gb200 | `kda_fused_decode` | shared | sglang | sglang:11 | 0 / 11 | 0 | — | — | — |
| gb300 | `causal_conv1d_fn` | shared | sglang, vllm | sglang:232, vllm:327 | 218 / 341 | 0 | 30.3 | 127.7 | 140.0 |
| gb300 | `causal_conv1d_fn_qkv3` | shared | sglang, vllm | sglang:396, vllm:848 | 396 / 448 | 0 | 24.8 | 196.3 | 197.7 |
| gb300 | `causal_conv1d_update` | shared | sglang, vllm | sglang:55, vllm:358 | 55 / 206 | 0 | 9.5 | 41.5 | 52.2 |
| gb300 | `chunk_kda` | shared | sglang | sglang:619 | 0 / 619 | 0 | — | — | — |
| gb300 | `flashkda_fwd` | shared | vllm | vllm:1173 | 0 / 773 | 0 | — | — | — |
| gb300 | `fused_kda_decode` | shared | vllm | vllm:44 | 0 / 44 | 0 | — | — | — |
| gb300 | `fused_kda_decode_mtp_dspark` | shared | sglang | sglang:107 | 0 / 107 | 0 | — | — | — |
| gb300 | `fused_recurrent_kda` | shared | vllm | vllm:269 | 0 / 161 | 0 | — | — | — |
| gb300 | `fused_recurrent_kda_packed_decode` | shared | sglang, vllm | sglang:33, vllm:86 | 32 / 44 | 1 | 11.8 | 61.4 | 65.1 |
| gb300 | `fused_sigmoid_gating_delta_rule_update` | shared | sglang | sglang:22 | 0 / 22 | 0 | — | — | — |
| gb300 | `kda_fused_decode` | shared | sglang | sglang:11 | 0 / 11 | 0 | — | — | — |
| h100_sxm | `causal_conv1d_fn` | shared | sglang, vllm | sglang:444, vllm:416 | 416 / 444 | 0 | 18.2 | 122.7 | 176.4 |
| h100_sxm | `causal_conv1d_fn_qkv3` | shared | sglang, vllm | sglang:396, vllm:428 | 396 / 428 | 0 | 181.9 | 193.1 | 194.3 |
| h100_sxm | `causal_conv1d_update` | shared | sglang, vllm | sglang:185, vllm:224 | 185 / 224 | 0 | 13.4 | 57.9 | 75.9 |
| h100_sxm | `chunk_kda` | shared | sglang | sglang:824 | 0 / 824 | 0 | — | — | — |
| h100_sxm | `flashkda_fwd` | shared | vllm | vllm:838 | 0 / 838 | 0 | — | — | — |
| h100_sxm | `fused_kda_decode` | shared | vllm | vllm:44 | 0 / 44 | 0 | — | — | — |
| h100_sxm | `fused_recurrent_kda` | shared | vllm | vllm:179 | 0 / 179 | 0 | — | — | — |
| h100_sxm | `fused_recurrent_kda_packed_decode` | shared | sglang, vllm | sglang:33, vllm:43 | 32 / 44 | 0 | 16.7 | 66.6 | 72.7 |
| h100_sxm | `fused_sigmoid_gating_delta_rule_update` | shared | sglang | sglang:152 | 0 / 152 | 0 | — | — | — |
| h100_sxm | `kda_fused_decode` | shared | sglang | sglang:11 | 0 / 11 | 0 | — | — | — |
| h200_sxm | `causal_conv1d_fn` | shared | sglang, vllm | sglang:447, vllm:419 | 419 / 447 | 0 | 17.7 | 119.3 | 179.3 |
| h200_sxm | `causal_conv1d_fn_qkv3` | shared | sglang, vllm | sglang:396, vllm:428 | 396 / 428 | 0 | 189.5 | 196.1 | 196.6 |
| h200_sxm | `causal_conv1d_update` | shared | sglang, vllm | sglang:185, vllm:224 | 185 / 224 | 0 | 15.7 | 67.0 | 79.3 |
| h200_sxm | `chunk_kda` | shared | sglang | sglang:824 | 0 / 824 | 0 | — | — | — |
| h200_sxm | `flashkda_fwd` | shared | vllm | vllm:844 | 0 / 844 | 0 | — | — | — |
| h200_sxm | `fused_kda_decode` | shared | vllm | vllm:44 | 0 / 44 | 0 | — | — | — |
| h200_sxm | `fused_recurrent_kda` | shared | vllm | vllm:179 | 0 / 179 | 0 | — | — | — |
| h200_sxm | `fused_recurrent_kda_packed_decode` | shared | sglang, vllm | sglang:33, vllm:43 | 32 / 44 | 0 | 19.4 | 49.9 | 52.6 |
| h200_sxm | `fused_sigmoid_gating_delta_rule_update` | shared | sglang | sglang:152 | 0 / 152 | 0 | — | — | — |
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
| b200_sxm | `causal_conv1d_fn_token_major` | shared | trtllm | trtllm:619 | 0 / 619 | 0 | — | — | — |
| b200_sxm | `causal_conv1d_update` | shared | trtllm | trtllm:132 | 0 / 132 | 0 | — | — | — |
| b200_sxm | `mamba_chunk_scan_combined` | shared | trtllm | trtllm:619 | 0 / 619 | 0 | — | — | — |
| b200_sxm | `selective_state_update_flashinfer` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |
| b300_sxm | `causal_conv1d_fn` | shared | trtllm | trtllm:586 | 0 / 586 | 0 | — | — | — |
| b300_sxm | `causal_conv1d_fn_token_major` | shared | trtllm | trtllm:624 | 0 / 624 | 0 | — | — | — |
| b300_sxm | `causal_conv1d_update` | shared | trtllm | trtllm:132 | 0 / 66 | 0 | — | — | — |
| b300_sxm | `mamba_chunk_scan_combined` | shared | trtllm | trtllm:624 | 0 / 624 | 0 | — | — | — |
| b300_sxm | `selective_state_update_flashinfer` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |
| gb200 | `causal_conv1d_fn` | shared | trtllm | trtllm:624 | 0 / 624 | 0 | — | — | — |
| gb200 | `causal_conv1d_fn_token_major` | shared | trtllm | trtllm:619 | 0 / 619 | 0 | — | — | — |
| gb200 | `causal_conv1d_update` | shared | trtllm | trtllm:132 | 0 / 66 | 0 | — | — | — |
| gb200 | `mamba_chunk_scan_combined` | shared | trtllm | trtllm:619 | 0 / 619 | 0 | — | — | — |
| gb200 | `selective_state_update_flashinfer` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |
| gb300 | `causal_conv1d_fn` | shared | trtllm | trtllm:586 | 0 / 586 | 0 | — | — | — |
| gb300 | `causal_conv1d_fn_token_major` | shared | trtllm | trtllm:577 | 0 / 577 | 0 | — | — | — |
| gb300 | `causal_conv1d_update` | shared | trtllm | trtllm:132 | 0 / 66 | 0 | — | — | — |
| gb300 | `mamba_chunk_scan_combined` | shared | trtllm | trtllm:576 | 0 / 576 | 0 | — | — | — |
| gb300 | `selective_state_update_flashinfer` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |
| h100_sxm | `causal_conv1d_fn` | shared | trtllm | trtllm:592 | 0 / 592 | 0 | — | — | — |
| h100_sxm | `causal_conv1d_fn_token_major` | shared | trtllm | trtllm:579 | 0 / 579 | 0 | — | — | — |
| h100_sxm | `causal_conv1d_update` | shared | trtllm | trtllm:132 | 0 / 66 | 1 | — | — | — |
| h100_sxm | `mamba_chunk_scan_combined` | shared | trtllm | trtllm:579 | 0 / 579 | 0 | — | — | — |
| h100_sxm | `selective_state_update_flashinfer` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |
| h200_sxm | `causal_conv1d_fn` | shared | trtllm | trtllm:619 | 0 / 619 | 0 | — | — | — |
| h200_sxm | `causal_conv1d_fn_token_major` | shared | trtllm | trtllm:609 | 0 / 609 | 0 | — | — | — |
| h200_sxm | `causal_conv1d_update` | shared | trtllm | trtllm:132 | 0 / 66 | 1 | — | — | — |
| h200_sxm | `mamba_chunk_scan_combined` | shared | trtllm | trtllm:609 | 0 / 609 | 0 | — | — | — |
| h200_sxm | `selective_state_update_flashinfer` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |
| l40s | `causal_conv1d_fn` | shared | trtllm | trtllm:559 | 0 / 559 | 0 | — | — | — |
| l40s | `causal_conv1d_fn_token_major` | shared | trtllm | trtllm:559 | 0 / 559 | 0 | — | — | — |
| l40s | `causal_conv1d_update` | shared | trtllm | trtllm:132 | 0 / 66 | 2 | — | — | — |
| l40s | `mamba_chunk_scan_combined` | shared | trtllm | trtllm:559 | 0 / 559 | 0 | — | — | — |
| l40s | `selective_state_update_flashinfer` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |
| rtx_pro_6000_server | `causal_conv1d_fn` | shared | trtllm | trtllm:598 | 0 / 598 | 0 | — | — | — |
| rtx_pro_6000_server | `causal_conv1d_fn_token_major` | shared | trtllm | trtllm:592 | 0 / 592 | 0 | — | — | — |
| rtx_pro_6000_server | `causal_conv1d_update` | shared | trtllm | trtllm:132 | 0 / 66 | 0 | — | — | — |
| rtx_pro_6000_server | `mamba_chunk_scan_combined` | shared | trtllm | trtllm:592 | 0 / 592 | 0 | — | — | — |
| rtx_pro_6000_server | `selective_state_update_flashinfer` | shared | trtllm | trtllm:66 | 0 / 66 | 0 | — | — | — |

## `mhc_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `sglang_tilelang_mhc_post` | shared | sglang | sglang:140 | 0 / 70 | 8 | — | — | — |
| b200_sxm | `sglang_tilelang_mhc_pre` | shared | sglang | sglang:140 | 0 / 70 | 6 | — | — | — |
| b200_sxm | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:140 | 0 / 140 | 0 | — | — | — |
| b200_sxm | `trtllm_mhc_pre_dg_nosplit` | shared | trtllm | trtllm:50 | 0 / 50 | 0 | — | — | — |
| b200_sxm | `trtllm_mhc_pre_dg_splitk` | shared | trtllm | trtllm:58 | 0 / 58 | 0 | — | — | — |
| b200_sxm | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:30 | 0 / 30 | 0 | — | — | — |
| b200_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:210 | 0 / 70 | 29 | — | — | — |
| b200_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:204 | 0 / 70 | 15 | — | — | — |
| b300_sxm | `sglang.hc_post[tilelang]` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| b300_sxm | `sglang.hc_pre[tilelang,prenorm_gemm=deepgemm,norm=fused]` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| b300_sxm | `sglang.kernels.ops.layernorm.mhc.hc_contract` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| b300_sxm | `sglang.kernels.ops.layernorm.mhc.hc_expand` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| b300_sxm | `sglang_tilelang_mhc_post` | shared | sglang | sglang:140 | 0 / 70 | 12 | — | — | — |
| b300_sxm | `sglang_tilelang_mhc_pre` | shared | sglang | sglang:140 | 0 / 70 | 4 | — | — | — |
| b300_sxm | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:140 | 0 / 70 | 3 | — | — | — |
| b300_sxm | `trtllm_mhc_pre_dg_nosplit` | shared | trtllm | trtllm:52 | 0 / 28 | 0 | — | — | — |
| b300_sxm | `trtllm_mhc_pre_dg_splitk` | shared | trtllm | trtllm:58 | 0 / 32 | 0 | — | — | — |
| b300_sxm | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:30 | 0 / 16 | 0 | — | — | — |
| b300_sxm | `vllm.mhc_fused_post_pre_tilelang[fused_post_gemm,norm=fused]` | shared | vllm | vllm:6 | 0 / 6 | 0 | — | — | — |
| b300_sxm | `vllm.mhc_fused_post_pre_tilelang[post+prenorm_gemm=deepgemm,norm=fused]` | shared | vllm | vllm:29 | 0 / 29 | 0 | — | — | — |
| b300_sxm | `vllm.mhc_post_tilelang` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| b300_sxm | `vllm.mhc_pre_tilelang[prenorm_gemm=deepgemm,norm=fused]` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| b300_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:210 | 0 / 70 | 28 | — | — | — |
| b300_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:203 | 0 / 70 | 15 | — | — | — |
| b300_sxm | `vllm.model_executor.layers.mhc.hc_contract` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| b300_sxm | `vllm.model_executor.layers.mhc.hc_expand` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| gb200 | `sglang.hc_post[tilelang]` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| gb200 | `sglang.hc_pre[tilelang,prenorm_gemm=deepgemm,norm=fused]` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| gb200 | `sglang.kernels.ops.layernorm.mhc.hc_contract` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| gb200 | `sglang.kernels.ops.layernorm.mhc.hc_expand` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| gb200 | `sglang_tilelang_mhc_post` | shared | sglang | sglang:140 | 0 / 70 | 3 | — | — | — |
| gb200 | `sglang_tilelang_mhc_pre` | shared | sglang | sglang:140 | 0 / 70 | 4 | — | — | — |
| gb200 | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:140 | 0 / 70 | 4 | — | — | — |
| gb200 | `trtllm_mhc_pre_dg_nosplit` | shared | trtllm | trtllm:48 | 0 / 26 | 0 | — | — | — |
| gb200 | `trtllm_mhc_pre_dg_splitk` | shared | trtllm | trtllm:60 | 0 / 33 | 0 | — | — | — |
| gb200 | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:30 | 0 / 16 | 0 | — | — | — |
| gb200 | `vllm.mhc_fused_post_pre_tilelang[fused_post_gemm,norm=fused]` | shared | vllm | vllm:6 | 0 / 6 | 0 | — | — | — |
| gb200 | `vllm.mhc_fused_post_pre_tilelang[post+prenorm_gemm=deepgemm,norm=fused]` | shared | vllm | vllm:29 | 0 / 29 | 0 | — | — | — |
| gb200 | `vllm.mhc_post_tilelang` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| gb200 | `vllm.mhc_pre_tilelang[prenorm_gemm=deepgemm,norm=fused]` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| gb200 | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:210 | 0 / 70 | 21 | — | — | — |
| gb200 | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:210 | 0 / 70 | 17 | — | — | — |
| gb200 | `vllm.model_executor.layers.mhc.hc_contract` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| gb200 | `vllm.model_executor.layers.mhc.hc_expand` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| gb300 | `sglang.hc_post[tilelang]` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| gb300 | `sglang.hc_pre[tilelang,prenorm_gemm=deepgemm,norm=fused]` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| gb300 | `sglang.kernels.ops.layernorm.mhc.hc_contract` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| gb300 | `sglang.kernels.ops.layernorm.mhc.hc_expand` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| gb300 | `sglang_tilelang_mhc_post` | shared | sglang | sglang:140 | 0 / 70 | 2 | — | — | — |
| gb300 | `sglang_tilelang_mhc_pre` | shared | sglang | sglang:140 | 0 / 70 | 4 | — | — | — |
| gb300 | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:140 | 0 / 70 | 0 | — | — | — |
| gb300 | `trtllm_mhc_pre_dg_nosplit` | shared | trtllm | trtllm:51 | 0 / 28 | 0 | — | — | — |
| gb300 | `trtllm_mhc_pre_dg_splitk` | shared | trtllm | trtllm:59 | 0 / 33 | 0 | — | — | — |
| gb300 | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:30 | 0 / 16 | 3 | — | — | — |
| gb300 | `vllm.mhc_fused_post_pre_tilelang[fused_post_gemm,norm=fused]` | shared | vllm | vllm:6 | 0 / 6 | 0 | — | — | — |
| gb300 | `vllm.mhc_fused_post_pre_tilelang[post+prenorm_gemm=deepgemm,norm=fused]` | shared | vllm | vllm:29 | 0 / 29 | 0 | — | — | — |
| gb300 | `vllm.mhc_post_tilelang` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| gb300 | `vllm.mhc_pre_tilelang[prenorm_gemm=deepgemm,norm=fused]` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| gb300 | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:210 | 0 / 70 | 20 | — | — | — |
| gb300 | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:210 | 0 / 70 | 26 | — | — | — |
| gb300 | `vllm.model_executor.layers.mhc.hc_contract` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| gb300 | `vllm.model_executor.layers.mhc.hc_expand` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| h100_sxm | `sglang.hc_post[tilelang]` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| h100_sxm | `sglang.hc_pre[tilelang,prenorm_gemm=deepgemm,norm=fused]` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| h100_sxm | `sglang.kernels.ops.layernorm.mhc.hc_contract` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| h100_sxm | `sglang.kernels.ops.layernorm.mhc.hc_expand` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| h100_sxm | `sglang_tilelang_mhc_post` | shared | sglang | sglang:136 | 0 / 68 | 0 | — | — | — |
| h100_sxm | `sglang_tilelang_mhc_pre` | shared | sglang | sglang:139 | 0 / 70 | 0 | — | — | — |
| h100_sxm | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:138 | 0 / 69 | 4 | — | — | — |
| h100_sxm | `trtllm_mhc_pre_dg_nosplit` | shared | trtllm | trtllm:45 | 0 / 24 | 0 | — | — | — |
| h100_sxm | `trtllm_mhc_pre_dg_splitk` | shared | trtllm | trtllm:61 | 0 / 34 | 0 | — | — | — |
| h100_sxm | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:28 | 0 / 16 | 0 | — | — | — |
| h100_sxm | `vllm.mhc_fused_post_pre_tilelang[fused_post_gemm,norm=fused]` | shared | vllm | vllm:6 | 0 / 6 | 0 | — | — | — |
| h100_sxm | `vllm.mhc_fused_post_pre_tilelang[post+prenorm_gemm=deepgemm,norm=fused]` | shared | vllm | vllm:28 | 0 / 28 | 0 | — | — | — |
| h100_sxm | `vllm.mhc_post_tilelang` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| h100_sxm | `vllm.mhc_pre_tilelang[prenorm_gemm=deepgemm,norm=fused]` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| h100_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:207 | 0 / 69 | 11 | — | — | — |
| h100_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:210 | 0 / 70 | 2 | — | — | — |
| h100_sxm | `vllm.model_executor.layers.mhc.hc_contract` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| h100_sxm | `vllm.model_executor.layers.mhc.hc_expand` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| h200_sxm | `sglang.hc_post[tilelang]` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| h200_sxm | `sglang.hc_pre[tilelang,prenorm_gemm=deepgemm,norm=fused]` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| h200_sxm | `sglang.kernels.ops.layernorm.mhc.hc_contract` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| h200_sxm | `sglang.kernels.ops.layernorm.mhc.hc_expand` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| h200_sxm | `sglang_tilelang_mhc_post` | shared | sglang | sglang:140 | 0 / 70 | 0 | — | — | — |
| h200_sxm | `sglang_tilelang_mhc_pre` | shared | sglang | sglang:140 | 0 / 70 | 0 | — | — | — |
| h200_sxm | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:140 | 0 / 70 | 0 | — | — | — |
| h200_sxm | `trtllm_mhc_pre_dg_nosplit` | shared | trtllm | trtllm:48 | 0 / 26 | 0 | — | — | — |
| h200_sxm | `trtllm_mhc_pre_dg_splitk` | shared | trtllm | trtllm:59 | 0 / 35 | 0 | — | — | — |
| h200_sxm | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:31 | 0 / 19 | 0 | — | — | — |
| h200_sxm | `vllm.mhc_fused_post_pre_tilelang[fused_post_gemm,norm=fused]` | shared | vllm | vllm:6 | 0 / 6 | 0 | — | — | — |
| h200_sxm | `vllm.mhc_fused_post_pre_tilelang[post+prenorm_gemm=deepgemm,norm=fused]` | shared | vllm | vllm:29 | 0 / 29 | 0 | — | — | — |
| h200_sxm | `vllm.mhc_post_tilelang` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| h200_sxm | `vllm.mhc_pre_tilelang[prenorm_gemm=deepgemm,norm=fused]` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| h200_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:210 | 0 / 70 | 21 | — | — | — |
| h200_sxm | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:210 | 0 / 70 | 13 | — | — | — |
| h200_sxm | `vllm.model_executor.layers.mhc.hc_contract` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| h200_sxm | `vllm.model_executor.layers.mhc.hc_expand` | shared | vllm | vllm:35 | 0 / 35 | 0 | — | — | — |
| l40s | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:134 | 0 / 67 | 6 | — | — | — |
| l40s | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:130 | 0 / 65 | 1 | — | — | — |
| l40s | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:134 | 0 / 67 | 6 | — | — | — |
| l40s | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:138 | 0 / 69 | 2 | — | — | — |
| rtx_pro_6000_server | `sglang_tilelang_mhc_post` | shared | sglang | sglang:67 | 0 / 67 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_tilelang_mhc_pre` | shared | sglang | sglang:35 | 0 / 35 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_torch_mhc_pre` | shared | sglang | sglang:67 | 0 / 67 | 0 | — | — | — |
| rtx_pro_6000_server | `trtllm_mhc_post_mapping` | shared | trtllm | trtllm:138 | 0 / 69 | 13 | — | — | — |
| rtx_pro_6000_server | `trtllm_mhc_pre_dg_nosplit` | shared | trtllm | trtllm:18 | 0 / 18 | 0 | — | — | — |
| rtx_pro_6000_server | `trtllm_mhc_pre_dg_splitk` | shared | trtllm | trtllm:37 | 0 / 37 | 0 | — | — | — |
| rtx_pro_6000_server | `trtllm_mhc_pre_fma` | shared | trtllm | trtllm:79 | 0 / 67 | 3 | — | — | — |
| rtx_pro_6000_server | `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | shared | vllm | vllm:69 | 0 / 69 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | shared | vllm | vllm:70 | 0 / 70 | 0 | — | — | — |

## `mla_bmm_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| a100_sxm | `default` | shared_fallback | sglang, trtllm | sglang:424, trtllm:424 | 224 / 224 | 0 | 1.4 | 18.4 | 115.9 |
| b200_sxm | `sglang_flashinfer_bmm_fp8` | shared | sglang | sglang:636 | 0 / 336 | 0 | — | — | — |
| b200_sxm | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:636 | 0 / 336 | 0 | — | — | — |
| b200_sxm | `sglang_torch_bmm` | shared | sglang | sglang:1272 | 0 / 336 | 1 | — | — | — |
| b200_sxm | `trtllm_bmm_out` | shared | trtllm | trtllm:1060 | 0 / 760 | 0 | — | — | — |
| b200_sxm | `trtllm_bmm_out_dequant_bf16` | shared | trtllm | trtllm:1060 | 0 / 760 | 0 | — | — | — |
| b200_sxm | `vllm_torch_bmm` | shared | vllm | vllm:1908 | 0 / 336 | 5 | — | — | — |
| b300_sxm | `sglang_flashinfer_bmm_fp8` | shared | sglang | sglang:636 | 0 / 336 | 0 | — | — | — |
| b300_sxm | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| b300_sxm | `sglang_torch_bmm` | shared | sglang | sglang:1484 | 0 / 336 | 1 | — | — | — |
| b300_sxm | `trtllm_bmm_out` | shared | trtllm | trtllm:1060 | 0 / 336 | 2 | — | — | — |
| b300_sxm | `trtllm_bmm_out_dequant_bf16` | shared | trtllm | trtllm:1060 | 0 / 336 | 0 | — | — | — |
| b300_sxm | `vllm_torch_bmm` | shared | vllm | vllm:1908 | 0 / 336 | 9 | — | — | — |
| gb200 | `sglang_flashinfer_bmm_fp8` | shared | sglang | sglang:636 | 0 / 336 | 0 | — | — | — |
| gb200 | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| gb200 | `sglang_torch_bmm` | shared | sglang | sglang:1484 | 0 / 336 | 0 | — | — | — |
| gb200 | `trtllm_bmm_out` | shared | trtllm | trtllm:1060 | 0 / 336 | 0 | — | — | — |
| gb200 | `trtllm_bmm_out_dequant_bf16` | shared | trtllm | trtllm:1060 | 0 / 336 | 1 | — | — | — |
| gb200 | `vllm_torch_bmm` | shared | vllm | vllm:1908 | 0 / 336 | 2 | — | — | — |
| gb300 | `sglang_flashinfer_bmm_fp8` | shared | sglang | sglang:636 | 0 / 336 | 0 | — | — | — |
| gb300 | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| gb300 | `sglang_torch_bmm` | shared | sglang | sglang:1484 | 0 / 336 | 0 | — | — | — |
| gb300 | `trtllm_bmm_out` | shared | trtllm | trtllm:1060 | 0 / 336 | 0 | — | — | — |
| gb300 | `trtllm_bmm_out_dequant_bf16` | shared | trtllm | trtllm:1060 | 0 / 336 | 2 | — | — | — |
| gb300 | `vllm_torch_bmm` | shared | vllm | vllm:2344 | 0 / 336 | 5 | — | — | — |
| h100_sxm | `sglang_flashinfer_bmm_fp8` | shared | sglang | sglang:636 | 0 / 336 | 0 | — | — | — |
| h100_sxm | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| h100_sxm | `sglang_torch_bmm` | shared | sglang | sglang:1484 | 0 / 336 | 0 | — | — | — |
| h100_sxm | `trtllm_bmm_out` | shared | trtllm | trtllm:1060 | 0 / 336 | 1 | — | — | — |
| h100_sxm | `trtllm_fp8_block_scaling_bmm_out` | shared | trtllm | trtllm:1060 | 0 / 336 | 0 | — | — | — |
| h100_sxm | `vllm_torch_bmm` | shared | vllm | vllm:1908 | 0 / 336 | 2 | — | — | — |
| h200_sxm | `sglang_flashinfer_bmm_fp8` | shared | sglang | sglang:636 | 0 / 336 | 0 | — | — | — |
| h200_sxm | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:424 | 0 / 224 | 0 | — | — | — |
| h200_sxm | `sglang_torch_bmm` | shared | sglang | sglang:1060 | 0 / 336 | 0 | — | — | — |
| h200_sxm | `trtllm_bmm_out` | shared | trtllm | trtllm:1060 | 0 / 336 | 0 | — | — | — |
| h200_sxm | `trtllm_fp8_block_scaling_bmm_out` | shared | trtllm | trtllm:1060 | 0 / 336 | 0 | — | — | — |
| h200_sxm | `vllm_torch_bmm` | shared | vllm | vllm:1908 | 0 / 336 | 5 | — | — | — |
| l40s | `default` | shared_fallback | trtllm | trtllm:848 | 0 / 448 | 0 | — | — | — |
| l40s | `sglang_flashinfer_bmm_fp8` | shared | sglang | sglang:636 | 0 / 336 | 0 | — | — | — |
| l40s | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:424 | 0 / 224 | 0 | — | — | — |
| l40s | `sglang_torch_bmm` | shared | sglang | sglang:1060 | 0 / 336 | 2 | — | — | — |
| l40s | `trtllm_bmm_out` | shared | trtllm | trtllm:1060 | 0 / 336 | 0 | — | — | — |
| l40s | `trtllm_fp8_block_scaling_bmm_out` | shared | trtllm | trtllm:1060 | 0 / 336 | 0 | — | — | — |
| l40s | `vllm_torch_bmm` | shared | vllm | vllm:636 | 0 / 336 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_flashinfer_bmm_fp8` | shared | sglang | sglang:636 | 0 / 336 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_sgl_kernel_bmm_fp8` | shared | sglang | sglang:848 | 0 / 224 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_torch_bmm` | shared | sglang | sglang:1484 | 0 / 336 | 0 | — | — | — |
| rtx_pro_6000_server | `trtllm_bmm_out` | shared | trtllm | trtllm:1060 | 0 / 336 | 1 | — | — | — |
| rtx_pro_6000_server | `trtllm_fp8_block_scaling_bmm_out` | shared | trtllm | trtllm:1060 | 0 / 336 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_torch_bmm` | shared | vllm | vllm:636 | 0 / 336 | 0 | — | — | — |

## `mla_context_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `FLASHINFER_MLA` | shared | vllm | vllm:2475 | 0 / 972 | 35 | — | — | — |
| b200_sxm | `FLASH_ATTN` | shared | vllm | vllm:21696 | 0 / 8136 | 118 | — | — | — |
| b200_sxm | `TRTLLM_RAGGED` | shared | vllm | vllm:10848 | 0 / 4068 | 58 | — | — | — |
| b200_sxm | `default` | shared_fallback | trtllm | trtllm:11712 | 0 / 11712 | 0 | — | — | — |
| b200_sxm | `mla_trtllm_mla` | shared | sglang | sglang:218 | 0 / 218 | 0 | — | — | — |
| b300_sxm | `FLASHINFER_MLA` | shared | vllm | vllm:2475 | 0 / 972 | 17 | — | — | — |
| b300_sxm | `FLASH_ATTN` | shared | vllm | vllm:21696 | 0 / 8136 | 27 | — | — | — |
| b300_sxm | `TRTLLM_RAGGED` | shared | vllm | vllm:10848 | 0 / 4068 | 8 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:11712 | 0 / 5856 | 44 | — | — | — |
| b300_sxm | `mla_trtllm_mla` | shared | sglang | sglang:218 | 0 / 218 | 0 | — | — | — |
| gb200 | `FLASHINFER_MLA` | shared | vllm | vllm:2475 | 0 / 972 | 16 | — | — | — |
| gb200 | `FLASH_ATTN` | shared | vllm | vllm:21696 | 0 / 8136 | 26 | — | — | — |
| gb200 | `TRTLLM_RAGGED` | shared | vllm | vllm:10848 | 0 / 4068 | 13 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:11712 | 0 / 5856 | 46 | — | — | — |
| gb300 | `FLASHINFER_MLA` | shared | vllm | vllm:4091 | 0 / 1944 | 15 | — | — | — |
| gb300 | `FLASH_ATTN` | shared | vllm | vllm:35197 | 0 / 16272 | 98 | — | — | — |
| gb300 | `TRTLLM_RAGGED` | shared | vllm | vllm:17591 | 0 / 8136 | 30 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:11712 | 0 / 5856 | 26 | — | — | — |
| h100_sxm | `FLASHMLA` | shared | vllm | vllm:2826 | 0 / 1080 | 11 | — | — | — |
| h100_sxm | `FLASH_ATTN` | shared | vllm | vllm:8704 | 0 / 3264 | 9 | — | — | — |
| h100_sxm | `FLASH_ATTN_MLA` | shared | vllm | vllm:4032 | 0 / 1512 | 20 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:7740 | 0 / 3880 | 14 | — | — | — |
| h200_sxm | `FLASHMLA` | shared | vllm | vllm:2826 | 0 / 1080 | 4 | — | — | — |
| h200_sxm | `FLASH_ATTN` | shared | vllm | vllm:8704 | 0 / 3264 | 0 | — | — | — |
| h200_sxm | `FLASH_ATTN_MLA` | shared | vllm | vllm:4032 | 0 / 1512 | 5 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:7739 | 0 / 3879 | 7 | — | — | — |
| l40s | `FLASH_ATTN` | shared | vllm | vllm:4502 | 0 / 2712 | 0 | — | — | — |
| l40s | `TRITON_MLA` | shared | vllm | vllm:360 | 0 / 216 | 0 | — | — | — |
| rtx_pro_6000_server | `FLASH_ATTN` | shared | vllm | vllm:11752 | 0 / 8136 | 0 | — | — | — |
| rtx_pro_6000_server | `TRITON_MLA` | shared | vllm | vllm:513 | 0 / 351 | 0 | — | — | — |
| rtx_pro_6000_server | `default` | shared_fallback | trtllm | trtllm:9719 | 0 / 5837 | 24 | — | — | — |

## `mla_generation_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `FLASHINFER_MLA` | shared | vllm | vllm:33480 | 0 / 13248 | 359 | — | — | — |
| b200_sxm | `default` | shared_fallback | trtllm | trtllm:17662 | 0 / 17662 | 0 | — | — | — |
| b300_sxm | `FLASHINFER_MLA` | shared | vllm | vllm:33480 | 0 / 13248 | 196 | — | — | — |
| b300_sxm | `default` | shared_fallback | trtllm | trtllm:17663 | 0 / 8832 | 256 | — | — | — |
| gb200 | `FLASHINFER_MLA` | shared | vllm | vllm:33480 | 0 / 13248 | 161 | — | — | — |
| gb200 | `default` | shared_fallback | trtllm | trtllm:17663 | 0 / 8832 | 204 | — | — | — |
| gb300 | `FLASHINFER_MLA` | shared | vllm | vllm:55460 | 0 / 26496 | 252 | — | — | — |
| gb300 | `default` | shared_fallback | trtllm | trtllm:17664 | 0 / 8832 | 182 | — | — | — |
| h100_sxm | `FLASHMLA` | shared | vllm | vllm:11776 | 0 / 4416 | 25 | — | — | — |
| h100_sxm | `FLASH_ATTN_MLA` | shared | vllm | vllm:11584 | 0 / 4344 | 99 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:11776 | 0 / 5888 | 199 | — | — | — |
| h200_sxm | `FLASHMLA` | shared | vllm | vllm:11776 | 0 / 4416 | 38 | — | — | — |
| h200_sxm | `FLASH_ATTN_MLA` | shared | vllm | vllm:11776 | 0 / 4416 | 39 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:11776 | 0 / 5888 | 244 | — | — | — |
| l40s | `TRITON_MLA` | shared | vllm | vllm:6977 | 0 / 4321 | 0 | — | — | — |
| rtx_pro_6000_server | `TRITON_MLA` | shared | vllm | vllm:10122 | 0 / 7132 | 0 | — | — | — |
| rtx_pro_6000_server | `default` | shared_fallback | trtllm | trtllm:14719 | 0 / 8832 | 31 | — | — | — |

## `moe_a2a_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `deepep` | shared | trtllm, vllm | trtllm:704, vllm:324 | 0 / 1028 | 0 | — | — | — |
| b300_sxm | `deepep` | shared | trtllm, vllm | trtllm:704, vllm:324 | 0 / 1028 | 0 | — | — | — |
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
| b200_sxm | `deepgemm` | shared | trtllm | trtllm:79080 | 0 / 79080 | 0 | — | — | — |
| b200_sxm | `moe_torch_flow` | shared | trtllm | trtllm:17334 | 0 / 13122 | 5 | — | — | — |
| b200_sxm | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:239760 | 0 / 239760 | 0 | — | — | — |
| b200_sxm | `moe_torch_flow_min_latency` | shared | trtllm | trtllm:75510 | 0 / 75510 | 0 | — | — | — |
| b200_sxm | `moe_torch_flow_nongated` | shared | trtllm | trtllm:50868 | 0 / 50868 | 0 | — | — | — |
| b200_sxm | `sglang_flashinfer_trtllm_moe` | shared | sglang | sglang:241292 | 0 / 137294 | 5 | — | — | — |
| b200_sxm | `sglang_fused_moe_triton` | shared | sglang | sglang:80352 | 0 / 43416 | 4 | — | — | — |
| b200_sxm | `sglang_mxfp4_flashinfer_trtllm_moe` | shared | sglang | sglang:28056 | 0 / 22872 | 0 | — | — | — |
| b200_sxm | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| b200_sxm | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | shared | vllm | vllm:1944 | 0 / 972 | 0 | — | — | — |
| b200_sxm | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutedsl_flashinfercutedslexperts` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| b200_sxm | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:1944 | 0 / 972 | 0 | — | — | — |
| b200_sxm | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmodular` | shared | vllm | vllm:10287 | 0 / 4131 | 7 | — | — | — |
| b200_sxm | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic` | shared | vllm | vllm:45318 | 0 / 16154 | 34 | — | — | — |
| b200_sxm | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:2106 | 0 / 1053 | 0 | — | — | — |
| b200_sxm | `vllm_compressedtensorswna16moe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| b200_sxm | `vllm_flashinfer_trtllm_moe_fp4` | shared | vllm | vllm:10287 | 0 / 9234 | 0 | — | — | — |
| b200_sxm | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmodular` | shared | vllm | vllm:12555 | 0 / 4374 | 3 | — | — | — |
| b200_sxm | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:37422 | 0 / 12474 | 21 | — | — | — |
| b200_sxm | `vllm_fp8moe_triton_tritonexperts` | shared | vllm | vllm:2754 | 0 / 2754 | 0 | — | — | — |
| b200_sxm | `vllm_fused_moe` | shared | vllm | vllm:83407 | 0 / 80329 | 0 | — | — | — |
| b200_sxm | `vllm_gptossmxfp4moe_flashinfer_trtllm_mxfp4_bf16_trtllmmxfp4expertsmonolithic` | shared | vllm | vllm:5832 | 0 / 1944 | 8 | — | — | — |
| b200_sxm | `vllm_marlin_int4_moe` | shared | vllm | vllm:10209 | 0 / 9237 | 0 | — | — | — |
| b200_sxm | `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:4779 | 0 / 1944 | 5 | — | — | — |
| b200_sxm | `vllm_modeloptfp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:36738 | 0 / 12503 | 39 | — | — | — |
| b200_sxm | `vllm_mxfp4_moe` | shared | vllm | vllm:1138 | 0 / 1138 | 0 | — | — | — |
| b200_sxm | `vllm_mxfp4moe_flashinfer_trtllm_mxfp4_mxfp8_trtllmmxfp4expertsmodular` | shared | vllm | vllm:6561 | 0 / 2187 | 4 | — | — | — |
| b200_sxm | `vllm_unquantizedfusedmoe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:5508 | 0 / 2835 | 5 | — | — | — |
| b200_sxm | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16experts` | shared | vllm | vllm:25720 | 0 / 13556 | 9 | — | — | — |
| b200_sxm | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16expertsmonolithic` | shared | vllm | vllm:16180 | 0 / 15127 | 4 | — | — | — |
| b200_sxm | `vllm_unquantizedfusedmoe_triton_tritonexperts` | shared | vllm | vllm:2106 | 0 / 1053 | 0 | — | — | — |
| b300_sxm | `deepgemm` | shared | trtllm | trtllm:79212 | 0 / 40497 | 57 | — | — | — |
| b300_sxm | `moe_torch_flow` | shared | trtllm | trtllm:19843 | 0 / 7371 | 18 | — | — | — |
| b300_sxm | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:239760 | 0 / 125712 | 123 | — | — | — |
| b300_sxm | `moe_torch_flow_min_latency` | shared | trtllm | trtllm:39465 | 0 / 39465 | 0 | — | — | — |
| b300_sxm | `moe_torch_flow_nongated` | shared | trtllm | trtllm:50868 | 0 / 27945 | 1 | — | — | — |
| b300_sxm | `sglang_flashinfer_trtllm_moe` | shared | sglang | sglang:320425 | 0 / 136404 | 15 | — | — | — |
| b300_sxm | `sglang_fused_moe_triton` | shared | sglang | sglang:104998 | 0 / 43416 | 5 | — | — | — |
| b300_sxm | `sglang_mxfp4_flashinfer_trtllm_moe` | shared | sglang | sglang:32562 | 0 / 22842 | 0 | — | — | — |
| b300_sxm | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| b300_sxm | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | shared | vllm | vllm:1944 | 0 / 972 | 0 | — | — | — |
| b300_sxm | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutedsl_flashinfercutedslexperts` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| b300_sxm | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:1944 | 0 / 972 | 0 | — | — | — |
| b300_sxm | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmodular` | shared | vllm | vllm:8181 | 0 / 4131 | 1 | — | — | — |
| b300_sxm | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic` | shared | vllm | vllm:44266 | 0 / 16180 | 38 | — | — | — |
| b300_sxm | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:2106 | 0 / 1053 | 0 | — | — | — |
| b300_sxm | `vllm_compressedtensorswna16moe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| b300_sxm | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmodular` | shared | vllm | vllm:12555 | 0 / 4374 | 7 | — | — | — |
| b300_sxm | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:38232 | 0 / 13284 | 33 | — | — | — |
| b300_sxm | `vllm_fp8moe_triton_tritonexperts` | shared | vllm | vllm:2835 | 0 / 2835 | 0 | — | — | — |
| b300_sxm | `vllm_gptossmxfp4moe_flashinfer_trtllm_mxfp4_bf16_trtllmmxfp4expertsmonolithic` | shared | vllm | vllm:5832 | 0 / 1944 | 7 | — | — | — |
| b300_sxm | `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:4779 | 0 / 1944 | 2 | — | — | — |
| b300_sxm | `vllm_modeloptfp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:36116 | 0 / 12555 | 26 | — | — | — |
| b300_sxm | `vllm_modeloptnvfp4fusedmoe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic` | shared | vllm | vllm:891 | 0 / 891 | 0 | — | — | — |
| b300_sxm | `vllm_mxfp4moe_flashinfer_trtllm_mxfp4_mxfp8_trtllmmxfp4expertsmodular` | shared | vllm | vllm:6561 | 0 / 2187 | 3 | — | — | — |
| b300_sxm | `vllm_unquantizedfusedmoe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:5508 | 0 / 2835 | 7 | — | — | — |
| b300_sxm | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16experts` | shared | vllm | vllm:25728 | 0 / 13582 | 15 | — | — | — |
| b300_sxm | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16expertsmonolithic` | shared | vllm | vllm:16102 | 0 / 15049 | 0 | — | — | — |
| b300_sxm | `vllm_unquantizedfusedmoe_triton_tritonexperts` | shared | vllm | vllm:2106 | 0 / 1053 | 0 | — | — | — |
| b60 | `vllm_xpu_moe` | shared | vllm | vllm:3996 | 0 / 2808 | 0 | — | — | — |
| b60 | `vllm_xpu_moe_mxfp4` | shared | vllm | vllm:2592 | 0 / 1296 | 0 | — | — | — |
| gb200 | `deepgemm` | shared | trtllm | trtllm:79299 | 0 / 40500 | 60 | — | — | — |
| gb200 | `moe_torch_flow` | shared | trtllm | trtllm:19683 | 0 / 7371 | 10 | — | — | — |
| gb200 | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:239760 | 0 / 125712 | 124 | — | — | — |
| gb200 | `moe_torch_flow_min_latency` | shared | trtllm | trtllm:75510 | 0 / 39465 | 48 | — | — | — |
| gb200 | `moe_torch_flow_nongated` | shared | trtllm | trtllm:50868 | 0 / 27945 | 1 | — | — | — |
| gb200 | `sglang_flashinfer_trtllm_moe` | shared | sglang | sglang:281187 | 0 / 142638 | 3 | — | — | — |
| gb200 | `sglang_fused_moe_triton` | shared | sglang | sglang:94041 | 0 / 43416 | 2 | — | — | — |
| gb200 | `sglang_mxfp4_flashinfer_trtllm_moe` | shared | sglang | sglang:30405 | 0 / 22872 | 0 | — | — | — |
| gb200 | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | shared | vllm | vllm:2916 | 0 / 972 | 1 | — | — | — |
| gb200 | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutedsl_flashinfercutedslexperts` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| gb200 | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:1944 | 0 / 972 | 0 | — | — | — |
| gb200 | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmodular` | shared | vllm | vllm:10287 | 0 / 4131 | 5 | — | — | — |
| gb200 | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic` | shared | vllm | vllm:45343 | 0 / 16232 | 40 | — | — | — |
| gb200 | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:2106 | 0 / 1053 | 0 | — | — | — |
| gb200 | `vllm_compressedtensorswna16moe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| gb200 | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmodular` | shared | vllm | vllm:12555 | 0 / 4374 | 3 | — | — | — |
| gb200 | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:38232 | 0 / 13284 | 35 | — | — | — |
| gb200 | `vllm_fp8moe_triton_tritonexperts` | shared | vllm | vllm:2835 | 0 / 2835 | 0 | — | — | — |
| gb200 | `vllm_gptossmxfp4moe_flashinfer_trtllm_mxfp4_bf16_trtllmmxfp4expertsmonolithic` | shared | vllm | vllm:5832 | 0 / 1944 | 5 | — | — | — |
| gb200 | `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:4779 | 0 / 1944 | 2 | — | — | — |
| gb200 | `vllm_modeloptfp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:35988 | 0 / 12555 | 41 | — | — | — |
| gb200 | `vllm_modeloptnvfp4fusedmoe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic` | shared | vllm | vllm:891 | 0 / 891 | 0 | — | — | — |
| gb200 | `vllm_mxfp4moe_flashinfer_trtllm_mxfp4_mxfp8_trtllmmxfp4expertsmodular` | shared | vllm | vllm:6561 | 0 / 2187 | 1 | — | — | — |
| gb200 | `vllm_unquantizedfusedmoe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:5508 | 0 / 2835 | 5 | — | — | — |
| gb200 | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16experts` | shared | vllm | vllm:25876 | 0 / 13556 | 12 | — | — | — |
| gb200 | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16expertsmonolithic` | shared | vllm | vllm:15945 | 0 / 14919 | 1 | — | — | — |
| gb200 | `vllm_unquantizedfusedmoe_triton_tritonexperts` | shared | vllm | vllm:2106 | 0 / 1053 | 0 | — | — | — |
| gb300 | `deepgemm` | shared | trtllm | trtllm:79299 | 0 / 40500 | 66 | — | — | — |
| gb300 | `moe_torch_flow` | shared | trtllm | trtllm:19845 | 0 / 7371 | 17 | — | — | — |
| gb300 | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:239760 | 0 / 125712 | 152 | — | — | — |
| gb300 | `moe_torch_flow_min_latency` | shared | trtllm | trtllm:39465 | 0 / 39465 | 0 | — | — | — |
| gb300 | `moe_torch_flow_nongated` | shared | trtllm | trtllm:50868 | 0 / 27945 | 1 | — | — | — |
| gb300 | `sglang_flashinfer_trtllm_moe` | shared | sglang | sglang:284303 | 0 / 142619 | 3 | — | — | — |
| gb300 | `sglang_fused_moe_triton` | shared | sglang | sglang:94043 | 0 / 43416 | 1 | — | — | — |
| gb300 | `sglang_mxfp4_flashinfer_trtllm_moe` | shared | sglang | sglang:30375 | 0 / 22842 | 0 | — | — | — |
| gb300 | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | shared | vllm | vllm:2916 | 0 / 972 | 0 | — | — | — |
| gb300 | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutedsl_flashinfercutedslexperts` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| gb300 | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:1944 | 0 / 972 | 0 | — | — | — |
| gb300 | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmodular` | shared | vllm | vllm:8181 | 0 / 4131 | 0 | — | — | — |
| gb300 | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic` | shared | vllm | vllm:44269 | 0 / 16180 | 40 | — | — | — |
| gb300 | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:2106 | 0 / 1053 | 0 | — | — | — |
| gb300 | `vllm_compressedtensorswna16moe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| gb300 | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmodular` | shared | vllm | vllm:12555 | 0 / 4374 | 5 | — | — | — |
| gb300 | `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:38232 | 0 / 13284 | 34 | — | — | — |
| gb300 | `vllm_fp8moe_triton_tritonexperts` | shared | vllm | vllm:2835 | 0 / 2835 | 0 | — | — | — |
| gb300 | `vllm_gptossmxfp4moe_flashinfer_trtllm_mxfp4_bf16_trtllmmxfp4expertsmonolithic` | shared | vllm | vllm:5832 | 0 / 1944 | 7 | — | — | — |
| gb300 | `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:4779 | 0 / 1944 | 5 | — | — | — |
| gb300 | `vllm_modeloptfp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | shared | vllm | vllm:36327 | 0 / 12555 | 46 | — | — | — |
| gb300 | `vllm_modeloptnvfp4fusedmoe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic` | shared | vllm | vllm:891 | 0 / 891 | 0 | — | — | — |
| gb300 | `vllm_mxfp4moe_flashinfer_trtllm_mxfp4_mxfp8_trtllmmxfp4expertsmodular` | shared | vllm | vllm:6561 | 0 / 2187 | 4 | — | — | — |
| gb300 | `vllm_unquantizedfusedmoe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:5508 | 0 / 2835 | 5 | — | — | — |
| gb300 | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16experts` | shared | vllm | vllm:25849 | 0 / 13608 | 15 | — | — | — |
| gb300 | `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16expertsmonolithic` | shared | vllm | vllm:16049 | 0 / 15023 | 0 | — | — | — |
| gb300 | `vllm_unquantizedfusedmoe_triton_tritonexperts` | shared | vllm | vllm:2106 | 0 / 1053 | 0 | — | — | — |
| h100_sxm | `moe_torch_flow` | shared | trtllm | trtllm:2592 | 0 / 1053 | 1 | — | — | — |
| h100_sxm | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:301482 | 0 / 134622 | 95 | — | — | — |
| h100_sxm | `moe_torch_flow_nongated` | shared | trtllm | trtllm:52164 | 0 / 25191 | 9 | — | — | — |
| h100_sxm | `sglang_flashinfer_cutlass_moe` | shared | sglang | sglang:17187 | 0 / 8601 | 0 | — | — | — |
| h100_sxm | `sglang_fused_moe_triton` | shared | sglang | sglang:256932 | 0 / 124497 | 10 | — | — | — |
| h100_sxm | `sglang_marlin_moe` | shared | sglang | sglang:28269 | 0 / 20493 | 0 | — | — | — |
| h100_sxm | `sglang_marlin_moe_situ_as_silu` | shared | sglang | sglang:541 | 0 / 541 | 0 | — | — | — |
| h100_sxm | `sglang_mxfp4_humming_moe` | shared | sglang | sglang:15 | 0 / 15 | 0 | — | — | — |
| h100_sxm | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | shared | vllm | vllm:2916 | 0 / 972 | 0 | — | — | — |
| h100_sxm | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:2106 | 0 / 1053 | 0 | — | — | — |
| h100_sxm | `vllm_compressedtensorswna16moe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| h100_sxm | `vllm_fp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:30213 | 0 / 10692 | 4 | — | — | — |
| h100_sxm | `vllm_fp8moe_triton_tritonexperts` | shared | vllm | vllm:22323 | 0 / 9801 | 5 | — | — | — |
| h100_sxm | `vllm_gptossmxfp4moe_triton_oaitritonmxfp4expertsmonolithic` | shared | vllm | vllm:5832 | 0 / 1944 | 5 | — | — | — |
| h100_sxm | `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:45927 | 0 / 16200 | 9 | — | — | — |
| h100_sxm | `vllm_unquantizedfusedmoe_triton_tritonexperts` | shared | vllm | vllm:52753 | 0 / 18306 | 11 | — | — | — |
| h200_sxm | `moe_torch_flow` | shared | trtllm | trtllm:3402 | 0 / 1134 | 1 | — | — | — |
| h200_sxm | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:292734 | 0 / 134622 | 125 | — | — | — |
| h200_sxm | `moe_torch_flow_nongated` | shared | trtllm | trtllm:52320 | 0 / 25272 | 14 | — | — | — |
| h200_sxm | `sglang_flashinfer_cutlass_moe` | shared | sglang | sglang:17349 | 0 / 8682 | 0 | — | — | — |
| h200_sxm | `sglang_fused_moe_triton` | shared | sglang | sglang:256932 | 0 / 124497 | 7 | — | — | — |
| h200_sxm | `sglang_marlin_moe` | shared | sglang | sglang:28269 | 0 / 20493 | 1 | — | — | — |
| h200_sxm | `sglang_marlin_moe_situ_as_silu` | shared | sglang | sglang:488 | 0 / 488 | 0 | — | — | — |
| h200_sxm | `sglang_mxfp4_humming_moe` | shared | sglang | sglang:15 | 0 / 15 | 0 | — | — | — |
| h200_sxm | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | shared | vllm | vllm:2916 | 0 / 972 | 0 | — | — | — |
| h200_sxm | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:2106 | 0 / 1053 | 0 | — | — | — |
| h200_sxm | `vllm_compressedtensorswna16moe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| h200_sxm | `vllm_fp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:30213 | 0 / 10692 | 17 | — | — | — |
| h200_sxm | `vllm_fp8moe_triton_tritonexperts` | shared | vllm | vllm:22356 | 0 / 9801 | 4 | — | — | — |
| h200_sxm | `vllm_fused_moe` | shared | vllm | vllm:36288 | 0 / 33210 | 0 | — | — | — |
| h200_sxm | `vllm_gptossmxfp4moe_triton_oaitritonmxfp4expertsmonolithic` | shared | vllm | vllm:5832 | 0 / 1944 | 4 | — | — | — |
| h200_sxm | `vllm_marlin_int4_moe` | shared | vllm | vllm:10206 | 0 / 9234 | 0 | — | — | — |
| h200_sxm | `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:45927 | 0 / 16200 | 11 | — | — | — |
| h200_sxm | `vllm_mxfp4_moe` | shared | vllm | vllm:1701 | 0 / 1296 | 0 | — | — | — |
| h200_sxm | `vllm_unquantizedfusedmoe_triton_tritonexperts` | shared | vllm | vllm:52893 | 0 / 18306 | 4 | — | — | — |
| l40s | `moe_torch_flow` | shared | trtllm | trtllm:1960 | 0 / 1960 | 0 | — | — | — |
| l40s | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:153460 | 0 / 76385 | 2095 | — | — | — |
| l40s | `moe_torch_flow_nongated` | shared | trtllm | trtllm:32919 | 0 / 18746 | 15 | — | — | — |
| l40s | `sglang_fused_moe_triton` | shared | sglang | sglang:120390 | 0 / 65988 | 77 | — | — | — |
| l40s | `sglang_fused_moe_triton_situ_as_silu` | shared | sglang | sglang:2997 | 0 / 2997 | 0 | — | — | — |
| l40s | `sglang_marlin_moe` | shared | sglang | sglang:6156 | 0 / 3078 | 2 | — | — | — |
| l40s | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| l40s | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:1026 | 0 / 1026 | 0 | — | — | — |
| l40s | `vllm_compressedtensorswna16moe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| l40s | `vllm_fp8moe_triton_tritonexperts` | shared | vllm | vllm:34573 | 0 / 19515 | 2 | — | — | — |
| l40s | `vllm_gptossmxfp4moe_marlin_marlinexperts` | shared | vllm | vllm:3888 | 0 / 1944 | 1 | — | — | — |
| l40s | `vllm_modeloptfp8moe_triton_tritonexperts` | shared | vllm | vllm:30345 | 0 / 16200 | 2 | — | — | — |
| l40s | `vllm_unquantizedfusedmoe_triton_tritonexperts` | shared | vllm | vllm:34042 | 0 / 18069 | 3 | — | — | — |
| rtx_pro_6000_server | `moe_torch_flow_cutlass` | shared | trtllm | trtllm:250452 | 0 / 131058 | 102 | — | — | — |
| rtx_pro_6000_server | `moe_torch_flow_nongated` | shared | trtllm | trtllm:50785 | 0 / 27943 | 0 | — | — | — |
| rtx_pro_6000_server | `moe_torch_flow_triton_fp8_block` | shared | trtllm | trtllm:65205 | 0 / 32967 | 25 | — | — | — |
| rtx_pro_6000_server | `sglang_flashinfer_cutlass_moe` | shared | sglang | sglang:91530 | 0 / 56052 | 2 | — | — | — |
| rtx_pro_6000_server | `sglang_fused_moe_triton` | shared | sglang | sglang:167184 | 0 / 89829 | 6 | — | — | — |
| rtx_pro_6000_server | `sglang_marlin_moe` | shared | sglang | sglang:4456 | 0 / 4456 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_marlin_moe_situ_as_silu` | shared | sglang | sglang:551 | 0 / 551 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | shared | vllm | vllm:972 | 0 / 972 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:35505 | 0 / 19683 | 5 | — | — | — |
| rtx_pro_6000_server | `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_compressedtensorswna16moe_marlin_marlinexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_fp8moe_deepgemm_deepgemmexperts` | shared | vllm | vllm:16821 | 0 / 16821 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_fp8moe_triton_tritonexperts` | shared | vllm | vllm:2754 | 0 / 2754 | 0 | — | — | — |
| rtx_pro_6000_server | `vllm_gptossmxfp4moe_marlin_marlinexperts` | shared | vllm | vllm:3888 | 0 / 1944 | 1 | — | — | — |
| rtx_pro_6000_server | `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:29484 | 0 / 16200 | 6 | — | — | — |
| rtx_pro_6000_server | `vllm_unquantizedfusedmoe_flashinfer_cutlass_flashinferexperts` | shared | vllm | vllm:32670 | 0 / 17901 | 9 | — | — | — |
| rtx_pro_6000_server | `vllm_unquantizedfusedmoe_triton_tritonexperts` | shared | vllm | vllm:1053 | 0 / 1053 | 0 | — | — | — |

## `msa_context_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `MiniMaxM3SparseMSAImpl` | shared | vllm | vllm:33079 | 0 / 14396 | 122 | — | — | — |
| b200_sxm | `msa_fmha_sm100` | shared | trtllm | trtllm:8106 | 0 / 5070 | 0 | — | — | — |
| b200_sxm | `sglang_minimax_prefill_triton_sparse` | shared | sglang | sglang:60326 | 0 / 33649 | 0 | — | — | — |
| b300_sxm | `MiniMaxM3SparseMSAImpl` | shared | vllm | vllm:33095 | 0 / 14431 | 147 | — | — | — |
| b300_sxm | `msa_fmha_sm100` | shared | trtllm | trtllm:8106 | 0 / 5070 | 0 | — | — | — |
| b300_sxm | `sglang_minimax_prefill_triton_sparse` | shared | sglang | sglang:60534 | 0 / 34014 | 0 | — | — | — |
| gb200 | `MiniMaxM3SparseMSAImpl` | shared | vllm | vllm:33080 | 0 / 14388 | 99 | — | — | — |
| gb200 | `msa_fmha_sm100` | shared | trtllm | trtllm:8106 | 0 / 5070 | 0 | — | — | — |
| gb200 | `sglang_minimax_prefill_triton_sparse` | shared | sglang | sglang:60001 | 0 / 33660 | 0 | — | — | — |
| gb300 | `MiniMaxM3SparseMSAImpl` | shared | vllm | vllm:33160 | 0 / 14403 | 95 | — | — | — |
| gb300 | `msa_fmha_sm100` | shared | trtllm | trtllm:8106 | 0 / 5070 | 0 | — | — | — |
| gb300 | `sglang_minimax_prefill_triton_sparse` | shared | sglang | sglang:60947 | 0 / 34014 | 0 | — | — | — |
| h100_sxm | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:23958 | 0 / 11567 | 103 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:5791 | 0 / 3382 | 12 | — | — | — |
| h100_sxm | `sglang_minimax_prefill_triton_sparse` | shared | sglang | sglang:58866 | 0 / 32870 | 0 | — | — | — |
| h200_sxm | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:23956 | 0 / 11567 | 268 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:5808 | 0 / 3388 | 8 | — | — | — |
| h200_sxm | `sglang_minimax_prefill_triton_sparse` | shared | sglang | sglang:59957 | 0 / 33505 | 0 | — | — | — |
| l40s | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:12688 | 0 / 7808 | 35 | — | — | — |
| l40s | `default` | shared_fallback | trtllm | trtllm:5671 | 0 / 3313 | 11 | — | — | — |
| l40s | `sglang_minimax_prefill_triton_sparse` | shared | sglang | sglang:15731 | 0 / 15731 | 0 | — | — | — |
| rtx_pro_6000_server | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:15128 | 0 / 10248 | 77 | — | — | — |
| rtx_pro_6000_server | `default` | shared_fallback | trtllm | trtllm:3630 | 0 / 3630 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_minimax_prefill_triton_sparse` | shared | sglang | sglang:32732 | 0 / 32732 | 0 | — | — | — |

## `msa_generation_module_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `MiniMaxM3SparseMSAImpl` | shared | vllm | vllm:14352 | 0 / 14352 | 0 | — | — | — |
| b200_sxm | `msa_fmha_sm100` | shared | trtllm | trtllm:6624 | 0 / 6624 | 0 | — | — | — |
| b200_sxm | `sglang_minimax_decode_triton_sparse_topk_radix` | shared | sglang | sglang:268 | 0 / 149 | 0 | — | — | — |
| b200_sxm | `sglang_minimax_decode_triton_sparse_topk_split` | shared | sglang | sglang:3056 | 0 / 1698 | 69 | — | — | — |
| b300_sxm | `MiniMaxM3SparseMSAImpl` | shared | vllm | vllm:14352 | 0 / 14352 | 0 | — | — | — |
| b300_sxm | `msa_fmha_sm100` | shared | trtllm | trtllm:6624 | 0 / 6624 | 0 | — | — | — |
| b300_sxm | `sglang_minimax_decode_triton_sparse_topk_radix` | shared | sglang | sglang:270 | 0 / 150 | 0 | — | — | — |
| b300_sxm | `sglang_minimax_decode_triton_sparse_topk_split` | shared | sglang | sglang:3060 | 0 / 1700 | 87 | — | — | — |
| gb200 | `MiniMaxM3SparseMSAImpl` | shared | vllm | vllm:14352 | 0 / 14352 | 0 | — | — | — |
| gb200 | `msa_fmha_sm100` | shared | trtllm | trtllm:6624 | 0 / 6624 | 0 | — | — | — |
| gb200 | `sglang_minimax_decode_triton_sparse_topk_radix` | shared | sglang | sglang:268 | 0 / 149 | 0 | — | — | — |
| gb200 | `sglang_minimax_decode_triton_sparse_topk_split` | shared | sglang | sglang:3056 | 0 / 1698 | 17 | — | — | — |
| gb300 | `MiniMaxM3SparseMSAImpl` | shared | vllm | vllm:14352 | 0 / 14352 | 0 | — | — | — |
| gb300 | `msa_fmha_sm100` | shared | trtllm | trtllm:6624 | 0 / 6624 | 0 | — | — | — |
| gb300 | `sglang_minimax_decode_triton_sparse_topk_radix` | shared | sglang | sglang:270 | 0 / 150 | 0 | — | — | — |
| gb300 | `sglang_minimax_decode_triton_sparse_topk_split` | shared | sglang | sglang:3060 | 0 / 1700 | 67 | — | — | — |
| h100_sxm | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:9544 | 0 / 9544 | 0 | — | — | — |
| h100_sxm | `default` | shared_fallback | trtllm | trtllm:4360 | 0 / 4360 | 0 | — | — | — |
| h100_sxm | `sglang_minimax_decode_triton_sparse_topk_radix` | shared | sglang | sglang:528 | 0 / 298 | 0 | — | — | — |
| h100_sxm | `sglang_minimax_decode_triton_sparse_topk_split` | shared | sglang | sglang:2755 | 0 / 1533 | 16 | — | — | — |
| h200_sxm | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:9562 | 0 / 9562 | 0 | — | — | — |
| h200_sxm | `default` | shared_fallback | trtllm | trtllm:4376 | 0 / 4376 | 0 | — | — | — |
| h200_sxm | `sglang_minimax_decode_triton_sparse_topk_radix` | shared | sglang | sglang:550 | 0 / 308 | 1 | — | — | — |
| h200_sxm | `sglang_minimax_decode_triton_sparse_topk_split` | shared | sglang | sglang:2768 | 0 / 1539 | 19 | — | — | — |
| l40s | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:9512 | 0 / 9512 | 0 | — | — | — |
| l40s | `default` | shared_fallback | trtllm | trtllm:4292 | 0 / 4292 | 0 | — | — | — |
| l40s | `sglang_minimax_decode_triton_sparse_topk_radix` | shared | sglang | sglang:370 | 0 / 370 | 0 | — | — | — |
| l40s | `sglang_minimax_decode_triton_sparse_topk_split` | shared | sglang | sglang:515 | 0 / 515 | 0 | — | — | — |
| rtx_pro_6000_server | `MiniMaxM3SparseTritonImpl` | shared | vllm | vllm:11402 | 0 / 11402 | 0 | — | — | — |
| rtx_pro_6000_server | `default` | shared_fallback | trtllm | trtllm:2730 | 0 / 2730 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_minimax_decode_triton_sparse_topk_radix` | shared | sglang | sglang:294 | 0 / 294 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang_minimax_decode_triton_sparse_topk_split` | shared | sglang | sglang:1531 | 0 / 1531 | 0 | — | — | — |

## `nccl_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b300_sxm | `NCCL` | shared | trtllm | trtllm:1512 | 0 / 126 | 0 | — | — | — |

## `scale_matrix_perf.parquet`

| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | dedup rows | median % | p95 % | max % |
|---|---|---|---|---|---|---|---|---|---|
| b200_sxm | `sglang` | shared | sglang | sglang:3256 | 0 / 1628 | 7 | — | — | — |
| b200_sxm | `static_scaled_fp8_quant` | shared | vllm | vllm:3256 | 0 / 1628 | 11 | — | — | — |
| b200_sxm | `torch_ops` | shared | trtllm | trtllm:3256 | 0 / 3256 | 0 | — | — | — |
| b300_sxm | `sglang` | shared | sglang | sglang:3256 | 0 / 1628 | 0 | — | — | — |
| b300_sxm | `static_scaled_fp8_quant` | shared | vllm | vllm:3256 | 0 / 1628 | 14 | — | — | — |
| b300_sxm | `torch_ops` | shared | trtllm | trtllm:3256 | 0 / 1628 | 15 | — | — | — |
| gb200 | `sglang` | shared | sglang | sglang:3256 | 0 / 1628 | 0 | — | — | — |
| gb200 | `static_scaled_fp8_quant` | shared | vllm | vllm:3256 | 0 / 1628 | 9 | — | — | — |
| gb200 | `torch_ops` | shared | trtllm | trtllm:3256 | 0 / 1628 | 11 | — | — | — |
| gb300 | `sglang` | shared | sglang | sglang:4884 | 0 / 1628 | 1 | — | — | — |
| gb300 | `static_scaled_fp8_quant` | shared | vllm | vllm:4884 | 0 / 1628 | 25 | — | — | — |
| gb300 | `torch_ops` | shared | trtllm | trtllm:3256 | 0 / 1628 | 7 | — | — | — |
| h100_sxm | `sglang` | shared | sglang | sglang:3256 | 0 / 1628 | 0 | — | — | — |
| h100_sxm | `static_scaled_fp8_quant` | shared | vllm | vllm:3256 | 0 / 1628 | 0 | — | — | — |
| h100_sxm | `torch_ops` | shared | trtllm | trtllm:3256 | 0 / 1628 | 17 | — | — | — |
| h200_sxm | `sglang` | shared | sglang | sglang:3256 | 0 / 1628 | 0 | — | — | — |
| h200_sxm | `static_scaled_fp8_quant` | shared | vllm | vllm:3256 | 0 / 1628 | 2 | — | — | — |
| h200_sxm | `torch_ops` | shared | trtllm | trtllm:3256 | 0 / 1628 | 24 | — | — | — |
| l40s | `sglang` | shared | sglang | sglang:3256 | 0 / 1628 | 66 | — | — | — |
| l40s | `static_scaled_fp8_quant` | shared | vllm | vllm:3256 | 0 / 1628 | 272 | — | — | — |
| l40s | `torch_ops` | shared | trtllm | trtllm:3256 | 0 / 1628 | 0 | — | — | — |
| rtx_pro_6000_server | `sglang` | shared | sglang | sglang:3256 | 0 / 1628 | 1 | — | — | — |
| rtx_pro_6000_server | `static_scaled_fp8_quant` | shared | vllm | vllm:3256 | 0 / 1628 | 7 | — | — | — |
| rtx_pro_6000_server | `torch_ops` | shared | trtllm | trtllm:3256 | 0 / 1628 | 2 | — | — | — |

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


### `shared` (211 kernel sources)

| kernel_source | frameworks | op files | systems | rows |
|---|---|---|---|---|
| `causal_conv1d_fn` | sglang, trtllm, vllm | gdn_perf.parquet, kda_perf.parquet, mamba2_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 223,690 |
| `causal_conv1d_fn_qkv3` | sglang, vllm | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 7,012 |
| `causal_conv1d_fn_token_major` | trtllm | mamba2_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 4,778 |
| `causal_conv1d_update` | sglang, trtllm, vllm | gdn_perf.parquet, kda_perf.parquet, mamba2_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 21,550 |
| `chunk_gated_delta_rule` | sglang, trtllm | gdn_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 72,299 |
| `chunk_gated_delta_rule_flashinfer` | trtllm, vllm | gdn_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 120,360 |
| `chunk_gated_delta_rule_triton` | vllm | gdn_perf.parquet | l40s, rtx_pro_6000_server | 12,265 |
| `chunk_kda` | sglang | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 5,103 |
| `chunk_kda_with_fused_gate` | vllm | kda_perf.parquet | l40s | 414 |
| `compressed_flashmla` | sglang | dsv4_csa_context_module_perf.parquet, dsv4_csa_generation_module_perf.parquet, dsv4_hca_context_module_perf.parquet, dsv4_hca_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 1,368,500 |
| `CutlassFp8BlockScaledMMKernel` | vllm | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 109,742 |
| `CutlassFP8ScaledMMLinearKernel` | vllm | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,137,698 |
| `deep_gemm.fp8_mqa_logits` | sglang | glm5_mqa_logits_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 38,762 |
| `deep_gemm.fp8_paged_mqa_logits` | sglang | dsv4_paged_mqa_logits_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 46,656 |
| `deepep` | sglang, trtllm, vllm | moe_a2a_perf.parquet, wideep_deepep_ll_perf.parquet, wideep_deepep_normal_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 23,512 |
| `deepepmoe` | sglang | wideep_context_moe_perf.parquet, wideep_generation_moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 6,333 |
| `deepgemm` | trtllm | gemm_perf.parquet, moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 551,330 |
| `deepgemm_megamoe` | sglang | dsv4_megamoe_module_perf.parquet | b200_sxm, gb200, gb300 | 776 |
| `DeepGemmFp8BlockScaledMMKernel` | vllm | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 641,938 |
| `deepseek_v3` | sglang | wideep_context_mlp_perf.parquet, wideep_generation_mlp_perf.parquet | h100_sxm, h200_sxm | 72 |
| `DeepseekV4TrtllmAttention` | trtllm | dsv4_csa_context_module_perf.parquet, dsv4_csa_generation_module_perf.parquet, dsv4_hca_context_module_perf.parquet, dsv4_hca_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 145,545 |
| `dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant` | vllm | computescale_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 27,513 |
| `fa3` | sglang | context_attention_perf.parquet, generation_attention_perf.parquet, wideep_context_mla_perf.parquet, wideep_generation_mla_perf.parquet | h100_sxm, h200_sxm | 296,574 |
| `fast_topk_transform_fused` | sglang | glm5_topk_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 48,196 |
| `flash_attention` | sglang | context_attention_perf.parquet, context_mla_perf.parquet, generation_attention_perf.parquet, generation_mla_perf.parquet | a100_sxm, h100_sxm, h200_sxm | 42,304 |
| `flash_attention_v3` | sglang | encoder_attention_perf.parquet | h100_sxm, h200_sxm | 31,816 |
| `flash_attention_v4` | sglang | encoder_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 46,074 |
| `FLASH_ATTN` | vllm | mla_context_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 133,947 |
| `FLASH_ATTN_MLA` | vllm | mla_context_module_perf.parquet, mla_generation_module_perf.parquet | h100_sxm, h200_sxm | 31,424 |
| `FLASH_ATTN_MLA_SPARSE` | vllm | dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet | h100_sxm, h200_sxm | 428,148 |
| `flash_mla_sparse_fwd` | sglang | glm5_dsa_attn_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 39,225 |
| `flashinfer` | sglang | context_attention_perf.parquet, generation_attention_perf.parquet, wideep_context_mla_perf.parquet, wideep_generation_mla_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 268,399 |
| `flashinfer_chunk_gated_delta_rule` | sglang | gdn_perf.parquet | h100_sxm, h200_sxm | 9,058 |
| `flashinfer_gated_delta_rule_decode` | sglang | gdn_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 208 |
| `FLASHINFER_MLA` | vllm | mla_context_module_perf.parquet, mla_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 167,416 |
| `FLASHINFER_MLA_SPARSE` | vllm | dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 1,180,816 |
| `FLASHINFER_MLA_SPARSE_DSV4` | vllm | dsv4_csa_context_module_perf.parquet, dsv4_csa_generation_module_perf.parquet, dsv4_hca_context_module_perf.parquet, dsv4_hca_generation_module_perf.parquet | rtx_pro_6000_server | 14,678 |
| `FLASHINFER_MLA_SPARSE_SM120` | vllm | dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet | rtx_pro_6000_server | 119,020 |
| `FlashInferCuteDslNvFp4LinearKernel` | vllm | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 696,414 |
| `FlashInferCutlassNvFp4LinearKernel` | vllm | gemm_perf.parquet | rtx_pro_6000_server | 73,260 |
| `FlashInferFp8BlockScaledMMKernel` | vllm | gemm_perf.parquet | h100_sxm, h200_sxm | 42,972 |
| `flashkda_fwd` | vllm | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 5,402 |
| `FLASHMLA` | vllm | mla_context_module_perf.parquet, mla_generation_module_perf.parquet | h100_sxm, h200_sxm | 29,204 |
| `FLASHMLA_SPARSE` | vllm | dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 1,181,841 |
| `FLASHMLA_SPARSE_DSV4` | vllm | dsv4_csa_context_module_perf.parquet, dsv4_csa_generation_module_perf.parquet, dsv4_hca_context_module_perf.parquet, dsv4_hca_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 264,167 |
| `fused_kda_decode` | vllm | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 308 |
| `fused_kda_decode_mtp_dspark` | sglang | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 428 |
| `fused_qkvzba_causal_conv1d_update_contiguous` | sglang | gdn_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 3,696 |
| `fused_recurrent_gated_delta_rule` | sglang, trtllm | gdn_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 774 |
| `fused_recurrent_gated_delta_rule_packed_decode` | sglang, vllm | gdn_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 16,950 |
| `fused_recurrent_kda` | vllm | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,309 |
| `fused_recurrent_kda_packed_decode` | sglang, vllm | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 651 |
| `fused_sigmoid_gating_delta_rule_update` | sglang | kda_perf.parquet | b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 630 |
| `fused_sigmoid_gating_delta_rule_update_flashinfer` | trtllm | gdn_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 2,772 |
| `fused_sigmoid_gating_delta_rule_update_triton` | trtllm | gdn_perf.parquet | l40s, rtx_pro_6000_server | 924 |
| `kda_fused_decode` | sglang | kda_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 77 |
| `mamba_chunk_scan_combined` | trtllm | mamba2_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 4,777 |
| `MarlinFP8ScaledMMLinearKernel` | vllm | gemm_perf.parquet | l40s | 37,518 |
| `MiniMaxM3SparseMSAImpl` | vllm | msa_context_module_perf.parquet, msa_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 189,822 |
| `MiniMaxM3SparseTritonImpl` | vllm | msa_context_module_perf.parquet, msa_generation_module_perf.parquet | h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 115,750 |
| `mla_trtllm_mla` | sglang | mla_context_module_perf.parquet | b200_sxm, b300_sxm | 436 |
| `moe_torch_flow` | trtllm | moe_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s | 93,599 |
| `moe_torch_flow_cutlass` | trtllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,957,168 |
| `moe_torch_flow_min_latency` | trtllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 229,950 |
| `moe_torch_flow_nongated` | trtllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 391,660 |
| `moe_torch_flow_triton_fp8_block` | trtllm | moe_perf.parquet | rtx_pro_6000_server | 65,205 |
| `msa_fmha_sm100` | trtllm | msa_context_module_perf.parquet, msa_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 58,920 |
| `NCCL` | trtllm | nccl_perf.parquet | b300_sxm | 1,512 |
| `NVLinkOneSided` | trtllm | trtllm_alltoall_perf.parquet | gb200 | 296 |
| `NVLinkTwoSided` | trtllm | trtllm_alltoall_perf.parquet | gb200 | 1,800 |
| `selective_state_update_flashinfer` | trtllm | mamba2_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 528 |
| `sglang` | sglang | computescale_perf.parquet, gemm_perf.parquet, scale_matrix_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 98,806 |
| `sglang.hc_post[tilelang]` | sglang | mhc_module_perf.parquet | b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 175 |
| `sglang.hc_pre[tilelang,prenorm_gemm=deepgemm,norm=fused]` | sglang | mhc_module_perf.parquet | b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 175 |
| `sglang.kernels.ops.layernorm.mhc.hc_contract` | sglang | mhc_module_perf.parquet | b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 175 |
| `sglang.kernels.ops.layernorm.mhc.hc_expand` | sglang | mhc_module_perf.parquet | b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 175 |
| `sglang.srt.layers.engram.Engram.apply_gate+sglang.srt.layers.engram.EngramEmbedding._owned_rows+sglang.srt.layers.linear.ReplicatedLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | sglang | dsv41_module_perf.parquet | b200_sxm, gb200, h100_sxm, h200_sxm | 240 |
| `sglang.srt.layers.linear.MergedColumnParallelLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | sglang | dsv41_module_perf.parquet | b200_sxm, gb200, h100_sxm, h200_sxm | 120 |
| `sglang.srt.layers.linear.RowParallelLinear.forward/sglang.srt.layers.quantization.fp8.Fp8LinearMethod` | sglang | dsv41_module_perf.parquet | b200_sxm, gb200, h100_sxm, h200_sxm | 120 |
| `sglang.srt.layers.vocab_parallel_embedding.ParallelLMHead.quant_method.apply/sglang.srt.layers.quantization.unquant.UnquantizedEmbeddingMethod` | sglang | gemm_perf.parquet | b200_sxm, gb200, h100_sxm, h200_sxm | 120 |
| `sglang.srt.models.deepseek_v2.MoEGate.forward` | sglang | gemm_perf.parquet | b200_sxm, gb200, h100_sxm, h200_sxm | 60 |
| `sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer._hc_mix_and_combine+sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_post` | sglang | dsv41_module_perf.parquet | b200_sxm, gb200, h100_sxm, h200_sxm | 60 |
| `sglang.srt.models.deepseek_v4.MQALayer.forward` | sglang | dsv41_module_perf.parquet | b200_sxm, gb200, h100_sxm, h200_sxm | 9,040 |
| `sglang0.5.20|DeepseekV2AttentionMLA|IndexerKPool|qkv_a=Fp8LinearMethod|q_b=Fp8LinearMethod|kv_b=UnquantizedLinearMethod|o=Fp8LinearMethod|attn_backend=dsa|method=MLA` | sglang | glm53_attention_module_perf.parquet | b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 3,659 |
| `sglang0.5.20|DeepseekV2AttentionMLA|IndexerKPool|qkv_a=UnquantizedLinearMethod|q_b=UnquantizedLinearMethod|kv_b=UnquantizedLinearMethod|o=UnquantizedLinearMethod|attn_backend=dsa|method=MLA` | sglang | glm53_attention_module_perf.parquet | b300_sxm, gb200, gb300 | 3,217 |
| `SGLang_CustomAllReduce_eager` | sglang | custom_allreduce_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 2,024 |
| `SGLang_CustomAllReduce_graph` | sglang | custom_allreduce_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 2,024 |
| `sglang_cutedsl_tgv_bf16` | sglang | gemm_perf.parquet | b300_sxm, gb200, gb300 | 3,929 |
| `sglang_deepgemm_gemm_nt_f8f8bf16` | sglang | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 629,739 |
| `sglang_dsa_dense_mha_fa3` | sglang | dsa_context_module_perf.parquet | h100_sxm, h200_sxm | 145,111 |
| `sglang_dsa_dense_mha_trtllm_ragged` | sglang | dsa_context_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 277,456 |
| `sglang_dsa_indexer_fa3` | sglang | dsa_generation_module_perf.parquet | h100_sxm, h200_sxm | 9,008 |
| `sglang_dsa_indexer_flashmla_kv` | sglang | dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet | h100_sxm, h200_sxm | 80,896 |
| `sglang_dsa_indexer_flashmla_sparse` | sglang | dsa_context_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 223,605 |
| `sglang_dsa_indexer_trtllm` | sglang | dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 176,983 |
| `sglang_dsa_skip_indexer_fa3` | sglang | dsa_generation_module_perf.parquet | h100_sxm, h200_sxm | 4,912 |
| `sglang_dsa_skip_indexer_flashmla_kv` | sglang | dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet | h100_sxm, h200_sxm | 49,141 |
| `sglang_dsa_skip_indexer_flashmla_sparse` | sglang | dsa_context_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 126,011 |
| `sglang_dsa_skip_indexer_trtllm` | sglang | dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 100,438 |
| `sglang_flashinfer_bf16_direct` | sglang | gemm_perf.parquet | b300_sxm, gb200, gb300 | 24 |
| `sglang_flashinfer_bf16_splitk` | sglang | gemm_perf.parquet | b300_sxm, gb200, gb300 | 105 |
| `sglang_flashinfer_bmm_fp8` | sglang | mla_bmm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 5,088 |
| `sglang_flashinfer_cutedsl_nvfp4` | sglang | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 464,350 |
| `sglang_flashinfer_cutlass_moe` | sglang | moe_perf.parquet | h100_sxm, h200_sxm, rtx_pro_6000_server | 126,066 |
| `sglang_flashinfer_cutlass_nvfp4` | sglang | gemm_perf.parquet | rtx_pro_6000_server | 59,126 |
| `sglang_flashinfer_trtllm_moe` | sglang | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 1,127,207 |
| `sglang_fused_moe_triton` | sglang | moe_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,211,403 |
| `sglang_fused_moe_triton_situ_as_silu` | sglang | moe_perf.parquet | l40s | 2,997 |
| `sglang_marlin_moe` | sglang | moe_perf.parquet | a100_sxm, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 86,185 |
| `sglang_marlin_moe_situ_as_silu` | sglang | moe_perf.parquet | h100_sxm, h200_sxm, rtx_pro_6000_server | 1,580 |
| `sglang_minimax_decode_triton_sparse_topk_radix` | sglang | msa_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 2,818 |
| `sglang_minimax_decode_triton_sparse_topk_split` | sglang | msa_generation_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 19,801 |
| `sglang_minimax_prefill_triton_sparse` | sglang | msa_context_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 409,094 |
| `sglang_mxfp4_flashinfer_trtllm_moe` | sglang | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 121,398 |
| `sglang_mxfp4_humming_moe` | sglang | moe_perf.parquet | h100_sxm, h200_sxm | 30 |
| `sglang_sgl_kernel_bmm_fp8` | sglang | mla_bmm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 5,724 |
| `sglang_sgl_kernel_fp8_scaled_mm` | sglang | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 760,646 |
| `sglang_tilelang_mhc_post` | sglang | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 903 |
| `sglang_tilelang_mhc_pre` | sglang | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 874 |
| `sglang_torch_bmm` | sglang | mla_bmm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 10,812 |
| `sglang_torch_linear` | sglang | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 908,794 |
| `sglang_torch_mhc_pre` | sglang | mhc_module_perf.parquet | rtx_pro_6000_server | 67 |
| `sglang_triton_w8a8_block_fp8` | sglang | gemm_perf.parquet | b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 9,842 |
| `static_scaled_fp8_quant` | vllm | scale_matrix_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 27,676 |
| `topk_transform_v1` | sglang | dsv4_csa_topk_calib_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 40,124 |
| `topk_transform_v2` | sglang | dsv4_csa_topk_calib_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 2,304 |
| `torch.nn.functional.linear` | vllm | gemm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,137,658 |
| `torch_flow` | trtllm | context_attention_perf.parquet, encoder_attention_perf.parquet, gemm_perf.parquet, generation_attention_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 3,825,706 |
| `torch_flow_flashinfer` | trtllm | context_attention_perf.parquet, generation_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 42,856 |
| `torch_ops` | trtllm | computescale_perf.parquet, scale_matrix_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 52,055 |
| `triton` | sglang | context_attention_perf.parquet, context_mla_perf.parquet, encoder_attention_perf.parquet, generation_attention_perf.parquet, generation_mla_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 350,168 |
| `TRITON_MLA` | vllm | mla_context_module_perf.parquet, mla_generation_module_perf.parquet | l40s, rtx_pro_6000_server | 17,972 |
| `TritonFp8BlockScaledMMKernel` | vllm | gemm_perf.parquet | l40s | 35,676 |
| `trt_flow_/smooth_quant_gemm_L96/PLUGIN_V2_SmoothQuantGemm_0` | trtllm | gemm_perf.parquet | a100_sxm | 6,048 |
| `trt_flow_/weight_only_quant_matmul_L257/PLUGIN_V2_WeightOnlyQuantMatmul_0` | trtllm | gemm_perf.parquet | a100_sxm | 12,096 |
| `TRTLLM` | trtllm | custom_allreduce_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,518 |
| `trtllm_bmm_out` | trtllm | mla_bmm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 8,480 |
| `trtllm_bmm_out_dequant_bf16` | trtllm | mla_bmm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 4,240 |
| `trtllm_fp8_block_scaling_bmm_out` | trtllm | mla_bmm_perf.parquet | h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 4,240 |
| `trtllm_mha` | sglang | context_attention_perf.parquet, generation_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 565,262 |
| `trtllm_mhc_post_mapping` | trtllm | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,110 |
| `trtllm_mhc_pre_dg_nosplit` | trtllm | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 312 |
| `trtllm_mhc_pre_dg_splitk` | trtllm | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 392 |
| `trtllm_mhc_pre_fma` | trtllm | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 388 |
| `trtllm_mla` | sglang | context_mla_perf.parquet, generation_mla_perf.parquet, wideep_context_mla_perf.parquet, wideep_generation_mla_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 70,527 |
| `TRTLLM_MNNVL_oneshot` | trtllm | custom_allreduce_perf.parquet | gb300 | 19 |
| `TRTLLM_MNNVL_twoshot` | trtllm | custom_allreduce_perf.parquet | gb300 | 27 |
| `TRTLLM_RAGGED` | vllm | mla_context_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 50,135 |
| `vllm.mhc_fused_post_pre_tilelang[fused_post_gemm,norm=fused]` | vllm | mhc_module_perf.parquet | b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 30 |
| `vllm.mhc_fused_post_pre_tilelang[post+prenorm_gemm=deepgemm,norm=fused]` | vllm | mhc_module_perf.parquet | b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 144 |
| `vllm.mhc_post_tilelang` | vllm | mhc_module_perf.parquet | b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 175 |
| `vllm.mhc_pre_tilelang[prenorm_gemm=deepgemm,norm=fused]` | vllm | mhc_module_perf.parquet | b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 175 |
| `vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang` | vllm | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,460 |
| `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` | vllm | mhc_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,455 |
| `vllm.model_executor.layers.mhc.hc_contract` | vllm | mhc_module_perf.parquet | b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 175 |
| `vllm.model_executor.layers.mhc.hc_expand` | vllm | mhc_module_perf.parquet | b300_sxm, gb200, gb300, h100_sxm, h200_sxm | 175 |
| `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | vllm | dsv4_paged_mqa_logits_module_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 62,688 |
| `vllm0.31.0|Glm5NextMLAAttention|SparseAttnIndexerKpool|projections=UnquantizedLinearMethod|mla_impl=FlashInferMLASparseImpl` | vllm | glm53_attention_module_perf.parquet | b300_sxm, gb200, gb300 | 6,019 |
| `vllm0.31.0|Glm5NextMLAAttention|SparseAttnIndexerKpool|projections=UnquantizedLinearMethod|mla_impl=FlashInferMLASparseSM90Impl` | vllm | glm53_attention_module_perf.parquet | h100_sxm, h200_sxm | 1,374 |
| `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm | 1,944 |
| `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 17,496 |
| `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutedsl_flashinfercutedslexperts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 3,888 |
| `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_cutlass_flashinferexperts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, rtx_pro_6000_server | 43,281 |
| `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmodular` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 36,936 |
| `vllm_compressedtensorsw4a4nvfp4moe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 179,196 |
| `vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 14,715 |
| `vllm_compressedtensorswna16moe_marlin_marlinexperts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 8,424 |
| `vLLM_custom_eager` | vllm | custom_allreduce_perf.parquet | a100_sxm, b200_sxm, b300_sxm, b60, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,840 |
| `vLLM_custom_graph` | vllm | custom_allreduce_perf.parquet | a100_sxm, b200_sxm, b300_sxm, b60, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 1,771 |
| `vllm_default` | vllm | gemm_perf.parquet | a100_sxm, b60 | 70,938 |
| `vllm_flash_attn` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | a100_sxm, b60 | 97,251 |
| `vllm_flash_attn_fa2` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, l40s, rtx_pro_6000_server | 245,795 |
| `vllm_flash_attn_fa3` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | h100_sxm, h200_sxm | 700,938 |
| `vllm_flash_attn_fa4` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | h100_sxm, h200_sxm | 9,256 |
| `vllm_flashinfer` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | b200_sxm | 71,600 |
| `vllm_flashinfer_fidecode` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | l40s, rtx_pro_6000_server | 92,723 |
| `vllm_flashinfer_fiprefill` | vllm | context_attention_perf.parquet | l40s, rtx_pro_6000_server | 90,646 |
| `vllm_flashinfer_flashinfertrtllmapidecode` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 268,320 |
| `vllm_flashinfer_flashinfertrtllmapidecode_trtllm_gen` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 264,396 |
| `vllm_flashinfer_flashinfertrtllmapidecode_xqa` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | rtx_pro_6000_server | 31,021 |
| `vllm_flashinfer_trtllm_moe_fp4` | vllm | moe_perf.parquet | b200_sxm | 10,287 |
| `vllm_flashinfer_trtllmdecode` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 254,018 |
| `vllm_flashinfer_trtllmprefill` | vllm | context_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 556,588 |
| `vllm_fp8moe_deepgemm_deepgemmexperts` | vllm | moe_perf.parquet | rtx_pro_6000_server | 16,821 |
| `vllm_fp8moe_flashinfer_cutlass_flashinferexperts` | vllm | moe_perf.parquet | h100_sxm, h200_sxm | 60,426 |
| `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmodular` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 50,220 |
| `vllm_fp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 152,118 |
| `vllm_fp8moe_triton_tritonexperts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 93,265 |
| `vllm_fused_moe` | vllm | moe_perf.parquet | a100_sxm, b200_sxm, h200_sxm | 126,055 |
| `vllm_gptossmxfp4moe_flashinfer_trtllm_mxfp4_bf16_trtllmmxfp4expertsmonolithic` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 23,328 |
| `vllm_gptossmxfp4moe_marlin_marlinexperts` | vllm | moe_perf.parquet | l40s, rtx_pro_6000_server | 7,776 |
| `vllm_gptossmxfp4moe_triton_oaitritonmxfp4expertsmonolithic` | vllm | moe_perf.parquet | h100_sxm, h200_sxm | 11,664 |
| `vllm_marlin_int4_moe` | vllm | moe_perf.parquet | b200_sxm, h200_sxm | 20,415 |
| `vllm_modeloptfp8moe_flashinfer_cutlass_flashinferexperts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 140,454 |
| `vllm_modeloptfp8moe_flashinfer_trtllm_trtllmfp8expertsmonolithic` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 145,169 |
| `vllm_modeloptfp8moe_triton_tritonexperts` | vllm | moe_perf.parquet | l40s | 30,345 |
| `vllm_modeloptnvfp4fusedmoe_flashinfer_trtllm_trtllmnvfp4expertsmonolithic` | vllm | moe_perf.parquet | b300_sxm, gb200, gb300 | 2,673 |
| `vllm_mxfp4_moe` | vllm | moe_perf.parquet | b200_sxm, h200_sxm | 2,839 |
| `vllm_mxfp4moe_flashinfer_trtllm_mxfp4_mxfp8_trtllmmxfp4expertsmodular` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 26,244 |
| `vllm_torch_bmm` | vllm | mla_bmm_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 13,156 |
| `vllm_triton_attn` | vllm | context_attention_perf.parquet, generation_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 78,559 |
| `vllm_unquantizedfusedmoe_flashinfer_cutlass_flashinferexperts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, rtx_pro_6000_server | 54,702 |
| `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16experts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 103,173 |
| `vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16expertsmonolithic` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 64,276 |
| `vllm_unquantizedfusedmoe_triton_tritonexperts` | vllm | moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 149,165 |
| `vllm_vit_flash_attn_fa2` | vllm | encoder_attention_perf.parquet | l40s, rtx_pro_6000_server | 30,716 |
| `vllm_vit_flash_attn_fa3` | vllm | encoder_attention_perf.parquet | h100_sxm, h200_sxm | 47,174 |
| `vllm_vit_flash_attn_fa4` | vllm | encoder_attention_perf.parquet | b200_sxm, b300_sxm, gb200, gb300 | 92,148 |
| `vllm_xpu_moe` | vllm | moe_perf.parquet | b60 | 3,996 |
| `vllm_xpu_moe_mxfp4` | vllm | moe_perf.parquet | b60 | 2,592 |
| `wideep_compute_cutlass` | trtllm | wideep_moe_perf.parquet | b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, rtx_pro_6000_server | 26,298 |

### `shared_fallback` (1 kernel sources)

| kernel_source | frameworks | op files | systems | rows |
|---|---|---|---|---|
| `default` | sglang, trtllm | context_mla_perf.parquet, dsa_context_module_perf.parquet, dsa_generation_module_perf.parquet, generation_mla_perf.parquet, mla_bmm_perf.parquet, mla_context_module_perf.parquet, mla_generation_module_perf.parquet, msa_context_module_perf.parquet, msa_generation_module_perf.parquet | a100_sxm, b200_sxm, b300_sxm, gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server | 598,936 |
