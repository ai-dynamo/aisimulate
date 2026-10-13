import sys; sys.argv=['x']
# trtllm DSV4 module collector: one shape = one row, in-process; mode comes from the perf
# filename (collect_dsv4_attn.run_dsv4_attn_worker). Serving record cell: isl 4096, kv auto
# (Hopper DSV4 = fp8_ds_mla pool; on SM100 rc29 cannot serve fp8_ds_mla and kv auto resolves to the bf16
# DeepseekV4CacheManager pool -> the sm100 lane is 'bf16', B200 2026-10-04), FP8 checkpoint -> fp8_block projection GEMMs, prefix 0.
from collector.trtllm.collect_dsv4_attn import run_dsv4_attn_worker, _serving_kv_cache_dtype
run_dsv4_attn_worker(4096, 1, 1, _serving_kv_cache_dtype(), 'bfloat16', 'fp8_block', 'sgl-project/DeepSeek-V4-Flash-FP8', 'csa', 0, perf_filename='/tmp/dsv4_csa_generation_perf.txt', device='cuda:0')
