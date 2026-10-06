import sys; sys.argv=['x']
# trtllm DSV4 module collector: one shape = one row, in-process; mode comes from the perf
# filename (collect_dsv4_attn.run_dsv4_attn_worker). Serving record cell: isl 4096, kv auto
# (Hopper DSV4 = fp8_ds_mla pool), FP8 checkpoint -> fp8_block projection GEMMs, prefix 0.
from collector.trtllm.collect_dsv4_attn import run_dsv4_attn_worker
run_dsv4_attn_worker(4096, 1, 1, 'fp8', 'bfloat16', 'fp8_block', 'sgl-project/DeepSeek-V4-Flash-FP8', 'hca', 0, perf_filename='/tmp/dsv4_hca_generation_perf.txt', device='cuda:0')
