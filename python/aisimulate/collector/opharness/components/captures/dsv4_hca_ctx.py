import sys; sys.argv=['x']
from collector.vllm.collect_dsv4_attn import run_dsv4_attn_worker
run_dsv4_attn_worker(4096, 1, 1, 'fp8', 'bfloat16', 'fp8_block', 'sgl-project/DeepSeek-V4-Flash-FP8', 'hca', None, 0, perf_filename='/tmp/dsv4_hca_context_perf.txt', device='cuda:0')
