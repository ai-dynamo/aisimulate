import sys; sys.argv=['x']
from collector.sglang.deepseekv4_sparse_modules import run_dsv4_sparse_kernel_worker
run_dsv4_sparse_kernel_worker('sgl-project/DeepSeek-V4-Flash-FP8', 'csa_attn', 1, perf_filename='/tmp/sgl_dsv4_csa_attn_perf.txt', device='cuda:0')
