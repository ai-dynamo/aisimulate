import sys; sys.argv=['x']
from collector.vllm.collect_dsv4_attn import run_dsv4_sparse_kernel_worker
run_dsv4_sparse_kernel_worker(1, 4096, 0, 1, 'paged_mqa_logits', 'sgl-project/DeepSeek-V4-Flash-FP8', perf_filename='/tmp/dsv4_paged_mqa_logits_perf.txt', device='cuda:0')
