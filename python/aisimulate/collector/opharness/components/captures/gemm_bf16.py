import sys; sys.argv=['x']
from collector.vllm.collect_gemm import run_gemm
# Llama-3.1-8B shape (4096x4096, k=4096): serving ref Llama-3.1-8B (cublas)
run_gemm('bfloat16', 4096, 4096, 4096, perf_filename='/tmp/gemm_perf.txt', device='cuda:0')
