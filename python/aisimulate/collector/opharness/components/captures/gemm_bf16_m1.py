import sys; sys.argv=['x']
from collector.vllm.collect_gemm import run_gemm
run_gemm('bfloat16', 1, 4096, 4096, perf_filename='/tmp/gemm_perf.txt', device='cuda:0')   # Llama decode shape
