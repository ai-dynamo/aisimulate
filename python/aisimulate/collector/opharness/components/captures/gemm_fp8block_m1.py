import sys; sys.argv=['x']
from collector.vllm.collect_gemm import run_gemm
run_gemm('fp8_block', 1, 7168, 16384, perf_filename='/tmp/gemm_perf.txt', device='cuda:0')  # DSV3.2 decode shape
