import sys; sys.argv=['x']
from collector.vllm.collect_gemm import run_gemm
# per-tensor fp8 (Qwen3-32B-FP8-Static-PerTensor shape 5120): serving ref that model
run_gemm('fp8', 4096, 5120, 5120, perf_filename='/tmp/gemm_perf.txt', device='cuda:0')
