import sys; sys.argv=['x']
from collector.vllm.collect_kda import run_kda_torch
run_kda_torch('context', 7168, 4, 96, 128, 96, 128, [1], [4096], 'moonshotai/Kimi-K3', perf_filename='/tmp/kda_perf.txt', device='cuda:0')
