import sys; sys.argv=['x']
from collector.vllm.collect_moe import run_moe_torch
run_moe_torch('fp8_block', [1, 4096], 7168, 2048, 8, 256, 1, 1, 'deepseek-ai/DeepSeek-V3', 'balanced', 0.0, perf_filename='/tmp/moe_perf.txt', device='cuda:0')
