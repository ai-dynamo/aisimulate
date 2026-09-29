import sys; sys.argv=['x']
from collector.vllm.collect_moe import run_moe_torch
run_moe_torch('w4a16_mxfp4', [1, 4096], 2880, 2880, 4, 128, 1, 1, 'openai/gpt-oss-120b', 'balanced', 0.0, perf_filename='/tmp/moe_perf.txt', device='cuda:0')
