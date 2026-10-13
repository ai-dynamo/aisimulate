import sys; sys.argv=['x']
from collector.sglang.collect_gdn import run_gdn_torch
run_gdn_torch('context', 1024, 4, 16, 128, 16, 128, [1], [4096], 'Qwen/Qwen3.5-0.8B', 'float32', perf_filename='/tmp/gdn_perf.txt', device='cuda:0')
