import sys; sys.argv = ['x']
from collector.trtllm.collect_mamba2 import run_mamba2_torch
run_mamba2_torch('generation', 8192, 256, 4, 256, 64, 8, 128, [1], None, 'nvidia/Nemotron-H-56B-Base-8K',
                 perf_filename='/tmp/mamba2_perf.txt', device='cuda:0')
