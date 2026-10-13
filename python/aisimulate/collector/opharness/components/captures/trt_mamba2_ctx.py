import sys; sys.argv = ['x']
# Nemotron-H-56B-Base-8K Mamba2 layer (case_generator common mamba2 row: d_model 8192, d_state 256,
# d_conv 4, 256 heads x 64, 8 groups, chunk 128); one prefill cell b=1 x 4096 = the serving probe's isl.
from collector.trtllm.collect_mamba2 import run_mamba2_torch
run_mamba2_torch('context', 8192, 256, 4, 256, 64, 8, 128, [1], [4096], 'nvidia/Nemotron-H-56B-Base-8K',
                 perf_filename='/tmp/mamba2_perf.txt', device='cuda:0')
