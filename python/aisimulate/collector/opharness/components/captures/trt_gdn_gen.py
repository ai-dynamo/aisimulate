import sys; sys.argv = ['x']
# One decode cell (b=1) of Qwen3.5-0.8B's GDN layer, matching the isl-4096 serving probe's decode phase.
from collector.trtllm.collect_gdn import run_gdn_torch
run_gdn_torch('generation', 1024, 4, 16, 128, 16, 128, [1], None, 'Qwen/Qwen3.5-0.8B',
              perf_filename='/tmp/gdn_perf.txt', device='cuda:0')
