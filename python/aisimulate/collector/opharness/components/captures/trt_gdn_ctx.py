import sys; sys.argv = ['x']
# One cell (b=1, isl=4096) instead of the registry sweep: FlashInfer's GDN chunk
# has a FullyFused variant for short T that the isl-4096 serving probe never runs.
from collector.trtllm.collect_gdn import run_gdn_torch
run_gdn_torch('context', 1024, 4, 16, 128, 16, 128, [1], [4096], 'Qwen/Qwen3.5-0.8B',
              perf_filename='/tmp/gdn_perf.txt', device='cuda:0')
