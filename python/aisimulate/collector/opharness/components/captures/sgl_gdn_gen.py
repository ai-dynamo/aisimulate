import sys; sys.argv=['x']
from collector.sglang.collect_gdn import run_gdn_torch
# Qwen3.5-0.8B GDN decode cell (same geometry as sgl_gdn_ctx / the vllm gdn_gen gate)
run_gdn_torch('generation', 1024, 4, 16, 128, 16, 128, [1], None, 'Qwen/Qwen3.5-0.8B', 'float32',
              perf_filename='/tmp/gdn_perf.txt', device='cuda:0')
