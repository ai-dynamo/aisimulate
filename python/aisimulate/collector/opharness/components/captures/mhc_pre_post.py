import sys; sys.argv=['x']
from collector.vllm.collect_mhc_module import run_mhc_module_worker
run_mhc_module_worker('pre', 4096, 4096, 4, perf_filename='/tmp/mhc_perf.txt', device='cuda:0')
run_mhc_module_worker('post', 4096, 4096, 4, perf_filename='/tmp/mhc_perf.txt', device='cuda:0')
