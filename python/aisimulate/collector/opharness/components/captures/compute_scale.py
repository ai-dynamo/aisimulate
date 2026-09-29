import sys; sys.argv=['x']
from collector.vllm.collect_computescale import run_computescale
run_computescale(4096, 7168, perf_filename='/tmp/computescale_perf.txt', device='cuda:0')
