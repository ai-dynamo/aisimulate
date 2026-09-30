import sys, os; sys.argv=['x']
# serving decodes WITHOUT speculation -> only the fused CUDA decode path is comparable
# (collect_kda._decode_paths; the sm90 recipe passed this via path_diff --env and was never committed)
os.environ.setdefault('AIS_KDA_DECODE_PATHS', 'fused')
from collector.vllm.collect_kda import run_kda_torch
run_kda_torch('generation', 7168, 4, 96, 128, 96, 128, [1], None, 'moonshotai/Kimi-K3', perf_filename='/tmp/kda_perf.txt', device='cuda:0')
