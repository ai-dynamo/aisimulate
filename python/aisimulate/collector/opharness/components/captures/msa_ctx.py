import sys; sys.argv=['x']
from collector.vllm.collect_msa_module import run_msa_module_worker
# MiniMax-M3 MSA context, isl 4096, 64 heads (tp1 shard), bf16
run_msa_module_worker(4096, 1, 64, 'bfloat16', 'bfloat16', 'bfloat16', 'MiniMaxAI/MiniMax-M3', 0, perf_filename='/tmp/msa_context_module_perf.txt', device='cuda:0')
