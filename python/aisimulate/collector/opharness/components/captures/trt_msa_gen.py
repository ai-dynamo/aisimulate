import sys; sys.argv=['x']
from collector.trtllm.collect_msa_module import run_msa_module_worker
# MiniMax-M3 MSA generation, kv len 4096, 64 heads, bf16 — mirrors captures/msa_gen.py (vllm); see trt_msa_ctx.py for why
# the gate is a floor on every SM at rc29.
run_msa_module_worker(4096, 1, 64, 'bfloat16', 'bfloat16', 'bfloat16', 'MiniMaxAI/MiniMax-M3', 0, perf_filename='/tmp/msa_generation_module_perf.txt', device='cuda:0')
