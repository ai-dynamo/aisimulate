import sys; sys.argv=['x']
from collector.vllm.collect_attn import run_attention_torch
# Llama-3.1-8B context cell (b=1, isl=4096, 32/8 heads, hd 128, bf16 KV): serving ref meta-llama/Meta-Llama-3.1-8B. Recorded 2026-09-30 (the sm90 gate was produced via op_smoke and never committed).
run_attention_torch(1, 4096, 32, 8, 128, False, True, 0, perf_filename='/tmp/context_attention_perf.txt', device='cuda:0')
