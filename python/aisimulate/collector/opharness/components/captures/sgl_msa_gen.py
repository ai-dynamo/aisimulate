import sys; sys.argv=['x']
from collector.sglang.collect_msa_module import run_msa_module
# MiniMax-M3 MSA generation, kv len 4096, 64 heads, bf16 — see sgl_msa_ctx.py
run_msa_module(64, 'MiniMaxAI/MiniMax-M3', 'bfloat16', 'bfloat16', 'bfloat16', False, 0,
               output_path='/tmp', quick_shape=(1, 4096, 0))
