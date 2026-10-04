import sys; sys.argv=['x']
from collector.trtllm.collect_attn import run_attention_torch
# Llama-3.1-8B context cell (b=1, isl/kv len 4096, 32/8 heads, hd 128, no window, fp8 KV + fp8 context FMHA, TRTLLM backend): serving ref
# meta-llama/Meta-Llama-3.1-8B. Recipe was missing from the repo (B200 handoff 2026-10-04: trtllm gates "MISSING CAPTURE").
run_attention_torch(1, 4096, 32, 8, 128, 0, True, True, True, 'TRTLLM',
                    perf_filename='/tmp/context_attention_perf.txt', device='cuda:0')
