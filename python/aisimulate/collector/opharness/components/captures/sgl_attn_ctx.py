import sys; sys.argv=['x']
from collector.sglang.collect_attn import run_attention_torch
# Llama-3.1-8B context cell (b=1, isl=4096, 32/8 heads, hd 128, bf16 KV, dense): serving ref meta-llama/Meta-Llama-3.1-8B.
# Recipe was missing from the repo (B200 handoff 2026-10-04: 11 sglang gates "MISSING CAPTURE"); backend left None so the
# collector picks the SM's serving default (fa3 sm90 / trtllm_mha sm100 / flashinfer sm89+sm120).
run_attention_torch(1, 4096, 32, 8, 128, False, False, True, 0, 128, -1, None, False, None, None, 'LlamaForCausalLM',
                    perf_filename='/tmp/context_attention_perf.txt', device='cuda:0')
