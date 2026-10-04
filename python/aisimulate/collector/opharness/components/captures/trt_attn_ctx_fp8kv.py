import sys; sys.argv=['x']
from collector.trtllm.collect_attn import run_attention_torch
# Llama-3.1-8B context cell (b=1, isl 4096, 32/8 heads, hd 128, no window, TRTLLM backend), fp8 KV cache with BF16 context FMHA:
# serving ref meta-llama/Meta-Llama-3.1-8B with kv_cache_config.dtype fp8. The collector's other fp8 precision case
# (fp8 KV + fp8 context FMHA) runs the e4m3 FMHA kernel, which the fp8-KV serving config does not select on sm89
# (L40 probe: serving prefill = fmha_v2_flash_attention_bf16_..._sm89, collector e4m3 variant diverged), so the gate
# captures the BF16-compute combo. Recipe was missing from the repo (B200 handoff 2026-10-04).
run_attention_torch(1, 4096, 32, 8, 128, 0, True, False, True, 'TRTLLM',
                    perf_filename='/tmp/context_attention_perf.txt', device='cuda:0')
