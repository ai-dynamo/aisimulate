import sys; sys.argv=['x']
from collector.sglang.collect_attn import run_attention_torch
# Llama-3.1-8B generation cell (b=1, kv len 4096, bf16 KV)
run_attention_torch(1, 4096, 32, 8, 128, False, False, False, 0, 128, -1, None, False, None, None, 'LlamaForCausalLM',
                    perf_filename='/tmp/generation_attention_perf.txt', device='cuda:0')
