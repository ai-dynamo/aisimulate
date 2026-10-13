import sys; sys.argv=['x']
import torch
from collector.sglang.collect_attn import run_attention_torch
# gemma-4-26B-A4B GLOBAL-layer context cell: 16 q heads / 2 kv heads (num_global_key_value_heads), global_head_dim 512,
# full attention (no window), bf16 KV; serving ref google/gemma-4-26B-A4B. The head-size-512 FMHA path is the part that
# differs from the head_dim 256 sliding layers.
# Backend = what sglang 0.5.21 serves Gemma4 with (arg_groups/model_overrides/gemma4.py: trtllm_mha on SM100, triton on
# every other SM) — the collector's generic SM table would say flashinfer on sm89/sm120 and fa3 on sm90.
backend = 'trtllm_mha' if torch.cuda.get_device_capability()[0] == 10 else 'triton'
run_attention_torch(1, 4096, 16, 2, 512, False, False, True, 0, 512, -1, None, False, None, backend,
                    'Gemma4ForConditionalGeneration', perf_filename='/tmp/context_attention_perf.txt', device='cuda:0')
