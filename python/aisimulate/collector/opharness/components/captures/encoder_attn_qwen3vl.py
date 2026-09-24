import sys; sys.argv=['x']
from collector.vllm.collect_attn_encoder import run_encoder_attention_torch
# Qwen3-VL-2B vision tower: hidden 1024 / 16 heads -> head_dim 64; one 1024-token image
run_encoder_attention_torch(1, 1024, 16, 64, perf_filename='/tmp/encoder_attention_perf.txt', device='cuda:0')
