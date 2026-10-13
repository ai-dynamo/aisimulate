import sys; sys.argv=['x']
from collector.vllm.collect_mla_module import run_mla_module_worker
# DeepSeek-V3.2 DSA context, isl 512, fp8 KV (graded against the dedicated isl-512 serving raw)
run_mla_module_worker(512, 1, 128, 'fp8', 'bfloat16', 'fp8_block', 'deepseek-ai/DeepSeek-V3.2', 'dsa', 0, perf_filename='/tmp/dsa_context_module_perf.txt', device='cuda:0')
