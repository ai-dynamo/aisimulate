import sys; sys.argv=['x']
from collector.vllm.collect_mla_module import run_mla_module_worker
# DeepSeek-V3.2 DSA generation, kv len 4096, fp8 KV (phase = 'generation' from the perf filename)
run_mla_module_worker(4096, 1, 128, 'fp8', 'bfloat16', 'fp8_block', 'deepseek-ai/DeepSeek-V3.2', 'dsa', 0, perf_filename='/tmp/dsa_generation_module_perf.txt', device='cuda:0')
