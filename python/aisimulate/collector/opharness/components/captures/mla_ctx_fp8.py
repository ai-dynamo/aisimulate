import sys; sys.argv=['x']
from collector.vllm.collect_mla_module import run_mla_module_worker
run_mla_module_worker(4096, 1, 128, 'fp8', 'bfloat16', 'fp8_block', 'deepseek-ai/DeepSeek-V3', 'mla', 0, perf_filename='/tmp/mla_module_context_perf.txt', device='cuda:0')
