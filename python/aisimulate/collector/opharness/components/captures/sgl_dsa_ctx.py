import sys; sys.argv = ['x']
# The sglang DSA module collector runs each case in a subprocess (invisible to the
# parent profiler); call the in-process entry with a one-batch filter instead.
from collector.sglang.collect_mla_module import run_mla_module
run_mla_module('dsa', 128, 'deepseek-ai/DeepSeek-V3.2', 'bfloat16', 'bfloat16', 'fp8_block',
               is_prefill=True, gpu_id=0, output_path='/tmp', batch_size_filter=1)
