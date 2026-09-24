import sys; sys.argv = ['x']
from collector.sglang.collect_mla_module import run_mla_module
run_mla_module('dsa', 128, 'deepseek-ai/DeepSeek-V3.2', 'fp8', 'bfloat16', 'fp8_block',
               is_prefill=False, gpu_id=0, output_path='/tmp', batch_size_filter=1)
