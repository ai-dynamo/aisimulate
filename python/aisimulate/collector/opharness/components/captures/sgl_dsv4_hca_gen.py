import sys; sys.argv=['x']
# sglang DSV4 module collector runs each case in a subprocess; capture its in-process
# entry at the serving record's cell (isl 4096, kv auto = fp8_e4m3 pool, FP8 checkpoint
# -> fp8_block projection GEMMs), one bs, prefix 0.
from collector.sglang.collect_dsv4_attn import run_dsv4_mla_module
run_dsv4_mla_module(model_path='sgl-project/DeepSeek-V4-Flash-FP8', mode='generation', attn_kind='hca', batch_sizes=[1], seq_lens=[], kv_cache_dtype='fp8_e4m3', gemm_type='fp8_block', tp_size=1, prefix_lens=(0,), seq_lens_by_prefix={0: [4096]}, output_path='/tmp', perf_filename_prefix='dsv4', device='cuda:0')
