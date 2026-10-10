import sys; sys.argv=['x']
from collector.sglang.collect_msa_module import run_msa_module
# MiniMax-M3 MSA context, isl 4096, 64 heads (tp1 shard), bf16: the in-process entry the collector's subprocess runs
# (collect.py drives run_msa_module_worker -> one subprocess per case; a parent-side profiler sees nothing, README rule).
# quick_shape pins the ONE (batch, seq, prefix) cell the serving record was probed at. Recipe added 2026-10-05 once the
# m3 dummy kept its dense head and MiniMax-M3 served on sm89 (it had been a MISSING CAPTURE gate before).
run_msa_module(64, 'MiniMaxAI/MiniMax-M3', 'bfloat16', 'bfloat16', 'bfloat16', True, 0,
               output_path='/tmp', quick_shape=(1, 4096, 0))
