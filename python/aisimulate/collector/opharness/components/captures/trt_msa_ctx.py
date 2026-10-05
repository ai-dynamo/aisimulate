import sys; sys.argv=['x']
from collector.trtllm.collect_msa_module import run_msa_module_worker
# MiniMax-M3 MSA context, isl 4096, 64 heads (tp1 shard), bf16 — mirrors captures/msa_ctx.py (vllm) and the sm89 trtllm
# smoke/sample cells (msa_context_module 40/40 clean). No serving record exists on any probed SM: trtllm 1.3.0rc29 does not
# boot MiniMax-M3 (MiniMaxM3SparseRuntimeBackend.forward requires k, v, k_cache ...), so verdicts_trt_rc29.sh declares the
# gate as a floor everywhere; it grades for real the day an rc serves M3.
run_msa_module_worker(4096, 1, 64, 'bfloat16', 'bfloat16', 'bfloat16', 'MiniMaxAI/MiniMax-M3', 0, perf_filename='/tmp/msa_context_module_perf.txt', device='cuda:0')
