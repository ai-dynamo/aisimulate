"""Single-cell capture: sglang GLM-5 DSA attention, module level, fp8 KV, DECODE (batch 1).
On Blackwell sglang 0.5.21 resolves DSA kv auto to fp8_e4m3 and serves GLM-5 with the TRT-LLM FMHA
token-sparse cubins; the kernel-level glm5_dsa_sparse_modules bench runs bf16 FlashMLA sparse_attn_fwd
(ALIGNED only against an explicit bf16-KV record). This drives the same module path the DSV3.2 fp8
gates use (collect_mla_module 'dsa' with fp8 KV) for GLM-5 — B200 2026-10-04."""
import sys; sys.argv = ['x']
from collector.sglang.collect_mla_module import run_mla_module
run_mla_module('dsa', 128, 'zai-org/GLM-5', 'fp8', 'bfloat16', 'bfloat16',  # bf16 checkpoint: bf16 projection GEMMs
               is_prefill=False, gpu_id=0, output_path='/tmp', batch_size_filter=1)
