"""Single-cell capture: sglang GLM-5 DSA attention, module level, fp8 KV, CONTEXT (batch 1, isl 4096).
On Blackwell sglang 0.5.21 resolves DSA kv auto to fp8_e4m3 and serves GLM-5 with the TRT-LLM FMHA
token-sparse cubins; the kernel-level glm5_dsa_sparse_modules bench runs bf16 FlashMLA sparse_attn_fwd
(ALIGNED only against an explicit bf16-KV record). This drives the same module path the DSV3.2 fp8
gates use (collect_mla_module 'dsa' with fp8 KV) for GLM-5 — B200 2026-10-04. One cell at the probe's
isl, like sgl_dsa_ctx_fp8_s4096 (a whole sweep unions length-conditional paths)."""
import dataclasses
import sys
sys.argv = ['x']
import collector.case_generator as cg  # noqa: E402
import collector.sglang.collect_mla_module as m  # noqa: E402
SEQ = 4096
_orig = cg.get_mla_module_sweep_spec
def _one_cell(backend=None):
    return dataclasses.replace(_orig(backend), context_sequence_lengths=[SEQ])
cg.get_mla_module_sweep_spec = _one_cell
m.get_mla_module_sweep_spec = _one_cell
m.run_mla_module('dsa', 128, 'zai-org/GLM-5', 'fp8', 'bfloat16', 'bfloat16',  # bf16 checkpoint: bf16 projection GEMMs
                 is_prefill=True, gpu_id=0, output_path='/tmp', batch_size_filter=1)
