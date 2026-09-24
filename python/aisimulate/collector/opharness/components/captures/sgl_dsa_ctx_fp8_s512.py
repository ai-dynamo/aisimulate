"""Single-cell capture: sglang DSA context, fp8-KV, seq_len 512 (batch 1).

A whole-sweep capture is a UNION of length-conditional paths (short cells run
dense FA3, long cells the indexer + sparse kernels) and can only be compared
with a serving record that also unions them — none does. Capture ONE cell at
the probe's isl instead (same lesson as trtllm attention decode, 2026-09-24).
The sweep spec is a frozen dataclass; replace it on both import sites."""
import dataclasses
import sys

sys.argv = ['x']
import collector.case_generator as cg  # noqa: E402
import collector.sglang.collect_mla_module as m  # noqa: E402

SEQ = 512
_orig = cg.get_mla_module_sweep_spec


def _one_cell(backend=None):
    return dataclasses.replace(_orig(backend), context_sequence_lengths=[SEQ])


cg.get_mla_module_sweep_spec = _one_cell
m.get_mla_module_sweep_spec = _one_cell
m.run_mla_module('dsa', 128, 'deepseek-ai/DeepSeek-V3.2', 'fp8', 'bfloat16', 'fp8_block',
                 is_prefill=True, gpu_id=0, output_path='/tmp', batch_size_filter=1)
