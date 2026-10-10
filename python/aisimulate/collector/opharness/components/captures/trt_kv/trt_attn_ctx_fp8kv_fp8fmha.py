"""trtllm fp8-KV single-cell capture: attention_context, the bf16 gate cell with fp8 KV
(--case-prefix '[1, 4096, 32, 8, 128, 0, True, True, True'; same batch/seq/heads/head_dim/window as the bf16 gate capture)."""
import runpy, sys
sys.argv = ["op_smoke.py", "--backend", "trtllm", "--op", 'attention_context', "--case-prefix", '[1, 4096, 32, 8, 128, 0, True, True, True', "--cases", "1", "--out-dir", "/tmp/smk"]
runpy.run_path('/work/ais/python/aisimulate/collector/opharness/components/op_smoke.py', run_name="__main__")
