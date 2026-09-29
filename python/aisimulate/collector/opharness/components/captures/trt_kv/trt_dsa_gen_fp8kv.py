"""trtllm fp8-KV single-cell capture: dsa_generation_module (--case-prefix "[4096, 1, 128, 'fp8', 'bfloat16', 'fp8_block'")."""
import runpy, sys
sys.argv = ["op_smoke.py", "--backend", "trtllm", "--op", 'dsa_generation_module', '--case-prefix', "[4096, 1, 128, 'fp8', 'bfloat16', 'fp8_block'", "--cases", "1", "--out-dir", "/tmp/smk"]
runpy.run_path('/work/ais/python/aisimulate/collector/opharness/components/op_smoke.py', run_name="__main__")
