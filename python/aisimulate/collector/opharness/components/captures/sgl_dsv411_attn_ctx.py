import sys; sys.argv=['x']
# dsv411 attention / context: one TP1 in-process producer cell at the serving probe's point (isl 4096 prefill
# or one decode token after it); plan frozen in place, SDK manifest from the workspace facts (see collector/dsv411/capture.py).
from collector.dsv411.capture import run_cell
run_cell("sglang", "attention", "context")
