#!/usr/bin/env python3
"""Component: execution-path alignment — collector op path vs serving.

Question this instrument answers (deterministically):
  "Does the collector's constructed op execute the SAME kernel families the
   framework executes in serving, for the same (model, phase, dtype, SM)?"

This mechanizes the bulk of the 'serving-parity audit': reading code does not
count as evidence; both sides run under the same profiler and translate
through the same kernel taxonomy, then their kernel sets are diffed.

Inputs
  - serving side: G1 identity record (probe capture, taxonomy-labeled)
  - collector side: re-run the collector's op path for the matching case
    under torch.profiler (same capture/normalization as the probes)

Output (machine verdict, per op family x model)
  serving_backends / collector_backends: canonical sets
  verdict: aligned | diverged (with the set difference)

A 'diverged' verdict blocks collection for that slice and hands off to AI
for the field-level metadata parity audit (which stays human/AI judgment —
kernel-set equality is necessary, not sufficient).

Status: contract stub — implementation lands with the upgrade_op workflow.
"""
from __future__ import annotations

import sys


def main() -> int:
    raise SystemExit("path_diff: not implemented yet — see module docstring for the contract")


if __name__ == "__main__":
    sys.exit(main())
