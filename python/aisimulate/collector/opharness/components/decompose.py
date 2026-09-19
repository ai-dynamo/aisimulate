#!/usr/bin/env python3
"""Component: model -> op decomposition, with residue detection.

Question this instrument answers (deterministically):
  "Which op families does this model's MEASURED execution consist of, and is
   any observed execution left over that no declared family covers?"

Inputs
  - a G1 identity record (archive/records.jsonl entry): observed modules,
    per-op kernel captures
  - components/kernel_taxonomy.yaml: kernel name -> canonical backend
  - families.yaml (opharness-local vocabulary; deliberately self-contained —
    a copy, not a reference, of the collector's family split, so this
    directory works without touching existing machinery)

Output (machine verdict, per model x framework x version)
  families_observed: [attention, moe, ...]        # covered execution
  residue: [kernel/module names with no family]   # DECOMPOSITION GAP
  verdict: covered | residue

A non-empty residue is the trigger for a human granularity decision (new
family? new table under an existing family?) — this instrument only detects,
it never decides granularity.

Status: contract stub — implementation lands with the onboard_model workflow.
"""
from __future__ import annotations

import sys


def main() -> int:
    raise SystemExit("decompose: not implemented yet — see module docstring for the contract")


if __name__ == "__main__":
    sys.exit(main())
