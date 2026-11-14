---
name: aisim-benchmark-evaluate
description: Evaluate AISim predictions against recorded truth, write an aisim-eval/v1 record with the GPU time of the collection, then verify it deterministically and attest the verified tier. Use after truth data exists for a cell and a predictor can be built.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags: [aisim, benchmark, evaluation, eval-record, verification]
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Evaluate and record

Stage 3 of [`aisim-benchmark`](../aisim-benchmark/SKILL.md). `aisim-eval` decides
what to compare and how to score (FPM step MAPE, replay-based E2E TTFT and TPOT); this skill makes
the result a verifiable record.

## Steps

1. Freeze the comparison as `aisim-eval` describes. Keep calibration inputs apart from truth
   inputs and list them as `Assumption` entries (`kind` `calibration` or `truth`, `status`
   `verified` or `estimated`).
2. Build the predictor through the canonical `best_available` interface. Compute its identity with
   `PredictionIdentity.build(code_path, data_pins, spec_sha256)`; pins are SHA-256 of the exact
   profiles and tables used. The fingerprint does not include the commit SHA, so unrelated commits do
   not orphan the record. If a record with the same fingerprint and truth already exists
   (`current_records`), report it instead of re-scoring.
3. Write truth with `write_truth_table` (columns `case_id`, `inputs_json`, `truth_value`), score
   with `compute_results`. Rows without truth or prediction become exclusions with reasons, never
   zero. Put collection quality, resolved memory and replay coverage in their own `Results` fields.
4. Build `EvalRecord` at tier `submitted`, with `gpu_time` from the collect stage (`None` or
   unknown if never recorded), `simulation_cost`, and `producer` (`kind="agent"`, this skill's
   name, evaluator revision). Partial or failed attempts are kept with `status_reason`.
5. Store: `write_record(dataset_root, record)`. It is content-addressed and refuses overwrites.
6. Verify and promote one tier:

   ```bash
   python/aisim/.venv/bin/python .agents/skills/aisim-benchmark-evaluate/scripts/check_eval_record.py \
     <record.json> --truth-root <dir> --predictor <predictor.py> --attest-root <dataset_root>
   ```

   `predictor.py` exposes `predict(inputs)` and `fingerprint()` from the same predictor as step 2.
   Exit code 1 means recomputation disagrees with the claim: fix the record, do not edit the check.
   Without `--predictor` the script only loads the record and prints `gpu_time` findings.
7. `qualified` needs a recorded `gpu_time` (the model rejects it otherwise), a threshold the
   user agreed, and a maintainer decision. Use `attest(record, report, "qualified")` only
   when the user asks for it; a library cannot qualify itself.

## Report

Per cell: MAPE with `n_scored / n_total`, coverage, exclusions, tier, record ID, `gpu_time` and its
authority. Without a threshold write "measured, no threshold".
