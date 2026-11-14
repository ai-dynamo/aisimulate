---
name: aisim-benchmark-plan
description: Plan an AISim benchmark before spending GPUs - query the coverage catalog for what exists, is stale or is missing for a model, GPU and framework, then estimate the GPU budget. Use at the start of any collection, qualification or re-evaluation request.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags: [aisim, benchmark, catalog, gpu-budget]
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Plan a benchmark

Stage 1 of [`aisim-benchmark`](../aisim-benchmark/SKILL.md). Do not re-collect what exists.

## Steps

1. Freeze the cell: model (with revision), hardware system ID, framework and exact version,
   parallelism, precision, layers wanted (`op`, `fpm`, `e2e`). Take unstated values from
   supplied configs; do not default them. Ask only for what cannot be derived.
2. Get the catalog. It is Tier 0 and small: `fetch.plan` / `fetch.fetch` with a `Selection`
   naming the hardware, or a local dataset root. If no catalog is reachable, say so and treat
   every cell as unknown, not missing.
3. Query it:

   ```bash
   python/aisim/.venv/bin/python .agents/skills/aisim-benchmark-plan/scripts/plan_catalog.py \
     --dataset-root <root> --hardware <system> --model <hf-id> --framework <fw> \
     --gpus <N> --wall-hours <H>
   ```

   Output groups cells into `reuse` (data and an accuracy record), `evaluate_only` (data, no
   record), `recollect_or_reevaluate` (`stale`: the system spec changed since the last record)
   and `collect` (`missing`). Check each reused cell's version and tier match the request;
   `submitted` records are not evidence for defaults or gates.
4. Estimate budget from the same script. The number is `estimated` (GPUs x wall hours x retry
   margin, queue wait excluded); show `prior_recorded_gpu_seconds` of comparable cells as a
   reference only. A null prior is "never recorded", not zero.
5. Check the [guidelines](../../../docs/perf-model/collector/README.md#benchmark-guidelines):
   list which rows (settings, workload, warmup, boundaries, ranks) the planned run will answer.
6. For E2E cells follow the InferenceX lookup in `aisim-eval` before planning a new run.

## Output

A plan with the cell, the catalog group per layer, the reuse decision, the collection commands
within `slurm-run` or `k8s-run`, the budget with its inputs, and acceptance thresholds if the
user gave any (otherwise "measured, no threshold"). Stop for approval before any GPU job when
the budget exceeds what the user authorized.
