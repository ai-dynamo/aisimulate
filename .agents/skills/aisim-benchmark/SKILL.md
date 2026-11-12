---
name: aisim-benchmark
description: Run the AISim benchmark flow for a model and GPU - plan from the coverage catalog, collect op, FPM or E2E data, record GPU time from job records, evaluate predictions into an eval record, and publish a candidate. Use when asked to benchmark, collect, qualify or re-evaluate a model on hardware, or to reuse existing evidence instead of re-measuring.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags: [aisim, benchmark, gpu-time, eval-record, dataset]
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISim benchmark flow

Orchestrates five stages. Each stage is its own skill; load only the one you are in.

| Stage | Skill | Deterministic check it must run |
| --- | --- | --- |
| 1 Plan | `aisim-benchmark-plan` | `Catalog` query, GPU budget |
| 2 Collect | `aisim-benchmark-collect` | guidelines rows answered, `gpu_time` captured |
| 3 Evaluate | `aisim-benchmark-evaluate` | `verify_record`, `check_record_gpu_time` |
| 4 Publish | `aisim-benchmark-publish` | manifest, catalog, parallelism rules |

Execution builds on `aisim-eval` (what to compare and how to score), and on
`slurm-run` or `k8s-run` (access and job lifecycle). These are installed
skills, not part of this repository; if one is missing, say so and stop at the
resolved plan instead of improvising cluster access.

## Rules that hold in every stage

- Follow the [benchmark guidelines](../../../docs/perf-model/collector/README.md#benchmark-guidelines).
  Do not copy them here; [record fields](references/record-fields.md) maps each row to where the library stores it.
- Never invent a measurement fact. Unknown stays null: GPU type, effective setting, cache state,
  GPU time, thresholds. Authority is `measured`, `declared` or `unknown`; only job records are `measured`.
- Run the library check instead of reasoning about its answer. Cite functions only from
  [library reference](references/library.md), which lists what exists on this branch.
- Authorization is not inherited across stages: submitting jobs, uploading to Hugging Face,
  opening PRs and promotion each need the user's explicit go-ahead. Planning and local scoring do not.
- Keep requested and observed settings, collection quality, resolved memory, replay coverage and
  measured accuracy as separate claims. A smoke run is not qualification.
- Raw logs, results and records go in a run-specific output directory or the dataset, never `docs/`.

## Flow

1. Plan. If the catalog shows the cell as `have` with a recent verified record, report it and stop.
2. Collect only what is `missing` or `stale`; use the smallest layer that answers the question.
3. Capture `gpu_time` from the Slurm or Kubernetes record of every collection and evaluation job.
4. Evaluate against truth that was not used for calibration; store and verify the record.
5. Publish a candidate; promotion is a maintainer decision.

Report per cell: stage reached, result or blocker, GPU time (with authority), record ID, next action.
