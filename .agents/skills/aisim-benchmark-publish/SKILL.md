---
name: aisim-benchmark-publish
description: Run the deterministic publication gates for collected AISim data and eval records and stage them as a candidate, following the candidate and promotion rules of the unified-data DEP. Use when data or records are ready to share; never promotes without a maintainer.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags: [aisim, benchmark, publish, candidate, promotion]
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Publish a candidate

Stage 4 of [`aisim-benchmark`](../aisim-benchmark/SKILL.md). Rules come from DEP
ai-dynamo/aisim#435; this skill does not widen them.

## Gates (run, then report the output)

1. Guidelines: every row answered in the artifact or the PR provenance section
   ([guidelines](../../../docs/perf-model/collector/README.md#benchmark-guidelines)). Missing rows are listed, not guessed.
2. Parallelism: `registry.evaluate(Context(...))`; `blocks_import()` stops the import, any error
   blocks qualification but allows archival. Unknown inputs surface as evidence gaps.
3. Structure: `check_manifest` / `check_dataset` for op-latency; FPM and E2E use their category
   protocol (FPM self-service validators). `check_published_systems` when a system spec changed.
4. Records: `load_record` on every record; `verify_record` must pass before tier `verified`.
   `check_record_gpu_time`: warn for archival, error for `qualified`.
5. Catalog: `build_catalog(root)` regenerates the shards; `check_catalog_size` must not be `refused`.
6. Hygiene: no tokens, private inventory or private payloads; redact logs; keep raw output hash-linked.

## Candidate, not promotion

- Stage a candidate as a path-identical directory under the dataset layout
  (`data/{op-latency,fpm,e2e,eval}/...`) with the regenerated catalog. Collector or packaged operation
  tables are converted to the op-latency layout with
  `python/aisim/tools/perf_database/publish_op_latency_tree.py build` and re-verified with `... audit`
  (a supported publish step until AISIM-2373 makes the collector emit the layout directly). Writes to a public repository go
  through a Hugging Face PR revision, and private or sensitive data goes only to the internal mirror.
- The library on this branch has no publish function and no `aisim data fetch` CLI. Stop at the
  validated local bundle plus the gate output and give the user the exact upload step; upload only
  with explicit authorization, using their existing `huggingface-cli login` or `HF_TOKEN`
  (never print or store it).
- Promotion (internal to public) is a path-identical copy of selected configuration directories after
  the same validator passes, redaction is applied, the public catalog is regenerated and a maintainer
  approves. Never promote, never push, and never open a PR on your own.
- A repository PR title is `<type>: <short description>`; link the candidate revision in the PR's
  provenance section.

## Report

Gates run with pass or fail, bundle path, record IDs and tiers, `gpu_time` totals with authority,
unanswered guideline rows, and the next action that needs a human.
