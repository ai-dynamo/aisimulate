<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded GLM FPM campaigns

Add `--fpm-shard-token-budget N` to the existing `collect.py --ops fpm_forward`
command with a frozen explicit GLM point manifest. `--plan-only` includes the
versioned partition in the normal collection plan. This is the existing FPM
runner with smaller child plans, not a second execution framework. Each child
loads a fresh native engine; model reload time is an explicit cost. No reuse of
an engine across children is claimed.

The budget counts requested real seed plus target tokens across all five warmup
and ten measurement repetitions: `15 * (total_kv_read_tokens + query_tokens)`.
Decode adds one query token per request. It is a work partition, not a GPU memory,
wall-time or runtime-admission estimate. Each original point's full repetitions
stay together. A single point over budget remains in its own child and is marked
`oversized_single_point`; no repetitions or requested points are removed. Select
the budget before freezing using observed runtime and available allocation time.
Large individual observations still require the streaming native readers.

The parent retains the complete original point bytes and phase-local point IDs.
`aic_fpm_shard_manifest` version 1 records each unique child plan/cell and its
local-native-ID to original-ID map. Parent, child and point digests bind the
partition, corpus, source/configuration, topology and execution options. The
ordinary point payload remains schema 2 or 3. No native qualification is inferred
from being present in a child plan: unsupported, failed and unmeasured points
remain part of the parent coverage obligation.

`--fpm-execution-timeout-seconds` optionally freezes the outer per-child timeout,
including initialization, for the existing Kubernetes or Slurm runner. Its
unchanged default is 14400 seconds. The vLLM benchmark timeout remains a separate
10800-second control. Partition work to fit the actual allocation and timers;
these controls do not prove a point can execute.

Execution writes `collection-plan.json`, `shard-manifest.json`, immutable child
plans under `plans/`, and normal child artifacts under `shards/<child-plan>/`.
Each child uses its own checkpoint beneath the supplied checkpoint directory.
The parent checkpoint is `fpm_forward_sharded.json`. Resume the same frozen plan
with the existing `--resume`; use `--resume-retry-failed` to retry failed children.
Changed parent/partition receipts fail rather than adopting another campaign.
Before a native retry, the complete prior working attempt is moved to
`attempts/<attempt-id>/`, with checkpoint and streaming SHA256 file receipts.
Archived attempts cannot be overwritten. Other independent children continue
after a failure, and the parent records every missing child.

Children collect only. Once all children pass native validation, the parent
rechecks the exact point union and publishes once through the existing database
writer. Every row retains its real child plan, cell, attempt, native run and grid
identity. A published child from another attempt blocks the whole new union;
first-publisher skipping cannot silently produce a partial campaign. The
`verified-union.json` receipt binds the selected raw files and archived attempt
receipts. Holdout campaigns never publish consumer data. Complete collection is
not accuracy acceptance; that remains `NOT_EVALUATED` until independent holdout
validation succeeds.

For accuracy, each calibration/holdout entry may reference the parent plan and
cell plus a `shard_manifest` file receipt and `shards`, a list of ordinary child
run specifications (`plan` receipt, `cell_id`, actual `attempt_id`, `raw_root`).
Include exactly the children belonging to that logical parent cell. The common
validator verifies the full parent partition, validates each native child, maps
latencies back to original point IDs and preserves separate native receipts.
It rejects reused requests, duplicate/missing points and unfrozen donor rows.
Ops uses its explicit `bind_sharded_calibration(paths, children, manifest)`
adapter for physical-operation ownership; FPM never averages Ops rows. All
original holdout points remain in the accuracy denominator if a child fails.
