---
name: aisimulate-agentx-gap-debug
description: Diagnoses AgentX GPU-versus-AISimulate replay gaps in TTFT, per-request TPOT, throughput, and latency distributions. Use to investigate workload, FPM, frontend, scheduler, or KV-cache fidelity and validate scoped corrections. Not for optimizing simulator host execution speed alone.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags: [aisimulate, agentx, replay, fpm, accuracy]
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Debug AgentX replay accuracy gaps

Find which part of the measured serving path the replay misrepresents. A good
headline match is not sufficient evidence that the workload, GPU model, or cache
behavior is right. Keep simulator host runtime separate from simulated latency.

## Establish the comparison

1. Resolve the user's acceptance statistics, tolerances, and reference run. Preserve
   an existing decision; do not invent universal thresholds. Freeze the original
   prediction before examining or calibrating against a new GPU result. Distinguish
   that prediction, diagnostic subsets, and post-result exploratory candidates.
2. Build an identity record from actual artifacts: source and loaded native hashes;
   image digest and overlay files; model, draft, tokenizer and trace revisions;
   observed driver/runtime policy; backend, topology, SD method/draft count/acceptance
   definition; graph/streaming policy; usable KV capacity and block size. Also record
   frontend/load-generator versions, seed, initial trajectory cuts/corpus cursor,
   lane count, warmup policy, duration, idle caps and cancellation/grace boundaries.
   Same seed or model name alone does not establish the same experiment.
3. Qualify execution before scoring: warmup and profile completed, expected ranks
   present, FPM gaps/resets explained, return codes and retained request counts
   checked. Inspect terminal scheduler evidence and owned-process cleanup separately.
   An unavailable accounting service or observation timeout is not a failed job;
   recheck the same live handle before deciding whether any retry is appropriate.

Prefer existing raw records and cached checkpoints. Stay within the authorized
task and resource budget; new GPU allocations, broader workloads, patches or
publication require the corresponding user authority. Diagnosis alone is not
permission to implement a fix. Keep failed attempts and use fresh result paths.

## Make the metric contract explicit

- Delimit primer, warmup, profile, grace and cooldown populations. Whole-capture
  FPM averages are not directly comparable to profile-only request metrics.
- Use the reference's first/last **visible** response definition. Per-request TPOT is
  `(last_visible - first_visible) / (output_tokens - 1)` for eligible requests with
  at least two output tokens. Its unweighted request mean is not pooled token ITL,
  chunk spacing, or GPU iteration duration. Keep one-token TTFT samples; retain
  zero-output successes in counts without inventing their undefined latencies.
- Record attempted/completed/error/canceled requests, latency sample counts and
  output tokens. Follow the declared error/cancellation policy; never drop a valid
  short-output request or outlier to improve agreement. Verify exported per-request
  records against the report, and independently recompute the acceptance statistics.
- Compare like percentile estimators and units. Keep official exports intact if
  computing additional same-estimator quantiles. Distinguish throughput over the
  observed successful-request span from throughput over a fixed admission/grace window.
- Use `abs(replay / GPU - 1)` on unrounded values for relative-error gates when that
  is the agreed contract. Preserve both complete closed-loop cohorts: changing
  timing changes future admissions and which turns finish. Unique source-turn pairs
  are a diagnostic subset, not a replacement headline cohort.

## Localize before changing the model

Read the relevant section of [diagnostic patterns](references/diagnostic-patterns.md):

| Evidence to inspect | Question it can answer |
|---|---|
| Same-geometry P-only, D-only and mixed FPM comparisons | Is the GPU estimator wrong at the execution shapes actually observed? |
| Dispatch, admission, GPU and client-visible boundaries | Is frontend, queueing or output work missing, rather than GPU time? |
| Short-output/tail contributions and neighboring iterations | Does a small population dominate the mean difference? |
| Exact-time admissions and coverage failures | Did artificial batching create a query outside the measured domain? |
| Prompt-prefix history, cache events and constrained FPM joins | Is work being recomputed or reused differently, and what remains unidentified? |

Choose one falsifiable hypothesis at a time. Establish a source path and a small
probe or event witness before modifying it. Compare full-population and diagnostic
subset results together; account for unmatched requests using original denominators.
Do not treat the best aggregate estimator as proof of the correct mechanism.

## Validate a correction in the right clock domain

For authorized code changes, first read applicable repository guidance and the
[canonical performance-model contract](../../../python/aisimulate/.claude/rules/perfmodel-api.md).
Use the existing performance-model construction/selection path rather than a
second estimator hidden in the caller. Keep experimental controls and calibration
lineage explicit; do not silently promote a case-specific coefficient to a default.

If modeling host-side latency, preserve GPU pass and physical KV/placement release
timestamps. Delay the appropriate client signals and causal completion feedback,
not the entire engine lifecycle. Test engine-complete/client-pending cancellation,
no late tokens or duplicate terminals, token-before-terminal ordering, SD bursts,
same-time deadlines, warmup barriers and honest unsettled-work accounting.

Use proportionate tests and a compiled small fixture before the full workload.
For optional behavior, check default-off parity and deterministic enabled behavior.
With fixed exogenous arrivals, a client-output-only change should preserve physical
GPU/KV evidence; subsequent GPU batches can legitimately change in closed-loop
replay because client feedback changes future arrivals. Keep host-cost qualification
separate; use the existing [replay performance skill](../aisimulate-replay-performance/SKILL.md)
when changing execution code, and the [replay parity skill](../../../python/aisimulate/.agents/skills/aisimulate-replay-parity/SKILL.md)
for affected semantics.

For a new measurement, pin the candidate's executable config, source/native identity,
input hashes and analysis version before launch. Preserve the full chosen workload,
including warmup and drain. Scope extrapolation explicitly and report every affected
query; do not silently switch estimators or enlarge coverage to make a run pass.

## Close with bounded claims and reproducible evidence

Report mean, p50/p90 and relevant tails, throughput, cohorts, coverage and resource
cleanup, even when acceptance uses only means. Keep the original prediction and
unsuccessful candidates alongside the selected result. Check for compensating
errors: overestimated GPU time can mask excessive cache reuse or missing CPU work.
Passing the requested mean bounds is not a claim of tail, per-request or cache parity.

Label calibration honestly: fitting only on an older run is still post-result
exploration if the new run influenced model selection. State support excursions,
shared-workload overlap and unresolved associations. If the cause is unidentifiable,
name the minimal missing observation and the next authorized experiment, not a
speculative fix. Archive raw data, terminal/cleanup receipts, exact inputs, configs,
source/native/analysis hashes, environment versions and a fresh-output reproduction
recipe. Do not rely only on temporary paths or a command that overwrites old results.

## Repository entrypoints

- Check the installed checkout's `aisimulate predict --help`; the relevant path is
  `predict --stack engine --config ... --capture-per-request --format json --output-dir ...`.
- [Continuous profiles](../../../docs/replay/agentic/continuous-profiles.md) and
  [warmup](../../../docs/replay/agentic/warmup.md) define phase clocks and deadlines.
- [AgentX qualification](../../../scripts/prediction_regression/qualify_agentx_replay.py)
  checks execution semantics with fixed timing; it is not a GPU accuracy benchmark.
- For missing independently measured FPM data, follow the existing
  [self-service workflow](../../../docs/perf-model/fpm-self-service/implementation.md)
  and its checkpoint/authorization rules instead of starting a parallel collector.
