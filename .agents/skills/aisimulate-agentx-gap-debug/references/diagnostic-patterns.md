<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Diagnostic patterns from AgentX alignment

These patterns were distilled from MiniMax AgentX GPU/replay debugging. They are
investigation branches, not claims that every backend has the same bottleneck.
No historical latency coefficient, driver policy, capture cap, matching window or
acceptance threshold is a reusable default. Validate behavior on the pinned build.

## Same-geometry FPM checks

Keep request-end latency separate from model-step latency. For each observed
P-only, D-only and mixed regime, query the selected estimator with the same batch,
new-prefill and past-KV geometry, SD convention, topology and available execution
profile. Aggregate sums can hide ragged per-request shapes or a different graph
regime; retain these limitations rather than claiming an exact match.

Sampling abundant decode records is useful, but preserve stratum/population weights
when reporting overall bias. Separate mean-of-relative-errors, relative error of
means, and absolute-error statistics. Keep all rare P/mixed records where practical.
Report measured, interpolated, extrapolated and unsupported counts separately.
No coverage failure is a valid low-latency result; do not substitute zero timing.

For SD, identify whether acceptance length includes the target/bonus token and
whether reported lengths describe inputs, committed outputs, or a later async step.
Do not infer support for all SD algorithms from one EAGLE/MTP test. Nominal work
`P + (draft_count + 1) * decode_requests` can help investigate a graph-size cliff,
but padding, mode and actual per-request scheduling decide which graph executed.
Changing capture sizes can change usable KV and workload behavior as well as timing.

If comparing interpolation and regression, keep the data and allowed coverage
policy fixed. Inspect local residuals before selecting by aggregate accuracy:
systematic P/mixed overprediction can accidentally compensate for other missing work.

## Frontend and output boundaries

Use this ordering as a map, then locate the actual backend events:

`client dispatch -> input processing -> engine admission -> GPU passes -> output processing -> visible response`

HTTP headers may precede lazy engine submission. FPM **listener receipt** follows
publication and is not a CUDA completion timestamp. Treat cross-process clock
alignment and publication delay as uncertainties. An additive timing decomposition
is arithmetic evidence, not proof that a residual is exclusively one CPU operation.

A useful pattern from the investigation was prompt-length-dependent first-output
latency. A tokenizer's streaming decoder can defer full-prompt decoding until the
first generated token even when its constructor is cheap. Check the actual loaded
tokenizer binary, backend and special-token policy. A CPU-only probe can separate
construction, first step and later steps; a local slope is not automatically the
server CPU's slope. A reasoning-parser optimization and detokenizer cost are
different paths; verify which path the measured request actually used.

GPU decode can continue while the frontend processes its first output. Consequently
client TPOT can shrink when first-visible delivery is delayed. Reproduce that causal
ordering rather than subtracting a constant from TPOT or moving CUDA events. If
introducing an output-stage model, document whether it serializes a shared batch,
models shared input/output CPU contention, and retains canceled queued work. Charge
first-output work once per request, not once per accepted speculative token.

Fit a proposed host model only on its declared calibration population, retain
residuals and evaluate on separately identified data. Report input-length support
and all out-of-support uses; no silent clipping or deletion. Combining a measured
receipt-to-visible residual with a queue model can double-count queueing already
inside that residual. A good fit does not remove this modeling uncertainty.

## Same-time admissions and coverage

Simulated equal-time requests can collapse into one batch even when native input
processing separates them. Inspect deterministic source/dispatch order and how a
pending-input feature treats exact ties. Excluding every equal-time peer but counting
the same peer at `t + epsilon` introduces a discontinuity.

If that matters, test a stable tie rule without changing causal arrival timestamps
or refitting unrelated costs. Count only earlier eligible peers, not the current
request or already-due work. Preserve the established cancellation and release
ordering. Stable enqueue order does not prove native FIFO completion.

Before collecting a larger FPM grid or enabling upper-KV extrapolation, examine the
batch that triggered the coverage error. A batching correction may remove an
artificial query. If coverage is genuinely missing, collect within the authorized
budget or use an explicitly approved bounded policy. Validate the exact supported
extrapolation kind and limits; a fully explained low-P policy is not an unsupported
query and does not authorize upper-KV or decode extrapolation.

## Tails, cohorts and attribution

For request TPOT, a long intervening prefill delay divided by a small `OSL - 1` can
dominate a mean error. Group by output length, input length, phase and source turn.
Look for exact event witnesses: another request's first token coinciding with the
target's last token, mixed-prefill work, readmissions or output batching. Transport
reads, SSE messages, visible content responses and generated tokens are different
counts; a short visible span alone does not establish parser suppression.

Keep complete cohorts for acceptance. For a subgroup's contribution to the mean
gap, use its latency sum divided by the **original full sample count** on each side:

`gap = sum(GPU latencies) / N_GPU - sum(replay latencies) / N_replay`

Partition those sums into matched groups and GPU-only/replay-only groups so the
parts reconcile to the full gap. Averaging only easy-to-match requests can erase
the most important slow requests. Use metric-specific populations: TTFT and TPOT
need not have the same denominator. Do not pair by row ordinal or silently merge
duplicate source-turn keys. Report matching criteria, exclusions and ambiguities.

## FPM association and cache hypotheses

Prefer real request IDs and native iteration IDs. If they are absent, keep inference
separate from measurement. A unique final length match plus an eligibility window
is stronger than nearest-time matching, but can still be a coincidental intermediate
chunk. List competing requests and account for clock/publication uncertainty.

For attention DP, determine whether an iteration ID is global or rank-local.
Separate known idle/dummy paths and missing measurements. Listener timestamps alone
do not prove global alignment; do not invent a universal millisecond matching cutoff.

A multi-prefill batch may have a unique eligible request-length sum. Preserve all
candidate combinations. A continuous single-prefill chain satisfying
`next_past_KV = past_KV + P`, followed by a constrained joint endpoint, can bound
recomputed work. It does **not** license splitting aggregate GPU duration or final
batch P equally across requests. State the assumptions needed for conservation and
keep counter continuity and request-lifetime evidence.

Final past-KV includes initial cache reuse **and earlier prefill chunks**. It is not
an initial cache-hit measurement. Compare native cached-token evidence if captured;
otherwise retain the unknown. Also distinguish trace/logical blocks from runtime
KV blocks and rendered token IDs, including template/control tokens and rounding.

Before declaring a cache implementation bug, compare preceding requests and
primer/warmup history, play/lane identity, idle gaps, reset/fork/spawn semantics and
cache-bust namespaces. Equal source hashes or a shared marker do not prove identical
rendered wire prefixes. Zero preemptions and low active KV utilization do not rule
out eviction of inactive cached blocks. Seek wire-token prefix hashes and cache
events when rendering differences and eviction remain indistinguishable.

## Outcome and implementation boundaries

Separate GPU/engine completion from client-visible terminal state in a proposed
CPU model. Physical KV release and placement retirement must not wait merely
because the client is still processing output. Conversely, a closed-loop client
slot and completion-triggered child request must follow the client clock.

Test cancellation while GPU work is active and while only client delivery remains;
handle exact-deadline ordering, pending output, warmup transition and token/terminal
counts. Count each unsettled request once. An output audit can bound pending CPU
work, but cannot alone prove the full GPU/CPU unsettled union or GPU cleanup.

For the handoff, explicitly separate contract acceptance, local FPM fit, tail
agreement, workload/cache fidelity and independence of validation. If means pass
while tails or a cache-work witness disagree, report the pass narrowly and retain
the contradictory evidence. Investigation results belong in `benchmarks/evidence/`
when publishing them in this repository; do not embed private hostnames, workspace
paths, raw traces or historical tuning constants in the reusable skill.
