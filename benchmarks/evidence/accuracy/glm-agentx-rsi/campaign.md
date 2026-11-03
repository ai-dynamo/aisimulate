<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM AgentX RSI campaign

## Objective and frozen protocol

Reduce pooled per-request TTFT and TPOT MAPE on six B300 TP8 GLM AgentX profiles.
A win improves either metric without worsening the other beyond measured repeat
variability. The aspirational target is both below 10%; point error and throughput
are diagnostics, not substitutes. No outlier removal or timing calibration.

Development lanes: 1/8/14. Validation lanes: 4/10/13, reserved from tuning during
this campaign; their historical results are already known. Freeze each scoring
cohort before experiments and report retained/lost/recovered identities separately.
Do not call an improvement from a smaller scored cohort a win.

Historical dashboard wheel: `1c753e4668310705b6451ba1ebf3061b3d5b30de`;
HF silicon revision: `da115d648f0be20a1c4e0408d552a4a1e8bebf9d`.
Research baseline: AISim main `9f4819e2b77afb4798bc8133c42ccc1d1b045e5c`
plus PR #665 `64ac539813677d15e58c8ba051a6b5fc95c5334b`, merged as
`eddb21246e0a371fe542f7cab4034b58173bce63`. Wheel SHA256: `79ed539e44052aca60f9e62d6427654417e2ce1c514c881cfee0f0949581d046`;
native SHA256: `07b39a7f2e76aba7e4092f58a133f574059da784d98ab3bec1cdef94aec0319d`.
Frozen scoring tooling: `a01b4b45d4a58f65332e7fb7646cc5eab2ce1cbe`.
Silicon is SGLang 0.5.20; historical predictions select op database 0.5.21.
This is cross-version evidence, not same-version qualification.

Raw inputs and the identity manifest remain private. `audit.py` refuses to replace
frozen files, verifies source uniqueness/output lengths, and scores requests directly.
Run `python audit.py --public-bundle PUBLIC_BRANCH_DIR --pr665-evidence PRIVATE_REPLAY_DIR --output NEW_PRIVATE_DIR`.
Run tests with `python -m pytest -c /dev/null -p no:cacheprovider test_audit.py`.

## Resource and stopping rules

Hard cap: 64 allocated B300 GPU-hours (eight TP8 node-hours), including startup,
profiling and failed jobs. Reserve active allocations at their full wall limit;
reconcile completed jobs against Slurm elapsed time and actual reserved GPUs.
Budget: 2 node-hours baseline/timing, 2 controlled experiments, 1 op/forward,
3 validation. Unused diagnostic time may move to validation. Qwen is separate.

Stop on validated attainment of both targets, exhausted budget, three consecutive
experiments without discriminating evidence or a verified improvement, or no
actionable hypothesis. Retain failed experiments; do not claim target attainment
when stopped with residual error. No automatic merge, HF promotion or dashboard
replacement. Use signed/DCO commits and focused draft PRs.

## Baseline reproduction — E000

**Hypothesis.** Dashboard and PR #665 differ because their cohorts and predictor
revisions differ; current data selection alone does not establish score equality.

**Experiment.** Independently compute both metric errors from saved request pairs,
freeze private source identities, preserve all small positive references, and verify
published dashboard denominators and arithmetic. Reproduce both #665 variants on
its original frozen cohort; no replay settings or observations changed.

**Conclusion — confirmed arithmetic/provenance.**

| Variant | TTFT MAPE | TTFT n | TPOT MAPE | TPOT n |
|---|---:|---:|---:|---:|
| Dashboard | 84.5587% | 8624 | 79.0844% | 8614 |
| #665 main | 85.0249% | 8260 | 146.7501% | 8251 |
| #665 plus data | 77.5517% | 8260 | 81.9830% | 8251 |

Private frozen cohort SHA256: `26e3a6b64e9a7a640d64a8d8e057c5e383a23f7a458a1f0bd496423922a33c1b`.
Baseline audit SHA256: `500b572212ffb25efdadcb1f0cc1bffb527e2ef18217dde4fc38ead617d12946`.

## H1 — Client measurement boundaries and denominator sensitivity

**Hypothesis.** Buffered client receipt and/or speculative bursts compress recorded
decode duration for a small subset; positive tiny denominators magnify MAPE.

**Experiment — E001.** Independently stratify the frozen dashboard pairs at
reference TPOT <0.1 ms; retain every request in headline scores. Compare raw
stream, HTTP receive, and server generation boundaries in targeted GPU controls.

**Conclusion — supported sensitivity; boundary mechanism unconfirmed.** At 14 lanes,
9/2429 requests contribute 65.43 percentage points of 115.63% TPOT MAPE; median
APE is 24.39%. This is not permission to exclude these requests or subtract their
contribution. Historical client timestamps alone cannot establish server generation.
Next: inspect raw examples, then compare server/client timing on matching controls.

## H2 — Replay recipe, tokenization, snapshots and cache history

**Hypothesis.** Effective settings or replay history differ from silicon.

**Experiment — E002.** Parse retained runtime `server_args` receipts and compare
with frozen replay configuration. Run one-variable settings controls on development
points before examining validation-point changes.

**Conclusion — supported configuration mismatch; performance impact pending.**
The one-lane runtime capped running requests at 2, while the dashboard config
allowed 256. The four-lane runtime used 8. High-lane settings remain separately
verified. **E004 cap-only result:** setting the C1 cap from 256 to 2 preserves all 324
scored requests and leaves TTFT/TPOT MAPE exactly unchanged (73.6567%/18.8688%).
The setting mismatch is confirmed; it does not explain the C1 accuracy gap.
Retain the correction as recipe alignment, not a claimed accuracy win.

## H3 — Speculative acceptance and token-delivery bursts

**Hypothesis.** Fixed acceptance fails to represent per-request acceptance and
burst delivery. Original runtime resolves NEXTN to EAGLE with steps=3/draft=4;
replay uses nextn=3/nextn_accepted=1.99. Do not assume an off-by-one without tracing
both definitions. The recipe's golden acceptance is explicitly inherited/provisional.

**Experiment.** Inspect runtime acceptance counters and implementation definitions;
measure matching speculation-on/off controls with actual accepted-token counts.

**Conclusion — open.** Configuration is present; measured request-specific acceptance
and its causal contribution are not yet established.

## H4 — Prefill/decode interference and closed-loop overlap

**Hypothesis.** Replay overlap or scheduling causes stalls absent on silicon.

**Experiment.** Reproduce the historical 25.7-second C14 decode-stall case on current
main; reduce to two requests with controlled arrivals, lengths, and cache state.
Compare admission, chunking, execution and token-delivery timestamps.

**Conclusion — open.** The older trace documents overlap divergence, not proof of
a current scheduler defect. No scheduler tuning before the discriminating control.

## H5 — Op timing, graph mode and interpolation

**Hypothesis.** Correct #665 selection leaves other kernels/shapes/timing boundaries
inaccurate. Compare graph coverage, full/reuse identity, precision, selected rows
and whole-forward critical path; collect only mismatching/uncovered shapes.

**Experiment.** Trace exact query routing, then matched-shape op and forward checks.

**Conclusion — open.** Data-file identity is verified; complete execution accuracy
is not established.

## H6 — Capacity, preemption and serving overhead

**Hypothesis.** Rank-local capacity, memory pressure, admission or response overhead
explains residual latency after the earlier controls.

**Experiment.** Reconcile capacity and lifecycle receipts. Instrument only unexplained
stalls, with unprofiled repeats to distinguish profiling overhead.

**Conclusion — open.** Do not infer eviction or preemption from cache utilization.

## Experiment ledger

| ID | Change | Status | GPU-hours | Next |
|---|---|---|---:|---|
| E000 | Independent frozen-cohort arithmetic | Verified | 0 | Research baseline |
| E001 | Positive tiny-denominator sensitivity | Verified diagnostic | 0 | Server/client control |
| E002 | Effective recipe comparison | Supported mismatch | 0 | Cap-only ablation |
| E003 | Current-main + #665 native baseline | Development complete | 0 | Full baseline accounting |
| E004 | C1 max sequences 256 to runtime 2 | No score change; all requests retained | 0 | Other mechanisms |

All evidence and commands are retained in the private campaign checkpoint.

## E003 — Current-main baseline versus historical dashboard

All frozen development requests are retained. On the identical C14 cohort, TTFT
MAPE changes 110.0450% to 104.7231%, TPOT 115.6302% to 84.8208%; C1 is unchanged.
C8 TPOT improves 54.4904% to 51.0948% but TTFT worsens 63.4054% to 63.7743%.
These are upstream-baseline effects, not new campaign fixes or a verified pooled
win. Current main includes #406's DSA reuse-accounting fix; attribution requires
a one-factor control, not a commit-title inference. Full baseline accounting and
validation-point results are recorded separately from development decisions.

## GPU timing controls — E005

CPU preparation job 1002174 failed before downloading because an interactive
shell variable was unset; no GPU time was used. Retry 1002194 downloads the exact
NVFP4 revision and retains weight/config checksums on a CPU data mover. Existing
shared GLM-5.3 has different precision and is not substituted.

The first diagnostic allocation is bounded at two TP8 node-hours, with fresh
servers for uninstrumented MTP, instrumented MTP, and instrumented plain decode.
`collect.sbatch` requires a reconciled remaining-GPU-seconds budget and freezes
code per job. `probe.py` controls synthetic contexts 8192/131072, output lengths
16/128 and lanes 1/8/14, with repeats and explicit cache flushes. These controls
are not AgentX accuracy scores. Cold/warm and profiler effects remain separate.

`instrument.py` adds one hook to the container's installed SGLang output streamer;
it records the original/modified file hashes and package version. The hook observes
host scheduler output readiness before detokenization/HTTP, not GPU kernel time.
`rsi_timing.py` records only request IDs, times, counts and batch membership. All
runtime source/log evidence remains private. Missing/incomplete requests stop the
control rather than being dropped. Compare against the original unmodified server
before interpreting instrumented timing.
