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

## E006 — Minimal overlap reduction

**Hypothesis.** A new current-main C14 high-error request is stalled by overlapping
prefill work, rather than an intrinsically 276 ms decode step.

**Experiment.** Use three numerical Mooncake controls with the same wheel, op data,
cache geometry and speculation. Warm a 287424-token prefix; the victim has 640
input tokens and 23 output tokens. Add a 361344-token prefill sharing the warm
prefix either 1 ms after the standalone first token or 1 ms after its last token.
Only interferer presence/arrival changes. `reduce_stall.py` retains each config,
trace, native report and result; this diagnostic is not a replacement benchmark.
The first attempt failed on assuming Mooncake preserves a request_id in native
telemetry; its evidence remains private. The corrected run selects the uniquely
specified victim shape and completes all three controls.

**Conclusion — confirmed simulated overlap mechanism; silicon fidelity open.**
Victim TPOT is 2.4626 ms standalone, 273.4086 ms with overlap, and 2.4626 ms when
separated; TTFT stays 58.6586 ms. The prior historical 25.7-second stall no longer
appears at that source request on current main (now 4.1313 ms TPOT). A different
current request produces 276.0665 ms TPOT versus silicon 3.7934 ms. This proves
sensitivity to overlap in replay, not that its scheduler is wrong: the measured
profile may have different arrivals and cache state. Next: verify the same
controlled interference on the GPU; do not tune scheduling to suppress the score.

## Full research baseline accounting

All 8624 frozen TTFT requests and 8614 frozen TPOT requests are retained across
all six points. Current-main pooled MAPE is 79.9935% TTFT and 70.0050% TPOT
(the exact sums and independent comparison are in private evidence). Both improve
against the historical dashboard; this is an upstream-baseline improvement, not
attribution to a campaign fix. C10 TPOT worsens 69.4362% to 73.8154%; C8 TTFT also
worsens slightly. Keep these regressions visible. The 10% targets remain unmet.

The next CPU ablation reverts only upstream #406's DSA reuse correction while
holding the new main baseline, #665 op table and frozen requests constant. Do not
attribute the upstream-baseline improvement from commit titles alone.

## Collection qualification update

All 50 weight shards and both configuration files match the immutable HF revision's
published checksums. Weight-manifest SHA256:
`a9a20139694f8d94eb4fc1fd1da8083da5de0fec3226555123ab4896c41cae50`.
The CPU preparation finished successfully. An overlapping CPU preflight step was
terminated when its enclosing preparation allocation ended; independent preflight
job 1002543 now checks the container interface. These attempts used no GPU time.

The timing probe also exercises the OpenAI chat frontend with the original
reasoning/tool parsers. It records content/reasoning character counts and usage,
never text, so first scheduler output, first visible reasoning/content, and final
client receipt can be distinguished. Native streaming, chat projection, frozen
scoring and resource accounting have seven passing focused tests. For stream
projection tests, run `uv run --no-project --with 'transformers>=4.57,<5' --with pytest
python -m pytest -c /dev/null -p no:cacheprovider test_probe.py`.

## E008 — Recorded cache-read alignment

**Hypothesis.** A broad cache-reuse mismatch explains most TTFT error.

**Experiment.** On development lanes only, join source-unique successful profile
records with equal output lengths to current-main telemetry and compare recorded
`usage_prompt_cache_read_tokens` with native first-admission reuse. Preserve
requests lacking this optional metric in headline accounting; they are unavailable
only for this cache diagnostic. All compared cache counts are within input lengths.

**Conclusion — broad mismatch not supported by averages; individual cases remain open.**

| Lanes | Diagnostic pairs | Mean silicon uncached tokens | Mean replay uncached tokens |
|---:|---:|---:|---:|
| 1 | 314 | 4163 | 4090 |
| 8 | 1388 | 3261 | 2987 |
| 14 | 2332 | 4253 | 3766 |

These averages do not prove matching batch/cache histories; they do not independently
explain the large TTFT gap. Keep frontend/tokenization and per-request tails in
the queue. Private paired rows and source identities are retained.

The runtime probe records distributed rank when available, keeping unknown ranks
explicit. Completed derived corpus copies are removed only after their checksums
are recorded; frozen source corpora, observed-output substitutions and numerical
results remain retained and reproduce them exactly.

## E007 — One-factor attribution of the upstream baseline improvement

**Hypothesis.** Upstream #406, rather than the other intervening main changes,
accounts for the observed development-point difference from the dashboard.

**Experiment.** Revert only commit `2be6a124f8310d35f0fa46c585b8c79ef56b01eb`
from the research baseline, rebuild its own wheel, and rerun lanes 1/8/14 with
identical #665 data and frozen inputs. Retain every baseline metric observation.

**Conclusion — confirmed commit-level effect on development cohorts.** The reverted
wheel exactly reproduces the historical dashboard TTFT/TPOT scores on all three
development points, retaining 4169 TTFT and 4166 TPOT observations. #406 therefore
explains the observed development-point change. Its lookup-slice and extrapolation
subchanges were not separately ablated, and this does not establish same-version
silicon accuracy or universal regression freedom.

## E009 — Deterministic replay repeat

**Hypothesis.** The C14 improvement is larger than native replay variability.

**Experiment.** Rerun the same current-main wheel, data, configuration and seed in
a fresh process. Compare all scored request pairs, not just aggregate averages.

**Conclusion — verified for this repeat.** Every scored C14 pair and both MAPEs
are exactly identical. This establishes zero observed replay variation for this
control; it does not measure GPU repeat variability.

## Preflight failures retained

CPU job 1002543 failed before any GPU allocation because this pinned image uses
`SchedulerOutputProcessorMixin`, not the newer output-streamer file layout. GPU
reservation 1002644 was canceled by its failed dependency with zero elapsed time.
The hook now checks both known module layouts and requires exactly one matching
`stream_output(reqs, ...)` interface; duplicate/instrumented/unknown interfaces
fail explicitly. Independent preflight retry: 1002662. Six audit/hook tests and
two native/chat projection tests pass. The CPU failures consume no GPU budget.

## Preflight interpretation correction

The earlier inference that the image necessarily uses the older mixin layout was
premature and is superseded. Retry 1002662 showed that distribution-relative paths
are invalid for this editable SGLang installation. The probe now resolves the actual
package origin without importing the GPU runtime and tests editable installations
explicitly. Retry 1002721 checks that corrected path discovery before GPU use.
These are probe setup failures, not evidence that SGLang or the model is unsupported.

## E010 — Qwen CPU compatibility smokes

The GLM-only wheel intentionally lacks PR #666's vLLM 0.31.0 context table; the
first three smokes correctly returned database-unavailable rather than borrowing
another version. Retain those outcomes. Reuse the same installed native binary
with an isolated data root that adds only the previously qualified #666 parquet
`d1188fe9cdb287bb5c87a9094026a17fe50d2401266a71a657facadbfec45700`.

All three 4K/cold/repeat-0 smokes at 1/4/16 lanes then complete: 8/32/128 requests
match with identical output lengths and all cache receipts agree, including
preparation. These are compatibility smokes, not a new Qwen accuracy campaign;
no Qwen GPU collection or active dataset changes belong to this campaign.

The reusable gap catalog records commit-level DSA reuse attribution as GLM-RSI-01,
with the residual hypotheses and regression limits preserved.

## Qualified CPU preflight

Retry 1002721 passes. Actual SGLang package version is 0.5.21; its implementation
is the newer `scheduler_components/output_streamer.py`. The older-layout hypothesis
is disproven: editable-install path discovery caused the prior failures. Original
streamer source SHA256 is `8bfe917edc87ae89f387c718944310e8f9002e1740ecae87b6fd7194f1669855`;
the tested hook SHA256 is `9d2ec980b62603532fa29443f86eb0183a2951037405f4187afc3e20eef06e1d`.
Launch help and client imports complete on CPU. The next GPU allocation remains
bounded at two node-hours and must retain its own actual source hashes and runtime
receipts; successful CPU preflight is not a claim of successful GPU serving.

## First GPU allocation correction

Job 1002810 was stopped during container startup before serving or measurements.
The launch did not explicitly scope compile/tokenizer caches to campaign storage;
that launch configuration is corrected and covered by a cache-location test.
Charge 96 seconds through cleanup at eight reserved GPUs (768 GPU-seconds),
conservatively including the longest Slurm cleanup record; parent elapsed was
87 seconds. Remaining budget before the next reservation: 229632 GPU-seconds.
No measured accuracy result is attributed to this canceled allocation.

## Active timing controls and accounting guard

Corrected GPU job 1002848 uses frozen collector commit
`c7bcd50b0dc7eeefea78754aaa00462c2332584c`; CPU preflight and checkpoint
checksums qualified before submission. It has a two-node-hour allocation limit.
Keep its own runtime hashes immutable while it runs.

A later client-retention guard records every concurrent attempted request before
stopping a failed case. Its regression test forces fourteen simultaneous HTTP
failures and verifies all fourteen are retained without error-body or prompt
content. This source update does not alter job 1002848. If that frozen run has a
client-case failure, reconstruct explicit missing-attempt accounting from the
case protocol and reject incomplete comparisons; do not infer missing timings.
