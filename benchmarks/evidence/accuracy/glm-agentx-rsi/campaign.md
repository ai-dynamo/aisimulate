<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM AgentX RSI campaign: per-request TTFT and TPOT MAPE

## Objective, protocol and current status

**Running; neither below-10% target is achieved.** Reduce pooled per-request TTFT
and TPOT MAPE across B300 TP8 GLM AgentX lanes 1/4/8/10/13/14. A win lowers
either metric without worsening the other beyond observed repeat variability.
Pool individual valid request errors, not concurrency MAPEs. Point error,
throughput, curves and forward timing are diagnostics, not substitutes.

Keep every measured request in accounting. Freeze metric-specific identities
before experiments and report lost/recovered matches separately; a smaller scored
cohort cannot establish a win. Preserve small positive references, undefined/zero
values, failures and unmatched requests. No outlier removal, unexplained timing
multipliers or score-driven calibration. Preparation is outside headline scoring.

Develop on lanes **1/8/14**; reserve **4/10/13** from tuning during this campaign.
Historical scores are already known: this is held-out-from-tuning validation,
not previously unseen data. New GPU controls are diagnostics, not replacements
for the frozen AgentX measurements.

**Verified numerical improvement:** current main plus #665 improves both pooled
MAPEs with every frozen metric observation retained. One-factor attribution to
upstream #406 is confirmed on development points; no new campaign modeling fix
has been accepted. C10 TPOT and C8 TTFT regressions remain visible below.

## Frozen baselines — E000/E003

| Variant | TTFT MAPE | TTFT n | TPOT MAPE | TPOT n |
|---|---:|---:|---:|---:|
| Dashboard | 84.5587% | 8624 | 79.0844% | 8614 |
| #665 original-main cohort | 85.0249% | 8260 | 146.7501% | 8251 |
| #665 with new data, same fixed cohort | 77.5517% | 8260 | 81.9830% | 8251 |
| Research main + #665, frozen dashboard cohort | 79.9935% | 8624 | 70.0050% | 8614 |

No metric observations are lost from the dashboard cohort. Its denominators differ
from #665's fixed cohort; the two comparisons remain separate.

| Lanes | Dashboard TTFT | Research TTFT | Dashboard TPOT | Research TPOT |
|---:|---:|---:|---:|---:|
| 1 | 73.6567% | 73.6567% | 18.8688% | 18.8688% |
| 4 | 60.8372% | 60.8380% | 42.4000% | 42.4000% |
| 8 | 63.4054% | 63.7743% | 54.4904% | 51.0948% |
| 10 | 65.0622% | 64.9230% | 69.4362% | 73.8154% |
| 13 | 93.1288% | 80.8748% | 81.6390% | 79.0883% |
| 14 | 110.0450% | 104.7231% | 115.6302% | 84.8208% |

Frozen-cohort diagnostics (inclusive linear percentiles; no trimming):

| Diagnostic | Dashboard TTFT | Research TTFT | Dashboard TPOT | Research TPOT |
|---|---:|---:|---:|---:|
| Median APE | 63.8256% | 63.7551% | 18.6451% | 18.4494% |
| P95 APE | 94.1059% | 93.9970% | 96.8281% | 94.3497% |
| P99 APE | 635.2183% | 611.6515% | 400.8936% | 357.7478% |
| Mean signed error | -31.1508% | -35.6382% | +59.1729% | +50.2411% |
| Mean absolute error | 598.6003 ms | 576.6278 ms | 1.6855 ms | 1.6889 ms |

The TPOT MAE increases slightly despite lower MAPE. Its 37 positive references
below 0.1 ms contribute 41.9348/31.9538 percentage points before/after, and remain
in the headline. Median errors also exceed 10%; tiny denominators are not the
only remaining problem. Per-concurrency diagnostics stay in the private manifest.

## H1 — Client timing boundaries and small positive denominators

**Hypothesis.** Buffered receipt, frontend work and/or speculative bursts compress
some client-observed decode intervals. Positive tiny denominators magnify MAPE.
Confidence: high in denominator sensitivity, moderate in the boundary mechanism.
Competing explanations: real short speculative bursts and scheduling differences.

**Experiment E001.** Independently recompute the original request errors and
stratify references below 0.1 ms without excluding them. Audit raw HTTP and metric
boundaries; retain all requests in the headline metric.

**Conclusion — sensitivity confirmed; timing mechanism open.** At 14 lanes,
9/2429 references contribute 65.43 percentage points to 115.63% TPOT MAPE; median
APE is 24.39%. One raw example has 222.7242 ms HTTP receive time but 0.54664 ms
recorded decode duration. These facts do not identify when the server generated
its tokens and do not justify changing historical observations.

**Experiment E015 — client boundary audit.** Pin the AgentX harness 1.0.12 wheel
and inspect its TTFT, request-latency, ITL and SSE-reader sources. All 2453
available TPOT values among 2462 raw profile records reconcile with
`decode_duration / (output_length - 1)`; this raw audit includes requests excluded
from the frozen eligible cohort. The reader assigns one receipt timestamp to SSE
messages parsed from the same transport chunk. **Conclusion:** the import formula
is correct and coalescing can compress the observed interval. Historical
coalescing remains unproven without server-ready/receipt observations; no baseline
correction is accepted. Keep the original last-content response boundary.

**E019 — raw chunk-count cross-check.** Audit profiling records on development
lanes without changing eligibility: 1/5/9 positive TPOT references below 0.1 ms
at lanes 1/8/14. Some contain many raw HTTP chunks: a 252-output-token C14 record
reports 77 chunks, 14.104 ms HTTP receiving and 14.130 ms full decode duration.
Server usage, output-token count and output-sequence length agree for all nine
C14 examples. A timestamp shared within one raw chunk therefore cannot alone
explain the entire pattern. Delayed buffering/draining upstream or client-side
remains a hypothesis; chunk counts do not locate the delay. Two C14 burst records
on different client workers begin visible content within 2.7 ms of each other;
this coincidence is descriptive, not causal proof. The original metric values
and every headline request remain retained. Private artifact
`E019-receipt-chunks-v2.json` explicitly supersedes a zero-row diagnostic that
used the wrong phase spelling; the invalid first attempt remains preserved.

**Experiment E005, readiness repair qualified for retry.** Compare uninstrumented MTP, instrumented MTP and
instrumented plain decode with fresh servers. Synthetic controls use 8192/131072
input tokens, 16/128 output tokens, lanes 1/8/14, repeats and explicit cache-flush
requests. Also exercise the OpenAI chat frontend with reasoning/tool parsers.
Capture host scheduler output-ready counts/timestamps by rank, client frame
receipt, numerical usage/acceptance fields, and one-second server metrics.
These are output-ready observations before detokenization/HTTP, not GPU kernel
completion timestamps. Chat frames without per-chunk token counts must not be
presented as exact per-token observations. Missing comparisons remain N/A.
For instrumentation effects, compare matching native matrix cases at repeat 1;
retain repeat 0 separately because first-use compilation may differ between fresh
servers sharing a compiler cache. Output-ready timestamps alone cannot establish
prefill-kernel overlap; that requires a qualified forward/profiler window.

**Conclusion — startup failed; GPU timing evidence unavailable.** Job 1002848
exceeded its 1200-second serving-readiness guard. A read-only Python stack sample
places the scheduler in `do_load_weights`, waiting for asynchronous expert-weight
copies (`expert_data.copy_`); the shard reader had finished. This establishes the
startup location, not a deadlock cause. No request was attempted. A native stack
sample failed to unwind and is retained as unavailable. Prepare the original
SGLang 0.5.20 image by immutable digest on CPU before another GPU attempt.

The historical attempt 1003379 passes weight loading and graph capture, then
fails its initial zero-timeout cache reset before any diagnostic request. Server
logs show pending warmup/pipeline work even though queue/running gauges are zero.
Its information endpoint was an insufficient readiness gate. Both pinned images
implement `/ready`, `/health_generate` and `/flush_cache?timeout=60`; the repaired
collector waits for readiness and a completed health generation, then permits a
bounded drain at each reset. A regression test keeps server-info available while
readiness fails and confirms no probe starts early. No scientific timing result
or measurement correction is claimed from either failed attempt.

Require actual counts, clocks, rank identity,
completion, output-length conservation and coverage before comparison. Server
retained/processed-token counts and client-delivered counts are distinct; preserve
any terminal over-generation rather than silently equating them.

`analyze_timing.py` accounts for 221 planned identities per mode, retains missing
records as unrecorded-or-unattempted (a killed case may have started), validates
counter/clock ordering and exact delivered lengths, and uses rank-0 readiness.
It ends native spans at first attainment of the delivered length, excluding
duplicate terminal frames. One-token intervals and chat token-interval scores
remain unavailable. Numerical tests cover units, terminal over-generation,
missing rank identity, backward counters/clocks and failed-attempt retention.
The frozen active run uses its container-local SGLang autotune cache; a later
collector revision explicitly scopes `SGLANG_CACHE_DIR` to campaign storage.
Neither this path fix nor the audit modifies running job 1003379.

**E005 natural-control result — qualified only as natural acceptance.** Job
1004446 completes both MTP modes, 221/221 requests each. The instrumented mode
has rank-0 observations for every request, no backward clocks/counters or
uncovered delivered lengths, and 201 requests with explicitly retained terminal
over-generation. Median first/final native receipt delays after readiness are
2.127/4.666 ms; maximum delays are 12.157/10.459 ms (first includes chat).
This does not establish the cause of historical AIPerf bursts. All 221 plain-mode
identities remain unrecorded-or-unattempted after intentional cancellation.
Original-policy qualification fails: this source omitted the forced acceptance
environment discovered in E016. Preserve all 442 completed diagnostics rather
than use them as matched AgentX evidence. Both headline MAPEs and coverage remain
unchanged. Audit artifact: private `E005-natural-audit-v1.json`.

**E017 — matching-policy controls and independent CPU predictions.** Job 1005806
freezes `8a9efa81612175eab4d4baa4490f7c2688225fa7`, uses the exact historical image,
and explicitly applies all three original acceptance overrides. The same 221
identities per mode, repetition policy and qualification rules are predeclared.
Generate eight cold C1 predictions afresh: 8192/131072 input × 16/128 output ×
MTP/plain. Keep the first failed plain configuration attempt; the corrected
eight-case run completes with exact lengths and zero prefix reuse. MTP TTFT is
243.863/5064.213 ms and TPOT spans 2.413–2.580 ms; plain TTFT is 242.361/5040.178
ms and TPOT spans 5.978–6.134 ms. Silicon 0.5.20 versus database 0.5.21 remains
cross-version. No synthetic result replaces either headline objective.

The matching-policy uninstrumented mode completes all 221 requests. For repeat 1
and 128 output tokens, 8K/131K cold C1 silicon TPOT is 2.6879/2.8277 ms versus
2.4516/2.5796 ms predicted (8.79%/8.77% error). TTFT is 446.002/5694.958 ms
versus 243.863/5064.213 ms predicted (45.32%/11.08% error). Short outputs retain
larger finite-acceptance variability: the 8K/16-token repeats measure 2.6632 and
3.1879 ms TPOT. These are untrimmed synthetic diagnostics, not a cohort win or
same-version op qualification. The independent private audit and complete
`E017-cold-unprofiled-comparison.json` preserve both repeats, cold cache receipts,
acceptance counters and all numerical comparisons. Remaining instrumented and
plain modes must finish before accepting an E017 conclusion.

## H2 — Recipe, tokenization, snapshots and cache history

**Hypothesis.** Effective serving settings or replay history differ from silicon.
Confidence: high in the recorded cap mismatch, low in a broad cache-history cause.
Competing explanations: correct inputs with incorrect execution or client timing.

**E018 — predeclared paired frontend experiment (pending).** Test whether large
mostly cached prompts incur frontend work missing from native token replay.
Use fresh uninstrumented and instrumented historical servers with the original forced-acceptance
policy, synchronous scheduling and all other E017 settings. For each of three
synthetic context labels (8192/131072/393216), construct one chat-template token
sequence. Compare native input IDs against chat messages, alternating endpoint
order over three repeats. Independently flush and seed exactly the same aligned
prefix before each request; retain 18 preparation and 18 measured requests per
mode (72 total), reporting instrumentation effects separately.
Preparation primes the allocator and remains unscored. Require exact server input
and output lengths, identical input hashes and equal reported cache reuse before
attribution; missing cache receipts leave the comparison unavailable. Serialize
requests before starting the measured timer, recording serialization separately.
Keep content, reasoning and tool-call boundary counts without text or token arrays.
Predicted observable: extra chat time before first scheduler readiness at equal
GPU input/cache state. Competing explanations include template mismatch, parser
buffering and drift; this synthetic repeated text cannot establish the entire
Weka workload's frontend cost. No fitted latency constant will be introduced.
The exact image's synchronous `_render_and_encode_chat_template` method renders
then encodes the full text without a thread offload. Its source SHA256 is
`047e4391dc87816a62da34392eea9379fcb1f93025e70c3b327de8b596f59b09`.
An optional timer retains stage duration, input count and a little-endian u32
token digest; it records projection cost separately and retains no text/arrays.
Retain wall, process-CPU and calling-thread CPU time separately, plus the serving
host CPU model/affinity, tokenizer package versions and thread settings. Host
measurements require these identities; B300 GPU identity alone is insufficient.
The local template explicitly requests `return_dict=False`, matching the server's
list-of-token-IDs contract across Transformers releases.
Join exactly one stage inside each chat request and require its digest to match
the native input. Missing/ambiguous stage, token or cache evidence disqualifies
the comparison. Use `analyze_timing.py --suite frontend` for independent checks.
The existing collector selects this suite explicitly; the default timing suite
and frozen running job remain unchanged. CPU runtime-source extraction (1005817)
uses no GPUs and pins the exact image before interpreting framework internals.
It completes successfully in 362 seconds. Existing `engine.tokenization` can
represent independently measured queued service costs; it is unset in the frozen
replay. Do not choose costs using MAPE or assume repeated synthetic text prices
the Weka corpus without further qualification.
CPU job 1006134 verifies all three template paths in the exact image in 217
seconds, producing 8204/131084/393228 input tokens. Direct template token IDs
match the server's separate render/encode path exactly. The GPU suite requires
this receipt and refuses mismatched model-config or tokenizer-runtime identities.
CPU-data-mover timings are retained as diagnostics and will not supply a serving
host cost model.

**Experiment E002/E004.** Parse retained runtime receipts. C1 used a running-request
cap of 2, while replay allowed 256; C4 used 8. Change only C1's cap to 2.

**Conclusion — recipe mismatch confirmed, C1 accuracy impact disproven.** All 324
scored requests remain; TTFT/TPOT MAPE are exactly unchanged at 73.6567%/18.8688%.
Keep the setting alignment separate from a claimed accuracy improvement.

**Experiment E008.** Join unique successful, equal-output-length development pairs
with recorded `usage_prompt_cache_read_tokens` and native first-admission reuse.
Requests without this optional metric remain in headline accounting.

| Lanes | Cache diagnostic pairs | Mean silicon uncached tokens | Mean replay uncached tokens |
|---:|---:|---:|---:|
| 1 | 314 | 4163 | 4090 |
| 8 | 1388 | 3261 | 2987 |
| 14 | 2332 | 4253 | 3766 |

**Conclusion — broad mismatch not supported by averages; individual cases open.**
All compared cache counts are within input lengths. The averages do not establish
matching histories, arrival overlap or a causal explanation of TTFT error.

**Experiment E014.** Use the existing native Weka qualifier to retain
source prompt-hash relationships. Replace the closed-loop controller with recorded
arrivals and observed output lengths on development lanes; retain preparation.
Predeclare the original frozen metric-specific cohorts. Preserve failed attempts
in accounting; missing observed lengths prevent simulating their GPU work. Compare
this diagnostic against E003, with no timing correction or input-length change.
It cannot establish an accuracy win or fully isolate historical cache behavior.

| Lanes | Closed-loop TTFT | Recorded-arrival TTFT | Closed-loop TPOT | Recorded-arrival TPOT |
|---:|---:|---:|---:|---:|
| 1 | 73.6567% | 65.3474% | 18.8688% | 18.1127% |
| 8 | 63.7743% | 85.8425% | 51.0948% | 65.3959% |
| 14 | 104.7231% | 68.4612% | 84.8208% | 116.0949% |

**Conclusion — controller/arrival sensitivity supported; no general improvement.**
All frozen TTFT 4169 / TPOT 4166 observations remain scored. Independently audit
331/1492/2578 native completions at lanes 1/8/14 and exact requested output lengths.
Failed measured attempts (6/25/39 including preparation) remain in accounting
but have no qualified lengths to simulate. The static replay retains source
prompt hashes/lengths, not HTTP-template tokens or verified historical cache
instances. Neither its extra recovered matches nor its lower C14 TTFT makes it a
candidate win. Next discriminate actual overlap and client timing on GPU.

## H3 — Speculative acceptance and output-delivery bursts

**Hypothesis.** Fixed acceptance misses request-dependent acceptance and bursts.
Confidence: low for a large mean mismatch after E012; request-level effects open.
Competing explanations: client receipt coalescing and prefill/decode interference.

**Experiment.** Verify the original NEXTN-to-EAGLE resolution (steps 3, draft tokens
4). Replay uses `nextn=3`, `nextn_accepted=1.99`, meaning 2.99 accepted output
tokens per round before request-tail truncation. The recipe's golden acceptance
is explicitly inherited/provisional; do not treat it as measured per-request truth.
E005 collects actual acceptance and output counters with speculation on/off.

**Experiment E016 — launcher policy audit, confirmed.** Inspect the original
InferenceX commit `c093d2052d2e4f6aeaa8162e9e81219f31556fe4` and C14 job
`119061192112` in run `39551262792`. The actual job log sets
`SGLANG_SIMULATE_ACC_LEN=2.99`, `SGLANG_SIMULATE_ACC_METHOD=match-expected` and
`SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`. The launcher's
[synthetic acceptance policy](https://github.com/SemiAnalysisAI/InferenceX/blob/c093d2052d2e4f6aeaa8162e9e81219f31556fe4/inferencex-e2e/infx/srt_slurm/synthetic_acceptance.py)
selects this golden value for speculative AgentX throughput runs; it is not
unconstrained natural acceptance. The inherited golden value is provisional as
a real-workload estimate, but it is the original benchmark's prescribed policy.
Actual job-log receipts at all six concurrencies confirm the same three values.
The private evidence artifact retains each job ID, URL and selected receipts.
The prediction lowering already samples integer acceptance from conditional rates
`1,0.99,0`; it does not advance requests by a fractional 2.99 or silently use the
scheduler's generic fallback distribution. A deterministic fractional-rounding
explanation is therefore unsupported by the current source.

**Protocol deviation.** E005 omitted these environment overrides. Its first
synthetic requests report natural acceptance 4.0 with rate 1.0; a live process
environment check confirms the three override variables are absent. E005 fails
original-policy qualification. Retain it only as a natural-acceptance diagnostic,
with no historical accuracy claim. Finish its instrumented natural control to
retain a server/client comparison, then stop before duplicating the plain control.
Next collect the original forced-acceptance controls, preserving both variants.
Do not change the frozen prediction setting: its 1.99 accepted drafts plus bonus
already targets the prescribed total 2.99. This is a collection repair, not a win.
The corrected collector records an explicit policy and the three environment
values, clears inherited overrides for natural/plain controls, and tests all
three cases. E017 will repeat the forced-policy controls with fresh servers;
natural data cannot substitute for them.

**Experiment E012.** Independently inspect retained profiling-phase server metric
receipts on development lanes. The mean `sglang:spec_accept_length`, explicitly
defined as accepted drafts plus bonus token per forward, is **2.9898103 at 8 lanes**
and **2.9867272 at 14 lanes**. The retained metric JSON is unavailable at 1 lane.
The 8/14-lane series contain 3600/3584 one-second intervals respectively.

**Conclusion — large mean-acceptance mismatch unsupported; request effects open.**
These runtime means closely match 2.99. They are gauge summaries, not per-request
or verify-count-weighted acceptance; do not infer individual acceptance, missing
intervals or verify counts from them. No new prediction setting or score change
is justified by this experiment. Request-level bursts remain an E005 question.

## H4 — Prefill/decode interference and closed-loop overlap

**Hypothesis.** Wrong overlap or scheduling creates decode stalls absent in silicon.
Confidence: high in simulated interference, unknown in silicon scheduler error.
Competing explanation: different controller arrivals create different valid overlap.

**Experiment E006.** The historical 25.7-second stall no longer occurs at that
source request on current main (now 4.1313 ms TPOT). A different current request
predicts 276.0665 ms versus silicon 3.7934 ms. Reduce it to numerical Mooncake
controls: warm a 287424-token prefix; victim input/output lengths are 640/23;
interfering input length is 361344 with that warm prefix. Add the interferer 1 ms
after the standalone first token, or 1 ms after its last token.

| Control | Victim TPOT | Victim TTFT |
|---|---:|---:|
| Standalone | 2.4626 ms | 58.6586 ms |
| Overlapping prefill | 273.4086 ms | 58.6586 ms |
| Separated prefill | 2.4626 ms | 58.6586 ms |

**Conclusion — simulated overlap mechanism confirmed; silicon fidelity open.**
Only interferer presence/arrival changes. This proves sensitivity to overlap, not
an incorrect silicon scheduler. E005 tests controlled GPU interference. Its
client-first-output trigger requires scheduler receipts to establish actual overlap.
The initial reduction attempt assumed Mooncake preserves request IDs; its failed
evidence is retained. The corrected control selects the uniquely specified victim.

**E014 follow-up — false overlap confirmed for this pair.** The silicon dispatch
records place that interferer 113.284 seconds after the victim starts and 113.129
seconds after it finishes. Closed-loop replay instead admits it about 1 ms after
the victim's first token. With recorded arrivals the victim predicts 3.9997 ms
TPOT versus silicon 3.7934 ms, compared with 276.0665 ms in closed-loop replay.
Its TTFT is 62.9373 ms versus silicon 71.6783 ms. This identifies wrong overlap
for this outlier; it does not establish a scheduler defect or a general fix.
HTTP input length is 663 versus source/replay 640; retain that qualification.
Next trace the upstream timing divergence and compare controlled silicon overlap.

## H5 — Op timing, graph mode, variant lookup and interpolation

**Hypothesis.** Data selection or execution timing differs despite including #665.
Confidence: high in the scoped upstream commit effect; remaining timing fidelity
unknown. Competing explanations: correct forward timing with wrong overlap or
client measurement boundaries.

**Experiment E007.** Revert only upstream #406 commit
`2be6a124f8310d35f0fa46c585b8c79ef56b01eb` from the research baseline, rebuild its
own wheel, and rerun development lanes with identical #665 data and frozen inputs.

**Conclusion — commit-level effect confirmed on development points.** The reversion
exactly reproduces the old dashboard TTFT/TPOT scores on all three development
cohorts, retaining 4169 TTFT and 4166 TPOT observations. The lookup-slice and
extrapolation subchanges were not separately ablated. Do not extrapolate that
attribution to all six points or claim universal regression freedom.

**Experiment E011.** Inspect current #665 generation and primary context tables at
GLM geometry: 8 local heads, BF16 MLA/projection and FP8 KV. Both full and skip
variants exist (2469 context rows and 404 generation rows per matching variant).

**Conclusion — matching geometry/precision coverage confirmed, timing fidelity open.**
A filename or row count does not establish actual runtime kernel selection,
interpolation weights or whole-forward accuracy. Collect exact discriminating
shapes only after runtime/selected-row reconciliation.

**Candidate question — separate MTP draft composition.** The pinned checkpoint
excludes layer 78 from quantization, and MTP startup exercises BF16 MoE tuning.
The current generation graph instead applies `mtp_scale_factor` to the target
layer mix. Quantify target verification and draft-forward shapes, precision,
indexer use and costs separately before proposing a change. This is an unverified
modeling hypothesis; an average-layer substitution is not enough evidence for a
new correction factor or an accepted fix.

## H6 — Capacity, preemption, frontend and serving overhead

**Hypothesis.** Admission, memory pressure or frontend work explains residual error.
Confidence: token capacity agrees; runtime overhead and lifecycle causes unknown.
Competing explanations: incorrect op timing and client buffering.

**Experiment E013.** Reconcile fixed replay capacity with actual runtime receipts.
All eight TP ranks on development lanes report 3783936 cache tokens for each of
the main/MTP allocations. Replay's 59124 blocks × 64 tokens equals 3783936.

**Conclusion — token-capacity mismatch disproven on development points.** Equal
capacity does not establish equal eviction, preemption or admission behavior.

**Next experiment.** Inspect admission, preemption and response timing. E005 distinguishes native
input-ID requests from chat frontend observations. Use uninstrumented repeats
before interpreting profiler effects.

**Conclusion — open.** Do not infer eviction/preemption from utilization, or infer
which operation uses FP8 merely from a model-level loader message. The pinned
checkpoint is mixed precision; match component-level settings and evidence.

`predict_cold.py` generates eight fresh cold C1 predictions at 8192/131072 input
tokens, 16/128 outputs, MTP with total acceptance 2.99 and plain decode. Compare
both recorded repeats with the forced-policy controls; repeat 1 separately checks
steady-state compilation. These are diagnostics, not the frozen headline cohort.
The 0.5.20 silicon versus 0.5.21 database comparison remains explicitly cross-version.
The initial cold-control helper incorrectly supplied `nextn_accepted=0` with
disabled speculation; plain control validation rejects that field. Retain this
failed attempt and its four completed MTP predictions, omit the field when off,
and generate all eight controls afresh in a separate output directory.

## Repeat and compatibility controls

**E009 hypothesis/experiment.** Test replay variability with a fresh C14 process
using the same wheel, data, seed and settings. **Conclusion:** every scored pair
and both MAPEs are exactly identical. This measures replay repeatability, not GPU
repeat variability.

**E010 hypothesis/experiment.** Run existing Qwen CPU smokes at 4K/cold/repeat 0,
lanes 1/4/16. The GLM-only wheel lacks #666's vLLM 0.31 context table and initially
returns database-unavailable; retain these outcomes. With the same native binary
and an isolated root adding only the qualified #666 parquet, all 8/32/128 measured
requests match and every cache receipt agrees, including preparation. **Conclusion:**
compatibility smoke passes with its required data, not a new Qwen accuracy result.
Qwen recollection and its GPU allocations remain separate.

## GPU budget, qualification and failed attempts

Hard cap: **64 allocated B300 GPU-hours / eight TP8 node-hours**. Include startup,
failed jobs and profiling. Reserve active allocations at their full wall limit;
reconcile terminal jobs against Slurm reserved GPUs and elapsed time. Intended
split: 2 node-hours baseline/timing, 2 controlled experiments, 1 op/forward,
3 validation; unused diagnostic time can move to validation.

- CPU preparation 1002174 failed on a shell variable; 1002194 then downloads and
  checksums the exact checkpoint. All 50 shards and two configuration files match
  immutable HF checksums. No substitute shared model is used.
- CPU preflights 1002543/1002662 expose a probe package-path error. The initial
  older-layout explanation is superseded: the image uses the newer streamer
  under an editable install. Actual package origin resolution and its regression
  test fix discovery. Preflight 1002721 passes with SGLang 0.5.21, Torch
  2.13.0+cu130 and Transformers 5.12.1. Earlier overlapping CPU steps and failures
  remain in private logs; they cost no GPU time.
- GPU reservation 1002644 never allocates after its dependency fails: charge zero.
- GPU 1002810 is stopped during startup to scope compile/tokenizer caches to
  campaign storage. Charge 96 seconds through cleanup at eight GPUs: 768 GPU-seconds
  (parent elapsed 87 seconds). No serving measurements are claimed.
- GPU **1002848**, frozen collector
  `c7bcd50b0dc7eeefea78754aaa00462c2332584c`, fails the startup guard. Slurm parent
  and cleanup elapsed 1236 seconds at eight GPUs: charge 9888 GPU-seconds.
  Total charged: **10656 GPU-seconds / 2.96 GPU-hours / 0.37 TP8 node-hours**.
  Remaining: **219744 GPU-seconds / 61.04 GPU-hours / 7.63 TP8 node-hours**.
- CPU historical-image preparation 1003104 rejects Enroot's architecture spelling
  `amd64`; corrected to its documented `x86_64` option in retry 1003117. Neither
  reserves GPUs. Runtime identity and hook compatibility must pass before use.
- CPU 1003117 qualifies the exact historical SGLang 0.5.20 image in 748 seconds.
  Squashfs SHA256: `7365b3e0ef2a45e2924540add0720f415b3e7c5c18b3fe02e779ea47105da079`.
- GPU **1003379**, source `389b183f7981d1be2cf50d864a22370a62ebb697`, attempts the
  historical timing controls with a 90-minute cap. The short queue was full;
  changing this pending job to the authorized normal queue keeps its resource
  limits. An initial launch on pool0-0203 fails before the batch script with
  `user_env_retrieval_failed_requeued_held`; charge six seconds through batch
  cleanup (48 GPU-seconds), retain that scheduler receipt, and release this held
  job once. Its next attempt on pool0-0095 fails the initial cache reset after
  843 allocated seconds, with zero diagnostic requests. Charge 849 seconds across
  both attempts × eight GPUs = 6792 GPU-seconds. Total campaign charge is now
  **17448 GPU-seconds / 4.8467 GPU-hours / 0.6058 TP8 node-hours**; remaining
  **212952 GPU-seconds / 59.1533 GPU-hours / 7.3942 TP8 node-hours**.

The historical retry also restores the recorded multithread/prefetch loading and
keep-alive settings. This is a qualification repair, not an isolated experiment
attributing a change to the framework version. The GPU runner accepts only the
two named runtime choices and requires the historical image's CPU receipt and
checksum before allocation. The failed 0.5.21 run remains separate evidence.
The historical first attempt is capped at 90 minutes, with an 85-minute collector
deadline and five minutes reserved for container startup/cleanup.
The readiness-repaired retry reduces these limits to 75/70 minutes. Persist
compiler caches per pinned image under campaign storage; the failed historical
attempt's cache is retained there with a relative symlink preserving its old path.
Do not share these caches across framework images. Fifteen focused audit,
readiness, counter and client-projection tests pass before this retry.

Retry **1004446** freezes source `b311598589516db220b716f079b486614ea0dd2d` and
runs on pool0-0095. Reserve its 4500-second wall limit plus 60 seconds of cleanup
at eight GPUs: 36480 GPU-seconds. With 17448 already charged, 176472 GPU-seconds
remain available after this reservation. Later submissions require this cleanup
margin explicitly; the current job was admitted with ample remaining budget.

Job 1004446 is cancelled after its two completed natural-acceptance controls;
the parent lasts 2094 seconds and cleanup ends at 2101 seconds. Conservatively
charge 2101 × eight = 16808 GPU-seconds. Total completed allocation cost is now
**34256 GPU-seconds / 9.5156 GPU-hours / 1.1894 TP8 node-hours**; remaining
**196144 GPU-seconds / 54.4844 GPU-hours / 6.8106 TP8 node-hours**. Job 1005806
reserves 36480 GPU-seconds including cleanup, leaving 159664 unreserved. Replace
this reservation with its actual terminal allocation through cleanup before the
next GPU submission. Preserve scheduler receipts and all failed attempts.

A later concurrent-retention guard writes every attempted request before stopping
a failed case; a test forces fourteen simultaneous failures and retains all of
them without prompt/error-body content. It does not modify frozen job 1002848.
If that run fails a case, retain explicit missing-attempt counts and disqualify
incomplete comparisons rather than inventing timings.

Stop when both targets validate, budget is exhausted, three consecutive completed
hypothesis experiments produce no discriminating evidence or verified improvement,
or no actionable hypothesis remains. Setup failures/regression tests are recorded
separately. Report residual error honestly. No automatic merge, HF promotion or
dashboard replacement; publication requires complete qualification/privacy checks.

## Provenance and reproduction

- Historical dashboard source: `1c753e4668310705b6451ba1ebf3061b3d5b30de`;
  HF silicon: `da115d648f0be20a1c4e0408d552a4a1e8bebf9d`.
- Research baseline `eddb21246e0a371fe542f7cab4034b58173bce63`: main
  `9f4819e2b77afb4798bc8133c42ccc1d1b045e5c` plus #665
  `64ac539813677d15e58c8ba051a6b5fc95c5334b`.
- Research wheel SHA256:
  `79ed539e44052aca60f9e62d6427654417e2ce1c514c881cfee0f0949581d046`;
  native SHA256 `07b39a7f2e76aba7e4092f58a133f574059da784d98ab3bec1cdef94aec0319d`.
- Frozen scoring tooling: `a01b4b45d4a58f65332e7fb7646cc5eab2ce1cbe`.
- #665 parquet SHA256:
  `287367dcd1b321a7b5ee15422b5b36b75268e76663c7f814a53d7151ee8d9c88`.
- #666 smoke-only parquet SHA256:
  `d1188fe9cdb287bb5c87a9094026a17fe50d2401266a71a657facadbfec45700`.
- GPU checkpoint: `nvidia/GLM-5.3-NVFP4@e3b8f9f2fad29727dfa95e766c5cac18ea1414bf`;
  weight-manifest SHA256 `a9a20139694f8d94eb4fc1fd1da8083da5de0fec3226555123ab4896c41cae50`.
- Actual SGLang streamer source SHA256:
  `8bfe917edc87ae89f387c718944310e8f9002e1740ecae87b6fd7194f1669855`.

Raw data, private identity cohorts, execution commands, failure evidence and the
machine-readable checkpoint remain in the private campaign output directory.
Completed derived trace copies are removed only after checksums are saved; frozen
source corpora and observed-output substitutions remain available to reproduce them.
The reusable gap catalog records scoped attribution as **GLM-RSI-01**.

From this directory:

```bash
python audit.py --public-bundle PUBLIC_BRANCH_DIR --pr665-evidence PRIVATE_REPLAY_DIR --output NEW_PRIVATE_DIR
python replay.py --tooling FROZEN_TOOLING --inputs FROZEN_HF_INPUTS --output NEW_EXPERIMENT_DIR --lanes 1 8 14
python reduce_stall.py --template FROZEN_CONFIG --predictor FROZEN_AGENTX_PREDICT --output NEW_REDUCTION_DIR
python analyze_timing.py --root PRIVATE_GPU_CONTROLS --output NEW_TIMING_AUDIT_JSON
python predict_cold.py --template FROZEN_CONFIG --predictor FROZEN_AGENTX_PREDICT --output NEW_COLD_PREDICTIONS
# Remote collector: --suite frontend selects the separately qualified E018 controls.
qualify_weka ORIGINAL_WEKA_TRACE --materialize PRIVATE_LOWERED_JSONL
PYTHONPATH=FROZEN_TOOLING python recorded_arrivals.py --inputs FROZEN_HF_INPUTS --lowered PRIVATE_LOWERED_JSONL --predictor FROZEN_AGENTX_PREDICT --frozen FROZEN_COHORTS --output NEW_DIAGNOSTIC_DIR
python -m pytest -c /dev/null -p no:cacheprovider test_audit.py test_timing.py
uv run --no-project --with 'transformers>=4.57,<5' --with pytest python -m pytest -c /dev/null -p no:cacheprovider test_probe.py
```

The GPU scripts require an immutable model receipt, successful CPU preflight,
explicit remaining GPU-seconds and campaign-root paths. Their synthetic controls
never change the frozen benchmark headline scores. Update this record and the
checkpoint after each meaningful experiment; preserve disproven conclusions.
