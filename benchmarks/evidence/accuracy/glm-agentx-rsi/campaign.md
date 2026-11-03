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
duplicate terminal frames. One-token intervals and exact chat per-token alignment
remain unavailable. Mean visible chat intervals use total output length and do
not require per-frame token counts; legacy probes lack tool-only boundaries, so
these remain separately labeled visible-content diagnostics, not qualified
historical TPOT. Numerical tests cover units, terminal over-generation,
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
The instrumented MTP mode also completes 221/221 with rank-0 coverage for every
request and no counter/clock/coverage qualification failures. Its repeat-1
128-token cold C1 TPOT is 2.6861/2.8256 ms at 8K/131K, close to the uninstrumented
2.6879/2.8277 ms. This does not establish instrumentation neutrality for every
schedule: the overlap victim is 23.7345 ms versus 2.6929 ms uninstrumented, while
standalone is 2.7341 versus 2.7258 ms. Preserve this behavioral difference and
require actual scheduling/forward boundaries; do not assign it to profiler cost
or a silicon scheduler bug from one control. Plain mode remains pending.

**E017 conclusion — complete, qualified diagnostic scope.** Job 1005806 completes
all 663 planned requests, including six unscored preparation requests; both
instrumented modes cover every request at rank 0 with no counter/clock/delivery
failures. Plain repeat-1 C1 TPOT at 8K/131K with 128 outputs is 6.4300/6.7868 ms
versus 5.9790/6.1341 predicted (7.01%/9.62% error); TTFT is 394.536/5454.561 ms
versus 242.361/5040.178 predicted (38.57%/7.60% error). Both speculation modes
retain a similar narrow cold-decode residual, so an MTP-only explanation of that
residual is not established. Correlated acceptance and overlap sensitivity remain
separate findings. Cross-version and synthetic-input limitations still apply.
Neither headline MAPE, coverage nor accepted modeling fixes change.

## H2 — Recipe, tokenization, snapshots and cache history

**Hypothesis.** Effective serving settings or replay history differ from silicon.
Confidence: measured caps are verified and the replay baseline already matches;
broad cache-history attribution remains low confidence.
Competing explanations: correct inputs with incorrect execution or client timing.

**E018 — predeclared paired frontend experiment (pending).** Test whether large
mostly cached prompts incur frontend work missing from native token replay.
Use fresh uninstrumented and instrumented historical servers with the original forced-acceptance
policy, measured overlap-loop setting and all other E017 settings. For each of three
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
Stage receipts capture the actual API process's affinity and thread environment;
collector process settings alone do not establish child-process settings.
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
GPU job **1006430** runs this experiment from immutable source
`9e52b65563b527712cb8428f2f72daf831932474`, with 22 source checksums verified
against a clean Git archive. Charge it to controlled-workload experiments;
independently audit both modes before selecting a cost model or next experiment.

**E018 conclusion — frontend service confirmed on controlled inputs.** Both
modes complete all 36 requests each (18 measured and 18 preparation). All 18
pairs match input/output lengths, input identity and observed prefix reuse.
The original validator expected every seeded raw token to be reusable and
rejected every pair. Preserve that initial audit. The exact image's `RadixKey`
counts EAGLE bigrams (N-1), then page-aligns; executing the archived class yields
8064/130944/393088 reusable tokens from raw seeds 8128/131008/393152, exactly
matching both endpoints. E017 independently confirms 287360 reused tokens with
MTP versus 287424 in plain decode from the same 287424-token seed. Qualification
v2 requires the resolved EAGLE/page64/cache-enabled receipt and this boundary;
missing receipts or other offsets still fail. This corrects a validation
assumption, not measurements or headline scoring.

On the measured Intel Xeon Platinum 8559C, synchronous render/encode takes
19.3–21.6 ms at 8204 input tokens, 268.3–316.5 ms at 131084, and 878.0–981.4 ms
at 393228. Process CPU time closely tracks wall time; numerical projection costs
0.16–4.33 ms and is separate. This accounts for most of the paired chat/native
TTFT gap at long contexts. First-use 8K chat TTFT remains separately visible at
1.50/1.61 seconds despite only 21.6 ms in this stage; do not attribute that entire
delay to tokenization or discard it. Every repeat remains in the evidence.

These synthetic prompts carry about 7.1 serialized bytes per input token, whereas
the original development workloads report about 4.0 transport bytes per token.
The harness labels KB using 1024 bytes; transport counts include headers, while
the control records JSON-body bytes. This descriptive comparison is not a cost
conversion formula. Measure representative Weka inputs before transferring a
per-token cost; do not choose a coefficient from headline MAPE. Both headline
MAPEs remain unchanged. Private evidence includes `E018-complete-audit-v2.json`
and `E018-bigram-cache-source-audit.json`.

**E025 — cache-key contract discrepancy, next discriminating check.** Existing
CPU E006 and silicon E017 use the same prefix-length relationship and one-output
seed, with no pressure. AISim reuses 287424 tokens; SGLang EAGLE reuses 287360,
while plain SGLang reuses 287424. The archived cache source SHA256 is
`150818b0a4eaa05171ca2e485698fa455608c13ee6b07f3945ac2095fa294de3`.
Inspect the native key representation and reproduce the lookahead/page-boundary
case before implementing a scoped fix. Test differing lookahead tokens, partial
pages, token conservation and unchanged ordinary-cache behavior. No blanket
one-page penalty or timing multiplier is an acceptable substitute for key
semantics.

Candidate `9d76c7de1387b3e28767b3ffc67ba1d6449dbc9c` on
`simonec/glm-agentx-eagle-cache` implements page-plus-lookahead lookup hashes,
logical cacheable extents, rollback invalidation and aligned Belady demand.
Ordinary cache keys, raw router hashes and physical page ownership remain
unchanged. Six EAGLE cases pass, including 287424→287360 reuse, changed lookahead,
materialized versus sampled output, admission, rollback and event identity;
all 40 KV-manager cases pass. The earlier full core run passes 2539 cases with
one ignored, before the last three added cases. Pinned adaptation source and
license are recorded with the fix. Focused draft: [#692](https://github.com/ai-dynamo/aisimulate/pull/692).

The candidate wheel SHA256 is
`bce6e6ba7c832751277a4853e4528ee1f266a9f98fe6b1ef58a5d58642534a71`;
native SHA256 `7851f1931a85d8f759ce52b73f7dcf8b4c16622231f88dfcf444e21f5c8d8de8`.
The #665 data SHA256 remains the frozen `287367dc…` value listed below.
Development replay uses the same HF revision, measurement settings and frozen
request identities at lanes 1/8/14. Only cache behavior changes. No extra GPU
time is allocated. Controlled replay confirms 287360 reused tokens, three successful requests and
unchanged input/output counts. The independent development audit retains all
4169 TTFT and 4166 TPOT identities. Pooled TTFT improves 88.4103→85.0238%, but
TPOT worsens 68.2526→73.5559%: **not a win**. Per-lane TTFT/TPOT are
73.2059/18.8691% (1), 62.9161/49.9483% (8), and 99.4730/94.5835% (14).
The C14 TPOT regression is material; C1 TPOT rises 0.00033 percentage points.
No losses or exclusions establish this result. Recovered observations outside
the original cohort remain separate; they do not enter the headline.

Conclusion: the cache-key discrepancy is supported and corrected, but this
causal repair alone does not reduce both request errors. Changed cache work
changes closed-loop overlap; do not relabel a physical correction as an accuracy
win. Preserve the research baseline and the focused draft. Validate this frozen
candidate on 4/10/13 and the Qwen CPU smoke cases before closing its regression
record. Evidence: `E025-reduction-audit.json`, `E025-independent-dev-audit.json`,
`E025-provenance.json`; all 2351 wheel op-data files are byte-identical to the
baseline. No GPU time is charged.

All-six validation is complete with the candidate unchanged. Four frozen TTFT
and four frozen TPOT matches are lost at C10, so full-campaign coverage does
**not qualify**. Retained-subset TTFT/TPOT are 77.0601/77.1057%, with 8620/8610
observations; these cannot establish a win against 8624/8614. Per-lane TTFT/TPOT:
C4 60.5839/42.5346%; C10 65.1852/79.6848% (four lost each); C13
75.6161/92.6798%. C10, C13 and C14 TPOT regressions remain explicit. Evidence:
`E025-independent-all-six-audit.json`, including complete measured accounting.
Qwen finite-campaign smoke cases retain exact numerical request parity at
1/4/16 lanes (8/32/128 requests), using the same approved #666 overlay and
explicit 0.31.0 backend as their baseline. An initial smoke attempt accidentally
let the frozen wrapper resolve latest packaged 0.30.0; its parity assertion
caught the mismatched identity. Keep it as unqualified evidence. The corrected
v2 uses the canonical replay factory and the unchanged explicit 0.31.0 config;
no measured or baseline files are edited. Evidence: `E025-qwen-smokes-v2/audit.json`.

**E026 — frontend corpus qualification.** All six original saved configurations
select `semianalysis_cc_traces_weka_062126` with seed 42 and no corpus override.
The exact pinned 1.0.12 plugin declares `default_prompt_corpus: coding`.
Reconstruct with that generator rather than assuming Shakespeare or transferring
the generic repeated-text CPU price. This is source qualification, not a latency
coefficient or a request-MAPE result. Evidence: `E026-corpus-source-audit.json`.

**E027 — batch-shared forced-acceptance ablation.** Hypothesis: a batch-shared
floor/ceil draw, as implemented in the original forced policy, changes joint
request timing despite matching the independent sampler's mean. Competing
explanations include replay/cache history and frontend buffering. Confidence in
the contract difference is high; its contribution to headline MAPE is unknown.
Change only sampling scope on `eddb21246e0a371fe542f7cab4034b58173bce63`:
acceptance rates `[1, 0.99]`, seed, op data, input corpus, and measurement settings
stay fixed. Do not include E025. Develop on frozen 1/8/14 metric cohorts; lost
matches disqualify wins and recovered matches remain separate. No held-out tuning.

Diagnostic commit `e55917de8771e52da9ae1df38823d6bc12001ea7` is signed/DCO;
wheel SHA256 `1c222fa2b044e328c9734d836ea23e443ce0ae3f8a7cef9e1ec3cddaad373aaa`,
native SHA256 `5779326ab84457948d04a544c6b3a14f63eca94e9c08578cab09eda2b4a584d0`.
All 2351 op-data files remain byte-identical. One test verifies common two/three
outputs across 100 multi-request rounds at a discriminating 50% conditional
acceptance; all 47 speculative tests pass. This matches correlation and marginal
distribution, not Torch's exact RNG stream, and is not a proposed generic default.
Use the existing `replay.py` command with this wheel on 1/8/14, followed by
`audit.py --frozen ... --baseline E003-baseline --candidate E027-batch-acceptance-dev`.
Resource cost is CPU only, zero allocated GPU-seconds. Independent audit retains
all 4169/4166 original TTFT/TPOT observations, with no lost matches. Development
pooled TTFT worsens 88.4103→91.8322% and TPOT worsens 68.2526→104.8737%.
Per-lane TTFT/TPOT: C1 73.6772/18.8789%, C8 64.9665/53.3170%, C14
109.8970/146.3360%. **Not a win; disprove this correction as a sufficient
remedy.** It changes joint timing materially but does not establish that
independent acceptance caused the headline gap. Keep the ordinary default and
research baseline. No held-out tuning or validation is needed for this rejected
diagnostic. Evidence: `E027-independent-dev-audit.json`, `E027-provenance.json`.

**E028 — isolate frontend blocking from GPU prefill.** Hypothesis: synchronous
chat rendering/tokenization prevents the API process from forwarding already-ready
GPU outputs, producing long TTFT and short client TPOT. Competing explanations
include GPU contention, transport buffering independent of API CPU work, and
client-side parsing. Evidence for synchronous CPU work is strong (E018), but
historical request attribution is unproven.

Use identical cold native requests (nominal 8K or 131K, output 16) with no, short
8K, or long 393K CPU-only `/v1/tokenize` work. Trigger the actor after a retained
scheduler-submission receipt, not the victim's first client output. Three repeats
per condition, alternating actor order. Run the tokenization client in a separate
process so response JSON parsing cannot hold the native client's Python lock.
Retain dispatch, CPU-stage, rank-zero output-ready and client-receipt timestamps;
no prompt/token arrays enter numerical evidence. Long GPU prefill and short CPU
work provide timing controls. Compare dispatch-only and detailed-instrumentation
arms on fresh servers with the original model/image/forced acceptance policy.

Predeclare 18 GPU victims plus 12 CPU-only actors and two unscored CPU warmups
per arm, with all attempts retained. Cache reuse must be zero; request lengths,
output counters, source hashes, clocks and separate client processes must qualify.
CPU-stage and submission receipts must also identify the same API process.
An exposed request must become ready while the CPU stage is active and reach the
client after it ends before attributing delay to frontend blocking. Missing
exposure is inconclusive. No historical MAPE claim or cost coefficient follows
from this synthetic control. Reuse the existing collector and stop on incomplete
qualification. Maximum reservation remains 4500 allocation seconds plus 60 for
cleanup (36480 GPU-seconds), admitted only against the reconciled ledger.
The exact-image CPU preflight checks the archived tokenizer-manager SHA256,
patch syntax, and separate actor startup/shutdown without GPU access. GPU
admission checks that receipt against runtime version and all five relevant
source files; changed hooks or clients require a fresh CPU preflight. Forty-one
focused tests pass, including exposure clocks, missing/ambiguous observations,
identity failures and startup-failure accounting. Numerical audits predeclare
all 64 attempted-request identities across both arms and never score preparation.
CPU job 1008523 completes successfully in 240 seconds with zero GPUs allocated;
the external cleanup step lasts 241 seconds, also CPU-only. It verifies the
historical source SHA256 `952fdfcf…`, patched SHA256 `7f065b67…`, runtime 0.5.20,
and separate actor lifecycle. Collector-side source checks must match that
receipt. The GPU batch also rejects anything other than eight B300 devices
before loading weights. Collection source, results and actual allocation cost
are pending; independent audit additionally requires a shared API process.

**E028 conclusion — synthetic blocking supported; historical attribution open.**
Retry 1009781 freezes collector `d66bcdad46cc6b67735803c958453fbe6b6d1a10`
and completes all **64/64 requests** on pool0-0107 (B300 TP8, Xeon 8559C,
64 logical CPUs available). Both fresh-server arms retain the original image,
model, settings and forced acceptance. There are 36 GPU victims, 24 CPU actors
and four unscored CPU warmups. Every client request succeeds, with zero missing
or duplicate identities. Native inputs agree and cold-cache receipts are zero.

All three instrumented small-GPU/long-CPU cases independently satisfy the
predeclared exposure rule: server outputs become ready during a synchronous CPU
stage in the same API process, and client delivery starts after that stage ends.

| Repeat | CPU stage ms | Server-ready interval ms/token | Client interval ms/token |
|---:|---:|---:|---:|
| 0 | 912.647 | 2.6038 | 0.06493 |
| 1 | 975.104 | 2.8507 | 0.06143 |
| 2 | 883.247 | 2.6168 | 0.05881 |

The dispatch-only arm reproduces small client intervals of 0.05263/0.07486/0.07559
ms/token with the long actor. Short actors finish before small-GPU outputs are
ready, and all six instrumented large-GPU/actor controls finish CPU work before
the first GPU output. Their client intervals remain about 2.6–2.7 ms/token.
This distinguishes API blocking from a GPU computation slowdown in these controls.
The `/v1/tokenize` actor also serializes a large token-array response: client
receipt begins about 196–199 ms after the measured CPU stage ends. Do not assign
that extra cost to ordinary chat or treat this method-only timer as full HTTP cost.

**Qualification limitation retained:** the original gate qualifies 18/18
unprofiled and **17/18 instrumented victims**. One no-actor repeat reports 18
internal tokens but delivers the requested 16. Original runtime source methods
`update_finish_state` and `output_ids_through_stop`, executed with controlled
synthetic IDs, verify that a speculative burst can cross the cap and trim two
tokens. The hook records untrimmed `output_ids`; the streamer emits the trimmed
view. Keep the original failed gate, since this run did not retain `finished_len`.
No exposed case has this discrepancy. Future instrumentation must distinguish
committed, emitted and discarded-tail counters rather than relaxing conservation.

Independent receipts: `E028-complete-audit-1009781.json`,
`E028-independent-audit-1009781.json`, `E028-counter-source-audit.json` and
`E028-loader-source-audit.json`. No historical request MAPE or baseline changes.
The next discriminating experiment measures actual coding-chat CPU service, not
a fitted latency multiplier; historical CPU/trajectory applicability remains open.

**E029 — coding-chat primitive preparation (CPU only).** Hypothesis: the
single-message generic controls do not establish the CPU cost of the measured
coding workload. Reconstruct the first frozen Weka play with harness 1.0.12,
seed 42, its full default coding pool, original saved configuration and pinned
GLM tokenizer. Reuse the harness session accumulator and chat endpoint, including
context resets; do not concatenate delta messages by hand. Select root turns
nearest nominal 8K/131K/393K source lengths before any timing or scoring.

Two clean reconstructions and the separate exploratory preparation produce
identical private payload/token hashes. Actual selected chat lengths are
**651 / 134720 / 393290**, with **1 / 38 / 219 messages**. The first play has no
near-8K request; retain the actual 651-token case rather than relabeling it 8K.
The default coding pool contains 261056 tokens with this exact tokenizer; the
pool's scale constant is not a guarantee of its realized token count. The
134720-token case agrees exactly with source length; the long case differs by
202 tokens from its 393088-token source target. Keep these differences explicit.

**Conclusion — preparation qualified, timing transfer still unqualified.** These
are reproducible frontend primitives, not measured historical request histories:
the original controller selections/cache trajectory and original CPU identity
remain separate qualifications. No latency coefficient, MAPE improvement or
historical measurement correction follows. Preparation uses no GPU time. Private
receipts `E029-prepared-v1/receipt.json`, `E029-prepared-v2/receipt.json` and
`E029-independent-audit.json` pin source/config/model assets and payload checksums;
payload text and token arrays stay private. Next use these primitives for
independently measured CPU-stage controls after E028 qualifies receipt blocking.

**E030 — coding frontend component cost (predeclared, CPU only).** Measure
E029's three private primitives with the exact historical
`_render_and_encode_chat_template` method, extracted unchanged from the pinned
image and checksum-checked before execution. Use model/tokenizer asset checksums,
Xeon 8559C and 64-CPU process affinity, with CUDA hidden. GPU-host policy rejects
zero-GPU normal-QOS requests and CPU QOS on this partition before allocation.
The 24-CPU-per-GPU admission rule requires three reserved B300s to preserve
64-CPU affinity. Reserve three under normal QOS for at most ten minutes, with a one-minute
cleanup margin: charge the actual three allocated GPUs and time against the campaign
(maximum 1980 GPU-seconds).
The component uses no CUDA kernels or model weights. Keep the rejected
admission checks as zero-allocation operational evidence.
Seven repeats per primitive alternate case order; retain every first-use sample.
Require byte-identical token identities against the prepared receipts and capture
wall, process-CPU and thread-CPU times. Preparation/JSON parsing/HTTP and GPU
execution are outside this component boundary. Do not transfer the result to
historical hardware or fit a coefficient to MAPE. The bounded ten-minute CPU job
and original source/configuration checksums accompany the results. Qualification
failure retains its observations and blocks cost-model use.

**E030 conclusion — component qualified; historical applicability incomplete.**
Job 1009935 finishes in 74 seconds at three reserved GPUs: charge 222 GPU-seconds,
including startup/cleanup, although CUDA remains hidden. All 21 measurements
qualify with exact input identities and independent arithmetic checks. Repeat
medians (six repeats following the retained first-use sample) are 1.1115 ms at 651
tokens, 168.3717 ms at 134720, and 539.0198 ms at 393290. Long-case thread CPU/wall
ratios are about 0.999; no multi-core throughput factor is assumed. Keep all seven
samples per case, including the 34.843 ms first use of the small case. These
coding-content costs are lower than generic prose controls; neither corpus
supplies a calibrated historical latency multiplier. Evidence:
`E030-independent-audit.json` and checksum-pinned component observations.

**E031 — original host provenance.** Original C14 job 119061192112 records Xeon
8559C/192 logical CPUs and launcher affinity 192 [0–191]. Thus CPU model agrees
with our controls, but the recorded launcher affinity differs from 64. A server
process affinity receipt is not separately available. Retain original job-log
checksum and exact resource-summary lines in `E031-historical-host-audit.json`;
verify all six source jobs before generalizing the host identity.

**E032 — affinity sensitivity (predeclared).** Run the same three primitives,
exact image/model assets and component method in fresh processes on one host:
192 CPUs, then a 64-CPU subset, then 192 again. Each arm retains seven samples per
case, 63 observations total. Require actual affinity, source and every input hash
to match its predeclared arm. Use CUDA-hidden processes with a policy-required
TP8/192-CPU reservation, capped at 600 seconds plus 60 seconds cleanup: at most
5280 GPU-seconds. Do not sum child processes as separate allocations. First-use
and repeat measurements remain separate; inspect the before/after 192 arms for
drift before attributing a difference to affinity. No historical MAPE claim or
HTTP cost follows from component qualification alone.

**E032 conclusion — affinity alone does not justify a cost multiplier.**
Job 1010000 completes all 63 observations in 100 allocated seconds at eight GPUs;
charge 800 GPU-seconds. Every token identity and requested affinity qualifies.
Repeat medians for 134720/393290 tokens are 164.9121/518.7693 ms in the first 192
arm, 164.1694/497.9141 ms at 64 CPUs, and 159.6700/500.1401 ms in the final 192 arm.
The 64-CPU result overlaps the drift between the 192-CPU arms; no blanket CPU-count
factor is established. Retain first-use samples and all arm/mask identities.
Evidence: `E032-independent-audit.json`. Host/source/input qualification is scoped
to this component, not original full HTTP service timing.

**E031 expanded provenance.** All six original job logs confirm Xeon 8559C,
192 logical CPUs and launcher affinity 192 [0–191], retained in
`E031-historical-host-audit-v2.json`. All six also explicitly declare, but fail to
verify at runtime, model revision `e3b8f9f2fad2` at `/model`. Our downloaded full
revision and 52 checksum-pinned weight/config files are verified independently;
they do not establish bit identity with the original serving files. Preserve
this qualification limit in `E031-historical-model-provenance.json`; the original
silicon scores remain unchanged.

**E033 — scenario mutation and measured input counts.** A checksum-identical
original C1 artifact (SHA256 `d58deb00ebc1e6305f8e4390064dcbb9bb13a4f6bb73f4829bfe3fb9f03cdd2b`)
contains the client log confirming scenario `agentx` auto-enables
`first_turn_prefix` cache busting. Directly loading the saved BenchmarkConfig
returns default `none` because it does not execute the scenario validator. The
pinned harness scenario requires the marker, minted per trajectory tree and
shared across its warmup/profile phases and descendants. Do not infer the
measured state from the authored/default-only configuration.

The unmarked E029 long primitive contains 393290 tokens. Using the original
benchmark IDs, saved initial lane 0/recycle 0 identity and the harness marker
builder/injector restores **393303 at C1 and 393302 at C8**, exactly matching their
recorded server usage at source outer 133. This confirms the selected count
reconciliation; byte identity and full cache history remain separate checks.
The E029/030/032 primitives deliberately omit worker markers and remain component
diagnostics. Their receipts now make that omission explicit for future runs;
old payloads and measurements stay immutable. Extend the audit over every
available first-play root request, retaining missing server-usage records
explicitly. The initial audit helper fails on such a record rather than
producing a result; retain its script/log and failure receipt. No MAPE correction
or modeling win is claimed from token-count agreement.

**E033 expanded count audit.** For the first frozen play's root profiling
requests at development lanes 1/8, retain 72 records: all 71 with server input usage
reconcile exactly after applying the actual scenario marker; one C1 record has
no server usage and stays unavailable. No valid count mismatches are discarded.
The audit uses the original worker's idempotent injection across context resets,
not a fixed token addition. Keep the failed first helper and its missing-field
receipt. Evidence: `E033-marker-all-first-play-audit-v2.json` and its script/log.
This does not establish byte-identical historical payloads or a cache-trajectory
repair, and leaves all headline metrics unchanged.

**E034 — measured CPU admission only (predeclared diagnostic).** Enable the
existing canonical tokenization stage on the frozen research wheel, changing no
scheduler/cache/op data or replay inputs. Freeze rate **1.3222043548296005 microseconds
per input token** before replay: 1000 times total measured component milliseconds
divided by total actual input tokens across all 42 full 192 before/after samples,
including first use. One CPU server, zero fixed cost. The latter is a conservative
diagnostic assumption, not measured zero HTTP/render overhead. This linear
amortization does not describe every message shape or cold cost. No MAPE values
enter its derivation. Profile SHA256:
`741c9b07352c57c267d0890403a0950435a048e32926c77456018d61025c8158`.

Predeclare all 4169 TTFT and 4166 TPOT frozen development observations, lanes 1/8/14;
report any lost/recovered observations and reject a win on a smaller cohort.
Verify native tokenized timestamps actually include at least each request's
component service and native completion qualifies. Keep closed-loop replay and
observed output lengths. API response buffering, JSON/HTTP and literal-marker
input/cache repairs remain outside this admission-only change. This is a scoped
causal diagnostic, not an accepted full serving model or historical measurement
correction. Run untuned lanes 4/10/13 only if the development result warrants
validation, preserving the same frozen rate and profile provenance.

**E034 conclusion — rejected as a sufficient remedy.** Native CPU service
and completion checks pass. The fixed admission cost reduces TTFT error among
retained development requests, but loses five frozen observations per metric
(four at C8, one at C14) and worsens pooled TPOT. The retained-subset MAPEs are
56.6126% TTFT and 71.7450% TPOT, versus the full frozen baseline 88.4103%/68.2526%.
Different denominators prohibit treating the TTFT change as a qualified win.
All 4169/4166 original identities remain in accounting; 4164/4161 retain scores.

| Lanes | TTFT MAPE, retained | TPOT MAPE, retained | Lost TTFT/TPOT |
|---:|---:|---:|---:|
| 1 | 25.0093% | 18.9029% | 0/0 |
| 8 | 28.8601% | 55.3671% | 4/4 |
| 14 | 76.9495% | 88.3007% | 1/1 |

Median TPOT APE decreases 18.9489→17.8600%, but the small-reference contribution
rises 28.2550→32.0316 percentage points and tails worsen. Preserve them. No rate
retuning, held-out replay, baseline replacement or accepted modeling fix follows.
Evidence: `E034-independent-dev-audit.json`, source 31adb4822 and the unchanged
research wheel/data. Source inspection of historical `serving_base.handle_request`
also confirms synchronous conversion on its asynchronous path; the admission-only
stage does not represent output callbacks sharing that API event loop.

**E035 — retained native matrix (predeclared CPU experiment).** Hypotheses:
GPU scheduling/forward gaps may remain after CPU input work is separated; competing
explanations are arrival-boundary and instrumentation effects. Reuse GPU 1005806,
not new collection. Replay cold native-input controls at 8/14 lanes, 8192/131072
input tokens and 128 outputs, both repeats, in all three original arms: 24 finite
cases / 264 request pairs. Preserve the plain/MTP factor, original forced mean 2.99
(1.99 accepted drafts), cap 256, chunk 8192, TP8 and exact input/output accounting.
Distinct first pages prevent prefix reuse across lanes as in the Task-i inputs.

Use each retained client's start times relative to its batch origin, not invented
simultaneous arrivals. These are client starts, not scheduler-submission clocks;
JSON/transport latency remains a limitation. Compare CPU predictions with both
client and rank-zero output-ready spans where recorded. The latter includes
scheduling stalls, not pure kernel time. End spans at first attainment of the
delivered 128-token cap and retain generated excess explicitly, consistent with
the source-verified trimming contract. Do not rewrite the E028 original gate.
Keep repeats/arms separate before assessing the instrumentation effect. This
cross-version 0.5.20-silicon/0.5.21-data diagnostic does not score AgentX headlines
or justify a global op multiplier. Retain failures and all planned pair identities.

**E002/E004 — prior cap conclusion superseded after identity audit.** Runtime
receipts establish caps 2/8/256 at lanes 1/4/higher. The earlier record incorrectly
claimed replay used 256 at C1 and a cap-only correction had no effect. Both E003
and E004 saved inputs actually use 2, and their checksums match their original
execution receipts. The public dashboard also reports 2. E004 did not change
the claimed variable, so its earlier causal conclusion is invalid. Preserve both
runs and the correction artifact rather than treating them as an ablation.

**E004b — actual counterfactual.** Change only C1 from its correctly aligned cap
2 to 256, with the frozen research wheel and silicon. All 324 matched requests
remain. TTFT MAPE changes 73.6567% to 71.0511%; TPOT changes 18.8688% to 19.2617%.
**Conclusion:** cap sensitivity is supported; a baseline cap mismatch is
disproven by the saved inputs. Do not adopt the mismatching cap for a lower TTFT
score, and do not claim a win while TPOT worsens. Frozen baseline scores remain
unchanged. Private receipts: `E004-config-identity-correction.json` and
`E004b-cap256/`.

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

**E022 — acceptance correlation audit, supported source/runtime difference.**
The exact historical `sglang/srt/speculative/spec_utils.py` (SHA256
`f7b6db9995357dd99a221b763a642fda1a26b5ec21bc33cb7e350f2a5bcd6343`)
samples one floor/ceil total acceptance value per verify batch, then fills every
request with that value. Current AISim samples the same marginal distribution
independently per running request. In E017's instrumented MTP counters, 394
multi-request output-ready groups comprise 390 uniformly three-token groups and
four uniformly two-token groups; no mixed increment group is observed. Terminal
over-generation remains counted. Both means are correct at 2.99; correlation
differs. **Conclusion:** batch-shared forced acceptance is confirmed for this
measurement policy; its contribution to MAPE is unmeasured. Queue a one-factor,
explicitly scoped batch-shared research ablation after the frontend experiment.
Natural acceptance must retain ordinary per-request behavior. Do not change the
general sampler default or claim that the historical mean was wrong.

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

**E023 — overlap-loop configuration audit and predeclared replay (pending).**
All original runtime receipts and E017's resolved serving configuration report
`disable_overlap_schedule=false`. The exact image sets `enable_overlap` from
that flag and selects `event_loop_overlap`, which overlaps host processing with
GPU work. An earlier description of these controls as synchronous was incorrect;
the collector already preserves the measured setting. The frozen AISim worker
leaves its optional `host_loop` model unset. Enable only this existing model on
development lanes 1/8/14, retaining the same wheel, data, cache settings and frozen
request cohorts. Predicted observable: different inbox/result visibility and
interference under concurrent arrivals; a free host loop preserves legacy token
times. Competing explanations include upstream frontend delays and op timing.
Require native completion, unchanged silicon identities, both MAPEs and lost/
recovered counts before judging the result. A flag difference alone does not
establish an accuracy fix; do not tune on lanes 4/10/13.

**E023 conclusion — no qualified win.** All frozen development observations
remain (4169 TTFT / 4166 TPOT, zero lost). Pooled TTFT MAPE falls 88.4103% to
83.6124%, but TPOT rises 68.2526% to 79.7859%. Keep the frozen baseline.

| Lanes | Baseline TTFT | Host-loop TTFT | Baseline TPOT | Host-loop TPOT |
|---:|---:|---:|---:|---:|
| 1 | 73.6567% | 73.8348% | 18.8688% | 18.8804% |
| 8 | 63.7743% | 64.0876% | 51.0948% | 56.9255% |
| 14 | 104.7231% | 96.2855% | 84.8208% | 101.2084% |

The loop flag alone is insufficient to resolve the request gaps; do not generalize
from its closer runtime configuration to better accuracy. Native completion and
frozen-cohort arithmetic independently pass. Held-out lanes remain untested for
this candidate because development evidence rejects it as a standalone win.

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

## H7 — Arrival clocks and request ordering

**Hypothesis.** Client-start timestamps recorded before body serialization do not
establish the GPU scheduler's arrival order. With cold concurrent prefills, a
request's position determines how much later prefill work stalls its decode.
Confidence is high in the boundary difference, moderate in its contribution to
paired errors. Competing explanations include actual scheduler ordering,
framework-version differences, graph timing and forced-acceptance correlation.

**E035 conclusion — modest batch-mean gaps conceal large request gaps.** All 24
finite cases / 264 requests complete and qualify cold/input/output accounting,
with no missing pairs. For MTP, batch-mean TPOT errors are about +13.6%/+8.0% at 8K
and −5.2%/−5.6% at 131K (8/14 lanes). Plain decode shows the same sign reversal,
not a universal MTP cost defect. Retain both repeats and all arms; these mean
comparisons do not replace request-level objectives.

In instrumented repeat 0 at 131K/8 lanes, mean TPOT is about 5% apart while matched
request TPOT MAPE is 512.7% against client receipt and 514.2% against output-ready
spans. All eight first-output positions differ; average rank displacement is 2.5.
Other cases also differ in ordering. The old client starts its clock before
`call` serializes the large request. Exact server-submission observations are
absent, so ordering causality remains unproven. Do not re-pair requests to improve
the score or apply a global timing multiplier. Evidence:
`E035-independent-audit.json`, 24 native case receipts and original GPU 1005806.

**E036 — prepared bodies and server submission (predeclared).** Reuse the
verified model/image/recipe, cold 131072-token native requests and 128 outputs at
eight lanes. Prepare all encoded bodies before releasing a batch barrier. Capture
client start, actual API submission, rank-zero eligible output-ready and client
receipt on shared monotonic clocks. Three repeats per arm; compare a minimal
dispatch-only arm with dispatch+output instrumentation on fresh servers. Retain
all 48 request identities, cached-token receipts, input/body hashes and numerical
stream counters. No prompt text, token arrays or raw bodies enter observations.

The v2 output probe separately records generated, eligible and discarded-tail
counts plus the actual finished-length boundary; preserve all old artifacts and
the E028 failed gate. Exact-image CPU preflight qualifies the prepared byte path,
original dispatch source and all ten relevant Python files. Forty-seven focused
tests pass, covering preflight drift rejection, no serialization inside the
prepared request, and generated 18/emitted 16/discarded 2 conservation. A changed
collector requires a new CPU receipt before GPU admission.

Bound GPU allocation to 45 minutes plus 60 seconds cleanup: reserve 22080 GPU-seconds.
The batch verifies Slurm's actual 45-minute limit, eight B300 devices, image and
preflight before model loading. Against the reconciled 146994 remaining
GPU-seconds, this leaves 124914 unreserved and fits the controlled-workload pool
(3466.75 used + 2760 reserved < 7200 TP8-equivalent node-seconds). CPU preflight
allocates no GPUs; startup/failure/cleanup all count when GPU allocated.

Predeclare two subsequent CPU replays per retained batch, changing only the
arrival clock: client-start versus recorded server-submission, with the same
frozen wheel/op data and paired request IDs. Compare first-output order and both
request metrics; keep batch means as diagnostics. These are finite native
controls, not an oracle fix for historical AgentX cohorts. Support the hypothesis
only if submission alignment improves ordering/error with qualified clocks and
all pairs retained; otherwise investigate scheduler/perf-model semantics.

E036 CPU preflight 1011181 completes in 296 seconds with zero GPUs allocated,
verifying the prepared-byte transport and ten source hashes against collector
`7ec97a9147d4b25bfd9c4bcf97c667865abe4472`. The original dispatch-manager source
SHA256 remains `952fdfcf…`; preserve the complete receipt. GPU job 1011234 uses
that frozen collector with its 45-minute Slurm limit and full 22080 GPU-second
reservation. Replace reservation with actual allocation/cleanup cost at termination.

The comparison helper keeps client-start→first-client-output TTFT as the metric
boundary for both replay clocks. For submission arrivals, derive predicted client
TTFT from the model's absolute first-output time and original client start;
do not compare its shorter native submission-relative TTFT directly. Two tests
verify that boundary and explicit generated/emitted/tail conservation. New matrix
observations require v2 counter receipts; old generated-only observations stay
readable in the original control mode. Generated count means retained accepted
`output_ids`, not all physically processed draft/verification tokens. No kernel
counter or output-throughput boundary is redefined.

**E036 attempt 1 conclusion — operational failure, no accuracy result.** Job
1011234 terminated after 410 seconds with eight allocated B300 GPUs: 3280
GPU-seconds charged, including startup and cleanup. All eight attempted requests
completed with 128 outputs, 131072 inputs and zero cached tokens; all eight lack
mandatory submission receipts, and the remaining 40 were not attempted. Preserve
the observations and failed qualification. The dispatch hook accepted only old
`rsi-block-` names and silently skipped new `rsi-submit-` names. Do not reconstruct
missing submission timestamps from client latency.

The retry changes only that name filter and qualification: exercise both real
receipt paths during CPU preflight and require a new versioned receipt matching
all ten collector sources. Tests cover both prefixes and source drift. The
reconciled total is 86686 GPU-seconds (24.079444 GPU-hours), leaving 143714
GPU-seconds (39.920556 GPU-hours), with no active reservation. A fresh 22080
GPU-second reservation fits the controlled-workload pool: 3876.75 used + 2760
reserved < 7200 TP8-equivalent node-seconds. Failed attempt evidence remains
separate from the new immutable source and receipt.

E036 retry CPU preflight **1012161** completes in 244 seconds with zero GPUs.
Receipt SHA256 `60ddf190be6f87a4bb8e655a4abb700365458d7a5ebd43f17b867d34fa97e07f`
qualifies both actual dispatch name prefixes, the prepared-byte transport and ten
source hashes against signed collector `a8a3363120b85d24ddae853ad24ec82fc433f87d`.
The versioned receipt is supplied explicitly to GPU job **1012931**, with requeue
disabled and a 45-minute limit. Hold 22080 GPU-seconds including cleanup, leaving
121634 unreserved; reconcile actual allocation when terminal. All 51 focused
collector/auditor/clock/counter tests pass. No accuracy result is accepted yet.
The separate matrix auditor derives request intervals and native TPOT directly
from first/last boundaries, verifies delivered/generated/tail counts, original
pair identities, cold-cache receipts and both arrival clocks. It recomputes
pooled equal-per-request errors without importing the prediction helper's scoring
functions. Undefined positive-reference errors reject control qualification;
no requests are silently dropped.

E036 GPU retry 1012931 completes all 48 planned requests in 881 allocated
seconds at eight GPUs, exit zero: charge 7048 GPU-seconds. Including the first
failed attempt, E036 costs 10328 GPU-seconds. Total campaign charge is 93734
GPU-seconds / 26.037222 GPU-hours, leaving 136666 GPU-seconds / 37.962778
GPU-hours, with no active GPU reservation. All 73 retained GPU artifacts have
checksum/byte-count entries in `E036-GPU-1012931-manifest.json`.

The first submission-clock CPU replay fails its arrival-identity guard. Native
trace loading shifts the earliest arrival to zero, removing the shared 40.493977
ms origin offset in repeat zero. Preserve that failed replay. The corrected
helper explicitly uses the earliest selected clock as the native origin, retains
all inter-arrival deltas, and translates predicted first-output times back to
client starts for TTFT. A regression test verifies identical client TTFT under
the shifted origin. Regenerate both comparisons in new directories; this is a
coordinate correction, not timing calibration or a prediction-model change.

**E036 conclusion — arrival ordering supported in finite native controls.** All
48 measured requests and all 48 predictions per clock qualify with zero lost
pairs. The independent auditor verifies 24 rank-zero output-ready identities,
explicit generated/emitted/discarded-tail conservation, cold caches, exact input
and output counts, prepared-body clocks, native completion and interval arithmetic.
The frozen native binary remains SHA256 `07b39a7f…`; no wheel, op data, acceptance
setting, request pairing or timing multiplier changes between conditions.

| Recorded arrival clock | Client TTFT MAPE / n | Client TPOT MAPE / n | Server-ready TPOT MAPE / n |
|---|---:|---:|---:|
| Client start | 66.3826% / 48 | 145.3488% / 48 | 87.4279% / 24 |
| API submission | 8.6952% / 48 | 7.3182% / 48 | 7.1682% / 24 |

Median client TTFT/TPOT APE improves from 39.4660%/53.6638% to
8.0907%/5.8454%; absolute errors improve from 11027.97/83.5809 ms to
1994.39/8.8422 ms. Signed errors become −8.6952%/−1.8016%. All six batches
have both request MAPEs below 10%, with all eight first-output positions matching
under API arrivals. Client-start replay mismatches six or seven positions in each
batch. Keep the matched request identities; no rank-based re-pairing occurs.

| Arm / repeat | Client-clock TTFT / TPOT | Submission-clock TTFT / TPOT |
|---|---:|---:|
| Dispatch-only / 0 | 49.623% / 54.127% | 9.297% / 8.322% |
| Dispatch-only / 1 | 46.094% / 280.541% | 8.624% / 6.865% |
| Dispatch-only / 2 | 45.664% / 275.051% | 8.471% / 7.050% |
| Output-instrumented / 0 | 66.541% / 74.199% | 8.804% / 7.439% |
| Output-instrumented / 1 | 76.169% / 71.516% | 8.453% / 7.228% |
| Output-instrumented / 2 | 114.205% / 116.658% | 8.522% / 7.006% |

The instrumented and dispatch-only submission-clock results have similar ranges;
retain their individual repeats rather than interpreting small differences as a
profiling correction. Client-clock v2 reproduces all 48 original client-clock
pairs exactly. The failed submission-origin replay remains separate evidence.

This is a diagnostic arrival-boundary result, **not a six-point AgentX accuracy
win** or a general model repair. Historical API-submission times are unavailable;
E014's recorded client arrivals do not supply them. The actual measured 0.5.20
versus database 0.5.21 comparison remains explicitly cross-version. Investigate
ordinary coding-chat admission and shared API delivery next, using real resolved
inputs and runtime observations rather than fitting historical errors.

Evidence: `E036-independent-audit.json`, SHA256
`983eb593f745a19cb85a11ad5ac5b6905270fbae79ec0ce63990a390911a7c13`; GPU manifest SHA256
`072a50b0a894343f2d64abcf0a0ceefe7a5deba993904475936a99754fcabf1c`,
original CPU qualification, Slurm accounting, both complete v2 replay bundles,
first-arm order audit and retained failed attempts. No held-out lanes, HF dataset,
public dashboard or default serving model changes.

## E037 — ordinary coding-chat admission and delivery (predeclared)

**Hypothesis (H1/H6).** The API buffering demonstrated with `/tokenize` also
occurs during ordinary coding-chat rendering, before the actor's own GPU work.
Predict that a short native victim finishes all server-ready outputs inside a
long coding-chat CPU stage, while client delivery begins after the stage ends.
Confidence: moderate; competing causes include actor GPU prefill interference,
response formatting and instrumentation. Only the strict enclosed-ready interval
can isolate CPU delivery blocking from the actor's later GPU work.

**Experiment.** Reuse E028's dispatch-triggered independent actor process and
E036's generated/emitted/tail probe, pinned checkpoint/image/forced acceptance
and unchanged server recipe. Replace the tokenize actor with real E029 coding
chat inputs: 651 and 393290 tokens (1/219 messages), checksum-pinned and verified
by the actual model tokenizer. These are **unmarked primitives**, not a replay
of historical scenario/cache/controller histories. Both actor and native victim
request 16 outputs; all requests start with reported cold cache.

Three repeats alternate none/short/long interference on an 8K-class native victim.
Two fresh arms: dispatch-only, then dispatch plus output-ready and frontend CPU
receipts. Plan **34 requests**: 18 native victims, 12 profile chat actors and four
unscored chat warmups. Flush cache before each warmup and case. Preserve all
attempts, incomplete requests and unmatched/ambiguous stages. Chat counts alone
do not prove token hashes; match actual instrumented frontend receipts to pinned
expected hashes and the native submission's API PID. Record that the actor also
executes GPU work. Do not assign `/tokenize` response-array cost to chat.

Qualify source hashes, both dispatch prefixes and both exact coding token
sequences in a fresh zero-GPU exact-image CPU preflight. Reserve at most
**39 minutes plus 60 seconds cleanup = 19200 GPU-seconds**, using 64 CPUs and
1 TiB memory on one eight-B300 node. The controlled pool has 4757.75/7200
TP8-equivalent node-seconds charged; 2400 reserved fits its remaining 2442.25.
Campaign remaining is 136666 GPU-seconds before reservation. Instrumented runs
must retain actual phase/counter/CPU boundaries and be compared with the minimal
arm. Native victim intervals are diagnostics; frozen six-point headline scores
remain unchanged. Stop and retain evidence on qualification failure.

CPU preflight 1013128 fails after 238 CPU-only seconds, allocating zero GPUs.
The original E029 token checksum covers compact JSON; the new loader incorrectly
compared it with the runtime's little-endian uint32 checksum. Preserve the failed
preflight. The repair separately verifies the original JSON checksum and records
the uint32-le hash for frontend-stage matching, with an explicit encoding label.
A regression test loads a checksum-pinned miniature primitive and verifies both
distinct hashes. Require a fresh source archive and CPU receipt before GPU use.

Repaired CPU preflight **1013188** completes in 246 seconds with zero GPUs.
Versioned receipt SHA256
`96b44e65d73ed1e0c489792b170e063b250f858f055629f9455e467a13221a4c`
qualifies both exact token sequences, both checksum encodings, both dispatch
prefixes and ten source hashes against collector
`c250a9a98d1022b5723105ff7133adafa15ff423`. GPU job **1013236** uses that immutable
archive, a 39-minute wall limit, no requeue and the full 19200 GPU-second
reservation. Charged cost remains 93734 GPU-seconds; 117466 remain unreserved
until actual allocation replaces this reservation. Preserve both CPU attempts.

E037 GPU attempt 1013236 stops at its first warmup after 354 allocated seconds
at eight B300 GPUs: charge 2832 GPU-seconds. The single attempted chat completes
with 651 inputs and 16 outputs but fails the cold-cache gate because projected
cache usage is unavailable. Preserve 1/34 attempted and 33 unattempted requests;
no profile victim runs and no mechanism conclusion follows.

Pinned `UsageProcessor` source SHA256
`883c4c3ba95eb52fa57d5bf732d128063bac0b032f484776331220f61b7498a5`
returns absent cache details for zero, including when reporting is enabled. The
saved resolved server configuration has `enable_cache_report=true`. Do not
rewrite the original failed gate. New collection may interpret this as zero
only after CPU qualification exercises the actual serializer at zero/64 cached
tokens, checks its source hash, and the live client verifies reporting enabled.
All other missing cache usage stays explicitly unavailable. Preserve the exact
interpretation as metadata; test explicit, omitted, missing and malformed cases.

Total charge is 96566 GPU-seconds / 26.823889 GPU-hours; remaining 133834
GPU-seconds / 37.176111 GPU-hours. No GPU reservation is active. Controlled
workloads use 5111.75/7200 TP8-equivalent node-seconds. The retry reduces its wall
limit to 33 minutes plus 60 seconds cleanup (16320 GPU-seconds), fitting the
remaining 2088.25 node-seconds; require a fresh immutable source and CPU receipt.

Actual-serializer CPU preflight **1013377** passes in 245 CPU-only seconds.
Receipt SHA256
`dcf26114b6b94811acd5d4a89c3082fba44ffc5613b013ec15449f4a0a2c0541`
verifies zero/64 cache counts against the real pinned serializer, both coding
sequences and all ten source hashes. Source
`6b65b5723550c62563c05d453d699d36d9566084` passes 68 focused tests; GPU retry
**1013407** uses its immutable archive and a 33-minute wall limit without requeue.
Reserve 16320 GPU-seconds including cleanup, leaving 117514 unreserved. The
client must additionally check the live resolved reporting flag before the first
warmup; otherwise it rejects collection rather than inferring cache state.

GPU retry 1013407 completes all 34 planned requests in 1108 allocated seconds,
exit 0, charging 8864 GPU-seconds. Total E037 GPU cost is 11696 GPU-seconds including
its first failed attempt. Campaign charge is 105430 GPU-seconds / 29.286111
GPU-hours; remaining 124970 GPU-seconds / 34.713889 GPU-hours. Release the GPU
reservation. Controlled workloads use 6219.75/7200 TP8-equivalent node-seconds.
Startup stack probes run inside the same allocation and are not charged again;
their retained timestamps precede the first profile request by approximately 38
seconds. No startup deadlock is established: later stacks show request reception.

The main audit retains 34/34 observations and qualifies 18/18 native victims.
An independent auditor checks original IDs, preparation separation, source and
runtime identity, actual coding hashes, both API submission clocks, v2
accepted-generated/emitted/discarded counts, CPU enclosure and the actor's
submission after victim generation. It does not import the timing analyzer.
Generated counts are accepted output IDs, not physically processed draft tokens.

**Conclusion — supported in ordinary coding chat; historical contribution open.**
Both audits retain all 34 new observations, four unscored warmups and 18/18
qualified profile victims; there are no duplicate or unexpected IDs. The
independent auditor verifies the measured source and model, both qualified coding
input hashes, actual reporting enabled, API PIDs, source-controlled phase
separation and explicit generated/emitted/discarded counters. In all three long
cases, every victim output is ready inside the CPU stage and client delivery
starts afterward. Actor GPU submission occurs only after all victim outputs are
ready, excluding its later GPU prefill as the cause of this buffering.

| Long coding repeat | CPU render/encode ms | Server-ready TPOT ms | Client TPOT ms |
|---|---:|---:|---:|
| 0 | 609.1428 | 2.615474 | 0.057021 |
| 1 | 651.9270 | 2.615066 | 0.060472 |
| 2 | 572.7944 | 2.617906 | 0.053552 |

API thread CPU time is at least 99.8% of wall time in these stages. Short coding
stages finish before the victim's first ready output and do not meet the
buffering predicate. Their client intervals and the no-actor controls remain
in the complete audit. The dispatch-only arm independently shows compression:
long-coding client TPOT 0.103560/0.071186/0.066411 ms. Thus the effect does not
require output/frontend instrumentation. The arms share a persistent compiler
cache, and the first short control differs; do not interpret their differences
as a fixed instrumentation correction or discard first-use samples.

Minimal-arm long actor client-to-API-submission intervals are 724.862/702.384/
740.843 ms. These include transport and queueing and are not isolated CPU service.
The active-server CPU stages above also differ from the earlier isolated CPU
primitives. Keep both measurement contexts and all repeats; do not fit a service
multiplier to historical TTFT/TPOT errors.

No historical headline correction, six-point accuracy win or new default model
setting follows. Existing frozen request identities remain unchanged. Next
quantify response blocking with the already-frozen E034 CPU bookings, preserving
its five lost observations and explicitly excluding closed-loop feedback from
that diagnostic, before attempting a broader replay-engine change.

Evidence: `E037-independent-audit.json`, SHA256
`c8fa6c60db617d6d45811d457d415ae90328c01cd5d542b2eec4ba407a2b854b`; complete numerical audit, 34-request source manifest, qualified
serializer/tokenizer receipt, Slurm allocation/cleanup receipt and all failed
attempts. The source script is pinned separately from measured runtime receipts.

 No new
modeling setting, fitted latency, HF promotion or dashboard replacement.

## H1/H6 — delivery-only CPU diagnostic (E038, predeclared)

**Hypothesis.** Missing shared-API output blocking contributes to the request
errors left by E034 admission-only modeling. Predict that retaining server work
but delaying receipt behind existing CPU bookings changes TPOT error in the
small-reference cohort. Competing causes are wrong arrivals/GPU ordering, op
cost and closed-loop feedback; confidence is moderate for this diagnostic.

**Experiment.** Use only frozen E034 development outputs at lanes 1/8/14 and the
unchanged component profile SHA256 `741c9b07…` (one CPU worker, zero base cost,
1.3222043548296005 us/input token). Pin an input manifest before execution,
including native observations, original scores/execution/configuration, E003
scores, frozen identities and the component profile. No GPU allocation or new
prediction wheel. Verify CPU service units, nonoverlapping bookings, original
request-to-score reconciliation and metric-specific eligibility.

For server-ready time t, assume zero-cost response jobs join FCFS behind input
jobs already queued at t: receipt=max(t, maximum booked completion among jobs
with dispatch<=t). Inputs win exact ties. Include preparation CPU bookings; keep
preparation outside accuracy. Translate first/last receipt into TTFT and TPOT
with the original client-start boundary and N−1 output denominator. A defined
zero predicted TPOT remains scored; undefined reference errors stay unavailable.
Keep signed error, median APE, tails, absolute error and small-reference counts.

Freeze the original 4169 TTFT/4166 TPOT development identities. E034 already loses
five per metric, retaining 4164/4161; retain and report those losses unchanged.
Report original baseline and common-retained baseline separately and pool
requests directly. Any recovered eligibility stays outside this frozen cohort.

This is **postprocessing only**: arrivals, cache histories, CPU bookings and
server work do not receive client-completion feedback. Flag delivery beyond the
original grace boundary without dropping it. The output is a separate diagnostic
schema, not a standard prediction/publication artifact or a qualified accuracy
win, even if a retained-subset metric improves. Use the result to decide whether
a replay-engine change is warranted; do not bypass profile-cancellation guards.

**Conclusion — arithmetic verified; not a physical-model validation or win.**
Input manifest SHA256 `90dfce382fa4f6d52762cdaec98c76c423d299b52975504f8708ee1d5d7f51c8`
pins 18 inputs before execution from source
`b446d2462778983110fb824b458249a38212366d`. All source CPU bookings have
consistent service units and do not overlap; original matched native timings
reconcile with E034 scores. Independent arithmetic reproduces these retained
cohort results, without recovering or dropping any additional observation:

| Metric | E034 admission only | Delivery-only diagnostic | Retained n / original frozen n |
|---|---:|---:|---:|
| TTFT MAPE | 56.6126% | 58.6335% | 4164 / 4169 |
| TPOT MAPE | 71.7450% | 69.7286% | 4161 / 4166 |

TPOT at 1/8/14 lanes changes 18.9029→21.0019%, 55.3671→56.8199% and
88.3007→83.7219%. TTFT changes 25.0093→24.9666%, 28.8601→30.8820% and
76.9495→79.2451%. Keep these regressions visible. Twenty-five zero predictions
remain scored; the 15 positive TPOT references below 0.1 ms remain included, with
contribution 32.0316→28.5335 percentage points. No delayed matched response crosses
the original grace boundary, which does not establish feedback-complete validity.

Source event-order audit finds 23/11/61 output-completion/input-dispatch ties at
8/1/14 lanes, including 4/2/14 same-conversation candidate successor ties. The
native loop processes worker completions and received terminals before agentic
feedback and new ready arrivals. A fixed-booking input-first postprocessor can
therefore queue a response behind an input that its receipt releases. These
counts identify potential causal ties, not proof that every tied input depends
on that completion. Preserve the predeclared diagnostic and its assumptions;
do not select a tie policy or service rate for a lower score.

The small retained TPOT reduction neither repairs the five pre-existing losses
nor validates a shared-API implementation. A proper test must schedule delivery
inside the replay and let the original controller react to actual receipt times,
with preparation and profile cancellation still qualified. That is the next
discriminating implementation experiment. No wheel/default/HF/dashboard changes.

Evidence: `E038-delivery-only-diagnostic.json`, independent arithmetic audit
SHA256 `dc1fafc7cc37fae2a7f23d30cedc118f1aa1e29392fa070b04e57c848fa25717`, causal-tie source audit and all pinned E034 inputs. GPU cost 0.


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

Job 1005806 completes with exit 0 after 2677 seconds including batch/extern
cleanup; charge 21416 GPU-seconds. Completed collection totals
**55672 GPU-seconds / 15.4644 GPU-hours / 1.9331 TP8 node-hours**, leaving
**174728 GPU-seconds / 48.5356 GPU-hours / 6.0669 TP8 node-hours**. Baseline/timing
uses 6959 of its allocated 7200 node-seconds. The remaining pools are controlled
workloads (7200), targeted op/forward (3600) and validation (10800), before any
documented transfer of unused time. E018 job 1006430 reserves 36480 GPU-seconds,
leaving 138248 unreserved. Its reservation must become actual charged time when
terminal; the eight-node-hour campaign limit remains the hard admission bound.

Job 1006430 completes with exit 0 after 1298 allocated seconds including cleanup;
charge 10384 GPU-seconds to controlled-workload experiments. Completed allocation
totals **66056 GPU-seconds / 18.3489 GPU-hours / 2.2936 TP8 node-hours**; remaining
**164344 GPU-seconds / 45.6511 GPU-hours / 5.7064 TP8 node-hours**. Release its
reservation. Controlled workloads have used 1298 of their 7200 node-seconds;
op/forward and validation reserves remain unused. No GPU job is active at this
checkpoint. All startup and diagnostic-step time is included.

E028 attempt **1009422**, source `d66bcdad46cc6b67735803c958453fbe6b6d1a10`,
fails the 1200-second server-readiness guard on pool0-0281. All 50 checkpoint
shards report read completion, but model weight processing never reports completion.
No benchmark request runs: the independent blocking audit records **0/64 requests**,
with both arms incomplete. This is an operational failure, not evidence against H1/H6.
The resolved server arguments match successful control 1006430 except for the
framework-generated random seed; no seed effect is established. The successful
control finishes weight loading in 402 seconds. The historical loader source has
a separate DeepSeek `ThreadPoolExecutor`; changing the checkpoint iterator flag
would not establish a repair to that pool. Keep inference settings unchanged for
a bounded retry on another node and retain startup stacks if it stalls.

Charge Slurm parent/extern elapsed **1250 seconds × eight GPUs = 10000 GPU-seconds**;
diagnostic steps are already inside that allocation. Cumulative cost is now
**76056 GPU-seconds / 21.1267 GPU-hours / 2.6408 TP8 node-hours**, leaving
**154344 GPU-seconds / 42.8733 GPU-hours / 5.3592 TP8 node-hours**. Release the
reservation. Controlled-workload allocation uses 2548 of 7200 node-seconds;
the next 4560-second reservation fits its remaining 4652 seconds. No scoring
baseline or hypothesis conclusion changes after this startup failure.

E028 retry **1009781** completes with exit 0 after **791 allocated seconds**
including cleanup, charging **6328 GPU-seconds**. Total completed allocation is
**82384 GPU-seconds / 22.8844 GPU-hours / 2.8606 TP8 node-hours**. Remaining:
**148016 GPU-seconds / 41.1156 GPU-hours / 5.1394 TP8 node-hours**. Release the
reservation; no GPU job is active. Controlled workloads use 3339 of 7200
node-seconds. Preserve the first failed startup and the single unqualified
counter comparison in this successful collection's complete accounting.

E030 job 1009935 completes with exit 0 after 74 allocated seconds at three GPUs:
charge 222 GPU-seconds. Total charge is **82606 GPU-seconds / 22.9461 GPU-hours**,
leaving **147794 GPU-seconds / 41.0539 GPU-hours**. Release its reservation. The
controlled-workload pool now uses 3366.75 TP8-equivalent node-seconds, accounting
for the partial three-GPU allocation. No GPU job remains active at this checkpoint.

E032 job 1010000 completes with exit 0 after 100 allocated seconds at eight GPUs:
charge 800 GPU-seconds. Total charge is **83406 GPU-seconds / 23.1683 GPU-hours**,
leaving **146994 GPU-seconds / 40.8317 GPU-hours**. Release the reservation. The
controlled pool uses 3466.75 TP8-equivalent node-seconds. No GPU job is active;
CPU-only source/reconstruction audits add no GPU allocation cost.

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
python audit.py --frozen FROZEN_COHORTS --baseline RESEARCH_BASELINE --candidate NEW_REPLAY_DIR --lanes 1 8 14 --output NEW_AUDIT_JSON
python replay.py --tooling FROZEN_TOOLING --inputs FROZEN_HF_INPUTS --output NEW_EXPERIMENT_DIR --lanes 1 8 14
python reduce_stall.py --template FROZEN_CONFIG --predictor FROZEN_AGENTX_PREDICT --output NEW_REDUCTION_DIR
python analyze_timing.py --root PRIVATE_GPU_CONTROLS --output NEW_TIMING_AUDIT_JSON
python analyze_timing.py --suite frontend --root PRIVATE_FRONTEND_CONTROLS --output NEW_FRONTEND_AUDIT_JSON
python analyze_timing.py --suite blocking --root PRIVATE_BLOCKING_CONTROLS --output NEW_BLOCKING_AUDIT_JSON
python analyze_timing.py --suite coding-blocking --root PRIVATE_CODING_CONTROLS --output NEW_CODING_AUDIT_JSON
python audit_coding.py --root PRIVATE_CODING_CONTROLS --preflight CODING_CPU_RECEIPT --output NEW_INDEPENDENT_CODING_AUDIT_JSON
python response_gate_diagnostic.py --campaign PRIVATE_CAMPAIGN_ROOT --input-manifest FROZEN_E038_INPUT_MANIFEST --output NEW_DELIVERY_DIAGNOSTIC_JSON
python prepare_weka_frontend.py --trace FROZEN_WEKA_TRACE --profile ORIGINAL_PROFILE_JSON --model-revision FULL_MODEL_SHA --output NEW_PRIVATE_PAYLOAD_DIR
python predict_cold.py --template FROZEN_CONFIG --predictor FROZEN_AGENTX_PREDICT --output NEW_COLD_PREDICTIONS
python predict_matrix.py --controls PRIVATE_GPU_CONTROLS --template FROZEN_CONFIG --predictor FROZEN_AGENTX_PREDICT --output NEW_NATIVE_MATRIX
# E036 requires the new submission receipts and keeps client TTFT boundaries.
python predict_matrix.py --dispatch-matrix --arrival-clock client_start --controls PRIVATE_SUBMISSION_CONTROLS --template FROZEN_CONFIG --predictor FROZEN_AGENTX_PREDICT --output NEW_CLIENT_CLOCK_MATRIX
python predict_matrix.py --dispatch-matrix --arrival-clock server_submission --controls PRIVATE_SUBMISSION_CONTROLS --template FROZEN_CONFIG --predictor FROZEN_AGENTX_PREDICT --output NEW_SUBMISSION_CLOCK_MATRIX
python audit_matrix.py --controls PRIVATE_SUBMISSION_CONTROLS --client-replay CLIENT_CLOCK_MATRIX --submission-replay SUBMISSION_CLOCK_MATRIX --output NEW_MATRIX_AUDIT_JSON
# Remote collector: --suite frontend selects the separately qualified E018 controls.
qualify_weka ORIGINAL_WEKA_TRACE --materialize PRIVATE_LOWERED_JSONL
PYTHONPATH=FROZEN_TOOLING python recorded_arrivals.py --inputs FROZEN_HF_INPUTS --lowered PRIVATE_LOWERED_JSONL --predictor FROZEN_AGENTX_PREDICT --frozen FROZEN_COHORTS --output NEW_DIAGNOSTIC_DIR
python -m pytest -c /dev/null -p no:cacheprovider test_audit.py test_timing.py test_timing_audit.py test_matrix.py
uv run --no-project --with 'transformers>=4.57,<5' --with pytest python -m pytest -c /dev/null -p no:cacheprovider test_probe.py test_frontend.py
```

The comparison mode retains original metric-specific eligibility, reports recovered
observations separately, and pools individual error sums and denominators. Its
diagnostics use the same frozen eligibility as the headline.

The GPU scripts require an immutable model receipt, successful CPU preflight,
explicit remaining GPU-seconds and campaign-root paths. Their synthetic controls
never change the frozen benchmark headline scores. Update this record and the
checkpoint after each meaningful experiment; preserve disproven conclusions.
