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

## H1 — Client timing boundaries and small positive denominators

**Hypothesis.** Buffered receipt, frontend work and/or speculative bursts compress
some client-observed decode intervals. Positive tiny denominators magnify MAPE.

**Experiment E001.** Independently recompute the original request errors and
stratify references below 0.1 ms without excluding them. Audit raw HTTP and metric
boundaries; retain all requests in the headline metric.

**Conclusion — sensitivity confirmed; timing mechanism open.** At 14 lanes,
9/2429 references contribute 65.43 percentage points to 115.63% TPOT MAPE; median
APE is 24.39%. One raw example has 222.7242 ms HTTP receive time but 0.54664 ms
recorded decode duration. These facts do not identify when the server generated
its tokens and do not justify changing historical observations.

**Experiment E005, awaiting a qualified retry.** Compare uninstrumented MTP, instrumented MTP and
instrumented plain decode with fresh servers. Synthetic controls use 8192/131072
input tokens, 16/128 output tokens, lanes 1/8/14, repeats and explicit cache-flush
requests. Also exercise the OpenAI chat frontend with reasoning/tool parsers.
Capture host scheduler output-ready counts/timestamps by rank, client frame
receipt, numerical usage/acceptance fields, and one-second server metrics.
These are output-ready observations before detokenization/HTTP, not GPU kernel
completion timestamps. Chat frames without per-chunk token counts must not be
presented as exact per-token observations. Missing comparisons remain N/A.

**Conclusion — startup failed; GPU timing evidence unavailable.** Job 1002848
exceeded its 1200-second serving-readiness guard. A read-only Python stack sample
places the scheduler in `do_load_weights`, waiting for asynchronous expert-weight
copies (`expert_data.copy_`); the shard reader had finished. This establishes the
startup location, not a deadlock cause. No request was attempted. A native stack
sample failed to unwind and is retained as unavailable. Prepare the original
SGLang 0.5.20 image by immutable digest on CPU before another GPU attempt.

Require actual counts, clocks, rank identity,
completion, output-length conservation and coverage before comparison. Server
retained/processed-token counts and client-delivered counts are distinct; preserve
any terminal over-generation rather than silently equating them.

## H2 — Recipe, tokenization, snapshots and cache history

**Hypothesis.** Effective serving settings or replay history differ from silicon.

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

## H3 — Speculative acceptance and output-delivery bursts

**Hypothesis.** Fixed acceptance misses request-dependent acceptance and bursts.

**Experiment.** Verify the original NEXTN-to-EAGLE resolution (steps 3, draft tokens
4). Replay uses `nextn=3`, `nextn_accepted=1.99`, meaning 2.99 accepted output
tokens per round before request-tail truncation. The recipe's golden acceptance
is explicitly inherited/provisional; do not treat it as measured per-request truth.
E005 collects actual acceptance and output counters with speculation on/off.

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

## H5 — Op timing, graph mode, variant lookup and interpolation

**Hypothesis.** Data selection or execution timing differs despite including #665.

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

## H6 — Capacity, preemption, frontend and serving overhead

**Hypothesis.** Admission, memory pressure or frontend work explains residual error.

**Experiment.** Reconcile fixed replay capacity with actual runtime receipts;
inspect admission, preemption and response timing. E005 distinguishes native
input-ID requests from chat frontend observations. Use uninstrumented repeats
before interpreting profiler effects.

**Conclusion — open.** Do not infer eviction/preemption from utilization, or infer
which operation uses FP8 merely from a model-level loader message. The pinned
checkpoint is mixed precision; match component-level settings and evidence.

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

The historical retry also restores the recorded multithread/prefetch loading and
keep-alive settings. This is a qualification repair, not an isolated experiment
attributing a change to the framework version. The GPU runner accepts only the
two named runtime choices and requires the historical image's CPU receipt and
checksum before allocation. The failed 0.5.21 run remains separate evidence.

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
python -m pytest -c /dev/null -p no:cacheprovider test_audit.py test_timing.py
uv run --no-project --with 'transformers>=4.57,<5' --with pytest python -m pytest -c /dev/null -p no:cacheprovider test_probe.py
```

The GPU scripts require an immutable model receipt, successful CPU preflight,
explicit remaining GPU-seconds and campaign-root paths. Their synthetic controls
never change the frozen benchmark headline scores. Update this record and the
checkpoint after each meaningful experiment; preserve disproven conclusions.
