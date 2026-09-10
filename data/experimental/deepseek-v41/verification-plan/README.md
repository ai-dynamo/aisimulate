# DeepSeek V4.1 measurement and prediction study

Status: the study design is frozen. GB300 completed the 126-configuration
calibration and 38 independent forward holdouts per Decoder profile, followed
by a separate ten-repeat attempt on every holdout. The [versioned report](https://github.com/ai-dynamo/aisimulate/tree/22068b86/data/experimental/deepseek-v41/gb300-silicon/report/precision-v2)
retains both attempts and all missing predictions. GB200 calibration and both
GPUs' complete HTTP E2E/FPM verification remain pending. The original GB300
16-workload qualification grid and its 309 derived module keys per profile
remain separate from independent verification counts.

## Scope and point counts

The first common domain is one node, TP4/EP1/DP1/PP1/CP1, text autoregression,
DSpark off, Engram in GPU memory, batch 1–2, at most 512 newly scheduled tokens
per iteration, prefill `prefix + new <= 2048`, and decode **past KV <= 2048**
per request. The measured decode's inclusive sequence length is past KV + 1;
native bookkeeping needs one further slot, so the maximum model length is
2050. Native allocator/page capacity must accommodate that padding for both
requests. Eager execution is a
separate identity from CUDA graph execution.

The supported runtime profiles are GB200/vLLM/decoder OFF and
GB300/SGLang/decoder OFF or ON. vLLM decoder ON remains a runtime dependency.
GB200 calibrates the whole-forward FPM approach; GB300 calibrates module-based
SILICON predictions. Both GPUs need independent whole-forward and E2E evidence.

| Set | Per-request new tokens | Prefix or decode KV tokens | Configurations per profile |
|---|---|---|---:|
| Calibration prefill | 1, 2, 3, 4, 8, 16, 32, 64, 127, 128, 129, 256, 512 | prefix 0, 128, 512, 1536 | 100 |
| Calibration decode | one token | KV 2, 3, 8, 32, 64, 127, 128, 129, 256, 512, 1024, 1536, 2048 | 26 |
| Geometric holdout prefill | 48, 96, 192, 384 | same four prefix buckets | 28 |
| Geometric holdout decode | one token | KV 96, 192, 384, 768, 1792 | 10 |
| Total | disjoint calibration and holdout geometries | batch 1 and 2 | **164** |

Prefill retains only points with `batch * new <= 512` and
`prefix + new <= 2048`: 25 batch/new combinations times four prefixes give
100 calibration points; seven held-out combinations times four prefixes give
28. The 13 and five decode KV anchors each cross two batch sizes. There are
126 calibration and 38 geometric holdout configurations per runtime profile,
or 492 profile/configuration combinations across the three supported profiles.
These are target counts, not a claim that every runtime has completed them.

Small values expose launch overhead and odd/even compressed-cache publication.
127/128/129 bracket the SWA and bounded-decoder transition. Prefix 1536 crosses
the 512-selected-entry threshold for both full-rate and half-rate owners.
SILICON prefix buckets are exact lookup dimensions: a curve at one prefix does
not establish interpolation at a missing prefix. The independent new-token
and decode-KV holdouts test interpolation within the stated domain.

`calibration.json` and `heldout.json` are native Dynamo schema-3 point manifests.
Their design sidecars record counts, bounds and file hashes. Points and prose
are authored for this study. The schema is defined by
[Dynamo benchmark_points.py at 5496017](https://github.com/ai-dynamo/dynamo/blob/54960177085413259859c88bd34ed0734d4c2ea9/components/src/dynamo/vllm/benchmark_points.py).
Freeze the canonical payload and hash in the Collector plan before rendering;
never patch generated launch scripts or admit held-out rows to calibration.

## E2E and semantic verification

The pilot has 17 cohorts and 24 HTTP requests per trial: 15 metric-bearing
scenarios and 22 requests, plus a smoke request and a prefix-warming setup
request. Scenarios cover short requests, 127/128/129 boundaries, off-grid
lengths, concurrent bursts, heterogeneous lengths, cold/warm prefix controls,
and repeated versus distinct text. Geometry overlap with calibration is
allowed for these semantic checks and must not be called a geometric holdout.

Freeze actual page size and cached-prefix behavior first. The observed GB300
page size is 256, so its cross-request test uses 512 shared tokens followed by
256 new tokens, with an independent cold 768-token control. Clear and verify
an idle cache before each cold cohort; preserve it only across the explicit
warm-A/reuse-B pair or a declared locality experiment. Record observed KV reuse,
not the intended prefix length. A 768-token cold prompt or a three-request burst
is a chunking, queueing or extrapolation challenge under the common bounds.

Use a fresh run ID, real-text token offset and seed for each independent trial.
Pair seeds across decoder OFF and ON for semantic and output comparisons. Keep
two additional original corpora with different content/locality as stress
strata; one short repeated fixture does not represent production traffic.
Replay-enabled heterogeneous FPM inputs need actual per-request extension
lengths; aggregate-only telemetry that cannot represent those tails is an
unsupported prediction case, not a passed homogeneous substitute.

## Repetitions and uncertainty

Run one complete warm-up suite before each runtime profile's pilot, exercising
all serving shapes and real decode while retaining its separate warm-up role
and timing/failure receipts. This excludes first-use JIT from the steady-serving
target without deleting pilot outliers after observing them. Continue clearing
KV before every designated cold cohort.

Collect ten independent pilot trials per E2E scenario. A trial is the unit of
replication: TP ranks and successive decode callbacks within one request are
correlated and do not count as independent trials. Report repeated calibration
timings separately from architectural coverage and request replication.

Use pilot trial-level TTFT, average time per output token and throughput to
estimate coefficient of variation `CV = sample_stddev / mean`. Freeze the main
sample size per scenario using the largest available metric CV:

`N = max(20, round_up_to_10((1.96 * CV / 0.05)^2))`.

Cap a single campaign stage at 100 main trials and explicitly report a precision
shortfall if the estimated requirement is larger. The 5% target concerns
measurement precision, not prediction accuracy. Pilot trials choose the budget
and are excluded from the main confidence interval. At the frozen main sample
size, report trial-level mean, median, variation and 95% bootstrap intervals;
bootstrap whole trials, not individual token gaps. The normal approximation
above is a planning estimate, so verify achieved interval width and report it
even if the target is missed. This follows the distinction between variance,
sample size and uncertainty in the
[NIST confidence-limit guidance](https://itl.nist.gov/div898/handbook/eda/section3/eda352.htm).

If prediction errors expose a transition missed by calibration, add neighboring
calibration points and reserve new, unseen geometric holdouts. Preserve the
original failed holdout result and the refinement history; do not report a
training point as successful independent verification after fitting it.

## Required report in PR descriptions

Each applicable PR description must include a compact result table plus links
to versioned evidence and reproduction instructions. The report must contain:

- Checkpoint/config revision; image architecture/digest; actual backend and
  instrumentation source hashes; GPU topology; precision; cache/page settings;
  decoder strategy; graph/fusion/collective dispatch; input and plan hashes.
- Planned, attempted, successful and failed logical configurations; repetitions,
  unique requests, native FPM iterations, physical operator keys and coverage.
  Report these counts independently and classify interpolation, extrapolation
  and unsupported cases. Preserve failed attempts without manufacturing rows.
- Paired observed/predicted forward latency and measured E2E TTFT, token timing
  and throughput. Define signed error as `(prediction / observation - 1) * 100`;
  also report median absolute percentage error, 90th-percentile absolute error
  across configurations and WAPE `sum(abs(predicted-observed))/sum(observed)`.
  State the weighting, sample count and prediction provenance for each table.
- Main-trial 95% uncertainty intervals; workload-level scatter/error plots;
  boundary/prefix breakdowns; source coverage and explicit remaining gaps.
  Do not imply a tail-latency percentile is well estimated by 20 requests.
- Full Dynamo FPM capture with zero-duration heartbeats excluded from latency
  error calculations but retained for counter/loss auditing. Require producer
  queue/send/drain evidence; continuity alone cannot detect pre-counter drops.
  If client frames coalesce tokens, report TTFT/TPOT/throughput where valid and
  mark exact client ITL unavailable.

SOL is an analytical lower-bound model; its bias must be shown without implying
calibrated accuracy. Compare full runtime predictions with E2E using the same
scheduler/workload assumptions. Summed GPU-module time or FPM forward time is
not automatically a prediction of frontend queueing, TTFT or network delivery.
Keep internal cluster identities, paths and raw infrastructure logs private.

Prefix/contexts above the common domain, candidate-budget frontiers around
16K/32K compressed-owner positions, batch above two, CUDA graphs, other parallel
layouts, vision and DSpark need separate qualified campaigns. Current Engram
module measurements omit the shared model-entry hash/history work; default
fused collectives also differ from an explicitly unfused NCCL execution policy.
These gaps belong in the error report and coverage table.

## Analysis and measurement boundaries

`analyze_e2e.py --plan pilot-plan.json --progress client/progress.json --output summary.json`
checks each planned cohort and reports client coverage. It freezes a main-stage
budget only after all ten pilot trials and required metrics are available.
For `sampling_role=main`, it computes percentile bootstrap intervals for the
mean using whole independent trial observations. It averages requests equally
within each trial; many correlated token gaps do not increase the trial count.
The script does not replace closed-run, transport or producer qualification.
Throughput here is output tokens over the complete finite cohort duration,
including prefill and queueing; it is not a saturation-throughput measurement.

`normalize_fpm.py` preserves raw telemetry and returns a separate prediction
input plus an explicit conversion receipt. The inspected SGLang producer uses
decode sequence length including the current query; Dynamo/vLLM's inspected
scheduler uses already computed past KV. The op graph's decode `s` includes the
current query, while `predict_decode_latency_total` and the collected whole-FPM
curve use past KV. Convert once via canonical past KV, adjusting by the number
of decode requests, and leave prefill/queued fields unchanged. Pin producer
source before selecting this bridge; test 127/128/129 boundaries so a one-token
semantic mismatch cannot masquerade as prediction error.

The vLLM FPM observation includes CPU schedule/output or adjacent-output timing.
The corrected SGLang FPM observation uses the existing GPU event interval.
SGLang native benchmark forward-only holdouts use its existing synchronized
wall-clock boundary, which includes batch preparation and sampling. HTTP E2E
adds frontend, network and queueing. Report these as separate observed targets.

Run the adjacent `test_analyze_e2e.py` and `test_normalize_fpm.py` with pytest to
check statistical independence, missing coverage and KV-axis conversions.

`compare_forward.py` consumes the SILICON collector's admitted independent
forward observations and the frozen holdout manifest. Run it with that
checkout's source and rebuilt extension available, supplying an explicit native
prediction configuration. It calls the shared Rust forward estimator without
tuning, validates geometry and decoder profile, preserves every missing
prediction, and records the actual loaded model, engine and binary hashes.
Run separate SOL, HYBRID and strict SILICON comparisons; a supported subset's
error must always be accompanied by its prediction coverage. The original
three-repeat and separate ten-repeat forward attempts estimate point medians;
neither establishes an E2E confidence interval.

`compare_trace.py` consumes a closed native/request-qualified audit, the frozen
main plan, normalized measurement identity and an explicit prediction config.
It compares every attributed native interval, retaining actual unreturned
overlap work and unsupported predictions. vLLM's original prompt-length
variance remains in native evidence; op-level prediction receives a separate
query-length variance with an explicit bridge. Whole-FPM retains its native
variance convention. Per-interval statistics are descriptive because intervals
within a trial are correlated. A separate summary resamples complete independent
trials within each scenario for total-forward signed bias and interval WAPE.
Every native interval, including unreturned overlap work, stays with its trial.
The final 95% intervals require at least twenty complete trials with complete
prediction coverage; a trial missing any prediction remains visible and prevents
that scenario from receiving a final interval. This uncertainty is conditional
on the frozen model and calibration, not variation across runtime lifecycles.

`compare_e2e.py` replays the actual input token IDs, output lengths and HTTP
submit offsets through the native scheduler and independent timing provider.
Each cold cohort starts a fresh model cache; the explicit prefix seed and reuse
requests share one replay instance. Only the reuse cohort contributes its
comparison metrics. Native real-KV/page controls and their source receipts are
pinned. HTTP response time is never an input to the timing provider. Its
TTFT/mean ITL/throughput comparison includes whole-trial paired bootstrap error
intervals only for complete main stages with at least twenty trials and complete
prediction coverage. Actual request ID/token hashes and successful cold-cache
acknowledgements are required, including the prefix seed.
Observed and predicted initial cached-token counts remain side by side for every
request; a disagreement is reported without dropping its timing error. Both
comparison tools retain the original plan's coverage and corpus roles. A declared
coverage candidate is not proof that every actual native geometry is interpolated.
Each comparison consumes one frozen main plan from one corpus stratum; separate
corpora retain separate trial budgets and uncertainty.

Both tools require source bindings to the exact audit/plan/execution/worker
artifacts and analysis source. E2E additionally binds the original HTTP summary
and resolved scheduler receipt. The normalized identity must be generated from
verified original receipts, not constructed by selecting arbitrary values.
`--diagnostic` accepts an explicitly closed partial lifecycle while preserving
the original main budget and missing coverage; it never reports a completed
study or final confidence interval. Partial stages cannot be silently pooled
across different measurement versions.

SGLang replay uses its measured logical KV capacity. vLLM's shared physical
pool has heterogeneous attention groups and per-request circular buffers, so it
cannot be mapped by summing group token capacities. Its E2E comparison uses a
declared unconstrained logical capacity calculated from the frozen workload,
only when every relevant native dispatch and the final receipt prove zero
allocation refusals/exceptions/preemptions and minimum free blocks above the
watermark. Any missing witness or actual capacity pressure makes that prediction
ineligible while preserving the real measurement. This tests serving timing
conditional on absence of capacity pressure; it does not validate allocator
memory accuracy. HTTP frontend/transport costs, native overlap details and the
existing SGLang replay's extra first-output decode remain explicit model limits.

The [original corpus strata](corpora/README.md) add English narrative and mixed
Chinese/English technical prose at fixed geometry. Their prepared inputs are
separate from actual collected coverage.
