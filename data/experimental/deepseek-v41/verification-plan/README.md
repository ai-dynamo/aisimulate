# DeepSeek V4.1 measurement and prediction study

Status: frozen study design, not completed measurement coverage. The current
GB300 16-workload grid per decoder profile is an implementation qualification
set. Its 309 module keys per profile are derived geometries, not 309 independent
workloads. Keep actual coverage and failed attempts alongside this plan.

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
