# Independent sampling assessment

The first 32 profile/workload cells qualified implementation paths. Their 618
module keys and 45 baseline keys were derived operator geometries, not
independent full-model cases. That pilot's `{0,256}` prefix grid did not cover
the serving verification's prefix 128. The completed common study and separately
versioned prefix refinement below expand coverage while preserving the pilot.
Repetitions estimate within-cell noise; they do not add architectural coverage.

## Completed designs and interpretation

| Study | Calibration configurations | Fresh holdout geometries | Repetitions and scope |
|---|---:|---:|---|
| [Common GB300 study](study/README.md) | 126 per profile | 38 per profile | Three measurements per configuration, after one warmup |
| [Precision attempt](report/precision-v2/README.md) | Unchanged | Same 38 per profile | Separate ten-repeat attempt; original attempt retained |
| [Prefix refinement](report/prefix-refinement-v1/README.md) | 18 additional bounded-profile configurations; two earlier pilot attention cells explicitly reused | 46 per profile, disjoint from original calibration, original 38 and new calibration | Ten-repeat holdouts; no forward-holdout timing becomes a module row |
| [Ordinary serving, OFF](report/serving-v4/off-prefix-refined/README.md) | Frozen original/refined overlays compared separately | 15 metric-bearing scenarios per trial | Ten pilot trials selected 40 main trials; 600 metric cohorts / 880 requests |

The 126/38 design is a bounded geometry and interpolation assessment. It is not
a statistical proof of accuracy across arbitrary contexts, content or hardware.
The new 46 geometries cannot be compared with the original 38 as if a change in
their aggregate error isolated the effect of calibration refinement. The OFF
serving report instead uses exactly the same observations for both overlays and
shows prediction coverage as well as error on the same supported subset.

For E2E precision, the frozen rule estimates `CV = sample_stddev / mean` from
ten pilot trials and takes the largest requirement across all 15 scenarios'
TTFT, mean time per output token and throughput:

`N_required = max(20, round_up_to_10((1.96 * CV / 0.05)^2))`.

The main-stage cap is 100. Pilot data chooses this budget and is excluded from
main confidence intervals. The normal approximation is a planning estimate
using an estimated variance, so achieved interval widths must also be reported;
it does not guarantee them. See the [NIST sample-size guidance](https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm).
The completed OFF run selected 40 trials and all 45 required mean intervals had
relative half-width at most 5% (maximum 4.38%). The ON pilot selected a requirement
of 150, above the 100-trial cap; this remains a recorded precision shortfall.
The ON pilot statistics are frozen at SHA256
`d7a0388ea15514f67165d2b06aeb2fb7b8faed52d11e7e2fda7a0abc393df2d6`;
this budget receipt does not qualify the still separate main observations.

Bootstrap intervals resample whole trials and are pointwise, conditional on the
observed physical runtime and frozen calibration. They do not give simultaneous
95% coverage of all 45 metrics or include uncertainty from collecting another
calibration table. Tokens, native dispatches and four TP ranks do not increase
the trial count. If a fixed allocation deadline divides the main plan across
physical lifecycles, retain the original trial IDs and all boundary observations,
report lifecycle-specific complete-trial intervals where eligible, and report
combined coverage and errors descriptively. Do not pool those segments into a
single-run confidence interval or claim the original precision target passed.

GB200's native FPM policy currently collects one timing per manifest point.
Its independent 38-point holdout therefore supports descriptive prediction
errors, with no per-geometry repeated-run confidence interval. HTTP E2E uses the
separate pilot/main rule. Different observation boundaries and replication
units must remain explicit in the combined PR report.

## Common bounded calibration and hold-out domain

For the common GB200/GB300 text domain, fix TP4/EP1, batch 1–2, total new tokens
per iteration at most 512, per-request past KV at most 2048 (native decode includes the current token, up to 2049), and eager execution.
The frozen explicit point manifest has the following independently checked
counts **per execution profile**:

| Purpose | Grid and constraint | Points |
|---|---|---:|
| Prefill calibration | B `{1,2}`, per-request Q `{1,2,3,4,8,16,32,64,127,128,129,256,512}`, P `{0,128,512,1536}`; `B*Q<=512`, `P+Q<=2048` | 100 |
| Decode calibration | B `{1,2}`, KV `{2,3,8,32,64,127,128,129,256,512,1024,1536,2048}` | 26 |
| Prefill hold-out | same B/P, Q `{48,96,192,384}`, same constraints | 28 |
| Decode hold-out | B `{1,2}`, KV `{96,192,384,768,1792}` | 10 |
| Total | calibration + disjoint hold-out points | **126 + 38 = 164** |

For SGLang replay OFF and ON this is 252 calibration and 76 held-out forward
configurations. vLLM's verified full profile needs one set. These counts exclude
warmup, statistical repetitions, content variants, and actual prefix-building
forwards. The original prefix-256 samples remain evidence but do not substitute
for an exact prefix-128/512/1536 bucket.

This is a reasonable bounded first calibration design, not a universal minimum
or a statistical sufficiency guarantee. Powers-of-two token anchors sample
launch, GEMM and routing regimes; 1/2/3 distinguish very small launches and
half-rate odd/even publication; 127/128/129 bracket SWA and bounded replay.
Prefix 1536 crosses the 512-entry sparse-index selection limit for both full-
and half-rate owners. Prefix is an exact table dimension, so each claimed
prefix/batch bucket needs its own attention curve.

Hold-out queries must remain out of the fitted tables. If they fail the
predeclared error threshold, add calibration points around the observed
transition and select fresh hold-outs. Determine repeat count from observed
noise/confidence; additional repetitions do not repair missing shapes. The
combined validation report specifies the adaptive repetition rule.

## Separate semantic and extended-domain challenges

Use at least two additional tokenized corpora with different repetition/locality
at four representative cells per profile: 16 further SGLang corpus/scenario/
profile combinations, each with its own pilot and main repetitions.
Preserve input hashes and actual expert histograms. One uniform-routing baseline
cannot establish Engram locality or expert-popularity accuracy.

Test heterogeneous extensions separately, including a short extension on a long
cached prefix paired with a longer uncached request. Aggregate FPM telemetry
cannot recover all per-request tail lengths; replay-enabled heterogeneous
telemetry remains rejected until its contract represents the actual requests.
Homogeneous substitutes must not be counted as passing those cases.

Prefix 4096 and real-KV decode near 8192/16384/32768 need separately qualified
runtime capacity. The coarse candidate budget is 2048 blocks × 8 entries;
frontier tests must straddle its different full-/half-rate context thresholds.
Neither the bounded common grid nor short-context extrapolation qualifies those
branches. Larger batches, other parallel layouts, CUDA graphs, DSpark and vision
also require separate qualification. There is no defensible fixed point count
for all of these unbounded extensions before their domain and error target are
specified.

## Executable expansion plan and predicted key coverage

`study-plan/calibration-plan.json` freezes all 126 configurations and their
original native point payload. `collector.sglang.dsv41_workloads` generates
that plan plus `study-plan/coverage-projection.json`; the latter is a CPU
projection, not measured data. The native module runner accepts the frozen
plan through `--workload-plan`. Explicit context cases measure only that
extension; explicit decode cases seed K real tokens on the same request,
then measure one decode at native inclusive K+1. No held-out timing enters this plan.

With one warmup and three measured repetitions per configuration, each profile
executes 908 native forwards including the 404 prefix/real-KV setup forwards,
and records 378 measured forwards. The baseline axis expands from 9 to 16 token
counts, producing 80 GEMM/MoE/NCCL keys. This is 4.05 times the pilot's 224
forward calls per profile before accounting for longer contexts, new JIT
shapes, model load and baseline work. A two-hour sequential GPU reservation is
a campaign budget, not a measured completion-time estimate; record actual
startup and per-case elapsed time before scheduling any subsequent repetition.
Original pilot evidence and tables remain separate.

The projected full-profile table has 836 physical module points and brackets
all 38 held-out module queries. The bounded-profile table has 830 points;
28 held-out configurations have complete interpolation domains, while **10
have missing attention curves**. For extensions beyond 128, current strict
keys use `effective_prefix = P + Q - 128` and `x = 128`; held-out Q values can
therefore need a prefix absent from calibration even when the original P
bucket exists. These are explicit missing strict-SILICON predictions. Any
Hybrid fallback must retain its SOL source label and is not a measured match.

A future bounded-attention contract could retain original P as the exact
bucket and original Q as a continuous axis, while computing the same native
tail scope internally. That requires stage/query metadata and a versioned
axis identity; existing rows must not be relabeled. It also needs separate
curves across native kernel regimes, including the 128-token cutoff, first
compressed publication, sparse selection saturation, and coarse candidate
limits. CSA role, compression ratio, batch, execution profile and runtime
provenance remain exact dimensions. This proposal is not implemented here.

## Independent native forward hold-outs

`study-plan/heldout-plan.json` freezes the 38 disjoint configurations. The
runner's `--forward-only` mode never constructs `ComponentRecorder`, changes
mHC streams, or intercepts attention output reductions. It refuses component
baselines and profiler timing. Use one warmup and at least three measured
repetitions with the two unfused communication flags and TP-sharded shared
experts. Its separate `forward-rank-*.jsonl` records cannot be published as
module rows.

The boundary matches the pinned native SGLang `one_batch.py`: synchronize,
wall-clock start, native `extend`/`decode`, synchronize, wall-clock stop. This
includes batch preparation, all model layers and shared Engram hashing, and
sampling. It is independent benchmark-forward ground truth, distinct from the
HTTP E2E and GPU-timed Dynamo FPM measurements. Source/input hashes, exact
coordinates, timing-boundary label and runtime arguments accompany each run.
Neither the held-out latency values nor their module timings enter calibration.

The native point manifest's decode axis is **past KV**, while SGLang attention
keys include the current token. Plans and forward records therefore retain
both `canonical_past_kv=K` and `native_inclusive_kv=K+1`; the runner seeds K
real tokens. At batch 2 and past KV 2048 the decode needs 4098 logical token
slots before page rounding. The expanded native-forward study uses 8192 allocator slots and a
4096-token prefill setup capacity, while preserving every measured geometry.
These are capacity settings, not an expansion of the statistical study domain.
