# Independent sampling assessment

The current 32 profile/workload cells qualify implementation paths. Three
repetitions estimate within-cell noise; they do not add architectural coverage.
The 618 module keys and 45 baseline keys are derived operator geometries, not
independent full-model cases. Prefix 128, which independent serving verification
uses, is absent from the current `{0,256}` grid and must be reported as missing.

## Common bounded calibration and hold-out domain

For the common GB200/GB300 text domain, fix TP4/EP1, batch 1–2, total new tokens
per iteration at most 512, per-request KV end at most 2048, and eager execution.
The proposed explicit point manifest has the following independently checked
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
forwards. The current prefix-256 samples remain evidence but do not substitute
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
at four representative cells per profile: 16 further SGLang E2E observations.
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
extension; explicit decode cases seed K-1 real tokens on the same request,
then measure one decode at exactly K. No held-out timing enters this plan.

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
