# Conditional dense collection components

The fitter identifies preparation and execution components for the recorded
second native pass in each worker. **Subsequent-in-worker does not prove a
compiler-warm state.** The results below use the immutable 876-record study; failed models remain unqualified.

From `python/aisimulate`:

```bash
python -m tools.perf_database.collection_cost_dense_components \
  --samples /path/to/frozen-dense-feature-samples.jsonl \
  --output /path/to/dense-component-fit.json
```

Each record must retain its sample/group/replicate identity, exact executed
case, actual native kernel/dispatch, source and observer hashes, runtime
protocol, actual allocation observations and deduplicated storage bytes. The
fitter requires the five nonoverlapping observed benchmark clocks to sum to
execution, and preparation plus execution to sum to the inclusive native call.
Missing observations, failed calls and incompatible runtime strata do not fit.

Preparation uses an intercept, actual unique cache-allocation bytes and actual
unique forward-input storage bytes. It never substitutes logical sequence size
for measured allocation; allocation floors and storage aliases matter.
Execution uses an intercept and causal attention work multiplied by observed
timed repetitions. That feature is an empirical **proxy** for all five measured
execution clocks under an identical recorded helper protocol. Warmup and capture
forward counts are unknown and are never inferred from requested warmups.

The predeclared design uses batches 1/4, context lengths 16384/32768 or decode
past lengths 65535/262143, with Q/KV geometries 4/1, 32/2 and 16/1. Only the first
two geometries from fresh worker replicates 1/2 fit coefficients. Geometry 16/1
is withheld across all three replicates. Replicate 3 independently checks the
sum of the same scoped component; it does not check a whole-worker forecast.

A component qualifies only with a full-rank, well-conditioned scaled design,
nonnegative coefficients, at most 20% maximum error on the withheld geometry,
and at most 15% error on the independent third component group. Validation
features must remain within observed training bounds. Kernel, precision,
checkpoint, device, framework, source or runtime changes form separate strata;
an incomplete stratum stays unqualified. Numerical tolerances only clip tiny
roundoff, never materially negative coefficients.

The native worker clock includes both recorded passes and shared initialization.
It is not a normal one-pass collection forecast. Cold/setup, whole-worker and
fleet totals remain unknown. Passing this conditional component study alone
cannot qualify the complete additional B/S/M collection population.

## Frozen study results

**One execution component qualified; no preparation component qualified.** The unchanged
20% whole-geometry and 15% independent-component gates rejected the other models. Negative
coefficients rejected the proposed physical model, not the positive observed time. No
post-hoc model changes or relaxed gates were used to obtain a passing result.

The input contains 864 full-study point clocks from 36 fresh workers plus 12 smoke diagnostics.
Only native pass 2 enters the design: 432 study observations form 12 complete runtime/phase
pools; six second-pass smoke observations lack a complete design. All 438 first-pass clocks
remain diagnostic inputs. Each complete pool trains on 16 observations (Q4/KV1 and Q32/KV2,
replicates 1/2), withholds all 12 Q16/KV1 observations across the three replicates, and checks
all 12 third-worker component predictions independently.

Failed cells below show **maximum whole-geometry error / third-component error**. A rejected
coefficient produces no qualified rate even if some individual predictions might agree.

| System | Framework | Phase | Preparation | Execution |
|---|---|---|---|---|
| b300_sxm | sglang | context | Rejected coefficient | Fail: 44.70% / 5.61% |
| b300_sxm | sglang | generation | Rejected coefficient | Rejected coefficient |
| b300_sxm | trtllm | context | Rejected coefficient | Rejected coefficient |
| b300_sxm | trtllm | generation | Rejected coefficient | Rejected coefficient |
| b300_sxm | vllm | context | Rejected coefficient | Fail: 90.98% / 10.27% |
| b300_sxm | vllm | generation | Rejected coefficient | Fail: 300.96% / 18.55% |
| h100_sxm | sglang | context | Rejected coefficient | Rejected coefficient |
| h100_sxm | sglang | generation | Rejected coefficient | Fail: 126.55% / 0.29% |
| h100_sxm | trtllm | context | Rejected coefficient | Pass |
| h100_sxm | trtllm | generation | Rejected coefficient | Fail: 48.42% / 6.24% |
| h100_sxm | vllm | context | Fail: 40.32% / 13.31% | Rejected coefficient |
| h100_sxm | vllm | generation | Fail: 22.98% / 6.66% | Fail: 57.85% / 2.27% |

The qualified **H100 / TensorRT-LLM context execution** component has maximum Q16/KV1 error
**3.5831%** and independent third-component error **0.3761%**.
For that recorded 12-point third-worker scope, execution measured **0.002313378 GPU-hours**;
the model predicted **0.002322079 GPU-hours**. This excludes preparation, worker initialization,
first-pass costs and progress/receipt I/O. It is not the total extra B/S/M collection cost.

That scope is full causal attention, head dimension 256, BF16 compute and KV, B1/4, sequences
16384/32768, and the recorded Q/KV geometries and `torch_flow` native path. It used six observed
timed repetitions and ten requested warmups under the exact recorded helper protocol.
Total warmup/capture forward invocation counts remain unknown; the timed-work feature is
an empirical proxy for the resolved five-phase execution clock. Preparation remains unqualified,
so no total point, one-pass worker, cold/setup or fleet forecast follows from this pass.

## Frozen provenance

- Inputs: [dense-feature-study.zip](dense-feature-study.zip) and [manifest](dense-feature-study.manifest.json).
- Input member: `dense-feature-samples-876-frozen.jsonl`.
- Input SHA256: `7af4e6b71f2fdca1f5eec5b596e446d0f7727995bf8ec114e1dd96743490be8e`.
- Archive SHA256: `b38c9e9ff5ba9c254fb68776aa959a063bb9e17d4252b1885d8f92deca2978ec`.
- Fit JSON SHA256: `010613c8c5184333254607be4a4df64aaabbee4a7ac3d039a768c08d4370fc08`.
- Fitter: `a8ba2dc16b6a057b3754435c5dbdb9e48df6b667` (public byte-identical equivalent `8666fe23e5d6df47f4113c1c614a6d7460e1b0a3`).
- Fitter source SHA256: `fd9c2f120c23aa1ca71ba728476e81c4d80f99d0dae14d62ad96ca451c13818a`.
- CPU fit used NumPy `2.4.6`; source code and unchanged gates define reproduction.
- Native producer: `c569abb1cd034c934f47974209515e3507f60b8a`; signed full-study audit tooling: `d577af159ec590ad44b492d87eca6a3a0dedc7c7`.
- Observer and driver hashes, actual framework/Torch/CUDA/image metadata, source sample IDs and all predictions/errors remain in the JSON/input archive.
- Smoke tooling has a distinct driver/timer protocol and is not pooled into the full study.

Allocation observations count unique actual storage IDs within each call, including distinct
initial/final vLLM cache buffers. Cache and forward-input storage sets are disjoint in these
records. Observer overhead remains within the recorded instrumented protocol; no subtraction
or uninstrumented-run equivalence is asserted. Neither pass 2 nor a cache footprint proves
compiler warmth or fixed preparation cost.
