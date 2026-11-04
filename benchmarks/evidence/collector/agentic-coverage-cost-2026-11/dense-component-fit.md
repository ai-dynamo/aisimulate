# Conditional dense collection components

The fitter identifies preparation and execution components for the recorded
second native pass in each worker. **Subsequent-in-worker does not prove a
compiler-warm state.** No measured feature snapshot has been fitted here yet;
numerical results will be attached only after an immutable input freeze.

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
