<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Modeled power and energy

Op-level FPE returns operation energy. Native Replay accumulates that evidence
and computes the coverage-gated power summary. Unified prediction and
recommendation preserve and display `power_w` and `power_coverage`; `predict
--detail energy` exposes available breakdown evidence. Whole-forward FPM,
fixed, and polynomial timing remain latency-only in Replay. Whole-forward FPM tables
with measured power (schema 8) publish watts only through the AIC-compatible
static and disaggregated paths; see [whole-forward FPM power](#whole-forward-fpm-power).
EPD's combined report keeps unified power unavailable even when its encoder has
separate analytical power.

The current producers and consumers are
[Replay reports](../../crates/core/src/replay/report.rs),
[power validation/presentation](../../python/aisimulate/src/aisimulate/power.py),
and [Sweeper results](../../python/aisimulate/src/aisimulate/sweeper/result.py).
The [synthetic contract fixture](../../tests/fixtures/power-contract-v1.json)
checks semantics, not measured accuracy.

## Scope and units

`power_w` is the active forward-pass average power of one GPU, in watts. It is
derived from modeled operation energy and modeled active latency. It is not:

- whole-cluster power or the sum across workers;
- wall-plug, CPU, host-memory, networking, storage, or cooling power;
- idle power between forward passes; or
- a measurement of the simulated deployment.

Operation energy uses watt-milliseconds (`energy_wms`). Operation and phase
latency use milliseconds (`latency_ms`). Dividing summed `energy_wms` by summed
`latency_ms` therefore yields watts without a unit conversion.

These values are modeling evidence. They must retain the timing provider,
performance-data identity, operation source, backend/version, hardware, and
role topology needed to interpret them. They are not a replacement for
hardware validation.

## Operation evidence and coverage

For every modeled operation `i`, let:

- `L_i` be its finite, non-negative active latency in milliseconds; and
- `E_i` be its finite, non-negative energy in watt-milliseconds.

An operation is power-covered only when `E_i > 0`. A zero energy value is the
missing-data sentinel; it must not be interpreted as a zero-power operation.
Coverage is weighted by active latency so many short covered operations cannot
hide one long uncovered operation:

```text
total_latency_ms   = sum(L_i)
covered_latency_ms = sum(L_i where E_i > 0)
power_coverage     = covered_latency_ms / total_latency_ms
power_w            = sum(E_i) / total_latency_ms
```

When `total_latency_ms` is zero, `power_coverage` is `0` and `power_w` is
unavailable. Implementations clamp a computed coverage ratio to `[0, 1]` after
validating the inputs; they must reject non-finite or negative evidence instead
of silently repairing it.

## Publication gate

AISimulate uses AIC's fail-closed 90% coverage threshold. A numeric `power_w`
may be published only when every condition below is true:

1. every replay role uses a timing provider that supplies operation-energy
   evidence;
2. total modeled active latency is positive;
3. `power_coverage >= 0.9`; and
4. the resulting power is finite and positive.

Coverage is based on modeled active time, not operation count. If operations
covering 90 ms of a 100 ms forward pass have energy data, `power_coverage` is
`0.90`. Because the threshold is inclusive, exactly `0.90` is sufficient;
`0.899` is not. Below the threshold, JSON output keeps numeric `power_coverage`
and sets `power_w` to `null`, allowing a consumer to distinguish insufficient
data from an unsupported energy path. A provider with an energy channel but
no covered operations therefore reports `power_coverage: 0` and `power_w: null`.

Fixed, polynomial, and forward-pass-metrics (FPM) timing providers give Replay
no operation energy, including FPM tables that carry measured power. A replay
using any of those providers, or mixing an
energy-aware role with an energy-unaware role, returns `power_w: null` and
`power_coverage: null`. The null values never mean zero watts or zero coverage.

## Aggregate and disaggregated deployments

Aggregation always happens on energy and active latency, never by taking an
unweighted mean of already averaged power values.

For an aggregated deployment, all prefill and decode forward-pass invocations
from the aggregated role contribute their operation energy, active latency,
and covered active latency.

For a disaggregated deployment, the prefill and decode roles are combined by
the same sums:

```text
power_w = (prefill_energy_wms + decode_energy_wms)
          / (prefill_active_latency_ms + decode_active_latency_ms)
```

`power_coverage` uses the corresponding covered-latency numerator across both
roles. This is a latency-weighted per-GPU mean across the modeled P+D request
path, not the sum of prefill-GPU and decode-GPU power and not a fleet power
estimate. Role-specific topology and timing provenance remain explicit.

If a configured timing speedup changes modeled active time, both that phase's
energy and latency evidence are scaled by the same factor before aggregation.
This preserves the phase-average power while changing its weight in the
end-to-end result.

Analytical encoder/EPD and attention/FFN-disaggregated (AFD) overlays do not
provide a qualified combined operation-energy report. Their unified power
summary remains unavailable; an encoder-specific field is not a combined
deployment estimate.

## Output contract

Replay JSON and prediction summaries include both fields. Each value is a
number or `null`:

| Field | Unit | JSON value (key always present) |
|---|---|---|
| `power_coverage` | Ratio in `[0, 1]` | Numeric for a supported energy-aware path, including below the gate and at zero coverage; otherwise `null`. |
| `power_w` | W/GPU | Numeric only when `power_coverage >= 0.9` and the other publication conditions hold; otherwise `null`. |

For example, insufficient coverage returns
`{"power_w": null, "power_coverage": 0.75}`. An unsupported energy provider
returns `{"power_w": null, "power_coverage": null}`. Qualifying synthetic
evidence can return `{"power_w": 450, "power_coverage": 0.9}`.

The machine-readable fragment is
[`schemas/power-metrics-v1.schema.json`](schemas/power-metrics-v1.schema.json).
It deliberately permits unrelated report metrics so it can validate both a
replay report and a Sweeper candidate's `metrics` object.

Recommendation results with a valid replay report retain both fields in
`candidates[].metrics` using the same names, units, and null semantics.
`candidates[].provenance.power` repeat both values, including nulls, and
may add evidence metadata such as the method, threshold, role, and source
identities. It must not contain numeric watts when candidate metrics contain
`power_w: null`. Failed attempts without a valid replay report retain the
existing empty-metrics envelope; they have no power summary to validate.

### Normal summaries and optional energy details

Normal human-readable `predict` summaries and each displayed `recommend`
candidate row must always show both `power_w` and `power_coverage` labels,
without requiring `--detail`. Watts are per GPU; display numeric coverage as a
percentage. When a value is unavailable, keep its label and show `unavailable`
with a short reason:

- Qualifying evidence: show numeric watts and coverage.
- Coverage below 90%: show unavailable watts and the computed coverage ratio,
  including `0%` when an energy-aware provider has no covered operations.
- Zero active latency on an energy-aware path: show unavailable watts and
  `0%` coverage, with a reason identifying the absence of active latency.
- Unsupported providers or topologies, including a role without an energy
  channel: show both values as unavailable; do not invent `0 W` or `0%`.

The `predict --detail energy` selector adds phase and per-operation
energy evidence to that normal summary. It must not change summary power,
coverage, or the publication gate. Missing breakdown evidence must carry an
unavailable reason. A detail request cannot promote partial or unsupported
evidence into a qualified summary value. Recommendation details use `predict`
on a saved candidate YAML; this contract does not add `recommend --detail`.

Both JSON keys are mandatory, independent of detail selection. JSON retains
numeric coverage when it can be computed, sets unavailable watts to `null`,
and sets both values to `null` when the provider or topology cannot supply the
required energy evidence. CSV exporters use an empty field for each null
value. Unavailable values must not be serialized as strings or zero watts.
Before validating or serializing a host-language metrics object, producers
must reject `NaN` and positive or negative infinity;
permissive encoder extensions are not valid JSON values under this contract.

## Whole-forward FPM power

Schema-8 whole-forward FPM tables record measured GPU power per row. The
[table contract and measurement method](methods/whole-forward.md#measured-power-schema-8)
describe how a row's `power_w` is measured and turned into operation energy.
This section describes where that power is published.

### Supported paths

In V1, numeric FPM watts come only from `fpm_interpolation` with the SOL
method over a schema-8 vLLM self-benchmark table, on the AIC-compatible
static and disaggregated paths:

- the SDK `InferenceSession.run_static` (`static`, `static_ctx`,
  `static_gen`) and `DisaggInferenceSession.run_disagg`;
- `aiconfigurator cli estimate` with `--estimate-mode static`, `static_ctx`,
  `static_gen` or `disagg`, and the disaggregated rows of the `default` and
  `exp` sweeps, all without `--enable-epd` (EPD rows are
  [withheld](#withheld-or-unsupported)).

Each FPM query contributes `E_i = power_w * L_i`. A query that touched a row
without power contributes `E_i = 0` and is uncovered. Every step of a static
decode trajectory counts as its own query, weighted by its repeat count, so
coverage is the latency share of the queries that touched only powered rows.
The [coverage](#operation-evidence-and-coverage) formulas above apply
unchanged. A disaggregated row weights each worker's watts by its time:
`(prefill_power_w * ttft + decode_power_w * decode_ms) / (ttft + decode_ms)`,
with `decode_ms = tpot * (osl - 1)`.

The SDK applies the 0.9 [publication gate](#publication-gate) to FPM models
itself, so direct SDK callers and the legacy CLI see the same result: below
the gate, `power_w` and the context, generation and end-to-end power getters
return `None`, while latency, energy and the true coverage stay available.
A disaggregated row has no watts when either worker withheld its own.
Op-level SDK outputs are not gated and keep their raw watts and coverage.

Coverage depends on the context length per request. Decode points below
roughly ten times the power-window iteration count in context tokens per
request carry no power, so short-context workloads can fall below the gate
even on a powered table.

A static or disaggregated FPM run over a schema-6 or 7 table now reports
`power_w: None` instead of `0.0`. Its latency is unchanged.

### Withheld or unsupported

These report no numeric FPM watts in V1. Latency is unaffected.

| Path | Result |
| --- | --- |
| Replay and unified `aisimulate predict`, at any attention-DP size | `power_w: null`, `power_coverage: null`, `publication_status: unsupported` with a reason. Tracked in AISIM-2001. |
| Unified `aisimulate recommend` | Candidate metrics carry `power_w: null` and `power_coverage: null`; candidate provenance reports `publication_status: unavailable` with "energy provider or topology unsupported". Tracked in AISIM-2001. |
| `estimate_forward_pass_time_ms()` iteration queries | Latency only; the query has no power output. |
| `fpm_regression` and `fpm_learned` (telemetry-trained), including roles restored from a state snapshot | Latency-only. |
| Legacy aggregate (mixed-step) FPM | `power_w: None` at any coverage; energy and coverage stay on the summary for diagnostics. Tracked in AISIM-2001. |
| Draft-scheme FPM (FPM target with operation-level draft, such as EAGLE-3) | Watts withheld pending qualification. |
| Speculative Option 1 (`draft_length`) and Option 2 (measured SD-round tables) | Latency-only. |
| A multimodal model with `fpm_interpolation.text_only` | Watts withheld whenever the model has encoder operations, including runs without images and the decode worker of a disaggregated run. |
| EPD (`--enable-epd`) with FPM language-model workers (`language_only`) | Watts withheld in E+agg (aggregate rule) and E+P+D, single point and sweep, because the row would mix op-level encoder energy with FPM energy. Single-point rows keep the true coverage. |
| `fpm_interpolation.method: direct` | Energy 0; never covered. |
| Schema-6 and 7 tables, including every table published before schema 8 | `power_w: null`. |
| GLM-5.3-Flash native-median, SGLang native-FPM and native-CUDA (`model_step_cuda`) tables | Latency-only. V1 covers the vLLM self-benchmark only. |
| DeepSeek-V4.1 real-KV and GLM-5.3-Flash prefix-seed adapter cells | Collected without a power window. |
| Decode points of hybrid, sliding-window and Mamba KV-warmup shadow layouts | Power unavailable per point. |

### Collection requirement

The FPM collector measures power by default. A power-measuring plan needs a
Dynamo image whose self-benchmark accepts `--benchmark-power-window-s`, and
fails before model loading on an older image. Opt out with
`--no-fpm-measure-power`, or `collection.measure_power: false` in guided
onboarding; see the
[onboarding guide](fpm-self-service/implementation.md#how-the-collection-grid-is-determined).

### Synthetic contract tests

These tests write synthetic schema-6, 7 and 8 pairs. They check semantics,
not accuracy:

- [`test_fpm_power_reconciliation.py`](../../python/aisimulate/tests/unit/sdk/test_fpm_power_reconciliation.py)
  checks that published static watts equal summed operation energy over summed
  latency at full coverage; that schema-6, 7 and all-null
  schema-8 tables give `None` with unchanged latency; that unpowered low-KV
  decode rows remove exactly their queries' latency from coverage; that Replay,
  `predict` and `recommend` publish no FPM power; that a draft-scheme static
  `cli estimate` withholds watts at full coverage; that aggregate sweeps,
  with and without a draft scheme, finish with no watts in any row or saved
  CSV; and that disaggregated sweep rows weight each worker's watts by its
  time.
- In [`test_fpm_forward.py`](../../python/aisimulate/tests/unit/sdk/test_fpm_forward.py),
  `TestFPMPowerPublication` checks each publication gate, and
  `TestFPMStaticPowerCoverage` checks per-query coverage of static decode
  trajectories, including the 0.9 threshold.

### Measured validation

FPM power accuracy has not been qualified yet. AISIM-1973 ends with a GPU
pipe-cleaning run that exercises the whole chain on a minimal matrix and
reports its errors for information only. Accuracy qualification comes with a
later power-collection campaign. It compares leave-one-out point power per
phase (median error at most 5%, p90 at most 10%) and end-to-end busy-iteration
power against DCGM energy (at most 10% per configuration at three or more load
levels). A configuration that misses these tolerances is left out of the
support matrix. Results go under `benchmarks/evidence/`, not these pages.

## Provenance boundary

At minimum, a consumer must be able to recover the concrete model, hardware,
backend and version, aggregate or disaggregated topology, timing provider per
role, performance-data source/revision, operation source, coverage threshold,
and computed coverage. Unknown runner metadata remains preserved under the
existing `runner_metadata` escape hatch until promoted to a typed field.

Modeled power and measured runtime or telemetry power must never share an
unqualified field. `power_w` is reserved for the modeled active-forward-pass
quantity in this contract. A future measured value must use a distinct name
and provenance that states its measurement boundary.

## Compatibility rules

This is the v1 modeled-power contract. Conforming producers
must emit both keys and use `null` for unavailable values. Legacy reports
produced before this integration may lack the keys; readers may treat that
legacy absence as unavailable, but such reports do not satisfy this schema.
Consumers must handle nulls before numeric formatting, arithmetic, or ranking.
Neither a null nor an absent legacy value may be coerced to zero.

Changing a field name, unit, scope, aggregation formula, missing-value rule, or
the `0.9` gate is a breaking semantic change. It requires a versioned schema
and an explicit converter or coordinated result-schema version bump. Adding
new optional provenance details is backward-compatible when existing meanings
do not change.

Support and accuracy remain separate gates. Execution with sufficient data
coverage establishes that a power result can be produced; parity with AIC and
accuracy against target-hardware measurements require their own qualification.
