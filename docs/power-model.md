# AIC-compatible modeled-power contract

This document makes AIC's existing modeled-power semantics explicit as the
target contract for AISimulate. It is normative for the `power_w` and
`power_coverage` fields planned for the unified `aisimulate predict` and
`aisimulate recommend` paths.

The migration boundary is:

- **Already present, unchanged:** FPE can return per-operation latency and
  `energy_wms`; AIC already derives power from energy over latency, computes
  latency-weighted coverage, and accepts coverage of at least 90%.
- **Added by this PR:** one documented AISimulate definition, a JSON Schema,
  and contract tests so each future consumer uses the same names, units, gate,
  aggregation, and missing-value rules.
- **Added by follow-up PRs:** wiring the existing FPE evidence through Replay,
  aggregating it, and publishing it through prediction, recommendation, and
  diagnostic outputs.

This PR does not change current AIC or FPE runtime behavior, and the contract
does not by itself make modeled power available in unified AISimulate commands.
The [AIC migration guide](cli/migrate-from-aiconfigurator.md) is authoritative
for which workflows are implemented in the current release.

Most semantics below match AIC directly. AISimulate normalizes unavailable
values to JSON `null`: both `power_w` and `power_coverage` are always present
in a conforming summary. A null value means unavailable, never zero watts or
zero coverage. Legacy AIC sentinels such as `0.0` or `NaN` are not published as
numeric power.

The existing implementation references are AIC's
[`InferenceSummary.get_power_data_coverage`](../python/aisimulate/src/aisimulate_core/sdk/inference_summary.py),
its CLI
[`POWER_DATA_COVERAGE_THRESHOLD` and publication gate](../python/aisimulate/src/aisimulate/legacy_cli/api.py),
and FPE's
[`PerOpValue` energy channel](../crates/core/src/perfmodel/engine/runtime.rs).
The machine-readable
[`power-contract-v1.json`](../tests/fixtures/power-contract-v1.json) fixture
provides synthetic, reproducible inputs and expected outputs for aggregate,
disaggregated, exact-threshold, below-threshold, speedup, uncovered, and mixed
provider cases. It is contract evidence, not measured data or an accuracy
claim.

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

Fixed, polynomial, and forward-pass-metrics (FPM) timing providers do not
synthesize energy. A replay using any of those providers, or mixing an
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

Encoder, attention/FFN-disaggregated (AFD), and other future roles follow the
same energy-over-active-latency rule once their unified runtime paths expose
typed evidence. Their presence in the compatibility AIC implementation does
not make them supported by the unified CLI.

## Output contract

Once the follow-up runtime work adds a conforming producer, Replay JSON and
prediction summaries will always include both fields. Each value is a number
or `null`:

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

Recommendation results with a valid replay report will retain both fields in
`candidates[].metrics` using the same names, units, and null semantics.
`candidates[].provenance.power` will repeat both values, including nulls, and
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

The planned `predict --detail energy` selector adds phase and per-operation
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

This is the initial, not-yet-released v1 power contract. Conforming producers
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
