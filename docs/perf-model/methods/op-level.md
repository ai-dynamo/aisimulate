<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Op-level timing

Python resolves the model's phase operations, dimensions, precision, and
parallel placement into an `EngineSpec`. Rust loads the resolved performance
database and evaluates that specification without re-entering Python on the hot
path. Compilation produces a versioned operation plan, not a native executable.
Use the [canonical estimator configuration](../configuration.md); low-level
compiled-engine entry points are documented in the [Python](../api/python.md)
and [Rust](../api/rust.md) references.

## Operation composition

An operation can use a collected table, a table interpolation, or its supported
analytical formula. The model graph determines repetition, communication,
overlap, and phase boundaries. Summing arbitrary kernel timings outside that
graph can double-count fused modules or omit dispatch work. The returned
per-operation tuple is `(name, latency_ms, energy_wms, source)`; zero energy
means missing energy evidence, not a zero-power operation.

| Database mode | Meaning |
| --- | --- |
| `SILICON` | Collected-data lookup/interpolation for table-backed operations, including approved shared sources; no HYBRID transfer rescue. Inherently analytical operations retain their own source. |
| `HYBRID` | Try collected data, then an implemented empirical/transfer path for a data gap. |
| `EMPIRICAL` | Use the implemented data-calibrated empirical path. Calibration is still required for table-backed operations. |
| `SOL` / `SOL_FULL` | Analytical speed-of-light paths where implemented; detailed decomposition can have narrower support. |

Data-source inheritance and empirical transfer are different. A row inherited
from an approved measured donor is still collected data. An estimated
cross-shape or cross-precision transfer must retain its transfer provenance.

## Empirical utilization and transfers

The native [utilization estimator](../../../crates/core/src/perfmodel/operators/util_empirical.rs)
models a calibrated operation as

```text
util = SOL(measured_shape) / measured_latency
latency(query) = SOL(query) / util(query)
```

`util` is an effective calibration factor, not a physical efficiency bounded
by one. Categorical/kernel slices are selected before numeric interpolation.
Within a slice, numeric coordinates are normalized in log space; the grid
preserves exact hits, clamps each axis to the measured range, and blends the two
nearest samples with inverse-distance weights (`k=2`, `p=1`). It does not need a
Cartesian measurement grid. No calibration and no permitted donor means an
explicit empirical-coverage error, not `SOL / constant`.

Transfer kinds are admitted by `transfer_policy`:

| Kind | Reference | Correction |
| --- | --- | --- |
| `xshape` | Another collected shape slice with the same precision | Query SOL with the selected reference utilization. |
| `xquant` | Another precision with the same memory/compute profile | Same SOL coefficients; no profile-level correction. |
| `xprofile` | A different memory/compute precision profile | Multiply reference utilization by the query/reference efficiency-level ratio. |
| `xop` | A related operation | Explicit operation-specific level alignment. |

Policies are `off`, `conservative`, `balanced`, and `aggressive`, or the
supported explicit set. The resolved policy is part of configuration and
provenance. No transfer policy turns a missing runtime implementation into
measured support. Windowed-attention fallbacks use a window-aware SOL ratio
only in their implemented empirical path.

## GEMM precision transfer

The [native GEMM operator](../../../crates/core/src/perfmodel/operators/gemm.rs)
uses the query's own SOL cost and a reference utilization:

```text
latency = SOL_query / (util_reference * level(query_profile) / level(reference_profile))
```

GEMM already pools all `(m, n, k)` shapes for one precision, so its `xshape`
reference list is empty. `xquant` uses the first same-profile sibling in table
order. Cross-profile candidates are ordered by compute distance, then memory
distance: a weight-only mode therefore prefers the matching arithmetic family.
MoE keeps its own historical profile-distance ordering. These tie-breaks are
observable prediction semantics, not opportunities to sort data arbitrarily.

`fp8_static` is composite: it needs the base FP8 and its overhead tables.
Transfer reachability alone cannot admit it. Validation requires known
precision profiles and the resolved transfer policy; a pure SILICON request
still needs data. New precision support requires native profile
levels, their exported Python reachability view, and tests described in [Extending](../extending.md).

## Diagnostics and limits

`static_phase_diagnostics` exposes operation timing, sources, measurement
substitutions, and available SOL detail before online correction. Missing SOL
breakdowns carry explicit reasons. The [FPE matrix](../support-matrix.md)
tests strict native coverage; it does not certify the HYBRID ladder or silicon
accuracy. See [DSA CP](context-parallel-dsa.md), [DeepEP-LL](deepep-ll.md),
and [execution limitations](../limitations.md) for specialized boundaries.
