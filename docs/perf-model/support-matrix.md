<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance-model support

Support is scoped to an estimator, model, hardware, runtime version, topology,
precision, worker role, and query domain. Successful construction, successful
queries, Replay feature support, and measured accuracy are separate results.

| Path | Required evidence | Coverage boundary |
| --- | --- | --- |
| `op_level` + `SILICON` | Registered graph and required collected operation data | A model can construct while a later shape lacks data. Analytical-only operations retain their own source labels. |
| `op_level` + `HYBRID` / `EMPIRICAL` | Applicable analytical operations and permitted calibration/transfer slices | A transfer is an estimate, not a newly measured cell. Missing calibration remains an error. |
| `op_level` + `SOL` | Analytical operation and system specifications | Lower-bound modeling does not certify a serving runtime or kernel accuracy. |
| `fpm_interpolation`, `method: direct` | Verified architecture metadata, resource profile, genuine phase measurements | Exact identity and measured brackets; no out-of-domain extrapolation or analytical graph. |
| `fpm_interpolation`, `method: sol` | Registered analytical model and matching forward table/sidecar | Only the implemented SOL transfer domain; not every analytical operation is supported by the FPM SOL evaluator. |
| `fpm_regression` | Valid observed iterations for the selected workload store | Cold or unusable stores return no nonempty prediction. One ready store does not train another. |

See [selection and fallback](configuration.md#selection-and-fallback) and
[whole-forward coverage](methods/whole-forward.md). Explicit invalid configuration
is not converted into a fallback estimator.

## Published FPE matrix

The [FPE Support Matrix](https://ai-dynamo.org/aisimulate/fpe-support-matrix/)
is a **strict-native op-level** probe of `EngineHandle.compile` and representative
forward queries. It does not call `best_available`, enable regression fallback,
or rescue missing data with HYBRID. It is not a matrix of all three estimator paths.

Rows retain model/architecture, system, backend/version, resolved precision,
topology, role, phase, source SHA, package version, and a reproducer. Aggregated
coverage requires prefill, early decode, late decode, and mixed probes; dedicated
prefill requires prefill, and dedicated decode requires both decode probes.
`PASS` requires positive finite latency. `SDK_UNREPRESENTABLE` means the public
builder cannot express the requested identity; it does not authorize probing a
simpler topology. Missing data, incompatible configurations, build failures,
and query failures remain distinct.

The site's branch selector identifies the tested main or release snapshot.
Check its source and qualification evidence instead of treating the page's
publication time as a new test run. Generation, nightly qualification, and
Pages publication belong to [CI accuracy](../ci/accuracy.md).

## Accuracy and replay

The [accuracy dashboards](https://ai-dynamo.org/aisimulate/) publish separate
FPM and end-to-end comparisons. Interpret each result with its measured runtime,
workload, source revision, and timing boundary. Separate passing prefill and
decode engine probes do not establish disaggregated rate matching or queueing.
Use [Replay features](../replay/features.md) for cache, scheduler, workload,
and topology combinations; use the [self-service support boundary](fpm-self-service/README.md)
for profile onboarding.
