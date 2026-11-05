<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance-model implementation

AISimulate's performance model lives under
[`src/perfmodel`](../src/perfmodel/). Python lowers registered model descriptions
to a versioned operation plan; Rust evaluates that plan with measured or
analytical operations. The canonical forward-pass estimator additionally owns
whole-forward interpolation, online correction, regression, readiness, and
resolved provenance.

Start with the [performance-model documentation](../../../docs/perf-model/README.md):

- [Configuration](../../../docs/perf-model/configuration.md) and [timing methods](../../../docs/perf-model/methods/op-level.md).
- [Python API](../../../docs/perf-model/api/python.md) and [Rust API](../../../docs/perf-model/api/rust.md).
- [Memory](../../../docs/perf-model/memory.md), [support](../../../docs/perf-model/support-matrix.md), and [limitations](../../../docs/perf-model/limitations.md).
- [Extending models and operations](../../../docs/perf-model/extending.md).

For estimator consumers, use `ForwardPassPerfModel::best_available` with the
canonical `ForwardPassPerfModelConfig`. `AicEngineBuilder` is a lower-level
compiled-engine builder. Native construction requires the matching `aisimulate`
wheel and Python feature; explicit regression construction does not.

The exported wire-schema constants define binary compatibility. Recompile
engine plans when updating the paired wheel/crate rather than relying on a
schema number copied into a README. Tests and parity instructions are in the
[core crate](../README.md) and [perfmodel parity suite](../parity_tests/perfmodel/README.md).
