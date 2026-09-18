---
description: Canonical performance-model interface and extension requirements.
paths:
  - "crates/core/src/perfmodel/**"
  - "crates/core/src/python.rs"
  - "crates/tests/public-api/**"
  - "python/aisimulate/src/aiconfigurator_core/**"
  - "python/aisimulate/src/aiconfigurator/**"
  - "python/aisimulate/src/aisimulate/**"
  - "python/aisimulate/tests/**"
  - "tests/**"
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# One Performance Model Interface

`ForwardPassPerfModel::best_available(ForwardPassPerfModelConfig)` in Rust and
`RustForwardPassPerfModel.best_available(config)` in Python are the **only
production performance-model construction interface**. All new performance-model
features MUST be exposed through this interface and its returned model. CLI,
Sweeper, Replay, Planner, and other consumers must use the same contract.

- Extend the Rust-owned typed configuration and model API first. Put estimator
  controls in `estimator_config`; keep model, hardware, backend, topology, and
  immutable `worker_type` in the canonical identity. Rust owns defaults,
  validation, selection, and estimation behavior. Python bindings and callers
  pass the complete configuration through without a second set of defaults or
  independent estimator selection.
- Do not add alternative constructors, caller-specific factory functions, or
  separate configuration schemas for new features. Existing low-level engine
  builders, compilation helpers, and compatibility adapters are implementation
  or migration machinery; they must not become a parallel public path for new
  performance-model features. Existing custom timing, AFD, and encoder providers
  do not authorize bypassing this interface for new estimator functionality.
- Preserve the complete resolved identity through candidate construction,
  caching, metadata, replay, and saved configurations. Hardware/version/capacity
  preflight must honor the same role-specific data roots. Resolve every default
  timing role even when another role uses a custom timing provider; unsupported
  combinations must fail explicitly rather than discard settings.
- Defaults are `estimation_mode: auto` and `fallback_policy: deny`. Auto always
  searches `op_level -> fpm_interpolation -> fpm_regression`, including with deny.
  Deny restricts an explicit mode; allow tries it first and then the remaining
  global priority. Invalid configuration must not trigger fallback. An untrained
  regression reports not-ready; offline simulation must reject it. Construction
  pins the selected estimator; queries do not silently switch models.
- Migrate legacy configuration at the input boundary. New configuration must
  preserve its selection and all controls after serialization and reload. Never
  infer an explicit user choice from a serialized legacy default.
- Cover the public Rust/Python contract and affected caller paths with behavioral
  regression tests, including saved-config round trips. Estimation math and
  data-selection changes must also follow [Rust core parity rules](rust-core/parity.md).

See the [Core API](../../../../docs/core-api.md#choosing-a-forward-pass-api) for
the schema and supported controls.
