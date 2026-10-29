<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR534 review and parent compatibility validation

Implementation `160ff6e941107b0805f70be2fb8b1a6bed5aa7d3` integrates PR533
head `f2fa38494c77f3ce428552096613338d2cf8859f` into PR534. It omits default
selector controls from serialized configuration and rejects nondefault controls
unless `fpm_learned.adaptation` is `causal_selector`. It also adds the missing
native diagnostics stub and documents the selector's SDK contract.

## Compatibility

The [native-reader probe](compatibility/README.md) loads both actual release
extensions in separate processes. The PR533 binary is bound to its existing
validation receipt; the candidate is freshly built from the implementation above.
The [result](compatibility/result.json) records binary, source, and input hashes.

- All 21 applicable candidate-normalized configurations are accepted by PR533.
  The nine standalone learned cases produce identical prediction trajectories
  and exported artifacts.
- All 27 candidate configurations round-trip exactly. Each nondefault selector
  control is retained. PR533 explicitly rejects the six causal-selector cases,
  which exercise a feature absent from that version.
- Runtime checkpoints resume in both directions. Each origin saves four fitted
  rows and one pending label; both readers then process the remaining 251 labels.
  All 256 query predictions, exported artifacts, and semantic diagnostics match
  the uninterrupted reference exactly. Timing and capacity-based memory counters
  are excluded from diagnostics comparisons.
- The runtime probe uses 10,000 historical rows compressed into 9,455 groups,
  one mutable tree out of 150, a seven-row recent window, and an observation age
  limit of nine. Both directions preserve the frozen prefix and finish with
  64 publications, 249 evictions, seven retained rows, and no pending labels.
  Saved schema-v5 checkpoint files are unchanged by loading or continuation.

## Regression coverage

The [validation receipt](validation.json) contains exact commands and log hashes
for the Rust FPM, Python SDK/API, caller, and frozen parity suites. The new Rust
tests cover each selector field across incompatible modes, default compatibility,
custom causal settings, and disabled speculation. Combined-model regressions
exercise global expiry with pending and fitted rows, rejected feedback, and the
inherited frozen-prefix cache under eager/lazy publication and grouped/raw history.
Flushes preserve calibration and selector evidence. Python tests cover early
validation, native/SDK serialization, seed identity, and the diagnostics stub.

Golden/parity artifacts, Cargo manifests, and lockfiles are unchanged. PR534
retains PR533's fallback and early engine/CLI/Sweeper capability guards. No new
dependency or runtime-checkpoint schema is introduced.

## Cost and scope

The selector fixes add a four-scalar default comparison at configuration
validation and serialization. They do not add work to prediction or feedback.
The merged online fitter is identical to PR533's version; its separate
[cost evidence](../fpm_pr533_review_v1/README.md) remains attributable to that
implementation. The combined regression tests exercise its frozen-prefix cache.

These are local macOS correctness and compatibility checks. They do not establish
fresh serving-workload accuracy or a combined-model speedup. Historical selector
accuracy and cost results retain the limitations stated in their original receipts.

## Diagnostic and export contract follow-up

Implementation `437a0777` addresses CodeRabbit's later export comment and
Cross-Layer Contract warning. The canonical reference now documents all seven
serialized `diagnostics().source` values. The learned-model guide and SDK
docstrings clarify that exporting a frozen forest excludes admitted calibration
feedback and correction stores; an online forest export excludes fitting state
and pending rows.

The [follow-up validation receipt](review-followup-validation.json) records
189 passing Python SDK/API tests and 283 passing Rust FPM tests. Added assertions
exercise `learned_online` through fitting and checkpoint continuation, and
`learned_causal_selector` through real expert switches and configuration reload.
The frozen export regression verifies a 10 ms seed, a 20 ms calibrated prediction,
and a 10 ms exported/reloaded prediction with empty correction state, while the
original model remains calibrated. Disabled calibration is covered as well.

The tests use the release extension built from merged revision `747458b0`.
The follow-up changes documentation, SDK docstrings, and tests only; Rust source
and dependencies are unchanged, and the SDK's executable Python AST is unchanged.
The receipt pins the extension and source hashes. This evidence is separate from
the earlier review and compatibility receipts above.

## Replay schema and shared documentation follow-up

Implementation `126ba627` addresses the remaining Cross-Layer Contract warning
from merged PR533. Replay's global and worker timing schemas and Sweeper's
search schema now advertise only their four supported estimator modes. A
before-validator retains the explicit learned-mode capability error. Shared
configuration and support references document learned source requirements,
explicit selection, and the standalone full-iteration API boundary.

The [Replay contract receipt](replay-contract-validation.json) records 904
passing caller tests and 189 passing SDK/API tests. Coverage includes validation
and serialization schemas, all supported modes through configuration round trips
and compiler serialization, normalized role controls, and learned rejection at
public and raw Replay boundaries. The standalone learned-model tests still pass.
The receipt includes generated schema excerpts, source hashes, and the unchanged
native extension hash. Caller tests include mocked paths; this is local contract
validation, with no new serving-accuracy, performance, or parity claim.
