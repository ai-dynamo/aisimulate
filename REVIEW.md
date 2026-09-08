<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate review contract

Review for behavior and evidence, not just whether the diff looks reasonable.
Start at the changed input, follow the real consumer to the externally visible
output, and identify every contract crossed along the way. Prefer one precise
root-cause finding over several comments on its symptoms.

Use these priorities:

- **P0**: exploitable security issue, data loss, or repository-wide outage.
- **P1**: incorrect public behavior, corrupted or materially wrong prediction,
  deadlock, release breakage, or an unsafe compatibility change.
- **P2**: reachable defect, missing validation, significant performance
  regression, or insufficient evidence for a changed contract.
- **P3**: low-risk maintainability concern with a concrete future failure mode.

Do not report preference-only naming, formatting, or documentation nits already
covered by automated tools. Do not treat bot approval, `MERGEABLE`, CODEOWNERS,
DCO, or a small green check set as merge readiness. Technical findings and
governance status are separate; merge readiness is assessed on the exact head.

## Risk-tiered review and CI

The `review-ready` label is the explicit admission gate for the inexpensive
evidence-gathering stage. Once it is applied, Fast CI and CodeRabbit run in
parallel for every non-draft PR. Medium- and high-risk PRs also require a Codex
review of the same commit. The risk level changes review depth, not merge
authority: every tier still requires the applicable CODEOWNER approval.

| Risk | Review before Full CI | Human merge gate |
| --- | --- | --- |
| Low | Fast CI and CodeRabbit | Applicable CODEOWNER |
| Medium | Fast CI, CodeRabbit, and Codex | Applicable CODEOWNER |
| High | Fast CI, CodeRabbit, and Codex | CODEOWNER plus relevant domain, architecture, security, or release owner |

Fast CI contains quick deterministic checks: source and legal policy, generated
CODEOWNERS integrity, lint, syntax compilation, whitespace, and Rust formatting.
Full CI contains the expensive multi-architecture dependency, Rust, Python,
public-API, build, and release-artifact tests. Dispatch Full CI only after the
required reviews have completed on the current commit with no unresolved P0/P1
finding. Lower-priority findings and CODEOWNER review may proceed while Full CI
runs, but all required conversations, approvals, and exact-head checks must be
complete before merge.

`Fast CI Success` and `Full CI Success` are the stable merge-gate results. Both
run with `always()` semantics and fail when required evidence is missing,
skipped unexpectedly, canceled, or failed. A non-draft PR without the
`review-ready` label fails `Fast CI Success`; making a PR ready or removing the
label retriggers the workflow. Keep the `ready_for_review`, `labeled`, and
`unlabeled` pull-request activity types so those state changes cannot retain a
stale green result. Direct pull-request runs publish `Fast CI Success`; Full CI
displays its reusable invocation as `Fast CI / Fast CI Success` and aggregates
that result into `Full CI Success`. Release staging is explicitly not
applicable to manual and trusted-copy PR validation, while it remains required
for `main` and `release/*` lifecycle pushes. Require the direct `Fast CI
Success` and aggregate `Full CI Success` results in branch rules rather than
individual conditional or reusable-workflow jobs.

During the review-acceleration pilot, a maintainer dispatches Full CI after
verifying those conditions, supplying the reviewed full commit SHA through the
required `expected_sha` input. Trusted copy-pr-bot `pull-request/*` branches also
run Full CI automatically as a temporary coverage backstop while the `main`
ruleset does not require the exact-SHA checks; treat such a run as PR evidence
only after confirming that its copied SHA equals the PR head. Automatic runs on
`main` and `release/*` remain lifecycle validation outside the pre-merge
sequence. Do not remove the copy-branch backstop until the ruleset enforces the
Fast and Full CI checks and requires branches to be current. Do not claim
conditional Codex or post-review Full CI automation until an approved service
credential and exact-head dispatcher are installed.

## Product invariants

- Python describes and orchestrates work. Rust computes per-operation latency,
  energy, and SOL values. Do not introduce a second performance oracle.
- Formulas, table selection, interpolation, quantization, fallback ordering, and
  performance data are product behavior. Changed answers require reproducible,
  explained before/after evidence; never refresh goldens merely to make tests
  pass.
- Public configuration, serialized schemas, CLI options, Python types/imports,
  Rust types/exports, defaults, docs, and fixtures must change together.
- Missing optional data may use an explicitly documented and observable absence
  path. Corrupt, ambiguous, or unsupported input must fail loudly rather than
  silently selecting another model, runtime, backend, kernel, or default.
- Keep provenance exact: distinguish measured values, estimates, proxies,
  synthetic fixtures, parity results, fake-runner results, and production
  traces. Do not turn parity into a predictive-accuracy claim.
- Binary parquet changes require a machine-readable summary of keys, shapes,
  units, row counts, coverage, and anomalies plus proof that a real consumer
  reaches the new data. A clean binary diff is not evidence.
- AISimulate is standalone. Dynamo is a downstream compatibility target, not a
  required runtime dependency. The release surface remains one `aisimulate`
  wheel and one `aisimulate-core` crate unless the artifact contract is changed
  deliberately.
- Only workflows under the repository-root `.github/workflows/` run for this
  repository. Imported workflows below `python/aisimulate/` are provenance, not
  hosted-CI evidence.

## High-value review paths

For schedulers and simulation state, inspect cancellation, preemption, draining,
retry, duplication, empty inputs, terminal events, queue fairness, time units,
and determinism. For performance modeling, compare cold, warm, and sweep paths;
selection rules and data-source precedence deserve the same scrutiny as
formulas. For cross-layer changes, require a test that exercises the final
consumer, not only object construction or an isolated mock.

Ask for the smallest evidence that would disprove the risky assumption: a
negative test, boundary case, exact command and result, explained golden diff,
held-out comparison, or production trace. When that evidence is unavailable,
say precisely what remains unmodeled or unverified.
