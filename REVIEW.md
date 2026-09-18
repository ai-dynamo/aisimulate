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

Fast CI runs automatically for every PR, including drafts. CodeRabbit reviews
non-draft PRs automatically, subject to its configured title and label
exclusions. Neither requires a `review-ready` label. Medium- and high-risk PRs
also require a Codex review of the same commit. The risk level changes review
depth, not merge authority: every tier still requires the applicable CODEOWNER
approval.

| Risk | Review before Full CI | Human merge gate |
| --- | --- | --- |
| Low | Fast CI and CodeRabbit | Applicable CODEOWNER |
| Medium | Fast CI, CodeRabbit, and Codex | Applicable CODEOWNER |
| High | Fast CI, CodeRabbit, and Codex | CODEOWNER plus relevant domain, architecture, security, or release owner |

Choose the highest tier warranted by the changed behavior and explain it in the
PR's review map. Low risk covers documentation and mechanical changes with no
runtime or enforcement change. Medium risk covers bounded behavior changes with
known consumers and reproducible validation. High risk includes numerical-model
or performance-data changes, public compatibility, cross-layer architecture,
and changes to security, runner trust, or release authority. A small diff is not
by itself low risk. The responsible CODEOWNER confirms the tier; a bot suggestion
can prompt escalation but cannot waive the required reviews.

### Review handoff and finding disposition

Before marking a PR ready for review, fill in the template's risk rationale,
responsible CODEOWNER, and expert escalation fields. Use the generated root
[CODEOWNERS](CODEOWNERS) and GitHub's requested-reviewer list to route the changed
paths. Name the owning team while an individual reviewer is being assigned;
record the responsible reviewer once that person accepts the handoff. For
changes spanning owners, identify who coordinates the handoff and request the
other affected owners. A GitHub code-owner approval can be satisfied by one of
several listed owners; it does not prove every relevant expert reviewed the
change.

For high-risk work, name the domain, architecture, security, or release owner
and the question requiring their decision. The same person may cover CODEOWNER
and expert roles when qualified; record both roles explicitly. For other tiers,
write N/A or name the expert needed for a specific uncertainty. Route an
unanswered request or a disputed tier/finding through the owning team, then the
repository maintainers if needed. Routine handoffs do not require a particular
maintainer or project lead.

Reviewers should acknowledge an accepted handoff, give an expected review time
or name a replacement, and distinguish **blocking** findings from **follow-up**
suggestions. Authors should answer each actionable thread with the fixing
commit and validation, a reasoned disagreement, or an agreed follow-up issue.
If review or a fix is delayed, update the PR with the owner and next step; do not
treat silence as approval.

Keep blocking findings in GitHub review conversations until corrected or the
reviewer accepts an evidence-backed disposition. P0/P1 findings block Full CI
admission; other required findings may be addressed while Full CI runs but still
block merge. A follow-up is non-blocking only when the responsible reviewer
agrees and the PR links an issue with an owner and scope. Do not hide a blocker
by resolving its thread or moving it to the backlog. After a push, refresh the
reviewed SHA, relevant reviews, CI links, and outstanding findings in the PR;
review completion requires the current head and resolved required conversations.

### CI evidence and admission

Fast CI contains quick deterministic checks: source and legal policy, generated
CODEOWNERS integrity, lint, syntax compilation, whitespace, and Rust formatting.
Full CI contains the expensive multi-architecture dependency, Rust, Python,
public-API, feature-mode, engine-golden, platform-wheel, collector-data,
prediction-regression, build, and release-artifact tests. The FPE support
matrix remains a scheduled/manual product-support audit rather than a PR gate.
`Full CI Success` is required for every admitted PR, but trusted
`pull-request/*` copies select only the components affected by the pull
request's complete changed-file set. Renames classify both the old and new
paths. Documentation and review-policy-only changes may mark every expensive
component explicitly N/A; unknown paths and changes to CI execution contracts
run the complete matrix. Manual, `main`, and `release/*` runs also execute the
complete matrix. The aggregate gate accepts a skipped component only when the
selector explicitly marked that component N/A; missing selection outputs,
unexpected skips, failures, and cancellations fail closed.
The independently maintained mapping oracle in
`.github/full-ci-selection-cases.yml` records the job-consumer rationale and
representative expected plans; Fast CI verifies the implementation against
that complete component inventory.

Dispatch Full CI only after the required reviews have completed on the current
commit with no unresolved P0/P1 finding. Lower-priority findings and CODEOWNER
review may proceed while Full CI runs, but all required conversations,
approvals, and exact-head checks must be complete before merge.

`Fast CI Success` and `Full CI Success` are the stable merge-gate results. Both
run with `always()` semantics and fail when required evidence is missing,
skipped unexpectedly, canceled, or failed. Fast CI checks depend on job results,
not draft status or labels. Opening, updating, reopening, or marking a PR ready
for review triggers Fast CI; label changes do not. Standalone Fast CI runs
publish `Fast CI Success`. Full CI's
`Require Fast CI` job verifies a successful run and all substantive jobs on the
same branch and commit; it does not rerun Fast CI internally. Branch pushes
require matching Fast push evidence; manual Full CI accepts matching Fast push
or manual evidence. The bounded wait and API checks fail closed. Release staging
depends on successful validation and remains a separate protected step on `main` and `release/*`
lifecycle pushes. Waiting for staging approval does not hold `Full CI Success`
open; a green validation result does not certify staging or publication.
Require the direct `Fast CI Success` and aggregate `Full CI Success` results in
branch rules rather than
individual conditional or reusable-workflow jobs.

The additive ruleset payload and runner-image rollout procedure are in the
[CI guide](docs/ci.md#required-checks-and-release-approval). A committed ruleset
payload is not evidence that repository enforcement has been activated.

During the review-acceleration pilot, a maintainer dispatches Full CI after
verifying those conditions, supplying the reviewed full commit SHA through the
required `expected_sha` input. For manual validation, dispatch standalone Fast CI
with the same ref and SHA first, then dispatch Full CI. Trusted copy-pr-bot
`pull-request/*` branches also run Full CI automatically as a temporary coverage backstop while the `main`
ruleset does not require the exact-SHA checks; treat such a run as PR evidence
only after confirming that its copied SHA equals the PR head. Automatic runs on
`main` and `release/*` remain lifecycle validation outside the pre-merge
sequence. Do not remove the copy-branch backstop until the ruleset enforces the
Fast and Full CI checks and requires branches to be current. Do not claim
conditional Codex or post-review Full CI automation until an approved service
credential and exact-head dispatcher are installed.

For an admitted PR, no second Full CI launch is needed after Fast CI. The
trusted `pull-request/*` push starts standalone Fast CI and Full CI automatically,
and every expensive component waits for the exact-branch/SHA Fast CI prerequisite
and scope selector to pass before it can acquire a protected runner. Application
tests additionally build one wheel per architecture and then fan out contracts, unit, integration, CLI-build,
support-matrix, and tool-build shards. Admission itself remains the maintainer security gate;
do not replace it with PR-authored credentials or a `pull_request_target`
workflow.

The [application test inventory](docs/ci.md#application-test-inventory-and-exceptions)
maps collected cases to their Full CI shard and records explicit manual and
optional-dependency exceptions. The contracts shard fails when a collected test
has no assignment.

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
