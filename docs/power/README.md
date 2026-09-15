<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Modeled-power qualification

This directory is the release ledger for the AISimulate 0.13 modeled-power
migration. It is deliberately separate from implementation tests: a green test
suite proves only the behavior exercised by that suite, while this ledger says
which product dimensions have qualifying evidence.

The checked-in ledger is **not qualified**. It contains pending evidence slots,
and AFD and EPD are explicitly blocked because those runner integrations are
not available on `main`. Nothing in this directory upgrades optional
runner-supplied metadata, zero-valued timing surfaces, or an empty evidence slot
into a modeled-power support claim.

## Artifacts

- [`qualification-matrix.json`](qualification-matrix.json) is the
  machine-readable release ledger.
- [`qualification-matrix.schema.json`](qualification-matrix.schema.json) is
  the versioned interchange schema for tooling and editors.
- [`../../scripts/validate_power_qualification.py`](../../scripts/validate_power_qualification.py)
  enforces the release-specific matrix and fail-closed evidence rules without
  requiring a schema-validation dependency.

The functional matrix covers the complete cross product of:

- dense and MoE models;
- aggregated and prefill/decode-disaggregated deployments;
- the built-in standard runner and optional Dynamo runner;
- covered and undercovered operation data; and
- the `default`/`op_level` timing path.

The negative matrix covers `default`/`fpm`, `fixed`, and `polynomial` timing for
both runners and deployment modes. Those paths must leave modeled power
unavailable. They must not synthesize `0 W` or reuse measured provenance.

Additional release-blocking gates track operation-level energy/source evidence,
power-data invariants, AIC parity, silicon accuracy, and the single-wheel
packaging contract. AFD and EPD appear as non-blocking, blocked integration
records so their absence remains visible without pretending they belong to the
0.13 support boundary.

Every automated entry labels its command as `available` or `planned`. Planned
commands name the deterministic suites that their dependency issue must add;
they are not executable evidence today and the validator will not allow such a
gate to pass. Manual gates use `not_applicable` rather than a placeholder
command.

## Validate the ledger

Run the structural check locally:

```bash
python scripts/validate_power_qualification.py
python -m pytest -q tests/test_power_qualification.py
```

Fast CI also executes the available data gate and uploads its generated ledger
and report, bound to the exact checkout commit. Reproduce that path from a clean
checkout with the project environment active:

```bash
python scripts/power_qualification_data.py --expected-revision "$(git rev-parse HEAD)"
python scripts/validate_power_qualification.py artifacts/power-qualification/qualification-matrix.json \
  --verify-execution --expected-revision "$(git rev-parse HEAD)"
```

The producer scans every parquet under the repository's data tree, records a
digest of the complete tree, and counts invalid values and paired `0.0/0.0`
unavailable sentinels. It does not locate data through an installed package.
The verifier rescans that same checkout and compares the complete report.
Changing a report's source revision or recomputing its JSON hash cannot replace
execution. Dirty source trees and stale revisions fail. The historical report
is retained for audit history; its gate stays pending in the committed ledger.
Only the generated CI ledger records a result for the current candidate.

Release mode always verifies automated execution. Future automated gates need
a reviewed execution verifier before they can qualify a release; a matching
JSON report alone is insufficient. Ledger commands are descriptive and are
never executed as arbitrary shell commands by the validator.

Run the fail-closed release decision only after every implementation and
evidence-producing job has completed:

```bash
python scripts/validate_power_qualification.py \
  --require-release-ready \
  --expected-revision <40-character-candidate-commit>
```

The second command fails while any release-blocking gate is not `passed`, while
the silicon threshold is unapproved, while `release_state` is not `qualified`,
or when `candidate_revision`, the explicit expected revision, and any passing
release evidence disagree. This is intentional: validating the ledger format
is not the same thing as qualifying a release.

## Attach evidence

For an automated or manual gate to move to `passed`, add at least one evidence
record containing:

1. a result of `pass`;
2. a repository-relative JSON evidence-report path;
3. the artifact's SHA-256 digest;
4. the exact 40-character candidate source commit; and
5. a UTC ISO-8601 recording time.

Failed evidence moves a gate to `failed`; do not delete it merely to make the
ledger green. Pending and blocked entries intentionally have empty evidence
arrays. Evidence should be public-safe and reproducible. Internal workflow IDs,
raw silicon measurements, and mutable branch names are not sufficient release
anchors.

The release-ready validator reads repository-relative artifacts from the
repository root, enforces a 16 MiB size limit, and checks their SHA-256 digest.
Remote URLs are intentionally rejected so a caller-supplied ledger cannot turn
release validation into an outbound network request. Missing, oversized, or
digest-mismatched content fails closed.

Each artifact is a versioned, gate-specific JSON report. Its gate ID, source
revision, matrix, assertion results, units, and anomaly list must agree with the
owning ledger gate. A passing report must cover every assertion and contain no
anomalies, so evidence from one gate cannot be reused to qualify another gate.

At closeout, set the top-level `candidate_revision` to the one integrated commit
that every release-blocking gate tested. Every passing evidence record must use
that same commit, and the release command must receive it independently through
`--expected-revision`. This prevents a plausible-looking ledger from mixing
passing artifacts produced from different or stale code revisions.

The AIC parity gate retains the existing 1% perfmodel relative tolerance and
requires non-zero, covered power fixtures. The silicon MAPE and minimum sample
count remain `null` until the release owner approves them. A universal accuracy
threshold is not inferred from the existing latency dashboard or from a single
hardware/model point.

## Ownership and sequencing

The `depends_on` fields refer to the Linear power migration issues that provide
each gate's implementation. Those dependencies explain sequencing; they do not
replace evidence. When a downstream implementation changes the support
boundary, update the matrix, validator, and documentation in the same review.

The user-facing migration guide remains conservative until the fail-closed
release check passes. See
[`../cli/migrate-from-aiconfigurator.md`](../cli/migrate-from-aiconfigurator.md)
for the currently supported migration boundary.

## Summary availability and evidence output

Every valid power summary must include both `power_w` and `power_coverage`.
Undercoverage uses null watts and numeric coverage; unsupported providers use
two nulls. Default CLI output and `--detail energy` must preserve identical
summary values, with both labels visible. The detail option only adds evidence.
These are required qualification assertions, not a claim that pending runtime
or downstream Dynamo integration gates have passed. See migration section 4.11;
section 4.10 continues to describe prediction details.

The data producer accepts only new output files under
`artifacts/power-qualification`. Use a new run subdirectory when retaining prior
evidence; existing outputs, traversal, and symlinks are rejected. Paired
float64 `0.0`/`0.0` values mean unavailable data; booleans and strings are invalid.
