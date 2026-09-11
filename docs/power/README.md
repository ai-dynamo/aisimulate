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

Run the deterministic structural check in normal CI:

```bash
python scripts/validate_power_qualification.py
python -m pytest -q tests/test_power_qualification.py
```

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
2. an artifact path or HTTPS URL;
3. the artifact's SHA-256 digest;
4. the exact 40-character candidate source commit; and
5. a UTC ISO-8601 recording time.

Failed evidence moves a gate to `failed`; do not delete it merely to make the
ledger green. Pending and blocked entries intentionally have empty evidence
arrays. Evidence should be public-safe and reproducible. Internal workflow IDs,
raw silicon measurements, and mutable branch names are not sufficient release
anchors.

The release-ready validator hashes repository-relative artifacts from the
repository root and downloads HTTPS artifacts before accepting their digest.
Missing content, non-HTTPS redirects, and SHA-256 mismatches fail closed.

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
