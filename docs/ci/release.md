<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Release artifacts and qualification

AISimulate 0.13.0 has one product version and exactly two release artifacts:

| Artifact | Build manifest | Public purpose |
| --- | --- | --- |
| `aisimulate` wheel | `python/aisimulate/pyproject.toml` | Application, CLI, estimator SDK, model/performance data, FPM Collector workflow/runtime, Replay, Sweeper, and the unified native runtime |
| `aisimulate-core` crate | `crates/core/Cargo.toml` | Engine-neutral estimator, simulation, and deterministic Replay for Rust consumers |

The external public-API test fixture is `publish = false` and excluded from the
product workspace. Imported AIConfigurator source does not retain another buildable `aiconfigurator`,
`aiconfigurator-core`, or Python `aisimulate-core` manifest. The
`aisimulate` and `aisimulate_core` namespaces live inside the `aisimulate` wheel.
The legacy `aiconfigurator` executable uses `aisimulate.legacy_cli`; the old
Python import namespaces are removed. See the [migration guide](../aic-backward-compatibility/migration.md#python-imports-and-resources).

The pinned-image `collector.sglang_rubin` operation collectors are source-checkout tools and are not wheel payloads. Collection runs from `python/aisimulate/` inside the pinned SGLang image; the wheel includes their qualified Vera Rubin NVL72 data and prediction APIs. This does not change the packaged FPM Collector workflow/runtime listed above. See the [pilot collector instructions](../../python/aisimulate/collector/sglang_rubin/README.md).

`scripts/release/build_release_artifacts.py` validates the manifest set before it
builds and validates the output directory afterward. A release build fails if
an additional wheel, source distribution, or crate appears.

Both artifacts use version `0.13.0`. The wheel builds its native extension from
the same Rust source as the published crate; it does not install a second core
distribution.

For published versions, wheel platform tags, source installation, and internal
nightly consumption, see the [installation guide](../getting-started/installation.md). The product
version in a manifest does not establish publication on an index.

## Packaged license files

The root `LICENSE` and `THIRD_PARTY_NOTICES.md` are the canonical repository
legal files. The wheel build is rooted at `python/aisimulate/`, so exact copies
are retained there and declared as wheel license files by `pyproject.toml`.
Both are installed under the wheel's distribution metadata; the nested copies
do not create a separate licensing boundary. `scripts/ci/check_packaged_legal_files.py`
fails CI if either packaging copy differs byte-for-byte from its root original,
and the release-artifact validator checks the bytes installed in the wheel.

Nightly builds stamp a dev suffix with `scripts/release/apply_dev_version.py` before
building: the wheel becomes `0.13.0.devYYYYMMDD` (PEP 440) and the crate
`0.13.0-dev.YYYYMMDD` (SemVer — cargo rejects the PEP 440 spelling, and the
dotted date is a numeric identifier so pre-release versions order
numerically). The wheel form follows the ai-dynamo/dynamo nightly
convention. The release script accepts only this suffix pair and still
anchors both artifacts to the one product version; any other version shape
fails the build.

The bundled performance database can exceed a registry's default per-file upload limit.
Verify the target index quota and the staged wheel before publication; do not
split the product into another distribution to work around an upload limit.

Full CI uploads `license-artifacts` for trusted PR copies and post-merge pushes
to `main` and `release/*`. It contains `deps.csv` (Python and Rust dependency
versions and licenses), `deps-diff.csv`, and `evidence.json` with the source and
comparison commit identities. PR evidence compares with the validated PR base;
post-merge evidence compares with the push's previous commit. Missing baseline
inputs fail CI. Python inventories resolve runtime requirements at collection
time on amd64/Python 3.12; they are not a wheel payload or a frozen PyPI lock.
An unchanged Python manifest reuses the same resolution for both commits.
The report runs independently of wheel builds and tests, and Full CI Success
requires it. Nightly CI uses the same report generator with its broader
Python/architecture matrix and previous successful scheduled nightly baseline.

## Source layout is not the publication boundary

The combined artifacts deliberately retain stable source subtrees:

- `python/aisimulate/src/aisimulate/` and
  `python/aisimulate/src/aisimulate_core/` mirror AIC Python code and data;
- `crates/core/src/perfmodel/` mirrors the AIC Rust estimator;
- AISimulate-owned facades and native integration stay outside those mirrors.

Keeping those folders separate makes an upstream AIC code, data, or test diff
mechanically path-rewritable while AIC remains active. It does not create a
package boundary: one Maturin manifest collects the Python trees and one Cargo
manifest compiles the Rust trees. See [AIC synchronization](aic-sync.md).

## Nightly CI and FPE qualification

Main branch nightly CI builds the approved release surface: one `aisimulate` wheel per Linux
architecture and one `aisimulate-core` Rust source crate. A changes guard compares
`main` with the last successful scheduled nightly. The build stamps a dev version using the original UTC run-creation date followed
by its zero-padded ten-digit workflow run number, for example
`0.13.0.dev202609170000001234`. Scheduled and manual runs have distinct versions;
retries retain the same version, and later dates sort after earlier dates. Builds
use pinned tooling and record checksums and provenance.

Development nightlies must be available before downstream consumers can validate
and merge an API migration. Pending entries in
[the stable-release migration checklist](../../.github/release-gates.json) therefore
do not block scheduled or approved manual nightlies. Build, compliance, wheel-smoke,
FPE qualification, and security requirements continue to apply. Publish the nightly,
validate and merge the downstream migration against that wheel, then complete the
migration checklist before a stable release.

Toolchain downloads (`uv` and `rustup-init`) retry transient failures up to five
times with a 10-second delay. Each attempt has a 30-second connection timeout
and a 120-second transfer timeout. The retry window is 300 seconds; an attempt
started within that window may finish afterward. Checksum verification remains
mandatory, and checksum, build, and test failures are not automatically retried.

Python dependency licenses are checked in isolated jobs on both architectures
before building or staging. Artifacts are then staged directly to internal
Artifactory through the protected `automated-release` environment. Each wheel
is downloaded again and checked against the build. Immutable copies and their
checksums/provenance are retained as `nightly-dist-<arch>` GitHub artifacts
before runtime dependencies execute, preserving the accuracy and installation
consumer contract.

Python license evidence covers the installed audit environment, including
the runtime dependency closure and audit tools such as `pip` and `pip-licenses`.
Package names are not exempted through a hard-coded ignore list. Detailed
license failures remain suppressed in public job logs; reproduce with
`pip-licenses --with-system` in the affected environment.

Two kinds of validation then run:

- **Wheel smoke tests:** fresh installations on amd64/arm64, each tested with
  Python 3.11, 3.12, and 3.13; dependencies, package identity/version, imports,
  and console commands are checked using the downloaded wheel.
- **FPE Support Matrix (scheduled and approved manual runs):** the amd64 nightly wheel is reused and checked against
  the expected source SHA and checksum. The installed SDK discovers live
  system/backend combinations, then shards native `op_level` evaluation across
  them. Qualification requires complete reports from the same wheel and the
  [required native probes](../../.github/fpe-required-probes.json).

FPE generates deterministic web-matrix artifacts. Its standalone manual mode
builds one wheel for the requested source SHA and uses that same wheel for all
shards. This is native operation-level support qualification; it does not
certify hardware accuracy, backend serving performance, or FPM coverage.

Internal staging uses `nightly/<run_id>/`; staging alone does not certify the
run. The GitLab security handoff requires both smoke tests, FPE qualification,
and license evidence to succeed, and only runs when
`GITLAB_SECURITY_TRIGGER_ENABLED` is enabled. Every rerun also requires approval
recorded for that run attempt through `manual-release-approver`; environment
reviewer protections must be configured to enforce it. FPE output is retained
as workflow artifacts; publishing dashboard pages is a separate Pages workflow
that consumes successful nightlies.

The GitLab request contract was checked against release-automation revision
`cdabacabb50e589c08b97b776b5c2f2644b5473e`, specifically the root
`.gitlab-ci.yml` variable forwarding and `projects/aisimulate.yml` consumer.
The endpoint comes from `GITLAB_PIPELINE_URL`; the authenticated multipart
request selects `ref=main` and forwards these fields:

| Variable | Nightly value |
| --- | --- |
| `PROJECT` | `aisimulate` |
| `PIPELINE_TYPE` / `RELEASE_TYPE` | `security` / `nightly` |
| `NIGHTLY_TAG` | `nightly-YYYYMMDDNNNNNNNNNN-<first-seven-commit-characters>` |
| `WHEEL_VERSION` | Exact stamped wheel version |
| `GITHUB_RUN_ID` / `COMMIT_SHA` | Producing GitHub run and full source commit |
| `AISIMULATE_TOOLING_SHA` | Trusted workflow commit providing the version stamper |
| `SLACK_THREAD_TS` / `SLACK_CHANNEL_ID` | Notification thread and channel, optionally empty |
| `DRY_RUN` | `false` |

The workflow-contract test executes the real trigger script against a fake
HTTP client, including missing credentials and HTTP errors. It validates the
request boundary; it does not certify GitLab scan or publication outcomes.

[Release branch nightly CI](../../.github/workflows/release-nightly-ci.yml) separately
discovers every `release/<version>` branch each day, including new releases and
days when their source is unchanged. From trusted `main`, it records each release
branch's current commit SHA and calls a reusable qualification workflow once per
release. Each builds an unchanged wheel and probes that release's inventory with
the current harness.
The release helper uses the same FPE probe and qualification code from `main`,
with the release's own installed package and model inventory. Its separate
wrapper lets older release branches use current qualification tooling without
modifying their source. It performs wheel identity, import, and dependency
checks needed for FPE; it does not run the main nightly's multi-architecture
smoke suites or package staging.
It records release and tooling commits separately. Releases run sequentially,
with up to 20 system/backend jobs and eight probe threads per job. Versioned
wheel, report, and web artifacts keep releases isolated. A failed release does
not cancel the rest, but Pages requires a successful overall nightly run.
Complete qualified results remain GitHub Actions artifacts for 90 days and
trigger Pages; this job does not publish packages.
Release branches without retained qualified CI evidence appear unavailable.
See the [FPE publication contract](accuracy.md#main-and-release-branches).

To build, stage, qualify, and publish a specific commit, dispatch from `main` and supply a full
40-character SHA reachable from `main` or a `release/*` branch:

```bash
gh workflow run nightly-ci.yml --ref main -f commit_sha=<full-source-sha>
```

An empty `commit_sha` selects `main` at dispatch time. Manual builds require
approval for the current run attempt, always build even when the source is
unchanged, and use current license-check tooling against the selected source's
package manifest. They run both architectures' wheel smoke tests and retain
checksums, source provenance, and license evidence. Staging paths include the
unique run ID, and manual runs neither block the scheduled concurrency group nor
count toward its unchanged-source guard.

Approved manual dispatches run FPE qualification and may trigger the GitLab
public publisher under the same successful-build, license, and FPE gates as the
cron. The selected source supplies the package and model inventory; trusted
workflow tooling supplies version stamping, the artifact contract, and FPE
probes, including for older release sources. Qualification records source and
tooling commits separately and uses the exact staged wheel. The GitLab crate
publisher uses that same pinned version stamper and unique numeric suffix.

The GitLab consumer must support `AISIMULATE_TOOLING_SHA` and the 18-digit
nightly suffix before this producer is enabled. Successful GitHub staging or
handoff alone does not establish that the asynchronous GitLab public publication
completed; inspect its wheel and crate publish jobs.


## Required checks and release approval

The intended required statuses are **`Fast CI Success`**, **`Full CI Success`**,
and **`codeowners`**, bound to the GitHub Actions application, with strict branch
currency. Use the direct Fast result and Full aggregate rather than requiring
each conditional/reusable job. Code review requirements remain independent.

The additive [ruleset payload](../../.github/required-main-checks.json) describes
that policy. Inspect the effective branch rules before relying on enforcement. Committing
the ruleset JSON does not activate it.

```bash
gh api repos/ai-dynamo/aisimulate/rules/branches/main
gh api repos/ai-dynamo/aisimulate/rulesets
```

After validating the workflow on `main`, a repository administrator can apply
the payload. Inspect existing rules first: update a rule with the same name
instead of creating duplicates, and preserve the existing review/CODEOWNER
rules. If the CI rule does not exist, create it with:

```bash
gh api repos/ai-dynamo/aisimulate/rulesets --method POST \
  --input .github/required-main-checks.json
gh api repos/ai-dynamo/aisimulate/rules/branches/main
```

Confirm all three status contexts and strict branch currency in the effective
rules. Maintainer access alone did not permit activation during rollout. Keep
the trusted-copy Full CI backstop until enforcement is verified.

Release staging has a separate control: `automated-release` must exist with
required reviewers before use, and Artifactory credentials belong in that
environment. A reference to a missing environment can create it without the
intended protection. Successful validation does not authorize publication.
