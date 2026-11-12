<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Release artifacts and qualification

AISimulate 0.14.0 has one product version and exactly two release artifacts:

| Artifact | Build manifest | Public purpose |
| --- | --- | --- |
| `aisim` wheel | `python/aisimulate/pyproject.toml` | Application, CLI, estimator SDK, model/performance data, FPM Collector workflow/runtime, Replay, Sweeper, and the unified native runtime |
| `aisim-core` crate | `crates/core/Cargo.toml` | Engine-neutral estimator, simulation, and deterministic Replay for Rust consumers |

The external public-API test fixture is `publish = false` and excluded from the
product workspace. Imported AIConfigurator source does not retain another buildable `aiconfigurator`,
`aiconfigurator-core`, or Python `aisimulate-core` manifest. The
`aisimulate` and `aisimulate_core` namespaces live inside the `aisim` wheel.
The legacy `aiconfigurator` executable uses `aisimulate.legacy_cli`; the old
Python import namespaces are removed. See the [migration guide](../aic-backward-compatibility/migration.md#python-imports-and-resources).

`scripts/release/build_release_artifacts.py` validates the manifest set before it
builds and validates the output directory afterward. A release build fails if
an additional wheel, source distribution, or crate appears.

Both artifacts use version `0.14.0`. The wheel builds its native extension from
the same Rust source as the published crate; it does not install a second core
distribution.

For published versions, wheel platform tags, source installation, and internal
nightly consumption, see the [installation guide](../getting-started/installation.md). The product
version in a manifest does not establish publication on an index.

The current distribution names are `aisim` (Python) and `aisim-core` (Rust).
Python imports remain `aisimulate` and `aisimulate_core`; `aisim` and
`aisimulate` invoke the same CLI. Current release tooling also validates and
qualifies historical `aisimulate` / `aisimulate-core` sources by their committed
manifest identities. Nightly GitLab triggers select the project from the built
wheel identity; renamed builds require release-automation MR !84 before cutover.

## Distribution migration

The renamed `aisim` wheel owns the same Python packages and CLI files as the
historical `aisimulate` wheel. Existing users must uninstall `aisimulate` before
installing `aisim`, including upgrades; mixed installations must be repaired
by removing the old distribution and force-reinstalling the selected new wheel,
or by creating a fresh environment. See the [migration procedure](../getting-started/installation.md#migrate-to-the-renamed-distribution).

Coordinate downstream dependency declarations, especially Dynamo, before the
cutover. A consumer requiring `aisimulate` can reinstall the old distribution
and recreate the overlap. New-name environments require compatible consumers
that declare `aisim`, followed by `pip check`. Historical release qualification
continues using its original distribution in a separate environment.

## Packaged license files

The root `LICENSE` and `THIRD_PARTY_NOTICES.md` are the canonical repository
legal files. The wheel build is rooted at `python/aisimulate/`, so exact copies
are retained there and declared as wheel license files by `pyproject.toml`.
Both are installed under the wheel's distribution metadata; the nested copies
do not create a separate licensing boundary. `scripts/ci/check_packaged_legal_files.py`
fails CI if either packaging copy differs byte-for-byte from its root original,
and the release-artifact validator checks the bytes installed in the wheel.

### Bundled model configuration provenance

Python and Cargo dependency inventories do not describe model configuration files
copied into the wheel. `scripts/ci/bundled_configs.json` records each configuration's
packaged path, SHA256, origin, upstream file URL and immutable commit, modification
status, license, copyright owner and canonical notice reference. The checker
validates the complete referenced license text against its recorded digest;
whitespace-only reflow is ignored. License identifiers are bound to registered
legal-text digests, so a custom agreement cannot accidentally be labelled MIT
without changing the license registry too. Explicit required attribution is checked
in notice prose outside the agreement itself. Canonical attribution remains in the root legal
files and their byte-identical packaging copies.

Fast CI runs `scripts/ci/check_bundled_configs.py` against source files. Full CI,
platform-wheel jobs (including macOS), and nightly builds check the exact repaired
wheel before it is staged or its runtime dependencies execute. These checks do not
import the wheel. New files without an inventory entry, changed bytes without
updated provenance, missing source/license information, incomplete notices, and
unexpected or missing wheel configurations fail validation. Inventory and checker
changes use the repository's existing code-owner review process; metadata is a
declaration to review, not proof of legal clearance.

The enforced scope is configuration files under `model_configs/`, plus JSON model
descriptors with top-level `architectures` or `model_type` found elsewhere in the
packaged source and wheel. This detects a Hugging Face config copied to another
package directory, but does not infer arbitrary copied data's origin from numbers
or detect every possible third-party asset. Project-authored files require an
explicit origin and description; generated files also require recorded inputs.

For an existing file whose provenance has not been individually established,
`origin: legacy_unresolved` and `license: UNKNOWN` preserve that uncertainty in
the report. This exception is limited to exact path/hash pairs in
`scripts/ci/bundled_config_legacy.json`, frozen at the recorded bootstrap commit.
The checker pins the baseline file's digest. Adding a file or editing its bytes
cannot use this exception, even if its inventory hash is updated. Resolving an
entry replaces it with verified source/license metadata; do not label unknown
content Apache-2.0 to make CI pass. The baseline is not an OSRB approval.

To validate and collect evidence locally:

```bash
python scripts/ci/check_bundled_configs.py check --output-dir /tmp/bundled-config-source
python scripts/ci/check_bundled_configs.py extract --check --wheel dist/*.whl \
  --output-dir /tmp/bundled-config-wheel
```

The `extract` subcommand unpacks the detected configurations and legal files
without importing or installing the wheel. Files retain their archive-relative
paths beneath `files/`; `extracted-files.json` records the wheel SHA256 and each
file's size, hash, and kind. Legal files include the wheel's `.dist-info/licenses/`
contents and conventional LICENSE, LICENCE, COPYING, NOTICE, and
THIRD_PARTY_NOTICES filenames. Use a new or empty output directory.

For inspection of an unrecorded or historical wheel, omit `--check`. Extraction
alone does not validate provenance or produce a compliance assessment. An
explicit `--inventory` can include declared configurations outside the usual
model-config locations. With `--check`, the same invocation applies the existing
source/wheel policy and writes the reports below; a failed check leaves no
extracted output. Existing invocations without a subcommand still mean `check`.

CI uses `extract --check` and retains only `bundled-deps.csv`,
`bundled-provenance.json`, and `extracted-files.json` as report artifacts. The raw
configuration and license copies remain in the runner's temporary directory.

The generated `bundled-deps.csv` uses the existing OSRB columns
`dependency_type,name,version,spdx_license`; each config has type
`bundled_model_config` and a content SHA256 as its version. Content changes
therefore appear in the existing nightly added/removed diff. The accompanying
`bundled-provenance.json` retains the upstream revision, complete metadata,
unresolved entries, and the exact wheel SHA256. Custom licenses keep their
`LicenseRef-*` identifiers and are not silently classified as permissive. These
checks establish inventory coverage and notice integrity, not contractual
compliance or automatic license approval.

Nightly CI uploads each architecture's report and includes both in the combined
`license-artifacts/deps.csv` and diff. Missing architecture reports fail evidence
generation. Historical source builds without their own inventory can use an
explicit trusted inventory with `--allow-subset`: every configuration actually
present must still match a catalog path/hash and its notices. A historical file
with different or unrecorded bytes fails; backport its provenance rather than
silently skipping the check.

Nightly builds stamp a dev suffix with `scripts/release/apply_dev_version.py` before
building: the wheel becomes `0.14.0.devYYYYMMDD` (PEP 440) and the crate
`0.14.0-dev.YYYYMMDD` (SemVer — cargo rejects the PEP 440 spelling, and the
dotted date is a numeric identifier so pre-release versions order
numerically). The wheel form follows the ai-dynamo/dynamo nightly
convention. The release script accepts only this suffix pair and still
anchors both artifacts to the one product version; any other version shape
fails the build.

The bundled performance database can exceed a registry's default per-file upload limit.
Verify the target index quota and the staged wheel before publication; do not
split the product into another distribution to work around an upload limit.

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

Main branch nightly CI builds the approved release surface: one `aisim` wheel per Linux
architecture and one `aisim-core` Rust source crate. A changes guard compares
`main` with the last successful scheduled nightly. The build stamps a dev version using the original UTC run-creation date followed
by its zero-padded ten-digit workflow run number, for example
`0.14.0.dev202609170000001234`. Scheduled and manual runs have distinct versions;
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
is downloaded again and checked against the build. Verified wheel/crate bundles and their checksums/provenance are stored in
Artifactory. Actions retains `nightly-dist-<arch>-manifest` completion manifests
before runtime dependencies execute; FPE and accuracy jobs download and verify
the same bundle through the shared wheel-storage action.

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
  Before probing, discovery resolves each required probe's `current` version
  through the selected source's installed SDK and checks it against that source's
  inventory. The resulting plan binds concrete versions to the source, tooling,
  and wheel identities; packaging validates that plan and records its versions in
  `fpe-qualification.json`. A passing older version cannot replace a failing
  required version. Release branches resolve their own `current` versions.

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
`cc63ecf80165209f630d4a5b3868983e980c6cc2` (release-automation MR !84), specifically
the root `.gitlab-ci.yml` variable forwarding and `projects/aisim.yml` consumer.
The endpoint comes from `GITLAB_PIPELINE_URL`; the authenticated multipart
request selects `ref=main` and forwards these fields:

| Variable | Nightly value |
| --- | --- |
| `PROJECT` | Built wheel identity: `aisim`, or `aisimulate` for historical source |
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


## Intermediate wheel storage

Use `.github/actions/wheel-storage` for CI, nightly, accuracy, FPE, and simulation wheel transfers. The action accepts `aisim`, historical `aisimulate`, and Dynamo wheel bundles without choosing a package by filename prefix. It preserves the original relative directory layout and companion provenance/requirements files. Reports and dashboard artifacts continue using normal Actions storage.

Each logical handoff name (for example `application-test-wheel-arm64`) produces an Actions artifact named `<name>-manifest`, containing only `wheel-manifest.json`. The manifest is published after every payload has passed an Artifactory read-back check. It binds source SHA, workflow SHA, repository ID, producer run/attempt, architecture, storage repository, scope/PR identity, relative filenames, sizes, and SHA-256 digests. The path uses the actual wheel source revision, independently of the workflow revision:

| Handoff category | Artifactory path inside the selected repository |
| --- | --- |
| Real PR | `pr/<pr-number>/<source-sha>/<run-id>/<attempt>/<handoff-name>/<sha256>/<relative-path>` |
| Non-PR preview | `preview/<source-sha>/<run-id>/<attempt>/<handoff-name>/<sha256>/<relative-path>` |
| Official main/release/nightly | `ci-wheels/<github-repository-id>/<run-id>/<attempt>/<handoff-name>/<sha256>/<relative-path>` |

CI propagates its verified PR target identity; manual simulation comparisons pass their selected PR number even when dispatched from main, and base/head source revisions retain distinct paths. Accuracy previews propagate the verified caller's PR number when present. A standalone FPE qualification provides an explicit `pr_number` for PR storage; dispatches on `pull-request/<N>` require the matching `N` and reject an omitted or mismatched value before transfer. Non-PR experiments use `preview` rather than a fabricated PR number. Consumers require the same expected scope and PR identity before any authenticated request. The official namespace is retained unchanged by this migration.

The numeric repository ID remains authenticated in every manifest and survives a GitHub repository rename. Downloads bind the caller's expected source SHA, architecture and producer run, then use the **producer attempt recorded in the manifest**. A consumer-only rerun therefore reuses the successful original build; it never guesses a path from its own `github.run_attempt`. Cross-run accuracy consumers also verify the existing nightly provenance. Old wheel-containing Actions archives are not selected as new handoffs; when no manifest exists, accuracy builds the exact selected source. Missing objects, checksum/size mismatches, wrong producers, and path escapes fail before a dependent install can proceed. Upload retries reuse verified objects; publication remains incomplete until the manifest upload succeeds.

### Environment prerequisites

Set repository variable `WHEEL_STORAGE_URL` to `https://artifactory.nvidia.com/artifactory`. Define `WHEEL_STORAGE_REPO` at the **environment level**, with no repository-wide fallback:

| GitHub environment | `WHEEL_STORAGE_REPO` | Used by |
| --- | --- | --- |
| `pr-wheel-staging` | `sw-dynamo-aisimulate-pr-pypi-local` | Copied-PR CI, PR accuracy previews, and simulation comparisons (including manual comparisons launched from main) |
| `automated-release` | `sw-dynamo-aisimulate-pypi-local` | Main/release CI, nightly builds, release FPE, and official accuracy/FPE consumers |

Full CI selects the environment from the workflow ref: `main` and `release/*` use `automated-release`, while PR/source refs use `pr-wheel-staging`. Standalone FPE uses the same rule unless an explicit `pr_number` selects PR storage, including a manual dispatch from main. E2E accuracy previews use PR storage; published accuracy campaigns and their cross-run nightly downloads use official storage. The manifest records the producer's storage repository, and downloads reject a repository mismatch before making an authenticated request.

Before enabling transfer jobs:

- Use the existing `ARTIFACTORY_TOKEN` from each job's selected GitHub environment. PR jobs receive the `pr-wheel-staging` token and repository; post-merge/release jobs receive the separate `automated-release` token and repository. The selected token must support read plus deploy/write for uploads (including read-back) and read for downloads under that repository's applicable `pr/**`, `preview/**`, or official `ci-wheels/1330347449/**` namespace. No new secret names or delete/admin permission are required. Existing final-publication configuration remains unchanged.
- Preserve branch admission: `automated-release` admits `main` and `release/*`; `pr-wheel-staging` admits trusted `pull-request/*` refs and the main/release refs used to launch manual PR comparisons. A manual PR comparison remains in PR storage even when its workflow is dispatched from main.
- Retain objects in **both** repositories for at least as long as their Actions manifests (30 days by default; FPE/accuracy handoffs use seven days). Expired objects require a producer rebuild, not a fallback to a different wheel.

The action never forwards credentials to redirects. The existing environment-scoped credential is passed only to the transfer step, and the manifest pins the producer's storage repository before any download. No credential or environment configuration is changed by the workflow code. Rollout should exercise one copied-PR build, a consumer-only rerun, and a main-branch nightly/accuracy reuse before closing infrastructure validation.

### Wheel producers and retained outputs

Persistent project and Dynamo wheel handoffs use Artifactory, including companion requirements, crate files, checksums, and build provenance in the same bundle. A wheel built, installed, and discarded within one job does not need a storage handoff or release credentials. Its result artifacts must select report directories rather than its build, dist, workspace, or virtual-environment directories.

| Workflow | Wheel generation or consumption | Retained output / disposition |
| --- | --- | --- |
| `ci.yml` | Application test wheel built once; application tests and application verification reuse it; protected final staging reuses that original wheel | Artifactory `application-test-wheel-<arch>`; Actions completion manifest and reports; no duplicate wheel payload |
| `ci.yml` same-job validation | Rust feature modes, Python compatibility, engine golden regression, and release artifact contract build/install locally | Disposable local wheel/build directories; only test/contract reports are retained |
| `validate-platform-wheels.yml` | Docker Linux amd64/arm64 and native macOS invoke `build-platform-wheel` | Disposable local `wheelhouse`; Actions provenance and installed-CLI JSON only |
| `nightly-ci.yml` | Trusted release builder produces wheel/crate/checksum/provenance bundle; staged wheels are downloaded and smoke-tested | Artifactory `nightly-dist-<arch>`; Actions completion manifest and metadata reports |
| `fpe-support-matrix.yml` | Standalone exact wheel build or nightly reuse; discovery, matrix shards, and packaging use the qualified wheel | Artifactory `fpe-qualified-wheel`; matrix/web reports remain Actions artifacts |
| `fpe-release-qualify.yml` | Unmodified selected release built with Maturin/auditwheel; generator and qualifier reuse its wheel/probe-plan bundle | Artifactory `fpe-release-wheel-<release>`; qualification and web reports remain Actions artifacts |
| `release-nightly-ci.yml` | Discovers pinned release refs and calls `fpe-release-qualify.yml` | Covered by the reusable release-FPE handoff; no independent wheel copy |
| `fpm-accuracy-branch.yml` | Exact-source nightly reuse or fresh manylinux build; campaign uses one shared wheel | Artifactory `fpm-accuracy-wheel-<key>`; results and comparisons remain Actions artifacts |
| `fpm-accuracy.yml` | Wrapper invokes the branch workflow and selects manifest-backed nightly reuse | Covered by FPM branch handoffs; measurement metadata remains in Actions |
| `e2e-accuracy-branch.yml` | Exact-source nightly reuse or fresh manylinux build; both InfX shards and AgentX shards consume its wheel | Artifactory `e2e-accuracy-wheel-<key>`; datasets, encrypted results, checkpoints, previews, and public summaries remain Actions artifacts |
| `e2e-accuracy.yml` | Wrapper resolves evaluated source/dataset and invokes the branch workflow | Covered by InfX/AgentX branch handoffs; only completion manifests qualify nightly wheel reuse |
| `simulation-performance.yml` | Builds AISim plus `ai_dynamo_runtime` and `ai_dynamo` wheels, requirements, lock, and build provenance | Entire base/head bundles in Artifactory; comparison results remain Actions artifacts |
| `performance.yml` | Builds base/head wheels into separate temporary dist directories, installs and compares in the same job | Disposable local dist/venv paths; only the separate benchmark results directory is uploaded |
| `prediction-regression-gate.yml` | Source install can build a local wheel before producing snapshots | Disposable install wheel; Actions gate snapshots and comparison reports only |
| `collector-check.yml` | Source install can build a local wheel | Disposable install wheel; cross-backend `.md`/`.json` reports only |
| `readme-checks.yml` | Source/development profiles perform editable installs in isolated workspaces | Disposable build/venv paths; Actions command logs, installation metadata, environment log, and harness reports only |
| `accuracy-digest.yml` | No product wheel generation; consumes qualified accuracy data | Actions preview, report, and delivery-state data only |
| `fast-ci.yml` | No product wheel producer; dependency installs/tests may create temporary fixture wheels | Actions source provenance only; fixtures are disposable |
| `pages.yml` | No product wheel producer; dependencies/tests and allowlisted site packaging | Pages/site and validation data only |
| `codeowners.yml` | No wheel generation | No wheel handoff |
| `slack-review-digest.yml` | No wheel generation | No wheel handoff |

The platform Dockerfile's legacy `aiconfigurator` migration-fixture wheel remains inside the local test build stage. The shared manylinux builder's unrepaired wheel is temporary; only its repaired caller-selected output enters a retained handoff. Simulation transfers include every wheel in the bundle, not just the AISim filename. No exported Actions or BuildKit cache is used as a wheel handoff; ordinary dependency-install caches do not change the producer/consumer storage contract.
