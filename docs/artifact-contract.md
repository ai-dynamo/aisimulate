# AISimulate artifact contract

AISimulate 0.13.0 has one product version and two base release artifacts:

| Artifact | Build manifest | Public purpose |
| --- | --- | --- |
| `aisimulate` wheel | `python/aisimulate/pyproject.toml` | Application, CLI, estimator SDK, model/performance data, FPM Collector workflow/runtime, Replay, Sweeper, and the unified native runtime |
| `aisimulate-core` crate | `crates/core/Cargo.toml` | Engine-neutral estimator, simulation, and deterministic Replay for Rust consumers |

The external public-API test fixture is `publish = false` and excluded from the
product workspace. Imported AIConfigurator source does not retain another buildable `aiconfigurator`,
`aiconfigurator-core`, or Python `aisimulate-core` manifest. The
`aisimulate` and `aisimulate_core` namespaces live inside the `aisimulate` wheel.
The legacy `aiconfigurator` executable uses `aisimulate.legacy_cli`; the old
Python import namespaces are removed. See the [migration guide](python-source-migration.md).

`scripts/build_release_artifacts.py` validates the manifest set before it
builds and validates the output directory afterward. The base release build
fails if an additional wheel, source distribution, or crate appears in its output.

Both artifacts use version `0.13.0`. The wheel builds its native extension from
the same Rust source as the published crate; it does not install a second core
distribution.

For published versions, wheel platform tags, source installation, and internal
nightly consumption, see the [installation guide](installation.md). The product
version in a manifest does not establish publication on an index.

## Optional Dynamo policy wheel

`aisimulate-dynamo-policy` is a separately installed adapter, built from
`python/aisimulate-dynamo-policy/pyproject.toml`. Its native crate at
`crates/dynamo-policy/` is `publish = false`, excluded from the core workspace,
and has a separate lockfile. The base wheel and crate have no Dynamo dependency.

The supported source build uses `scripts/build_dynamo_policy.py` to produce
matching base and adapter wheels from one checkout. Python package versions,
Rust package versions and exact dependency pins must agree; both native modules
must report the same core source digest and serialized replay contract. The
adapter imports the existing public APIs of immutable, merged Dynamo commit
`d9eb42db1168131fdae318eef77255637e4d3495`, without a local override. Both
wheels build with Rust 1.96.1 and committed Cargo lockfiles. The builder verifies
wheel metadata and legal files and records source identity and artifact hashes
in `manifest.json`. Archive/container builds require an explicit source SHA and
record source cleanliness as unknown because Git metadata is absent.

This adapter is source-built until matching wheels are published. The existing
base release and nightly jobs continue to emit their two base artifacts; they
do not publish the adapter. The optional Full CI job builds and installs a
matching pair, then exercises native policy lifecycle and real YAML CLI tests.
See the [AgentX quickstart](agentx-quickstart.md) for installation and supported
routing configurations.

## Packaged license files

The root `LICENSE` and `THIRD_PARTY_NOTICES.md` are the canonical repository
legal files. The base and optional adapter wheel builds are rooted at their
respective Python package directories, so exact copies are retained in both
and declared as wheel license files by each `pyproject.toml`.
Both are installed under the wheel's distribution metadata; the nested copies
do not create a separate licensing boundary. `scripts/check_packaged_legal_files.py`
fails CI if either packaging copy differs byte-for-byte from its root original,
and the release-artifact validator checks the bytes installed in the wheel.

Nightly builds stamp a dev suffix with `scripts/apply_dev_version.py` before
building: the wheel becomes `0.13.0.devYYYYMMDD` (PEP 440) and the crate
`0.13.0-dev.YYYYMMDD` (SemVer — cargo rejects the PEP 440 spelling, and the
dotted date is a numeric identifier so pre-release versions order
numerically). The wheel form follows the ai-dynamo/dynamo nightly
convention. The release script accepts only this suffix pair and still
anchors both artifacts to the one product version; any other version shape
fails the build.
The same stamp updates the optional adapter's package versions, exact base pins,
and local package records in both Cargo lockfiles while retaining resolved
third-party dependency versions.

The bundled performance database makes the unified wheel about 164 MiB, above
the default 100 MiB per-file upload limit on PyPI and TestPyPI. Before the first
unified release, the release owner must obtain a project-specific upload-limit
increase for `aisimulate` on both indexes and verify the release wheel through
the normal staging workflow. This is a release prerequisite, not a reason to
split the payload into another distribution.

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
