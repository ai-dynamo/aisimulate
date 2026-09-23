# AISimulate artifact contract

AISimulate 0.13.0 has one product version and two release artifacts:

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
builds and validates the output directory afterward. The release build
fails if an additional wheel, source distribution, or crate appears in its output.

Both artifacts use version `0.13.0`. The wheel builds `aisimulate._runtime` from
`crates/python/`, whose `aisimulate-python` package is `publish = false`. This
binding depends on the same `aisimulate-core` source as the published crate;
it does not install a second core distribution.

For published versions, wheel platform tags, source installation, and internal
nightly consumption, see the [installation guide](installation.md). The product
version in a manifest does not establish publication on an index.

## Native Dynamo routing in the application wheel

The `aisimulate` wheel includes the Python adapter and native Dynamo routing in
its existing `aisimulate._runtime` extension. There is no separate policy wheel
or second native module. The `aisimulate-python` binding owns the Dynamo
dependency; `aisimulate-core` remains independent of Dynamo. Both are members of
the root Cargo workspace and share its committed `Cargo.lock`, with core as the
default member.

The binding uses public APIs from immutable, merged Dynamo commit
`d9eb42db1168131fdae318eef77255637e4d3495`. Builds require Rust 1.96.1 and use no
local dependency override. Python, binding, core and workspace versions, exact
core dependency pins and local lockfile records remain synchronized and checked
by the release builder and version-stamping tools. The compiled native runtime
also identifies its Dynamo revision and replay API. No `ai-dynamo` Python package
is required for this built-in routing path.

Use the [standard source installation](installation.md#use-current-source) or
[AgentX quickstart](agentx-quickstart.md). A merged source feature is separate
from publication of a wheel containing it; check the actual source revision of
the installed artifact. The legacy `--stack dynamo` provider remains a separate
optional integration with its own compatible distribution.

## Packaged license files

The root `LICENSE` and `THIRD_PARTY_NOTICES.md` are the canonical repository
legal files. The wheel build is rooted at `python/aisimulate/`, so exact copies
are retained there and declared as wheel license files by its `pyproject.toml`.
Both files are installed under the wheel's distribution metadata; the nested copies
do not create a separate licensing boundary. `scripts/check_packaged_legal_files.py`
fails CI if either legal-file copy differs byte-for-byte from its root original,
and the release-artifact validator checks the bytes installed in the wheel.

Nightly builds stamp a dev suffix with `scripts/apply_dev_version.py` before
building: the wheel becomes `0.13.0.devYYYYMMDD` (PEP 440) and the crate
`0.13.0-dev.YYYYMMDD` (SemVer — cargo rejects the PEP 440 spelling, and the
dotted date is a numeric identifier so pre-release versions order
numerically). The wheel form follows the ai-dynamo/dynamo nightly
convention. The release script accepts only this suffix pair and still
anchors both artifacts to the one product version; any other version shape
fails the build.
The same stamp updates the private binding version, its exact core pin, and the
local core/binding records in the shared Cargo lockfile while retaining resolved
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
package boundary: one Maturin manifest collects the Python trees and builds the
binding with the shared Rust core. See [AIC synchronization](aic-sync.md).
