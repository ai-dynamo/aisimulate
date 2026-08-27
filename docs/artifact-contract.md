# AISimulate artifact contract

AISimulate 0.12.0 has one product version and exactly two release artifacts:

| Artifact | Build manifest | Public purpose |
| --- | --- | --- |
| `aisimulate` wheel | `python/aisimulate/pyproject.toml` | Application, CLI, estimator SDK, model/performance data, FPM Collector workflow/runtime, Replay, Sweeper, and the unified native runtime |
| `aisimulate-core` crate | `crates/core/Cargo.toml` | Engine-neutral estimator, simulation, and deterministic Replay for Rust consumers |

The external public-API test fixture is `publish = false` and excluded from the
product workspace. Imported AIConfigurator source does not retain another buildable `aiconfigurator`,
`aiconfigurator-core`, or Python `aisimulate-core` manifest. The preserved
`aiconfigurator`, `aiconfigurator_core`, and `aisimulate_core` namespaces all
live inside the `aisimulate` wheel and therefore do not add artifacts.

`scripts/build_release_artifacts.py` validates the manifest set before it
builds and validates the output directory afterward. A release build fails if
an additional wheel, source distribution, or crate appears.

Both artifacts use version `0.12.0`. The wheel builds its native extension from
the same Rust source as the published crate; it does not install a second core
distribution.

Nightly builds stamp a dev suffix with `scripts/apply_dev_version.py` before
building: the wheel becomes `0.12.0.devYYYYMMDD` (PEP 440) and the crate
`0.12.0-devYYYYMMDD` (SemVer — cargo rejects the PEP 440 spelling). This
follows the ai-dynamo/dynamo nightly convention. The release script accepts
only this suffix pair and still anchors both artifacts to the one product
version; any other version shape fails the build.

The bundled performance database makes the unified wheel about 164 MiB, above
the default 100 MiB per-file upload limit on PyPI and TestPyPI. Before the first
unified release, the release owner must obtain a project-specific upload-limit
increase for `aisimulate` on both indexes and verify the release wheel through
the normal staging workflow. This is a release prerequisite, not a reason to
split the payload into another distribution.

## Source layout is not the publication boundary

The combined artifacts deliberately retain stable source subtrees:

- `python/aisimulate/src/aiconfigurator/` and
  `python/aisimulate/src/aiconfigurator_core/` mirror AIC Python code and data;
- `crates/core/src/perfmodel/` mirrors the AIC Rust estimator;
- AISimulate-owned facades and native integration stay outside those mirrors.

Keeping those folders separate makes an upstream AIC code, data, or test diff
mechanically path-rewritable while AIC remains active. It does not create a
package boundary: one Maturin manifest collects the Python trees and one Cargo
manifest compiles the Rust trees. See [AIC synchronization](aic-sync.md).
