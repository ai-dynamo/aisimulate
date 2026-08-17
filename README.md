# AISimulate

AISimulate is the standalone home for GPU-free inference simulation and
deployment configuration. Its three release artifacts do not declare a Dynamo
installation dependency. The imported AIC generator retains its existing,
function-local integration with Dynamo's deployment config modifiers for the
explicit Dynamo-manifest workflow; normal simulation and CLI startup do not
load that optional integration.

Starting with 0.12.0, this repository owns the complete AIConfigurator product
surface—not only its native core. The full AIC application, CLI, generator,
SDK, Collector, tests, documentation, and development tooling are preserved
under `python/aisimulate/`. The native estimator remains split into its
independently consumable wheel and crate.

## Release artifacts

This repository produces exactly three release artifacts:

1. `aisimulate` Python wheel — the complete application and CLI;
2. `aisimulate-core` Python wheel — the estimator SDK, native extension,
   model metadata, and performance data;
3. `aisimulate-core` Rust crate — the native estimator for Rust consumers.

It does **not** publish an `aiconfigurator` wheel or an `aiconfigurator-core`
wheel/crate. The legacy Python import namespaces and the `aiconfigurator`
console command remain compatibility surfaces inside the two AISimulate
wheels for the 0.12 transition.

## CLI transition

The `aisimulate` wheel installs both commands:

```bash
aisimulate cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm
aiconfigurator cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm
```

Both currently execute the complete, proven AIC CLI. The newer
`predict`/`recommend` design must satisfy the tracked AIC-to-AISimulate parity
matrix, product requirements, and approved exceptions before it replaces this
delegation.

## Repository layout

```text
crates/
  aisimulate-core/      Rust estimator and native PyO3 extension
  tests/public-api/     external-consumer compile contract
python/
  aisimulate/           complete AIC application and compatibility CLI
  aisimulate-core/      Python estimator SDK, metadata, and performance data
docs/
  artifact-contract.md  the three-artifact release boundary
  core-api.md           public core API and compatibility contract
  migration.md          AIC-to-AISimulate source and history mapping
scripts/
  build_release_artifacts.py
```

## Build the release set

```bash
python -m pip install build maturin
python scripts/build_release_artifacts.py --output-dir dist
```

The build fails unless the output directory is empty, the manifests describe
only the approved packages, and the final directory contains exactly the three
artifacts above.

For focused development and test instructions, see
[`python/aisimulate/README.md`](python/aisimulate/README.md) and
[`docs/core-api.md`](docs/core-api.md).

## Source provenance

PR #2 retains its earlier path-filtered core ancestry. The complete upper
application is an exact migration snapshot from AIConfigurator `main` commit
`13b5cf2697876692b0a52098266c81162add11fc`; that source SHA remains the
review and future-sync boundary. See [`docs/migration.md`](docs/migration.md).
