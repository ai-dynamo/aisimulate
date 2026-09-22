<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Install AISimulate

Use Python **3.11–3.13** in an isolated environment. Simulation runs on the
host CPU; `engine.hardware` selects the GPU being modeled. A GPU serving
environment is required for collection and deployment benchmarks.

## Choose a package and matching documentation

Documentation on `main` describes the current source. A published wheel may
predate a documented feature. Record your installed version before comparing
results or reporting a problem:

```bash
python -c 'from importlib.metadata import version; print(version("aisimulate"))'
python -c 'import aisimulate, aisimulate._runtime; print(aisimulate.__file__); print(aisimulate._runtime.__file__)'
```

### Published packages

As checked on **September 14, 2026**, PyPI publishes `0.12.0.dev1`; the GitHub
`v0.12.0` release is still a draft. This is a dated publication snapshot, not
a promise that `main` is included in that wheel. Check the
[PyPI release files](https://pypi.org/project/aisimulate/#files) and
[GitHub releases](https://github.com/ai-dynamo/aisimulate/releases) for newer
artifacts and use the documentation associated with the selected release.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --pre 'aisimulate==0.12.0.dev1'
aisimulate --help
```

An exact version pin makes the choice explicit. To discover newer published
prereleases, use `python -m pip index versions --pre aisimulate`. Once a stable
release is published, `python -m pip install aisimulate` normally selects a
stable release; `--pre` allows prereleases. Neither command requests the
latest repository source or an internal nightly automatically.

When replacing standalone AIConfigurator, first follow the
[package migration instructions](../README.md#upgrade-from-standalone-aiconfigurator)
in the environment you intend to use. The `aisimulate` wheel owns both console
commands. Python imports use `aisimulate` and `aisimulate_core`; see
[Python source migration](python-source-migration.md) for the breaking import change.

## Platform matrix

| Host | Published `0.12.0.dev1` wheel | Current source wheel-build target |
|---|---|---|
| Linux x86-64 | `manylinux_2_34_x86_64`: glibc 2.34 or newer | `manylinux_2_28_x86_64`: glibc 2.28 or newer |
| Linux ARM64 | `manylinux_2_34_aarch64`: glibc 2.34 or newer | `manylinux_2_28_aarch64`: glibc 2.28 or newer |
| macOS Apple Silicon | `macosx_11_0_arm64` | macOS ARM64, deployment target 11.0 |
| macOS Intel, native Windows, musl-based Linux | No wheel in this release | No corresponding platform-wheel job |

The native wheels use `cp311-abi3`; the complete package still requires
Python 3.11–3.13 and compatible dependency wheels. A build target is not
evidence that an artifact was published or that every OS/dependency combination
was tested. The
[platform-wheel workflow](../.github/workflows/validate-platform-wheels.yml)
defines build validation; the filenames in your selected release define what
you can install. In particular, the glibc 2.28 repair on `main` does **not**
change the already-published `0.12.0.dev1` wheels.

On Linux, check `uname -m` and `ldd --version`; on macOS, check `uname -m` and
`sw_vers`. To check wheel availability without compiling AISimulate:

```bash
python -m pip download --no-deps --only-binary=:all: \
  'aisimulate==0.12.0.dev1' --dest wheel-check
```

This checks only AISimulate's wheel, not all transitive dependencies. A missing
wheel on an unlisted host does not establish source-build support there.

## Use current source

For features documented on `main`, use a source checkout and record its commit:

```bash
git clone https://github.com/ai-dynamo/aisimulate.git
cd aisimulate
git rev-parse HEAD
uv sync --project python/aisimulate --extra dev
source python/aisimulate/.venv/bin/activate
aisimulate --help
```

Install `uv`, a Rust toolchain with Cargo, and a C/C++ compiler plus platform
linker before syncing: Maturin compiles the native extension. On macOS, install
the Xcode Command Line Tools (`xcode-select --install`); Linux builds need
the equivalent compiler and linker tools. The workspace uses Rust edition
2024; the current macOS wheel job pins Rust 1.96.0. See the
[build action](../.github/actions/build-platform-wheel/action.yml) for CI's
toolchain choices and [DEVELOPMENT.md](../DEVELOPMENT.md) for validation.

Current performance data is checked-in Parquet. Git LFS is needed only for
retained legacy text assets and tests that use them. Re-run `uv sync` after
changes to Rust or packaging, and verify the imported paths shown above point
to the intended environment.

## Use an internal nightly

The [nightly workflow](../.github/workflows/nightly-ci.yml) produces a wheel
version such as `0.13.0.devYYYYMMDD`, then stages artifacts to access-controlled
Artifactory through the protected release environment. The run subsequently
checks the downloaded wheel and qualifies its FPE support matrix. Use a
successful completed nightly: a successful build, staging step, or dev suffix
alone does not establish that validation completed. The nightly path is
`nightly/<run_id>/`, not a public PyPI release channel.

Obtain the wheel for your platform from Artifactory using your organization's
authenticated artifact access. Obtain `provenance.json` and `SHA256SUMS.txt`
from the same run's `nightly-dist-amd64` or `nightly-dist-arm64` GitHub artifact.
Verify the recorded source revision and wheel SHA-256, then install that exact
downloaded file:

```bash
python -m pip install /absolute/path/to/downloaded/aisimulate-VERSION-PLATFORM.whl
```

Replace that illustrative filename with the actual wheel filename. Retain the
source SHA, workflow run URL, wheel filename and SHA-256 with your results.
The [artifact contract](artifact-contract.md) describes wheel/crate versioning.

## Optional Dynamo integration

The built-in engine stack does not require Dynamo. There are two optional
integrations with separate packages, router schemas and runtime capabilities.
Explicit `--stack` always wins. Without it, a `router` key after `--set` selects
`dynamo-policy`, while no `router` selects `engine`; the selected adapter then
validates the configuration. A Planner-only YAML needs explicit `--stack dynamo`.
See [execution-stack selection](cli/user-guide.md#choose-an-execution-stack).

### Native Dynamo policy plugin

For `aisimulate predict` with `router.policy: kv_router` and optional
`session` or `sibling_group` affinity, install the separate
`aisimulate-dynamo-policy` plugin together with the matching base `aisimulate`
wheel. This integration supports offline aggregated/P-D vLLM/SGLang prediction,
including AgentX snapshots, warmup and duration profiles. It does not provide
Planner, dynamic scaling, online execution or routing recommendation.

The plugin is currently source-built; the base release and nightly jobs do not
publish it. Follow the [AgentX quickstart installation](agentx-quickstart.md#1-install-from-source)
for the combined source checkout, Rust 1.96.1 prerequisites, clean paired-wheel
build/install commands and optional container. Initial adapter qualification is
Linux x86-64. The builder records source revision and wheel hashes in
`manifest.json`. Both native modules must agree on package version, replay API
and core source digest, and the adapter checks its immutable merged Dynamo
revision. Even equal package versions from different core source trees are
rejected; see the [optional artifact contract](artifact-contract.md#optional-dynamo-policy-wheel).

This plugin imports existing native Dynamo policy APIs without a full
`ai-dynamo` installation or a running Dynamo service. Installing `ai-dynamo`
does not install `aisimulate-dynamo-policy`, and the historical AISimulate 0.12
pairing below is not a compatible substitute for the current paired source
build. Missing or incompatible plugins fail explicitly, without falling back to
round-robin. A source feature being merged does not establish that its matching
plugin wheel is published on a package index.

### Legacy full Dynamo stack

For explicit `--stack dynamo`, install a Dynamo distribution that supplies its
runner and legacy Router/Planner adapters into the same environment. The legacy
Router tuning and recommendation schemas remain separate from the native policy
plugin; keep `--stack dynamo` when predicting YAML saved by that stack's
recommendation run. The verified **source** pairing on September 14, 2026 is:

| Dynamo source | Declared AISimulate dependency | Integration registration |
|---|---|---|
| [`cf944aebb23aafd758ffa2c3fa0ecfdd8804926b`](https://github.com/ai-dynamo/dynamo/blob/cf944aebb23aafd758ffa2c3fa0ecfdd8804926b/pyproject.toml), manifest version `1.5.0` | `aisimulate==0.12.0.dev1` on Python 3.11–3.13 | `aisimulate.runner_factories`, `aisimulate.config_adapters`, and `aisimulate.sweep_config_providers` |

This source manifest does not establish a published or runtime-qualified
package pair. At that date, PyPI's latest `ai-dynamo` release is `1.4.2`, and
`1.5.0` is not published there. For source integration, follow the build and
integration instructions at the matching Dynamo revision; for published
packages, check the selected wheel's dependency and entry-point metadata.
Do not assume an unpinned `pip install ai-dynamo` contains the source pairing
above. Record both package versions and source/build identities; successful
dependency installation alone does not establish adapter compatibility.

The [Dynamo deployment guide](../python/aisimulate/docs/dynamo_deployment_guide.md)
separately explains generated serving artifacts and runtime version pins.
Installing a Dynamo adapter for simulation and launching a GPU serving
container are different steps in that workflow.
