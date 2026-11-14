<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Install AISim

Use Python **3.11–3.13** in an isolated environment. Simulation runs on the
host CPU; `engine.hardware` selects the GPU being modeled. A GPU serving
environment is required for collection and deployment benchmarks.

## Choose a package and matching documentation

Documentation on `main` describes the current source. A published wheel may
predate a documented feature. Record your installed version before comparing
results or reporting a problem:

```bash
python -c 'import aisim; print(aisim.__version__)'
python -c 'import aisim, aisim._runtime; print(aisim.__file__); print(aisim._runtime.__file__)'
```

### Published packages

The following example pins release `0.12.1`. Use its
[versioned documentation](https://github.com/ai-dynamo/aisim/tree/v0.12.1). Check the
[PyPI release files](https://pypi.org/project/aisim/#files) and
[GitHub releases](https://github.com/ai-dynamo/aisim/releases) for newer
artifacts and use the documentation associated with the selected release.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'aisimulate==0.12.1'
aisim --help
```

An exact version pin makes the choice explicit. To discover newer published
prereleases, use `python -m pip index versions --pre aisim`.
`python -m pip install aisim` normally selects a stable release; `--pre`
allows prereleases. Neither command requests the latest repository source or
an internal nightly automatically.

When replacing standalone AIConfigurator, first follow the
[package migration instructions](../../README.md#upgrade-from-standalone-aiconfigurator)
in the environment you intend to use. The `aisim` wheel owns both console
commands. Python imports use `aisim` and `aisim_core`; see
[Python source migration](../aic-backward-compatibility/migration.md#python-imports-and-resources) for the breaking import change.

## Platform matrix

| Host | Current source wheel-build target |
|---|---|
| Linux x86-64 | `manylinux_2_28_x86_64`: glibc 2.28 or newer |
| Linux ARM64 | `manylinux_2_28_aarch64`: glibc 2.28 or newer |
| macOS Apple Silicon | macOS ARM64, deployment target 11.0 |
| macOS Intel, native Windows, musl-based Linux | No corresponding platform-wheel job |

The native wheels use `cp311-abi3`; the complete package still requires
Python 3.11–3.13 and compatible dependency wheels. A source build target does
not establish which artifacts were published or which dependency combinations
were tested. The [platform-wheel workflow](../../.github/workflows/validate-platform-wheels.yml)
defines build validation. Check the filenames of the release you select;
source build changes do not alter previously published wheels.

On Linux, check `uname -m` and `ldd --version`; on macOS, check `uname -m` and
`sw_vers`. To check wheel availability without compiling AISim:

```bash
python -m pip download --no-deps --only-binary=:all: \
  'aisimulate==0.12.1' --dest wheel-check
```

This checks only AISim's wheel, not all transitive dependencies. A missing
wheel on an unlisted host does not establish source-build support there.

## Migrate to the renamed distribution

Current source builds `aisim` / `aisim-core`; historical published releases use
`aisimulate` / `aisimulate-core`. Changing a manifest does not publish the new
names. Use a verified `aisim` wheel or wait for its availability on your chosen
index before migrating.

The Python distributions install overlapping package files and console scripts.
Do not install them together. Prefer a fresh environment; for an existing
engine-only environment, remove the old distribution first, then install the
selected new wheel (or `aisim==<published-version>`):

```bash
python -m pip uninstall -y aisimulate
python -m pip install /absolute/path/to/aisim-VERSION-PLATFORM.whl
python -m pip check
aisim --help
aisimulate --help
```

AISim checks distribution metadata during import and CLI startup. If both `aisim` and `aisimulate` are installed, it raises an `ImportError` with repair commands before loading the native runtime. This check does not prevent pip from installing or overwriting shared files. An old distribution installed last can overwrite the legacy entry points and bypass the check through those old paths.

Use that same ordering for upgrades. If both distributions have already been
installed, uninstall `aisimulate` and use `pip install --force-reinstall` for
the selected `aisim` wheel, because uninstalling the old distribution can remove
files listed in both RECORDs. Both CLI names remain available after migration;
Canonical Python imports are `aisim` and `aisim_core`; `aisimulate` and `aisimulate_core` remain forwarding aliases.

Dynamo and other consumers must change their **distribution dependency** to
`aisim` before sharing the new environment. Do not migrate the historical Dynamo
pair documented below in place; its dependency still selects `aisimulate`.
Use a compatible downstream release and run `pip check` after installing the
complete environment. Skipping dependency installation is not a substitute for
updating that requirement.

## Use current source

For features documented on `main`, use a source checkout and record its commit:

```bash
git clone https://github.com/ai-dynamo/aisim.git
cd aisim
git rev-parse HEAD
uv sync --project python/aisim --extra dev
source python/aisim/.venv/bin/activate
aisim --help
```

Install `uv`, a Rust toolchain with Cargo, and a C/C++ compiler plus platform
linker before syncing: Maturin compiles the native extension. On macOS, install
the Xcode Command Line Tools (`xcode-select --install`); Linux builds need
the equivalent compiler and linker tools. The workspace uses Rust edition
2024; the current macOS wheel job pins Rust 1.96.0. See the
[build action](../../.github/actions/build-platform-wheel/action.yml) for CI's
toolchain choices and [DEVELOPMENT.md](../../DEVELOPMENT.md) for validation.

Current performance data is checked-in Parquet. Git LFS is needed only for
retained legacy text assets and tests that use them. Re-run `uv sync` after
changes to Rust or packaging, and verify the imported paths shown above point
to the intended environment.

## Use an internal nightly

The [nightly workflow](../../.github/workflows/nightly-ci.yml) produces a wheel
version such as `0.15.0.devYYYYMMDD`, then stages artifacts to access-controlled
Artifactory through the protected release environment. The run subsequently
checks the downloaded wheel and qualifies its FPE support matrix. Use a
successful completed nightly: a successful build, staging step, or dev suffix
alone does not establish that validation completed. The nightly path is
`nightly/<run_id>/`, not a public PyPI release channel.

Obtain the wheel for your platform from Artifactory using your organization's
authenticated artifact access. For new runs, download the same run's `nightly-dist-amd64-manifest` or
`nightly-dist-arm64-manifest` Actions artifact. Its `wheel-manifest.json` identifies
the verified wheel, `provenance.json`, and `SHA256SUMS.txt` objects in Artifactory.
Fetch them with authenticated artifact access and verify the recorded sizes and
SHA-256 digests. Older runs retain these files in `nightly-dist-<arch>` archives;
see the [handoff contract](../ci/release.md#intermediate-wheel-storage).
Verify the recorded source revision and wheel SHA-256, then install that exact
downloaded file:

```bash
python -m pip install /absolute/path/to/downloaded/aisim-VERSION-PLATFORM.whl
```

Replace that illustrative filename with the actual wheel filename. Retain the
source SHA, workflow run URL, wheel filename and SHA-256 with your results.
The [artifact contract](../ci/release.md) describes wheel/crate versioning.

## Optional Dynamo integration

The engine stack does not require Dynamo. For `--stack dynamo`, install a
Dynamo distribution that supplies the required runner and adapters into the
same environment. The root [README](../../README.md#with-dynamo) records a validated pair: `ai-dynamo==1.7.0.dev20261028` and
`aisim==0.13.0.dev202610240000000082`, from the NVIDIA prerelease index.
Installation, `pip check`, CPU prediction, an eight-trial recommendation, and
prediction of the best candidate pass with Python 3.12 on Linux.

The AISim wheel corresponds to source
[`f3a3e3156530574f5e7a10446f19e571b82db871`](https://github.com/ai-dynamo/aisim/commit/f3a3e3156530574f5e7a10446f19e571b82db871)
and [nightly run 40076769417](https://github.com/ai-dynamo/aisim/actions/runs/40076769417).
The Dynamo wheel is the October 28 nightly build.
This is a README simulation qualification, not a claim that Dynamo's entire
nightly pipeline passed. Full CI README checks retest the installed artifacts
when relevant files change.

Keep this environment separate from a current AISim source checkout.
Dynamo pins an exact AISim nightly; substituting TOT violates that package
contract. Public Dynamo 1.4.2 lacks the modern stack registration, and 1.5.0 uses
removed AISim imports when forced alongside TOT. Do not bypass dependency
resolution with `--no-deps` or independently upgrade AISim in this pair.

The [Dynamo deployment guide](../sweeper/deployment-generation.md)
separately explains generated serving artifacts and runtime version pins.
Installing a Dynamo adapter for simulation and launching a GPU serving
container are different steps in that workflow.
