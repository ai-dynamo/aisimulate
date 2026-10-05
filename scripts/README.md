<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CI and maintenance scripts

Group scripts by the workflow responsibility that owns them. Workflows call the
files directly; imports use the `scripts.<group>` namespace. FPM entrypoints use
`python -m scripts.fpm_accuracy.run_fpm_accuracy` and
`python -m scripts.fpm_accuracy.prepare_fpm_measurements` so their sibling
`types/` package cannot shadow Python’s standard library. Run commands from
the repository root unless a script documents another working directory.

| Directory | Jobs and responsibility |
| --- | --- |
| `ci/` | Fast/Full CI selection, repository policy, licenses, documentation links, test inventory, and CI image setup |
| `release/` | Wheel/artifact builds, version stamping, release migration gates, and installed CLI checks |
| `readme/` | README command checks and their reports |
| `e2e_accuracy/` | E2E dataset acquisition, prediction campaigns, and summary generation |
| `fpm_accuracy/` | FPM dataset preparation, prediction campaigns, and shared FPM contracts |
| `fpe/` | FPE support-matrix and release qualification |
| `pages/` | Qualified artifact selection, site assembly, and browser checks |
| `performance/` | Forward/simulation performance selection and wheel provenance |
| `prediction_regression/` | Numerical sentinels and trace replay qualification |
| `notifications/` | Accuracy and review digests |
| `aic_sync/` | Upstream synchronization patch generation and its ledger |

Keep shared code with its owning job family; consumers import it instead of
copying it. Colocated fixtures and small script tests move with their scripts.
Cross-workflow contract tests remain in the repository's `tests/` directory.

## Dependencies

`pyproject.toml` is the source for script dependency declarations. Its standard
`[dependency-groups]` keep tooling out of the installed AISimulate wheel. Jobs
using only the standard library need no dependency group. Campaigns still
install their exact evaluated wheel separately; this manifest must not install
the current checkout in its place.

Hashed `requirements.txt` files are generated per job family for reproducible
`pip` installs. Edit the dependency group, then regenerate its lock; do not edit
the lock manually. For example:

```bash
uv pip compile --group scripts/pyproject.toml:e2e-accuracy --generate-hashes \
  --python-version 3.12 --universal -o scripts/e2e_accuracy/requirements.txt
python -m pip install --require-hashes -r scripts/e2e_accuracy/requirements.txt
```

The other group/output pairs are `fpm-accuracy` → `fpm_accuracy/`, and `ci`,
`pages`, `readme`, and `release` → the same-named directories. Do not use
`uv sync` in a wheel-evaluation environment: it can remove the wheel under test.
Historical release builds retain a fallback to the source revision’s old wheel
builder location. Full CI may resolve a dependency group together with the selected wheel's dev
extra where a joint resolution is required.

The application/runtime dependencies remain in
[`python/aisimulate/pyproject.toml`](../python/aisimulate/pyproject.toml).
