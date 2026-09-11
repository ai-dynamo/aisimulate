<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CI test inventory

The active workflows are in the repository-root `.github/workflows/`.
Fast CI validates policy, workflow/selection contracts, lint, formatting, and
syntax. Full CI runs the compiled, installed-package, and numerical validation
components selected by `scripts/select_full_ci.py`. Manual, main, and release
runs select every Full CI component.

## Application tests

Full CI builds one application wheel per architecture, installs that wheel in
each shard, and reuses it for final wheel verification. These paths are relative
to `python/aisimulate/`, except the explicitly identified repository-root suite.

| Tests | Full CI assignment |
| --- | --- |
| Repository-root `tests/` | `contracts`, entire suite |
| `tests/cross_package/` and the CLI compatibility test | `contracts`; cross-package tests also run on Python 3.11 and 3.13 |
| `tests/unit/` and `tests/golden/` | `unit`, entire directories including tests without markers |
| `tests/integration/` | `integration`, entire directory against the installed native wheel and packaged performance data |
| Build-marked `tests/e2e/cli/` | `cli-build`; recommendation runs separately to avoid parallel resource contention |
| Build-marked `tests/e2e/support_matrix/` | `support-matrix` smoke tests |
| Build-marked `tests/e2e/tools/` | `tools-build`; installed FPM verification reuses the same wheel |

The integration shard includes configuration-adapter estimates, memory
estimation, configuration picking, and TRT-LLM KV-capacity tests. Its entire
directory is selected without a marker filter, so newly added integration
modules cannot silently fall outside the shard.

The contracts shard runs `scripts/check_application_test_inventory.py` against
actual pytest collection and uploads `application-test-inventory-<arch>`.
Every collected case has a shard or a documented manual destination. Unknown
test categories, unexplained collection skips, and collection errors fail the
inventory check. This is collection/assignment evidence; passing test execution
is established by the corresponding shard results.

## Explicit exceptions

`.github/application-test-inventory.json` names the exceptions individually:

* Five extended CLI suites retain manual non-build coverage: API equivalence,
  the model/system/backend compatibility sweep, static estimates,
  estimate-versus-default comparison, and the non-build experiment cases.
  These are broader compatibility qualification, beyond the PR build subset.
* Three Collector tensor suites may skip collection when real PyTorch is not
  installed: DeepSeek V4 MegaMoE workload, helper MoE distribution, and SGLang
  MoE EP routing. Run them manually in a Collector development environment with
  real PyTorch. Their absence is recorded as a skip, never as a test pass.

Run the extended compatibility suite from `python/aisimulate/` with:

```sh
python -m pytest tests/e2e/cli -m 'not build'
```

The existing integration tests can also report fixture-specific runtime skips;
the integration shard displays their reasons. Collection assignment does not
turn unavailable fixture coverage into a successful evaluation.

## Other validation and ownership

Full CI also owns Rust workspace/public-API/feature checks, engine goldens,
collector-data validation, prediction regression, platform wheels, and release
artifact contracts. FPE support-matrix generation remains scheduled/manual;
Artifactory staging remains on trusted main/release lifecycle runs.

AIC-1931 owns this execution inventory and selection. AIC-1911 owns enforcement
of the stable Fast/Full results. AIC-1916 owns the separate combined Model Data
Quality Gate; the collector and prediction jobs do not complete that issue.
