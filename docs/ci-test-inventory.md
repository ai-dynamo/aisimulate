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
each shard, and verifies that same wheel concurrently with the test shards. These paths are relative
to `python/aisimulate/`, except the explicitly identified repository-root suite.

| Tests | Full CI assignment |
| --- | --- |
| Repository-root `tests/` | `contracts`, entire suite on four pytest workers |
| `tests/cross_package/` and the CLI compatibility test | `contracts`; cross-package tests also run on Python 3.11 and 3.13 |
| `tests/unit/` and `tests/golden/` | `unit`, four disjoint groups covering entire directories including tests without markers |
| `tests/integration/` | `integration`, entire directory against the installed native wheel and packaged performance data |
| Build-marked `tests/e2e/cli/` | `cli-build`, four disjoint groups; recommendation runs serially in group 1 |
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

## Ten-minute Full CI target

The complete-matrix run at `3fc9a13f` took 14m58s, compared with 36–39 minutes
on the unsharded companion branches. These are individual observed runs, not a
latency guarantee. Queue time, cold builds, and shared runner capacity count
toward the ten-minute target.

Application tests now use 12 jobs per architecture (24 total). The four unit
and four CLI partitions use pytest-split's `least_duration` algorithm over the
same complete collection; every group is scheduled, including unmarked unit
cases. Each partition uses four pytest workers. No test workloads or assertions
are reduced. The repository contract suite also uses four workers; recommendation
E2E remains serial within group 1. The installed wheel and its dev dependencies
use uv's installer to reduce repeated setup time.

Wheel verification starts as soon as the application wheel exists. Protected
staging still depends on the validation jobs. The Rust embedding job prepares
its Python runtime and Rust test binaries concurrently in separate Cargo target
directories, requires both builds to succeed, then runs the tests.

The informational data sanity scan uses four processes for independent
system/op tables. The parent collects every group's fingerprints before running
cross-system and cross-op comparisons, preserving the serial report ordering
and all detector coverage. FPE support qualification remains a separate
nightly/manual audit; it is not included in this Full CI latency target.
