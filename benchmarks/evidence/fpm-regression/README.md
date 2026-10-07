<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Regression comparison inputs and results

The JSON files are preserved, byte for byte, from source revision
`acaca5d169769b41d4c594d68dd0964c35d0f3bf`. They record paired regression
configuration measurements on integrated source `5a7e86dd`; this is not a
comparison of two different executables and not a current default recommendation.

- [FPM Gym results](fpm-lazy-gym-results.json)
- [Captured-workload results](fpm-lazy-capture-results.json)

The Gym dataset is pinned to
[`5487a4599a7fbc012c07bcd3699754bdf4a8bef7`](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/tree/5487a4599a7fbc012c07bcd3699754bdf4a8bef7).
Prediction precedes each update. Missing predictions are excluded from error
scores and reported through coverage; common-support scores use the same points.
Captured-workload settings were selected during exploration, not an untouched
holdout. Features, retention, constraints and update policy vary together, so
CPU differences cannot be attributed to the lazy-update gate alone.

## Configurations and scoring

The Gym JSON stores `default_estimator_config` and
`signed_lazy_estimator_config`, together with source, binary, and harness
hashes. The capture JSON stores `resolved_configurations_by_group`,
`canonical_default_semantics`, `methods`, and input fingerprints. These are
the settings and inputs for the retained comparison, not aliases for today's
defaults.

Gym includes all 15 registered cases with measurements and excludes five
without measurements. Full-stream scoring has no warmup exclusion. Its final
30% suffix is scored after the 70% prefix, with updates continuing through the
suffix; it is not a frozen-model holdout. Gym macro MAPE weights cases equally.
Capture role macros first average captures within each of six backend/workload
groups, then weight groups equally. Its primary comparison uses common
prediction support after the first ten input rows. The two protocols must not
be pooled into one MAPE.

## CPU measurement boundary

Timings measure process user plus system CPU around native prediction/update
loops on an Apple M5 Pro macOS desktop. Parsing, model construction, dispatch
preparation, diagnostics, serialization and destruction are outside the timed
region.
Fresh models are used for each pass; short inputs repeat passes to target at
least 100 ms per timing block. Twelve matched rounds balance configuration
order. Original-row weights count observations once regardless of repetitions;
reported changes are medians of within-round ratios, not ratios of medians.

Controlled runs share a fixed bucket hasher for reproducible retention
tie-breaking. The Gym JSON's `production_randomized_crosscheck` separately
records unmodified-build behavior. The timing intervals describe variation
between those rounds, not population confidence intervals or Linux end-to-end
serving performance.

Current behavior is documented under [online regression](../../../docs/perf-model/methods/online-regression.md).
