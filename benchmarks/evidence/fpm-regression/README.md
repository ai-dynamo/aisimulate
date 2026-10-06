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

The original [measurement protocol and configuration definitions](https://github.com/ai-dynamo/aisimulate/blob/acaca5d169769b41d4c594d68dd0964c35d0f3bf/docs/fpm-lazy-regression-validation.md)
remain pinned provenance. CPU timings describe the measured macOS host and
timed prediction/update loop, not Linux end-to-end serving performance.
Current behavior is documented under [online regression](../../../docs/perf-model/methods/online-regression.md).
