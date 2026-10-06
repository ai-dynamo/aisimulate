<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1 profile qualification evidence

These results apply to separately collected `full` and `decoder_bounded`
DeepSeek-V4.1-Flash profiles, TP4, DP1/PP1/EP1, GPU-resident Engram, eager
execution, text input and speculation disabled. TP2 and H100 TP4 weight-only
capacity rejection is not a measured OOM or an available FPM profile.

Each retained profile uses 145 calibration and 38 heldout geometries with ten
attempts per point. MAPE equally weights geometry errors against the heldout
medians. Calibration and heldout runs, requests and geometries are distinct.

| System and profile | Explicit-query coverage | MAPE | Maximum geometry error |
| --- | --- | --- | --- |
| GB200 full | 38/38 | 1.18418035% | 14.2899673% |
| GB200 bounded | 38/38 | 1.12313769% | 8.85221426% |
| H200 full | 38/38 | 1.50639143% | 3.66066412% |
| H200 bounded | 38/38 | 2.93239322% | 9.21125404% |
| B200 full | 38/38 | 8.48723299% | 152.99519567% |
| B200 bounded | 38/38 (380/380 attempts) | 2.81290416% | 7.90214539% |

Bounded aggregate queries admit 26/38 geometries: the 12 multi-prefill shapes
require per-request extend lengths. Conditional aggregate MAPE is 1.23254318%
on GB200, 2.71502976% on H200, and 2.68722182% on B200. These conditional
scores are not interchangeable with explicit-query full coverage. The B200
full outliers remain in the result; technical admission does not imply an
accuracy threshold was met.

The measured pool, native precision/runtime routes, memory limits and immutable
dataset/source pins are recorded in the packaged
[profile evidence](../../../python/aisimulate/src/aisimulate_core/systems/profiles/dsv41_fpm/README.md).
Use the [model contract](../../../docs/perf-model/models/deepseek-v41.md) for
current usage. The [source qualification record](https://github.com/ai-dynamo/aisimulate/blob/acaca5d169769b41d4c594d68dd0964c35d0f3bf/python/aisimulate/docs/fpm/deepseek-v41-four-gpu.md)
pins the detailed acquisition and original failed-admission evidence; this
summary does not generalize to different precision, memory placement or loads.
