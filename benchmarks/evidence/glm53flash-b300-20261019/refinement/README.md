<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# B300 small-input coverage refinement

The expanded table adds 243 disjoint native calibration cases to the original
359 unchanged measurements: 427 prefill and 175 decode cases in total. The
native producer remains pinned to AISimulate
`8d021d5a41f6f47fd3d4c8064be6361b7c8b3c85` and Dynamo Python instrumentation
`54960177085413259859c88bd34ed0734d4c2ea9`. The coherent Python/Rust consumer stays
at `c9e7363bf5983b980fe6622f7b064ee7adff5f5d`. Runtime configuration, core
interpolation code and thresholds are unchanged.

The original measurements show observed PIECEWISE execution at or below 64
scheduled prefill tokens and NONE above 64. The new grid fills small-input gaps
and measures both 64 and 68 tokens, respecting the native four-token alignment.
`boundary-diagnosis.json` is based on observed per-measurement dispatch metadata,
not a mode inferred from timing. Every new prefill request is align-4.
Unaligned request lengths are outside this validation study.

The 68 new cases were frozen before collection and are disjoint from both
calibration grids and the original holdouts. New validation supports
53/53 prefill cases with mean absolute percentage error
2.6830% (maximum 10.6763%), and
15/15 decode cases with mean error 1.1002%
(maximum 3.6872%). The frozen 10% per-phase
mean-error gate result is `true`.

On the 45 identical new prefill cases supported by both tables,
mean error changes from 76.7040% to
2.4991%. `validation.json` separately reports coverage
for all new cases, so unsupported baseline cases cannot disappear into an average.

The original failed validation is preserved in the parent directory and the
original HF commit `4dc9dda1ccc5f6b778c0bbea5138b28237082b2f`. Results on those
old holdouts are now diagnostic: their failures informed refinement and some
shapes were remeasured as calibration. Their old timing observations were never
added to the table. No new holdout timing was used for calibration.

Each new point has five warmups and ten measurements, and a separate full-grid
engine launch repeats the 243 new calibration points. All valid slow samples
are retained. Source-inclusive two-launch variation is diagnostic and does not
replace complete canonical fresh-population qualification.

Admission remains `development`; normal materialization is blocked. Finite CPU
replay and downloaded-file hash verification are recorded separately. These
results do not establish matched serving accuracy, cache-reuse accuracy, 1M
context coverage or a supported DynoSim dependency pair. GPUs were released.

A preliminary alignment-invalid attempt was stopped before measurements. Its
resource use counts within the unchanged two-hour deadline; its logs are
retained locally. The corrected frozen policy and input hashes are included.
