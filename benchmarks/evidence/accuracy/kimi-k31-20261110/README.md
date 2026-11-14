<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Kimi K3.1 Pro measured accuracy

These are two separate full-model timing experiments on GB300 TP8. Op uses native SGLang eager CUDA-event timing; FPM uses Dynamo/vLLM native scheduler-wall estimates. They do not establish an apples-to-apples method ranking or a SemiAnalysis serving comparison.

| Method / phase | Paired / selected | MAPE | P95 APE | Max APE |
|---|---:|---:|---:|---:|
| Op / all | 8 / 8 | 45.5272% | 69.6650% | 69.7419% |
| Op / prefill | 4 / 4 | 22.4106% | 42.1176% | 43.4415% |
| Op / decode | 4 / 4 | 68.6438% | 69.7089% | 69.7419% |
| FPM / prefill | 10 / 10 | 8.4120% | 23.3388% | 33.7880% |
| FPM / decode | 16 / 16 | 2.4558% | 11.5408% | 13.0172% |
| FPM / all selected points | 26 / 26 | 4.7466% | 12.5251% | 33.7880% |

**FPM accuracy acceptance: FAILED.** prefill P95 APE is 23.3388% against the original 20% gate. The aggregate MAPE does not override this phase failure. Both formal collections completed successfully; the native measurements remain development calibration data, with every held-out point and outlier retained. Neither the table nor the threshold was adjusted.

Op predictions were frozen before the eight full-model measurements, using a private 20-point GB300 DSA calibration snapshot and existing Op profiles. The later 96-point ARM coverage table is not the calibration snapshot scored here. All eight shapes and all ten retained observations from each TP rank remain in the JSON; the truth is the maximum rank median after three warmup forwards. Native untuned MoE tactic fallback warnings are retained as runtime context, without assigning them as the cause of error.

FPM withholds coordinates from the formal table simultaneously, predicts them from the remaining rows, and retains every selected point and outlier. This assesses direct interpolation at the same GPU/TP/runtime; it does not assess SOL transfer across TP sizes or GPUs. Each phase has one formal collection and one retained native estimate per point. The JSON records the actual internal sample counts and native reducers; no independent repeatability certification has been completed.

`project_reports.py` is an explicit field projection. It does not select points, recompute predictions, fit, or tune. CSV copies coordinates, prediction, truth and signed/absolute percent error from the sealed reports; JSON adds source/frozen/native hashes and scopes. APE is `100 * abs(predicted_ms - measured_ms) / measured_ms`; MAPE gives equal weight to paired points, and P95 linearly interpolates sorted APE at `(N-1)*0.95`. Hashes identify the unchanged private original reports and source artifacts. Hostnames, PIDs, environments, logs and absolute paths are excluded.

To reproduce the projection from the private audit archive, run `python project_reports.py --accuracy-root AUDIT_ARCHIVE --output OUTPUT_DIRECTORY`. The archive must contain the relative report names used in the script. No GPU or package upload is performed.
