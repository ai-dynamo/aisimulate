<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.3-Flash vLLM v0.31.0 FPM: evaluation and provenance (GB300)

This directory holds the evaluation, fidelity and provenance records of the five vLLM `0.31.0`
native-FPM calibration tables: fp8-tp2, fp8-tp4, nvfp4-tp2, nvfp4-tp4 and nvfp4-tp1.

The tables themselves (`fpm_forward_perf.parquet` and its sidecar) are external runtime inputs under
the repository policy in `.gitignore`. They are distributed through the FPM Hugging Face dataset.
`package.json` pins every table by SHA256. `source-metadata.json` is the byte copy of each table's
sidecar.

Method, serving configuration, inputs and runs are recorded in each `provenance.json` and in
[the FPM design notes](../glm53flash-fpm.md#vllm-v0310-recollection-stock-runtime).
Holdout predictions use `best_available` with `fpm_interpolation` and fallback `deny`.
`source-recompute.json` is an independent exact-fraction MAPE recomputation. Clean truth is the
vLLM 0.31.0 stock-server holdout, matched at identical executed geometry (221/221 points per deployment).

| Deployment | Phase | n | FPM holdout MAPE (gate 10%) | Max APE | Recompute | MAPE vs clean truth | FPM/clean-truth median [p10, p90] |
|---|---|---|---|---|---|---|---|
| fp8-tp2 | prefill | 144 | 2.16% | 8.60% | 2.158% (match) | 4.47% | 1.002 [0.917, 1.120] |
| fp8-tp2 | decode | 77 | 1.99% | 6.82% | 1.987% (match) | 3.91% | 1.035 [1.005, 1.073] |
| fp8-tp2 | table | 360 rows | `c590d709a0ddc7d0b432f4fc607a6f97a902756d955555ed8e45921e02a9804c` | | | | |
| fp8-tp4 | prefill | 144 | 1.28% | 8.02% | 1.282% (match) | 5.33% | 0.986 [0.916, 1.099] |
| fp8-tp4 | decode | 77 | 1.63% | 5.97% | 1.633% (match) | 3.97% | 1.038 [1.020, 1.092] |
| fp8-tp4 | table | 359 rows | `887b74ff40469e8c39e3839d52203a9b0e65a0810a0425b11987a848f7df6260` | | | | |
| nvfp4-tp2 | prefill | 144 | 3.68% | 13.72% | 3.683% (match) | 5.49% | 1.001 [0.954, 1.127] |
| nvfp4-tp2 | decode | 77 | 1.53% | 5.21% | 1.534% (match) | 5.31% | 1.051 [1.030, 1.101] |
| nvfp4-tp2 | table | 360 rows | `960b3aa187de794fba07e0ee8d4b9583b324a64b5ca5eab435247b7d44958c1f` | | | | |
| nvfp4-tp4 | prefill | 144 | 2.47% | 20.84% | 2.473% (match) | 5.25% | 0.987 [0.934, 1.116] |
| nvfp4-tp4 | decode | 77 | 1.13% | 4.40% | 1.134% (match) | 5.03% | 1.025 [1.010, 1.112] |
| nvfp4-tp4 | table | 359 rows | `0a7d62334768aa61b99fc0b9ad96220eebb1961b0e8c702a642a834f12933f80` | | | | |
| nvfp4-tp1 | prefill | 144 | 2.35% | 12.15% | 2.355% (match) | 4.59% | 1.003 [0.988, 1.101] |
| nvfp4-tp1 | decode | 77 | 1.98% | 8.06% | 1.983% (match) | 3.03% | 1.029 [0.996, 1.063] |
| nvfp4-tp1 | table | 377 rows | `a3d3ad924935474380f8b9cabf25cb51b8b3d3abdaa71de567be837b2ecf5577` | | | | |
