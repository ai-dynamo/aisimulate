<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance model

The performance model estimates forward-pass latency and GPU resource capacity
for a concrete model, hardware, backend, and parallel configuration. [Replay](../replay/README.md)
uses those estimates while simulating request scheduling; [Sweeper](../sweeper/README.md)
compares the resulting deployment metrics. A forward-pass estimate alone does
not predict request TTFT, TPOT, or throughput.

## Choose a timing method

| Method | Inputs | Use it when |
| --- | --- | --- |
| [Op-level](methods/op-level.md) | Registered analytical model, system specification, operation tables and data policy | You need operation-level timing, source diagnostics, or supported analytical estimates. |
| [Whole-forward FPM](methods/whole-forward.md) | Exact-deployment forward measurements and metadata; a resource profile for direct self-service | You have measured forward timings and want to replay the measured workload domain. |
| [Online regression](methods/online-regression.md) | Observed iteration workloads and wall times from one worker identity | A running integration can train and update a role-bound predictor. |

Use `RustForwardPassPerfModel.best_available(config)` in Python or
`ForwardPassPerfModel::best_available(config)` in Rust. Construction resolves one
estimator and records its identity. It does not switch estimators during queries.
A cold regression is not usable for offline prediction.

## Read by task

- [Configuration](configuration.md): identity, estimator selection, fallback, and precision.
- [Support matrix](support-matrix.md): distinguish query coverage, replay support, and accuracy.
- [Memory](memory.md): weight/KV budgets, graph reservations, and grouped cache profiles.
- [FPM self-service](fpm-self-service/README.md): collect or import whole-forward measurements;
  preserve the existing [implementation guide](fpm-self-service/implementation.md)
  and [worked examples](fpm-self-service/examples.md) as the workflow reference.
- [Collector](collector/README.md): operation data, runtime pins, and publication.
- [Extend the model](extending.md): model structure, native operations, and SOL integration.
- [DeepSeek-V4.1](models/deepseek-v41.md): model-specific execution and physical memory contracts.
- [Power](power.md) and [limitations](limitations.md): interpretation and boundaries.
- API reference: [Python](api/python.md), [Rust](api/rust.md).

Operation-specific mechanisms include [DSA context parallelism](methods/context-parallel-dsa.md)
and [DeepEP-LL communication](methods/deepep-ll.md).
