<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Library functions the skills call

All names below were checked with `grep` on this branch under
`python/aisim/src/aisim_core/data/`. Run with that package on the path
(`python/aisim/.venv/bin/python`). Re-grep before relying on a name that
is not listed; if a function you expect is absent, report the gap instead of
re-implementing it.

| Need | API |
| --- | --- |
| Coverage query | `catalog.Catalog(root).rows / empty_cells / stale_cells / unevaluated_cells / worst_cells / systems / hardware` |
| Catalog build and size | `catalog.build_catalog`, `catalog.check_catalog_size` |
| Select and fetch data | `fetch.Selection`, `fetch.plan`, `fetch.fetch` (selectors required; `max_bytes`, `offline`) |
| Slurm GPU time | `gpu_time.parse_sacct`, `gpu_time.gpu_time_from_sacct`, `gpu_time.gpu_time_from_slurm` |
| Kubernetes GPU time | `gpu_time.gpu_time_from_kubernetes`, `gpu_time.capture_gpu_time_kubernetes` (with a `KubeReader`) |
| GPU time model and checks | `gpu_time.GpuTime` (`authority`: measured, declared, unknown), `check_gpu_time(gpu_time, "archival"\|"qualification")`, `gpu_time_ok` |
| Eval record | `eval.EvalRecord`, `PredictionIdentity.build`, `prediction_fingerprint`, `write_truth_table`, `compute_results` |
| Store and read | `eval.write_record` (content-addressed, never overwrites), `load_record`, `current_records` |
| Verify and promote tier | `eval.verify_record`, `eval.attest`, `eval.check_record_gpu_time` |
| Parallelism validity | `registry.evaluate(Context(...))` returns a `Report` (`blocks_import`, `blocks_qualification`) |
| Manifest checks | `aisim_core.data.check_manifest`, `check_dataset` (op-latency protocol only) |
| System spec sync | `systems.check_published_systems`, `systems.publish_system_spec` |

Not on this branch: a Hugging Face publish function and the `aisim data fetch`
CLI. Do not call or imitate them; see the publish skill for what to do instead.
