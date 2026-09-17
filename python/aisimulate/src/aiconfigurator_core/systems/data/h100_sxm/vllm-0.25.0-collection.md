# H100 SXM vLLM 0.25.0 collection

This is an in-progress checkpoint for PR #244. It does not claim complete operator coverage or whole-model accuracy.

- Published checkpoint: 1 tables and 75,036 measured rows.
- Completed full sweeps: 75,036 passed cases, 0 failed cases, zero unattempted cases. Failed cases remain in the adjacent compressed failure archive.
- FP8-block GEMM and the other PR #219 operator families are still being collected. Smoke data are excluded.
- Runtime: vLLM 0.25.0, source dd10e03f95f94edbea1975c67ace3a35ec9a8a40; Torch 2.11.0+cu130; CUDA 13.0; FlashInfer 0.6.13.
- Collector: unchanged cbaf51b64fa460e5ec6146bde407a4c64958212d.
- One exclusive node per job; one Slurm task manages eight GPU workers.
- Slurm requested high GPU frequency; sampled clocks are retained. A hard clock lock was not verified.
- Validation: hashes, schemas, finite positive latencies, unique physical keys, complete case IDs, strict native provenance, and exact native GEMM queries passed with shared-layer fallback disabled.
- All 0.24.0 data and query-version defaults remain unchanged. The 0.24.0 GEMM donor policy excludes only old FP8-block rows; explicit 0.24.0 primary queries preserve historical values.

| Full sweep | Cluster | Job | Passed | Failed |
| --- | --- | ---: | ---: | ---: |
| gemm-bfloat16 | aws-iad-slurm-2 | 2189 | 37,518 | 0 |
| gemm-fp8 | aws-iad-slurm-2 | 2190 | 37,518 | 0 |

| Table | Rows |
| --- | ---: |
| gemm_perf.parquet | 75,036 |

Full case plans, checkpoints, logs, GPU identity, and sampled clocks are retained under the personal cluster storage directory results/vllm025-pr244-multigpu/h100_sxm/production/. Cluster and job identifiers above locate each attempt.
