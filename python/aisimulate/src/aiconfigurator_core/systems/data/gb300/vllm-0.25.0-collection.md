# GB300 vLLM 0.25.0 collection

This is an in-progress checkpoint for PR #244. It does not claim complete operator coverage or whole-model accuracy.

- Published checkpoint: 4 tables and 173,705 measured rows.
- Completed full sweeps: 173,705 passed cases, 1,632 failed cases, zero unattempted cases. Failed cases remain in the adjacent compressed failure archive.
- FP8-block GEMM and the other PR #219 operator families are still being collected. Smoke data are excluded.
- Runtime: vLLM 0.25.0, source dd10e03f95f94edbea1975c67ace3a35ec9a8a40; Torch 2.11.0+cu130; CUDA 13.0; FlashInfer 0.6.13.
- Collector: unchanged cbaf51b64fa460e5ec6146bde407a4c64958212d.
- One exclusive node per job; one Slurm task manages four GPU workers.
- Slurm requested high GPU frequency; sampled clocks are retained. A hard clock lock was not verified.
- Validation: hashes, schemas, finite positive latencies, unique physical keys, complete case IDs, strict native provenance, and exact native GEMM queries passed with shared-layer fallback disabled.
- All 0.24.0 data and query-version defaults remain unchanged. The 0.24.0 GEMM donor policy excludes only old FP8-block rows; explicit 0.24.0 primary queries preserve historical values.

| Full sweep | Cluster | Job | Passed | Failed |
| --- | --- | ---: | ---: | ---: |
| gemm-bfloat16 | oci-aga-slurm-1 | 771306 | 37,518 | 0 |
| gemm-fp8 | oci-aga-slurm-1 | 771336 | 37,518 | 0 |
| gemm-nvfp4 | oci-aga-slurm-1 | 771337 | 37,518 | 0 |
| attention_context | oci-aga-slurm-1 | 771366 | 52,836 | 1,632 |
| encoder_attention | oci-aga-slurm-1 | 771370 | 7,679 | 0 |
| mla_bmm_gen_pre | oci-aga-slurm-1 | 771371 | 300 | 0 |
| mla_bmm_gen_post | oci-aga-slurm-1 | 771372 | 336 | 0 |

| Table | Rows |
| --- | ---: |
| context_attention_perf.parquet | 52,836 |
| encoder_attention_perf.parquet | 7,679 |
| gemm_perf.parquet | 112,554 |
| mla_bmm_perf.parquet | 636 |

- Context attention retains 1,632 failed cases: the selected FA4 runtime rejects symmetric head dimension 192. All 52,836 successful cases are published; no shapes were removed from the full plan.

Full case plans, checkpoints, logs, GPU identity, and sampled clocks are retained under the personal cluster storage directory results/vllm025-pr244-multigpu/gb300/production/. Cluster and job identifiers above locate each attempt.
