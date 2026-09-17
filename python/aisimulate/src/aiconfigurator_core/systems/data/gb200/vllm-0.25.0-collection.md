# GB200 vLLM 0.25.0 collection

- Published: **17 tables / 420,134 measured rows** for the PR #219 table scope.
- All **350,380 retained cases** were attempted: **339,614 passed / 10,766 failed / 0 unattempted**. A case can produce multiple rows; failed aggregate tasks can retain successful partial rows.
- Failures remain explicit. This is not complete operator support or a whole-model accuracy qualification. Smoke measurements are excluded.
- Full case plans, checkpoint IDs, error logs, and prior retry evidence are retained in `vllm-0.25.0-failures.json.gz`; the adjacent JSON report records source hashes and measurement jobs.

## Runtime and measurement

- Unchanged collector: `cbaf51b64fa460e5ec6146bde407a4c64958212d`.
- vLLM 0.25.0 source: `dd10e03f95f94edbea1975c67ace3a35ec9a8a40`; Torch 2.11.0+cu130; CUDA 13.0; FlashInfer 0.6.13.
- Image: `vllm/vllm-openai@sha256:2f726db9babd627fb59addaed2038577fc767108680901271b069d91759d1286`.
- Cluster / driver: `oci-hsg-cs-001` / `580.126.20`.
- Exclusive nodes, one Slurm launcher task, four GPU workers. Slurm requested high GPU frequency; a hard clock lock was not verified.
- Raw evidence on the listed clusters: `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec/results/vllm025-pr244-multigpu/gb200/production/`.

## Table coverage

| Table | Rows |
| --- | ---: |
| `context_attention_perf.parquet` | 52,836 |
| `dsv4_csa_context_module_perf.parquet` | 5,840 |
| `dsv4_csa_generation_module_perf.parquet` | 1,544 |
| `dsv4_hca_attn_module_perf.parquet` | 3,918 |
| `dsv4_hca_context_module_perf.parquet` | 5,840 |
| `dsv4_hca_generation_module_perf.parquet` | 1,544 |
| `dsv4_paged_mqa_logits_module_perf.parquet` | 3,918 |
| `encoder_attention_perf.parquet` | 7,679 |
| `gdn_perf.parquet` | 10,326 |
| `gemm_perf.parquet` | 148,962 |
| `generation_attention_perf.parquet` | 65,408 |
| `mhc_module_perf.parquet` | 140 |
| `mla_bmm_perf.parquet` | 636 |
| `mla_context_module_perf.parquet` | 13,059 |
| `mla_generation_module_perf.parquet` | 11,400 |
| `moe_perf.parquet` | 72,976 |
| `msa_context_module_perf.parquet` | 14,108 |

## Failed cases

| Run | Failed | Final source job |
| --- | ---: | --- |
| `gemm-fp8_block-00` | 285 | `7211915` |
| `gemm-fp8_block-01` | 285 | `7211916` |
| `gemm-fp8_block-02` | 270 | `7211918` |
| `gemm-fp8_block-03` | 270 | `7211920` |
| `attention_context` | 1,632 | `7212224` |
| `mla_context_module` | 117 | `7212225` |
| `msa_context_module` | 3,460 | `7213456` |
| `moe` | 367 | `7212796` |
| `attention_generation` | 2,228 | `7212228` |
| `mla_generation_module` | 1,848 | `7212229` |
| `gdn` | 4 | `7212230` |

- Hardware: NVIDIA GB200, PCI device 0x294110DE, 189,471 MiB, 1,200 W maximum board power.
- All production measurements use the HSG cluster. BF16, ordinary FP8, and NVFP4 GEMM each completed all 37,518 cases; FP8-block retains 1,110 runtime assertion failures.
- Context and generation attention retain 1,632 and 2,228 assertions rejecting symmetric head dimension 192 in the selected FA4 runtime. MLA context retains 117 runtime failures because the selected FMHA kernel rejects the Q/KV head ratio.
- MoE fresh-worker retry recovered 19 tasks and retained 367 failures. Successful original measurements were preserved; partial rows of retried failed tasks were remeasured, with all prior physical keys retained. Observed runtime failures are recorded by case, not treated as universal hardware limits.
- GDN retains four CUDA grid-y limit failures.
- MLA generation retains 1,848 failures from the selected FMHA kernel's Q/KV head-ratio restriction. All six DSV4 tables, MHC, encoder attention, and MLA BMM completed without failed cases.
- MSA fresh-worker retry recovered 48 cases: 14,108 passed / 3,460 failed, with all original successful rows unchanged. Persistent errors include CSR head-ratio restrictions, Triton compilation failures, and CUDA/worker failures. This retry completed normally without manual worker intervention.

## Validation and limits

- Hashes, B200 reference schemas, finite positive latencies, unique physical keys, complete case IDs, and metadata passed. Four FP8-block shards cover exactly 37,518 retained case IDs.
- All 17 tables loaded through the native engine with strict provenance and shared-layer fallback disabled. Exact measured GEMM and MSA queries passed; the adjacent validation JSON records B200 shape-key differences.
- DSA, MSA generation, compute-scale, communication, and fresh KDA measurements are outside this publication. Older eligible donors can still contribute to predictions.
- All 0.24.0 measurements and metadata remain present. Source-owned reuse policies exclude only old FP8-block GEMM donors measured eagerly with host launch gaps; other eligible donors remain usable. Explicit 0.24.0 primary queries preserve historical data. Query-version defaults are unchanged.
