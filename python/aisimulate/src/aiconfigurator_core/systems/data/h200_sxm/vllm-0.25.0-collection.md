# H200 SXM vLLM 0.25.0 collection

- Published: **17 tables / 353,658 measured rows** for the PR #219 table scope.
- All **294,391 retained cases** were attempted: **290,997 passed / 3,394 failed / 0 unattempted**. A case can produce multiple rows; failed aggregate tasks can retain successful partial rows.
- Failures remain explicit. This is not complete operator support or a whole-model accuracy qualification. Smoke measurements are excluded.
- Full case plans, checkpoint IDs, error logs, and prior retry evidence are retained in `vllm-0.25.0-failures.json.gz`; the adjacent JSON report records source hashes and measurement jobs.

## Runtime and measurement

- Unchanged collector: `cbaf51b64fa460e5ec6146bde407a4c64958212d`.
- vLLM 0.25.0 source: `dd10e03f95f94edbea1975c67ace3a35ec9a8a40`; Torch 2.11.0+cu130; CUDA 13.0; FlashInfer 0.6.13.
- Image: `vllm/vllm-openai@sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97`.
- Cluster / driver: `neb-cdg-slurm-1` / `570.211.01`.
- Exclusive nodes, one Slurm launcher task, eight GPU workers. Slurm requested high GPU frequency; a hard clock lock was not verified.
- Raw evidence on the listed clusters: `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_lights-out-inf/users/simonec/results/vllm025-pr244-multigpu/h200_sxm/production/`.

## Table coverage

| Table | Rows |
| --- | ---: |
| `context_attention_perf.parquet` | 53,771 |
| `dsv4_csa_context_module_perf.parquet` | 5,840 |
| `dsv4_csa_generation_module_perf.parquet` | 1,544 |
| `dsv4_hca_attn_module_perf.parquet` | 3,918 |
| `dsv4_hca_context_module_perf.parquet` | 5,840 |
| `dsv4_hca_generation_module_perf.parquet` | 1,544 |
| `dsv4_paged_mqa_logits_module_perf.parquet` | 3,918 |
| `encoder_attention_perf.parquet` | 7,679 |
| `gdn_perf.parquet` | 10,324 |
| `gemm_perf.parquet` | 111,444 |
| `generation_attention_perf.parquet` | 66,708 |
| `mhc_module_perf.parquet` | 140 |
| `mla_bmm_perf.parquet` | 636 |
| `mla_context_module_perf.parquet` | 5,820 |
| `mla_generation_module_perf.parquet` | 8,832 |
| `moe_perf.parquet` | 54,432 |
| `msa_context_module_perf.parquet` | 11,268 |

## Failed cases

| Run | Failed | Final source job |
| --- | ---: | --- |
| `gemm-fp8_block-00` | 285 | `8687` |
| `gemm-fp8_block-01` | 285 | `8680` |
| `gemm-fp8_block-02` | 270 | `8681` |
| `gemm-fp8_block-03` | 270 | `8688` |
| `attention_context` | 697 | `8693` |
| `mla_context_module` | 36 | `8752` |
| `attention_generation` | 928 | `8728` |
| `msa_context_module` | 444 | `8760` |
| `moe` | 174 | `8754` |
| `gdn` | 5 | `8715` |

- Hardware: NVIDIA H200, SM90, 143,771 MiB, 700 W maximum board power, NV18 links between all eight GPUs.
- FP8-block retains 1,110 runtime assertion failures. Context and generation attention retain 697 and 928 FA4 CuTe FP8 assertions requiring SM100.
- Fresh-worker retries recovered one MLA context case, 69 MSA cases, and four MoE tasks. Successful original measurements were preserved; partial rows of retried failed MoE tasks were remeasured, with all prior physical keys retained.
- Remaining MLA context failures: 28 CUDA launch errors and eight cuBLAS execution errors. Remaining MSA failures: 144 Triton compilation errors, 245 CUDA illegal-access errors, and 55 runtime driver illegal-address errors.
- Remaining MoE failures: 138 value errors and 36 runtime errors, including quantization block divisibility and scalar-versus-vector dequantization contracts. These are observed runtime failures, not claims that every alternate implementation is unsupported.
- GDN retains four CUDA grid-y limit failures and one OOM: the failing task requested another 80 GiB with about 80 GiB already allocated on a 139.81 GiB device.
- All six DSV4 tables, MHC, encoder attention, MLA BMM, and MLA generation completed without failed cases.
- Three initial jobs failed before runner startup with signal 53 on `neb-cdg-slurm-1-gpu-11`; replacements excluded that node. Two early generation jobs exceeded the AF_UNIX path-length limit before measurements; replacements used short temporary paths.

## Validation and limits

- Hashes, B200 reference schemas, finite positive latencies, unique physical keys, complete case IDs, and metadata passed. Four FP8-block shards cover exactly 37,518 retained case IDs.
- All 17 tables loaded through the native engine with strict provenance and shared-layer fallback disabled. Exact measured GEMM and MSA queries passed; the adjacent validation JSON records B200 shape-key differences.
- DSA, MSA generation, compute-scale, communication, and fresh KDA measurements are outside this publication. Older eligible donors can still contribute to predictions.
- All 0.24.0 measurements and metadata remain present. Source-owned reuse policies exclude only old FP8-block GEMM donors measured eagerly with host launch gaps; other eligible donors remain usable. Explicit 0.24.0 primary queries preserve historical data. Query-version defaults are unchanged.
- NVFP4 is excluded by the stock SM100 capability gate on Hopper.
