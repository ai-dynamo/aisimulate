# GB300 vLLM 0.25.0 collection

- Published: **17 tables / 420,175 measured rows** for the PR #219 table scope.
- All **350,380 retained cases** were attempted: **339,630 passed / 10,750 failed / 0 unattempted**. A case can produce multiple rows; failed aggregate tasks can retain successful partial rows.
- Failures remain explicit. This is not complete operator support or a whole-model accuracy qualification. Smoke measurements are excluded.
- Full case plans, checkpoint IDs, error logs, and prior retry evidence are retained in `vllm-0.25.0-failures.json.gz`; the adjacent JSON report records source hashes and measurement jobs.

## Runtime and measurement

- Unchanged collector: `cbaf51b64fa460e5ec6146bde407a4c64958212d`.
- vLLM 0.25.0 source: `dd10e03f95f94edbea1975c67ace3a35ec9a8a40`; Torch 2.11.0+cu130; CUDA 13.0; FlashInfer 0.6.13.
- Image: `vllm/vllm-openai@sha256:2f726db9babd627fb59addaed2038577fc767108680901271b069d91759d1286`.
- Cluster / driver: `oci-aga-slurm-1` / `580.167.08`; `oci-jhb-slurm-1` / `580.159.03`.
- Exclusive nodes, one Slurm launcher task, four GPU workers. Slurm requested high GPU frequency; a hard clock lock was not verified.
- Raw evidence on the listed clusters: `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec/results/vllm025-pr244-multigpu/gb300/production/`.

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
| `moe_perf.parquet` | 73,002 |
| `msa_context_module_perf.parquet` | 14,123 |

## Failed cases

| Run | Failed | Final source job |
| --- | ---: | --- |
| `attention_context` | 1,632 | `771366` |
| `mla_context_module` | 117 | `771367` |
| `attention_generation` | 2,228 | `771519` |
| `mla_generation_module` | 1,848 | `771520` |
| `gdn` | 4 | `771521` |
| `gemm-fp8_block-00` | 285 | `437143` |
| `gemm-fp8_block-01` | 285 | `437144` |
| `gemm-fp8_block-02` | 270 | `437145` |
| `gemm-fp8_block-03` | 270 | `437146` |
| `msa_context_module` | 3,445 | `437323` |
| `moe` | 366 | `437705` |

- Production measurements are split across AGA (non-block GEMM and attention/helper families) and JHB (FP8-block GEMM, MoE, and MSA). Both use NVIDIA GB300, PCI device 0x31C210DE, 284,208 MiB, and 1,400 W maximum board power.
- Framework packages and hardware identity match, but driver patches differ: AGA 580.167.08 and JHB 580.159.03. An eight-case-per-dtype smoke comparison gave JHB/AGA median ratios of 0.99754 (BF16), 1.00002 (FP8), 0.97741 (FP8-block), and 0.999717 (NVFP4). Individual ratios vary; this small sample does not establish full workload equivalence. Smoke timings are not in the dataset. The comparison is retained in `vllm-0.25.0-cross-cluster-smoke.json`.
- FP8-block retains 1,110 runtime assertion failures. Context and generation attention retain 1,632 and 2,228 assertions rejecting symmetric head dimension 192 in the selected FA4 runtime.
- MLA context and generation retain 117 and 1,848 runtime failures because the selected FMHA kernel rejects the Q/KV head ratio.
- MoE fresh-worker retry recovered 21 tasks. Successful original measurements were preserved; partial rows of retried failed tasks were remeasured, with all prior physical keys retained. Remaining failures are recorded by case rather than treated as universal hardware limitations.
- GDN retains four CUDA grid-y limit failures. All six DSV4 tables, MHC, encoder attention, and MLA BMM completed without failed cases.
- Two early generation jobs exceeded the AF_UNIX path-length limit before measurements; replacement jobs used short temporary paths.
- MSA first pass: 14,123 passed / 3,445 failed. Fresh-worker retry 439769 is still running; this checkpoint publishes the finalized first-pass table and does not claim the retry is complete. Its recovered cases will be added after validation.

## Validation and limits

- Hashes, B200 reference schemas, finite positive latencies, unique physical keys, complete case IDs, and metadata passed. Four FP8-block shards cover exactly 37,518 retained case IDs.
- All 17 tables loaded through the native engine with strict provenance and shared-layer fallback disabled. Exact measured GEMM and MSA queries passed; the adjacent validation JSON records B200 shape-key differences.
- DSA, MSA generation, compute-scale, communication, and fresh KDA measurements are outside this publication. Older eligible donors can still contribute to predictions.
- All 0.24.0 measurements and metadata remain present. Source-owned reuse policies exclude only old FP8-block GEMM donors measured eagerly with host launch gaps; other eligible donors remain usable. Explicit 0.24.0 primary queries preserve historical data. Query-version defaults are unchanged.
