# H100 SXM vLLM 0.25.0 collection

- Published: **17 tables / 353,449 measured rows** for the PR #219 table scope.
- All **294,319 retained cases** were attempted: **290,824 passed / 3,495 failed / 0 unattempted**. A case can produce multiple rows; failed aggregate tasks can retain successful partial rows.
- Failures remain explicit. This is not complete operator support or a whole-model accuracy qualification. Smoke measurements are excluded.
- Full case plans, checkpoint IDs, error logs, and prior retry evidence are retained in `vllm-0.25.0-failures.json.gz`; the adjacent JSON report records source hashes and measurement jobs.

## Runtime and measurement

- Unchanged collector: `cbaf51b64fa460e5ec6146bde407a4c64958212d`.
- vLLM 0.25.0 source: `dd10e03f95f94edbea1975c67ace3a35ec9a8a40`; Torch 2.11.0+cu130; CUDA 13.0; FlashInfer 0.6.13.
- Image: `vllm/vllm-openai@sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97`.
- Cluster / driver: `aws-iad-slurm-2` / `595.58.03`.
- Exclusive nodes, one Slurm launcher task, eight GPU workers. Slurm requested high GPU frequency; a hard clock lock was not verified.
- Raw evidence on the listed clusters: `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_lights-out-inf/users/simonec/results/vllm025-pr244-multigpu/h100_sxm/production/`.

## Table coverage

| Table | Rows |
| --- | ---: |
| `context_attention_perf.parquet` | 53,771 |
| `dsv4_csa_context_module_perf.parquet` | 5,789 |
| `dsv4_csa_generation_module_perf.parquet` | 1,544 |
| `dsv4_hca_attn_module_perf.parquet` | 3,918 |
| `dsv4_hca_context_module_perf.parquet` | 5,793 |
| `dsv4_hca_generation_module_perf.parquet` | 1,544 |
| `dsv4_paged_mqa_logits_module_perf.parquet` | 3,918 |
| `encoder_attention_perf.parquet` | 7,679 |
| `gdn_perf.parquet` | 10,284 |
| `gemm_perf.parquet` | 111,444 |
| `generation_attention_perf.parquet` | 66,708 |
| `mhc_module_perf.parquet` | 139 |
| `mla_bmm_perf.parquet` | 636 |
| `mla_context_module_perf.parquet` | 5,820 |
| `mla_generation_module_perf.parquet` | 8,760 |
| `moe_perf.parquet` | 54,432 |
| `msa_context_module_perf.parquet` | 11,270 |

## Failed cases

| Run | Failed | Final source job |
| --- | ---: | --- |
| `gemm-fp8_block-00` | 285 | `2212` |
| `gemm-fp8_block-01` | 285 | `2213` |
| `gemm-fp8_block-02` | 270 | `2214` |
| `gemm-fp8_block-03` | 270 | `2215` |
| `attention_context` | 697 | `2195` |
| `mla_context_module` | 36 | `2196` |
| `attention_generation` | 928 | `2197` |
| `msa_context_module` | 442 | `2216` |
| `moe` | 174 | `2218` |
| `gdn` | 9 | `2204` |
| `mhc_module` | 1 | `2205` |
| `dsv4_csa_context_module` | 51 | `2219` |
| `dsv4_hca_context_module` | 47 | `2208` |

- FP8-block fresh-worker retries recovered all 28 first-attempt OOM cases. The final 111,444-row GEMM table contains the full BF16 and ordinary FP8 sweeps and 36,408 successful FP8-block cases; 1,110 runtime assertions remain.
- Context and generation attention retain 697 and 928 FA4 CuTe FP8 assertions requiring SM100.
- Fresh-worker retries recovered 71 MSA cases, eight MoE tasks, and nine DSV4 CSA context cases. Successful original timings were preserved; partial rows of retried failed MoE tasks were remeasured with prior physical keys retained.
- Remaining MSA failures: 144 Triton compilation errors, 241 CUDA illegal-access errors, and 57 runtime driver illegal-address errors. MoE retains 174 failed tasks.
- DSV4 CSA context retains 27 OOM, 23 `aten::new_empty` dispatcher runtime errors, and one accelerator error. HCA context retry retains 18 OOM and 29 dispatcher runtime errors. These are observed failures of this runtime and measurement setup, not claims that alternate implementations cannot run these shapes.
- MLA context retry 2217 and HCA context retry 2220 recovered no cases. The stock resume finalizer then rejected the absent new staging tables. Original finalized sources 2196 and 2208 are published after verifying identical Parquet hashes and identical done/failed case sets. Both failed retry attempts remain archived as diagnostic evidence; no staging rows or checkpoints were fabricated.
- GDN retains four CUDA grid-y limit failures and five OOM failures. MHC retains one OOM for the post-op case with hidden size 7168, hc_mult 4, and 524288 tokens: another 28 GiB was requested with about 70 GiB already allocated on a 79.18 GiB device.
- Encoder attention, MLA BMM, MLA generation, DSV4 CSA/HCA generation, paged MQA logits, and HCA attention completed without failed cases.

## Validation and limits

- Hashes, B200 reference schemas, finite positive latencies, unique physical keys, complete case IDs, and metadata passed. Four FP8-block shards cover exactly 37,518 retained case IDs.
- All 17 tables loaded through the native engine with strict provenance and shared-layer fallback disabled. Exact measured GEMM and MSA queries passed; the adjacent validation JSON records B200 shape-key differences.
- DSA, MSA generation, compute-scale, communication, and fresh KDA measurements are outside this publication. Older eligible donors can still contribute to predictions.
- All 0.24.0 measurements and metadata remain present. Source-owned reuse policies exclude only old FP8-block GEMM donors measured eagerly with host launch gaps; other eligible donors remain usable. Explicit 0.24.0 primary queries preserve historical data. Query-version defaults are unchanged.
- NVFP4 is excluded by the stock SM100 capability gate on Hopper.
