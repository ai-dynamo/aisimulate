# B300 / vLLM 0.25.0 operator data — partial publication

Collected **17 tables / 420,111 measurements** on NVIDIA B300 SXM6 AC (SM103), matching the 17-table scope of [PR #219](https://github.com/ai-dynamo/aisimulate/pull/219).

All **350,380 retained cases** were attempted: 339,641 passed and 10,739 failed. Some tasks emit multiple rows. Failed tasks can retain valid partial measurements. Collection completion does not mean every shape succeeded.

## Coverage

Physical keys include kernel source and shape fields; device identity and measurement values are excluded. Fifteen tables exactly match B200 coverage. MoE and MSA differences remain explicit.

| Table | B300 rows | B200 rows | B300-only keys | B200-only keys |
|---|---:|---:|---:|---:|
| context_attention | 52,836 | 52,836 | 0 | 0 |
| dsv4_csa_context_module | 5,840 | 5,840 | 0 | 0 |
| dsv4_csa_generation_module | 1,544 | 1,544 | 0 | 0 |
| dsv4_hca_attn_module | 3,918 | 3,918 | 0 | 0 |
| dsv4_hca_context_module | 5,840 | 5,840 | 0 | 0 |
| dsv4_hca_generation_module | 1,544 | 1,544 | 0 | 0 |
| dsv4_paged_mqa_logits_module | 3,918 | 3,918 | 0 | 0 |
| encoder_attention | 7,679 | 7,679 | 0 | 0 |
| gdn | 10,326 | 10,326 | 0 | 0 |
| gemm | 148,962 | 148,962 | 0 | 0 |
| generation_attention | 65,408 | 65,408 | 0 | 0 |
| mhc_module | 140 | 140 | 0 | 0 |
| mla_bmm | 636 | 636 | 0 | 0 |
| mla_context_module | 13,059 | 13,059 | 0 | 0 |
| mla_generation_module | 11,400 | 11,400 | 0 | 0 |
| moe | 72,924 | 72,378 | 702 | 156 |
| msa_context_module | 14,137 | 14,110 | 202 | 175 |

## Runtime and provenance

- Official vLLM 0.25.0 image, amd64 digest `e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97`.
- vLLM source `dd10e03f95f94edbea1975c67ace3a35ec9a8a40`; Torch 2.11.0+cu130, CUDA 13.0, FlashInfer 0.6.13.
- Collector commit `cbaf51b64fa460e5ec6146bde407a4c64958212d`, with no collector or framework source changes.
- Exclusive eight-GPU Slurm nodes; at most four production nodes concurrently. Four disjoint block-FP8 partitions cover the complete retained GEMM plan.
- Slurm requested high GPU frequency. **Hard clock locking was not verified.** One-second clock traces and per-job statistics are retained; samples cannot assign an exact clock to each timing.
- The job-local environment adds Arrow 25.0.1. Finalization recovery adds pandas 2.3.2 and pinned utility dependencies without changing vLLM, Torch, CUDA, or NumPy.

## Retry and failure evidence

- MSA retry recovered 44 cases, reaching 14,137 rows. One worker blocked with idle GPUs at 3,474/3,475 retry tasks; its verified task-owned process was terminated and the collector recorded the failure. Process checks, signals, and timestamps are retained.
- MoE retry recovered 14 tasks, adding 365 row keys. The stock finalizer replaced 37 partial-row keys from previously failed tasks. Validation confirms every original key survives and changed timings belong only to those failed tasks.
- Both retries initially failed during Parquet merging because pandas was missing. Finalization-only resumes skipped all passed/failed tasks and completed the merges; measurement and finalization job IDs are recorded separately.
- `vllm-0.25.0-failures.json.gz` contains initial, retry, and final checkpoint/summary/error-log evidence. Full logs cover failed IDs missing from stock JSON summaries.
- Remaining failures include block-FP8 scale-shape assertions, dense-attention head dimension 192, MLA head-ratio constraints, MSA CSR/Triton/CUDA errors, MoE alignment/kernel/CUDA errors, and GDN grid-size limits. These observations do not establish that every failed shape is an inherent B300 hardware limit.

## Validation

- All 17 tables passed hash, schema, finite-positive-latency, physical-key uniqueness, complete-plan, and public provenance-loader checks.
- All seven collector-data rules passed. All 27 targeted metadata/discovery tests passed.
- The native engine loaded all 17 tables with strict provenance and shared-layer fallback disabled. Four GEMM dtype queries and three MSA projection-dtype queries returned exact measured values with silicon provenance.
- UTF8 offset widths were aligned losslessly to B200 schemas. No timings were averaged or synthesized.

## Scope limits

- DSA context/generation and MSA generation are outside the 17-table scope; smoke diagnostics recorded runtime/kernel failures.
- Compute-scale output is withheld because stock finalization rejects its undeclared scale-matrix output.
- KDA requires a different preview runtime. No B200 timings or reuse declarations were copied. CAR/NCCL are outside this collection.
- Use version `0.25.0` explicitly. Whole-model prediction accuracy was outside this collection.

The adjacent JSON report contains table hashes, source jobs, coverage, clocks, recovery history, and archive details. Data is collected and validated, with publication pending.

## Evidence archive

- Archive: `b300-vllm025-219-evidence.tar.gz` (desktop artifact directory recorded in the JSON report).
- SHA-256: `29d6b379dbfbc96ff383fd77d753d8c2ef8636fc219718226b96dd1799b5742e`.
- Contains original and resumed outputs, checkpoints, complete error logs, clock traces, runtime records, job states, and campaign scripts.

## Reuse safety for historical FP8-block measurements

- All B200/B300 0.24.0 data, sidecars, reuse declarations and query-version slots
  are retained. The 0.25.0 collection remains partial and is not promoted to the
  current version. Load this unlisted version explicitly in the SDK with
  `allow_unlisted_version=True`.
- The 0.24.0 GEMM directories now restrict donor kernels through `reuse.yaml`.
  BF16, ordinary FP8 and NVFP4 remain eligible; the 35,742 FP8-block rows on
  each GPU are excluded from declared, implicit and cross-backend reuse.
  Other operator families, including communication, keep their existing reuse.
- The old collector used `use_cuda_graph=gemm_type != "fp8_block"`. Host launch
  gaps were included in eager FP8-block latency but published under the same
  performance contract as graph-timed GEMMs. [PR #219](https://github.com/ai-dynamo/aisimulate/pull/219)
  documents a B200 mechanism A/B at vLLM 0.25.1: eager versus one-op graph
  timings were 148.55 versus 12.32 us at `(M,N,K)=(1,768,7168)` and 152.77
  versus 8.21 us at `(1,7168,384)`. These are mechanism evidence, not a fresh
  reproduction of every old 0.24.0 row.
- The policy only restricts donors. Explicit 0.24.0 primary queries still read
  the original measurements, including the problematic FP8-block rows. It is
  not a correction of historical primary predictions or an assertion that
  other 0.24.0 operators are invalid. Fresh 0.25.0 graph-timed kernels are not
  blocked by the source-specific restriction.
- Reuse-policy validation: 22 Rust source-resolution tests, 66 Python tests,
  both engine parity suites (365 tests, unchanged goldens), and eight numerical
  sentinels passed. Live B200/B300 source resolution confirms the GEMM donor
  filter and unrestricted 0.24.0 custom-allreduce reuse.
