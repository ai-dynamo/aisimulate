# Qwen3.5 hybrid allocator versus kernel-page check

- B300 Slurm job **1001566** on `aws-pdx-slurm-1`, node `pool0-0001`: **COMPLETED, exit 0**, elapsed 77 seconds, one GPU.
- **30/30 attention measurements passed**: five shapes × three repeats × two mappings. Kernel source was `vllm_flashinfer_trtllmprefill` throughout; BF16 query/KV, four query heads, one KV head, dimension 256 (Qwen3.5-397B-A17B TP8-local geometry).
- Actual prior TP8 serving workers recorded **528-token allocator blocks**. The pinned vLLM runtime chooses **16-token attention kernel pages** by virtual block splitting: 528 = 33 × 16. Runtime `select_common_block_size(528, [FlashInferBackend])` returned 16. A 528-token FlashInfer kernel-page comparison would be invalid.
- Median paired grouped/native timing ratios across the five shapes range **0.9938–1.0082**. All 15 individual paired ratios range **0.9798–1.0178**. No numerical acceptance threshold was specified, so no formal equivalence pass is claimed.

| Batch | Sequence | Native page16 median ms | Manager528/kernel16 median ms | Median paired ratio |
|---:|---:|---:|---:|---:|
| 1 | 4,096 | 0.056315 | 0.055952 | 0.9938 |
| 1 | 24,576 | 0.956619 | 0.956571 | 1.0005 |
| 4 | 16,384 | 1.545376 | 1.535269 | 0.9960 |
| 4 | 32,768 | 6.139355 | 6.047595 | 1.0080 |
| 16 | 40,960 | 41.092196 | 41.277247 | 1.0082 |

## Scope and limits

This is a fresh single-GPU operation comparison using the exact TP8-local geometry, not a new TP8 end-to-end serving campaign. Actual allocator geometry comes from the preserved eight-rank serving receipt from job 946660. Native vLLM source and fresh B300 runtime establish virtual kernel-page selection. The comparison uses the unchanged collector and a test-only block-mapping wrapper: native independently randomized page16 IDs versus randomized manager blocks each expanded into 33 consecutive virtual page16 IDs. It measures full causal prefill and the native KV update; it does not cover cached-tail approximation, scheduler chunking, decode, or all 95 published rows. Measured SM clocks varied (see per-measurement GPU state); `--gpu-freq=high` is not a hard clock lock.

## Provenance and reproduction

- Collector revision: `2579763d1882b3259843c7d2ee4084b8b95d1ef5` (#664 head checked at collection).
- Runtime: vLLM `0.31.0`, source `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
- Existing pinned image: `vllm-0310-cu130-a4a4c043.sqsh`; exact squashfs hash is in `1001566/image.sha256`.
- Raw timing records: `results.json` includes all 30 native CSV rows, per-call wall times and observed GPU state; `summary.json` contains the paired summary. `mapping-records.json` records the virtual page mappings.
- Prior actual eight-rank serving receipt: `cache-receipt.json`, from job 946660. This is geometry evidence, not a new serving run.
- Measurement harness: `compare.py`; runtime and selected-source identity: `identity.json`; executed image hash: `image.sha256`. The identity stores a hash of the runtime selection source instead of copying that upstream implementation.
- Remote result root: `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec/results/pr664-gpu-review-20261102/qwen-page/1001566`.
- Reproduce inside the pinned vLLM 0.31.0 image on one B300: check out collector commit `2579763d1882b3259843c7d2ee4084b8b95d1ef5`, create an empty writable output directory, and run the following with the absolute checkout path. Runtime/geometry assertions run before measurements. The harness writes only validation results to the working directory; never import these diagnostic rows into packaged tables.

  ```bash
  PYTHONPATH=/absolute/aisimulate/python/aisimulate python /absolute/aisimulate/benchmarks/evidence/collector/qwen35-page-layout-20261102/compare.py
  ```
- These are validation-only artifacts. No PRs, packaged tables, shared model storage, or unrelated jobs were modified.

## Original 95-row table provenance

The page experiment validates a specific layout concern; it does not change the
original measurements or their identity. The original table's
`collector_ref: unknown` remains unchanged. Collection job
[518571254](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-auto-collector/-/jobs/518571254)
in pipeline
[77828291](https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-auto-collector/-/pipelines/77828291)
checked out `13e15164522c495af025cfb4d683c7dd06666328`. That collector predates
the environment-variable provenance support; Git was absent in the container.
The 40-file source closure at the original revision reproduces the recorded hash:

- Collector hash: `sha256:ac01f264c808ed1a8a041b938735ecadbde1595a8242259b6ba27b9541381ffc`.
- Packaged Parquet: `d1188fe9cdb287bb5c87a9094026a17fe50d2401266a71a657facadbfec45700`.
- Packaged sidecar: `6c16ec43beed1f94ff28b9c6ff46020a552da03f8789d886dd0ae4a3d0f175a4`.

The original source revision, original data hashes and later validation revision
are separate evidence. No later collector revision is assigned to historical data.
