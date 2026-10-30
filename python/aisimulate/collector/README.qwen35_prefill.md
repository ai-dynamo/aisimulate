# Qwen3.5 prefill recollection

Collect B300 BF16 full-attention measurements for the existing
Qwen3.5-397B-A17B TP8 agentic alignment campaign. The original context lookup
selected vLLM 0.30 decode-labelled rows with only one-token coverage. This
campaign measures actual vLLM 0.31 prefill and checks requested-version routing.

## Frozen scope

- vLLM 0.31.0, source `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
- Official amd64 image pinned in `qwen35_prefill_runtime.yaml`, matching the
  original E2E image. One B300 GPU measures one TP8 rank; no weights are needed.
- Four query heads, one replicated KV head, head dimension 256, BF16 Q/K/V,
  no sliding window, causal full prefill (`query_len = seq_len`, `step = 0`).
- Kernel page 16 via `AISIM_VLLM_ATTENTION_BLOCK_SIZE=16`. The original
  cache receipt measured allocator/APC block 528. The kernel page is derived
  from the exact runtime source: the largest supported divisor of 528 is 16.
  Collection logs must confirm the resolved kernel page.
- Framework-selected FlashInfer `TRTLLMPrefill` for query lengths greater
  than one. A one-token smoke control must retain its actual decode label.
- Preserve the collector's graph-timed KV update plus attention boundary.
  Serving prefill attention runs eagerly at piecewise graph boundaries; this
  remaining timing-mode difference is not qualified by this campaign.

## Run and acceptance

1. Pin and publish the collector source commit after CPU checks.
2. Run `qwen35_prefill_smoke.yaml` through `aisim-auto-collector` using launcher
   ref `simonec/glm-agentx-op-collection`, commit
   `f3ce58a3e20c29c9b31221073182956efd556324`.
3. Verify runtime/GPU identity, kernel page, actual metadata/kernel source,
   positive finite timings, five completed cases and zero failures.
4. Run `qwen35_prefill_full.yaml`: 95 full-prefill cases across batches
   1/2/4/8/16 and sequence anchors 16 through 40,960. Existing sweep parameters
   add long-context anchors; exact runtime filters bound this campaign.
5. Retain checkpoints, summaries, errors, raw rows and provenance. Require
   95 unique successful rows, no unexplained omissions or duplicate keys.
6. Load only the full-run prefill table into an isolated systems root. Keep
   smoke and one-token control rows separate. Confirm the unchanged consumer
   selects the fresh 0.31 prefill source over older decode-labelled donors.
7. Replay all 36 existing exact-token runs and compare all 2,016 measured
   requests. Preserve baseline metrics and check all 2,142 cache observations.

Set `AISIM_OPS=attention_context`, `AISIM_FRAMEWORKS=vllm:0.31.0`,
`AISIM_SYSTEMS=b300_sxm`, `AISIM_GPU_NUM=1`, `AISIM_SHARDS_PER_JOB=1`,
`RUN_JOBS_FOR_MISSING_DATA_ONLY=0`, and `CREATE_GITHUB_PR=0`. Pin
`AISIM_REVISION` and the raw shard-plan URL to the published source SHA.
`AISIM_COLLECTOR_ENV` supplies the kernel-page variable and both hash-bound
runtime-manifest variables. The manifest container path is
`/mnt/ci_job/aisimulate/python/aisimulate/collector/qwen35_prefill_runtime.yaml`.
Keep performance data and campaign logs outside Git until data publication is
requested. Do not trigger the launcher's manual publication job.

## Interpretation limits

The table stores full causal sequence measurements. AISim queries total
sequence length, then scales by `(full_length² - prefix_length²) / full_length²`.
Cached-tail timings cannot be relabelled as these full-prefill rows. Anchors
above the serving token budget are needed for this existing table contract.
No prefix axis, SDK dispatch rule, scheduler setting or numerical calibration
is changed here. GDN, MoE, communication and older donor data remain separate
accuracy questions. This is alignment evidence, not official InferenceX data.

## Source audit

All runtime links use the pinned vLLM revision above:

- [Kernel page selection](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/worker/utils.py#L288)
  and [V2 runner use](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/worker/gpu/attn_utils.py#L251).
- [Supported pages](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/flashinfer.py#L430),
  [query-length split](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/flashinfer.py#L1337),
  and [actual prefill API](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/flashinfer.py#L2387).
- [BF16 backend selection](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/utils/flashinfer.py#L706).

Original E2E evidence: `benchmarks/evidence/qwen35-agentic-20261029/` at
`fad97a78a87081f99db2dbf931f0fc59540beff2`.
