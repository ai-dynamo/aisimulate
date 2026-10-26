# B200 vLLM 0.31 GEMM: failure and resume reconciliation

- Pipeline **76706015**, job **509647192** accounts for all **75,543** selected callbacks: **74,433 succeeded; 1,110 failed; zero unattempted**.
- Keep successful evidence. Publish a separately attested canonical derivative with **74,433 rows**, using stable keep-last for the two resume duplicates. Preserve the raw archive and both original values. No GPU rerun is needed for these bookkeeping defects.
- The **1,110 failures exactly match the frozen vLLM 0.25 historical failed-key set**. They remain explicit unsupported coverage, not successful measurements. There are no new failed keys relative to that historical set.

| dtype / selected class | Unique successes | Failures |
|---|---:|---:|
| BF16 / torch.nn.functional.linear | 37,518 | 0 |
| Tensor FP8 / CutlassFP8ScaledMMLinearKernel | 507 | 0 |
| Block FP8 / CutlassFp8BlockScaledMMKernel | 5,328 | 1,110 |
| Block FP8 / DeepGemmFp8BlockScaledMMKernel | 31,080 | 0 |

## Native failure cluster

All failures are the same scale-contract assertion: **15 N/K lanes × 74 original M values**.

| N | K |
|---:|---|
| 1 | 2048, 3072, 4096, 8192 |
| 4 | 1024, 2048, 2560, 4096 |
| 8 | 1024, 2048, 2560, 3072, 4096 |
| 12 | 5120 |
| 24 | 5120 |

Native Cutlass block-FP8 dispatch calls `cutlass_scaled_mm`. With N not divisible by16, that wrapper falls back to `triton_scaled_mm`, which requires one activation-scale column. Block-FP8 supplies `[M,K/128]`; every failed K exceeds128. These are unsupported native paths. Forcing another backend or adding collector-only padding would change the measured serving contract.

Original historical report/failure artifacts were hash-verified. The 0.27.1 subset also documents74 assertions at N1/K8192, but its complete exact-key failure list was not recovered; the full equality proof uses0.25.

## Two resume duplicates

The first allocation5307688 timed out at05:42:16 PDT. Its last periodic archive was saved at05:42:09. The runner tarred a live directory; CSV append and checkpoint persistence were not an atomic snapshot. Restart loaded46,733 successful checkpoint keys, but restored46,735 data rows.

| dtype / M,N,K | First raw row / latency(ms) | Resumed row / latency(ms) |
|---|---|---|
| block FP8 / 96,5120,6144 | 46,733 / 0.016789333273967106 | 46,735 / 0.01718933383623759 |
| BF16 / 96,5120,5120 | 46,734 / 0.011866666376590729 | 46,736 / 0.01203733351495531 |

Row indices are zero-based. The collector writes each CSV row before marking its callback done; checkpoint persistence occurs later. The live archive can also copy these files at different moments. The final unique row set exactly equals checkpoint successes. Retained finalization transactions report `merged_existing=false` and `previous_target=null`, ruling out an existing-Parquet merge as the source.

**Canonical policy:** stable keep-last over every column except latency, limited to this same-provenance artifact. Keep rows46,735/46,736, preserving the completed resumed attempts regardless of latency. Do not select the faster result or average attempts. Record the new artifact hash/count and replacement audit; the raw transaction attestation cannot be reused for the derivative. The raw artifact remains untouched. Per-row timestamps are absent, so the precise write-versus-tar instant cannot be distinguished.

## Error reporting defect and success status

- The checkpoint and full collector log contain **1,110 distinct failures**. Error JSON contains only **786 case records + one UnresolvedFailures summary**; the missing324 IDs are listed in the JSON report. The earlier770 figure was a progress snapshot.
- Final queue draining uses `Queue.empty()` before worker join, with no drain during or after join. Worker587 then exceeded the42-second join timeout and was terminated. Pending feeder records can be lost through this lifecycle. The exact number buffered at termination is inferred, but missing JSON records and complete logs are directly verified.
- The collector exits nonzero only for `ModuleCollectionFailure`; individual failures do not fail the job. `status: complete` and GitLab success therefore do not mean full measurement coverage.

## Narrow next actions

- Preserve all evidence; root prepares the canonical derivative and publishes explicit unsupported keys. Do not rerun unrelated successes.
- Repair resume recovery within existing checkpoint/output machinery: reconcile or reject stale duplicate output identities without changing case IDs or shared-output contracts. Test interruption after append/before checkpoint and a live snapshot restore.
- Drain error records through worker completion and reconcile against failed checkpoint IDs. Test burst errors larger than pipe capacity and delayed delivery. Keep partial successful data.
- If the native unaligned block-FP8 path is repaired later, qualify that new source/runtime and rerun only the15 N/K lanes ×74 original M values. No blind retry is useful now.

## Immutable source evidence

- https://github.com/ai-dynamo/aisimulate/blob/8a411395d3161f1320a9de0e8670deacb40b314c/python/aisimulate/collector/collect.py#L1595
- https://github.com/ai-dynamo/aisimulate/blob/8a411395d3161f1320a9de0e8670deacb40b314c/python/aisimulate/collector/collect.py#L2057
- https://github.com/ai-dynamo/aisimulate/blob/8a411395d3161f1320a9de0e8670deacb40b314c/python/aisimulate/collector/collect.py#L6214
- https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/_custom_ops.py#L806
- https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py#L313
- https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py#L201
- https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-auto-collector/-/blob/afb07bdba36c87af72674d968376fe3fac8d8f13/data_collection/gpu_node.sh#L265

Detailed hashes, exact failed M values, missing JSON error IDs, counts, and duplicate rows are in `b200-gemm-full-failure-triage.json`.
