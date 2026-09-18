# PR #244 collection evidence

This non-packaged bundle makes the vLLM 0.25.0 collection claims reproducible
without a GPU, host paths, usernames, scheduler IDs, or raw diagnostic logs.
It is outside the Python package's source/data tree.

## Verify

From the repository root, after `uv sync --project python/aisimulate --extra dev`:

```sh
python/aisimulate/.venv/bin/python scripts/audit_pr244_data.py
```

The check recomputes all case counts and summaries of the 85 shipped parquet
files. It checks each file's hash and row count against the original collection
report, then compares the result with `manifest.json`. No network or historical
Git objects are needed for this check. Use `--write` only when intentionally
regenerating the manifest after reviewing a change.

| System | Tables | Rows | Retained cases | Done | Failed | Unattempted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B300 | 17 | 420,111 | 350,380 | 339,641 | 10,739 | 0 |
| GB200 | 17 | 420,134 | 350,380 | 339,614 | 10,766 | 0 |
| GB300 | 17 | 420,226 | 350,380 | 339,681 | 10,699 | 0 |
| H100 | 17 | 353,449 | 294,319 | 290,824 | 3,495 | 0 |
| H200 | 17 | 353,658 | 294,391 | 290,997 | 3,394 | 0 |

All 85 tables have zero reported null cells, duplicate physical keys, and
nonfinite or nonpositive latency values.

Cases and rows are different units: a single MoE/GDN case can emit many rows.
Zero unattempted cases means every **retained** case reached a final checkpoint
outcome. It does not mean every possible shape succeeded or that capability
filtering retained every raw candidate. Failed cases remain failed; they are
not converted into timing rows or silently counted as successful coverage.

## Files and provenance

- `manifest.json`: per-system and per-attempt counts; per-table SHA-256, rows,
  physical key columns, Arrow types, numeric shape ranges, categorical domains,
  distinct counts, latency ranges in **milliseconds**, and anomaly counts
  (nulls, duplicate physical keys, nonfinite/nonpositive latency). Sidecar hashes
  bind the summary to the packaged provenance. Physical row identity includes
  `kernel_source`; engine lookup keys can be narrower.
- `case-outcomes.json.gz`: deterministic gzip of JSON. `plans[sha256]` contains
  the complete sorted retained task-ID list. The digest is SHA-256 of the IDs
  joined by newline, with **no final newline**. Each system's attempt references
  one plan and an equally long `outcomes` string: `D` = done, `F` = failed,
  `U` = unattempted. Zip the IDs with that string to inspect any failed shape.
  Plans shared across attempts/systems are stored once. Original report/archive
  hashes and reported table hashes/counts are retained.
- Source: the final collection reports and failure archives at repository commit
  [`1f29ee459882796db312b8a288c040b83edbb211`](https://github.com/ai-dynamo/aisimulate/commit/1f29ee459882796db312b8a288c040b83edbb211),
  before diagnostic cleanup. Only final top-level checkpoints are extracted;
  earlier attempts and retry logs are not summed into the results.
- B300's archived standalone case-plan object was absent. Its exact retained IDs
  were recovered from the final checkpoint union, then checked against **both**
  the independently recorded plan digest and expected count for every attempt.
  This is recorded as `verified_checkpoint_union`; other systems use their
  `archived_case_plan`. A mismatch stops extraction instead of inferring zero
  unattempted cases from the observed outcomes alone.
- Collector revision: `cbaf51b64fa460e5ec6146bde407a4c64958212d`.
  Actual vLLM runtime revision: `dd10e03f95f94edbea1975c67ace3a35ec9a8a40`.
  The runtime revision differs from the v0.25.0 tag's commit.

To reproduce the one-time sanitized extraction when the source Git objects are
available, run the same command with `--extract-cases --write`. Extraction
allowlists only case IDs, outcomes, counts, and hashes; it does not copy arbitrary
report fields, paths, error messages, or runtime environment dictionaries.
The evidence supports collection coverage and table integrity, not diagnosis of
each failure or end-to-end prediction accuracy.

## Why table-wide GEMM reuse is intentional

The six 0.24.0 declarations (B200/B300/GB200/GB300/H100/H200) fill **every**
unshadowed 0.25.0 GEMM key. They add 36,408 FP8-block keys, 1,776 BF16 keys,
and 1,776 FP8 keys per system, plus 1,776 NVFP4 keys on Blackwell. Retained
0.24.0 primary rows always win, including their exact measured latency.

The non-FP8-block donor rows record these selected kernels. An immutable source
comparison between vLLM
[`v0.24.0 / ee0da84ab9e04ac7610e28580af62c365e898389`](https://github.com/vllm-project/vllm/tree/ee0da84ab9e04ac7610e28580af62c365e898389)
and the
[actual 0.25.0 runtime](https://github.com/vllm-project/vllm/tree/dd10e03f95f94edbea1975c67ace3a35ec9a8a40)
found:

| Precision | Recorded kernel | Source comparison |
| --- | --- | --- |
| BF16 | `torch.nn.functional.linear` | `RowParallelLinear` and `UnquantizedLinearMethod` in [`layers/linear.py`](https://github.com/vllm-project/vllm/blob/dd10e03f95f94edbea1975c67ace3a35ec9a8a40/vllm/model_executor/layers/linear.py) have identical Python ASTs. |
| FP8 | `CutlassFP8ScaledMMLinearKernel` | [`kernels/linear/scaled_mm/cutlass.py`](https://github.com/vllm-project/vllm/blob/dd10e03f95f94edbea1975c67ace3a35ec9a8a40/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py) is byte-identical. |
| NVFP4 | `FlashInferCuteDslNvFp4LinearKernel` | [`kernels/linear/nvfp4/flashinfer.py`](https://github.com/vllm-project/vllm/blob/dd10e03f95f94edbea1975c67ace3a35ec9a8a40/vllm/model_executor/kernels/linear/nvfp4/flashinfer.py) and the [`compressed-tensors NVFP4 scheme`](https://github.com/vllm-project/vllm/blob/dd10e03f95f94edbea1975c67ace3a35ec9a8a40/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py) are byte-identical. |

This supports using the same operation, precision, shape, GPU, and kernel path
as an explicit cross-version proxy for missing measurements. It does **not**
prove identical latency across PyTorch, CUDA, FlashInfer, driver, or clock
versions. Consumers requiring only native 0.24.0 measurements must disable
shared-layer reuse; provenance identifies donor rows as `declared_reuse`.

`test_corrected_024_gemm_uses_declared_025_measurements` compares the **entire**
native loaded GEMM map with a first-source-wins merge of all primary and donor
rows on all six systems. It catches omitted non-FP8 donor keys, unexpected extra
keys, and overwritten primary latencies.
