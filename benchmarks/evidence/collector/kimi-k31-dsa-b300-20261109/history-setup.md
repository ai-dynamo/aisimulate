# Native history setup failures

Campaign source commit: `9718c10c8cc74f7901b10fd4dd2922a66409915b`.
Runtime source: `afcd7d6b487a9cbe73e25941971b4cc107e6f9ae`, official SGLang image.

The collector's `_prepare_forward` calls native `one_batch.BenchRunner.extend`
with the complete history (`total_length - 1`) before `decode`. This creates real
latent and index-K cache state, but the native one-batch helper executes one
monolithic extend; it does not use the online scheduler's chunking lifecycle.
The timed decode boundary is never entered when this setup fails.

The original campaign retained 60/72 rows (39 context, 21 generation).
Its 12 failures comprise two OOMs at TP1 and ten cuBLAS errors across TP1/2/4.
Exact cases and error text are in the adjacent `receipt.json`. The source logs
remain in the campaign evidence archive.

- `torch.OutOfMemoryError` in
  `sgl_kernel/flash_mla.py:181 -> sparse_decode_fwd` during native prefill.
  TP1 B4/history131072 and B16/history32768 each requested 96.03 GiB while
  61.83 GiB was free. This is observed setup allocation failure, not a
  measured decode-capacity limit. No allocator or reduced-workload workaround
  was applied.
- `CUBLAS_STATUS_EXECUTION_FAILED` at
  `sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:1141`
  in the native BF16 value-up-projection `torch.bmm`, also during prefill.
  TP1 B1/history131072, B4/history32768, and B16/history8192 exposed this path.
  TP2 also failed all five large-total-token cases; TP4 failed B4/history131072
  and B16/history32768 in the same native cuBLAS operation.
  No root cause beyond the native failing operation is inferred.

These are recorded by the unchanged full-node orchestrator as unexpected
case failures. The original selected grid and native error text remain intact.
Final artifacts must retain partial coverage and list all observed failed
cases; successful rows are not rewritten or synthesized.

A future improvement can audit and use native scheduler chunked prefill for
history construction while keeping the measured decode module unchanged.
That needs a source-backed state-equivalence test (KV slots, index-K history,
per-request lengths and continuation) and separate collection provenance.
It cannot be treated as a generic retry or silent change to this campaign.
