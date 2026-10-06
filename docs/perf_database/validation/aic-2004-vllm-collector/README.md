# B200 native vLLM DSA recollection

This change repairs module measurements used by GLM-5.2 and DeepSeek-V3.2.
It preserves native RoPE/scale buffers, initializes real FP8 projection weights
with finite nonzero values, populates historical indexer keys with the native
quantizer, requires CUDA graph timing, and releases graphs and the vLLM
workspace between cases. Checkpoints declaring index reuse emit separate full
and reuse rows through the native `skip_topk` path. A full forward populates the
actual shared top-k buffer before reuse is timed.

Decode metadata describes total KV length, including its one new token. The
export now records `isl=1, step=total_length-1`, matching both native MLA/DSA
loaders' `isl + step` key. Existing rows are not silently relabeled.

The native execution contract is pinned to vLLM
[`752a3a504485790a2e8491cacbb35c137339ad34`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/model_executor/models/deepseek_v2.py#L1080).
The reuse flag bypasses the indexer while retaining attention projections and
sparse attention. These are runtime API references; no new upstream code is
copied in this change.

The target manifest includes all 23 distinct GLM-5.2 coordinates from the
AIC-2004 historical prefill cohort, local head counts corresponding to TP
1/2/4/8, BF16 and FP8 KV, native BF16/block-FP8 attention projections, decode,
and sparse/dense and long-prefix boundaries. Synthetic module inputs and
local head geometry do not establish checkpoint-value fidelity, distributed
TP performance, or whole-model accuracy acceptance. Packed NVFP4 attention
projection collection is outside this qualification; the GLM NVFP4 checkpoint's
attention projections are BF16.

Fresh measurements retain their real `vllm/0.25.1` path. The existing public
`next` slot resolves newer non-DSA data and these DSA rows, with older vLLM rows
filling unmeasured coordinates. This PR does not change the current/next
version-slot contract. Completion of the attested target manifest does not
mean completion of the full collector grid or qualification of other GPUs.

The paired simulation PR consumes full/reuse measurements using the physical
21/57 GLM layer split; DeepSeek-V3.2 remains 61 full-indexer layers. Historical
whole-forward comparisons are cross-version diagnostics and are reported in
that PR, separately from native module measurements.

`independent-review.json` records the independent review, the two findings
subsequently fixed (decode coordinates and engine backend selection), and the
limits of CPU lifecycle tests. GPU receipts retain the actual collector revision `8f7dfc09`. The subsequent
source-only import ordering change does not change the measurement functions.

## Measured data

The B200 campaign completed 154 target cases and 2 repeat controls on GPU
`7adde65f-566f-24de-aef0-1167ac85806d`, node `umb-b200-263`, Slurm job
4763305. It produced 184 context rows (106 full, 78 reuse) and 80 decode rows
(48 full, 32 reuse). All 267 timing observations used CUDA graphs, executed
30 iterations after 10 warmups, and reported no throttling. The three repeated
full/reuse observations changed by at most 4.47%. Rows have unique consumer
identities and finite positive latencies; all seven collector-data rules pass.

`data-validation.json` records row counts, hashes, repeated observations and
coverage limits. `runtime-identity.json` and `runtime-source-hashes.json` pin
the image, native files and dependency versions. `plan.json` is the frozen
attempted workload. `timing-receipts.json` preserves raw full-precision timings;
production parquet uses the collector's normal four-decimal millisecond
serialization. `repeat-controls/` stays outside the production dataset.

The original first pass completed too, but preceded the decode-coordinate
repair. It is not shipped as corrected data. Both phases were recollected
from the fixed source, so no old row was relabeled.

## Reproduce

Check out collector revision `8f7dfc09383f19bb9eda247f01289385202b4e67`.
In the pinned vLLM image from `runtime-identity.json`, expose exactly one
B200, mount that revision's `python/aisimulate` at `/task/payload`, this evidence
directory at `/repro`, and an empty writable directory at `/output`. Set
`PYTHONPATH=/task/payload`, `HF_HUB_OFFLINE=1`, `COLLECTOR_MEASURE_POWER=0`,
and allocation-private FlashInfer/Triton/CUDA caches before imports. Run:

```sh
python /repro/reproduce.py
```

The reproduction driver is the frozen campaign driver with a license header
and formatting added. It calls the native collector directly, does not alter
benchmark arguments or results, and fails on a rejected graph or runtime case.
The runtime receipt records the original frozen driver's SHA, not the formatted
copy's SHA. This is a bounded reproduction; run the normal collector case
matrix separately for full-grid coverage.
