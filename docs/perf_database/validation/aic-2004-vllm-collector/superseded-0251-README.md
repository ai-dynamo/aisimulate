# Superseded: invalid current-query KV initialization

This retained historical report describes the earlier diagnostic pilot. Its
production data are being replaced: the collector omitted native slot_mapping,
so current-query MLA KV writes were skipped. It does not qualify the repaired
collector or support final accuracy claims. Keep the associated old receipts
for comparison; do not load the 0.25.1 pilot as production data.

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
whole-forward comparisons use the original vLLM 0.25.1 truth cohort and are
reported in that PR. They are historical diagnostics, not fresh deployment-matched
whole-model accuracy acceptance; other operation tables can resolve mixed versions.

`independent-review.json` records the independent review, the two findings
subsequently fixed (decode coordinates and engine backend selection), and the
limits of CPU lifecycle tests. GPU receipts retain the actual collector revision `36f40cd4`. Since the
reviewed revision, collector source changed only in import ordering.

## Measured data

The B200 campaign completed 154 target cases and 2 repeat controls on GPU
`2eba6257-9bb3-2747-fc09-864946248a91`, node `umb-b200-235`, Slurm job
4763807 (driver 615.71.09). It produced 184 context rows (106 full, 78 reuse) and 80 decode rows
(48 full, 32 reuse). All 267 timing observations used CUDA graphs, executed
30 iterations after 10 warmups. The three repeated full/reuse observations
changed by at most 1.16%. The node had no drain flag before or after collection;
periodic active-run samples recorded 1965 MHz SM and 3996 MHz memory clocks,
and the postflight clock-event bitmask was zero. These are periodic observations,
not per-kernel clock telemetry. Power measurement was disabled, so the helper
`throttled=False` flag alone is not independent clock-health evidence. Rows have unique consumer
identities and finite positive latencies; all seven collector-data rules pass.

`data-validation.json` records row counts, hashes, repeated observations and
coverage limits. `runtime-identity.json` and `runtime-source-hashes.json` pin
the image, native files and dependency versions. `plan.json` is the frozen
attempted workload. `timing-receipts.json` preserves raw full-precision timings;
production parquet uses the collector's normal four-decimal millisecond
serialization. `repeat-controls/` stays outside the production dataset.

A subsequent node-level pegged-clock flag was recorded at 00:30:26 UTC,
16 minutes 12 seconds after this run finished at 00:14:14 UTC.
`later-node-observation.json` preserves it. The retained pre/postflight checks
had no flag, and the sampled clocks and repeat controls remain documented.
The later flag identifies neither the affected GPU nor an earlier onset; it
neither invalidates these timings nor proves they were unaffected.

The original first pass preceded the decode-coordinate repair. A later pass
on that same node was also superseded after the node was administratively
flagged for pegged clocks; this does not establish which GPU was affected.
`superseded-node-observation.json` preserves the old identity and data hashes.
Both production tables come entirely from the final job on node 235, from fixed
source; no old row was relabeled or blended into these two tables.

The 20 new backend-fact entries were drafted from the new native labels and
reviewed against `platforms/cuda.py:91–116` at the pinned vLLM commit: SM100
FP8 KV selects FlashInfer; BF16 KV selects FlashInfer at local heads <=16 and
FlashMLA above that. Both appear in a BF16 precision slice because the registry
intentionally does not key on head count. Existing registry entries are unchanged.

## Reproduce

Check out collector revision `36f40cd4a611adb1590ccf0f2f8f006154c6bb72`.
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
