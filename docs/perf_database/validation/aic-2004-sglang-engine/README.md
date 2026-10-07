# DSA reuse-layer accounting validation

SGLang already models GLM-5.2 as 21 full-indexer layers and 57 reuse layers.
The defect was in the reuse-layer roofline: decode charged the full indexer,
and context retained indexer projection work even when the indexer was skipped.
Beyond a decode table's sequence range, this also scaled a measured reuse row
by the growing full-indexer cache cost.

The fix excludes the indexer's Q, K and weighting projections, logits and cache
access from reuse-layer compute/memory traffic. Attention QKV, absorbed BMMs,
sparse MLA and output projection remain. Decode table extrapolation, empirical
calibration, SOL diagnostics and FPM's SOL-based transfer use the same variant.
The FPM decode roofline now applies the existing `21/78` full-layer fraction.
This changes timing calculations, not memory-capacity or scheduler accounting.

Checkpoint exclusion parsing also misclassified norm exclusions:
GLM FP8's `self_attn.q_a_layernorm` and `kv_a_layernorm` entries were treated as
exclusions of the entire attention block. Named norms and `indexers_proj` now
leave the FP8 projection table key intact. Across bundled configurations, this
changes GLM-5, 5.1, 5.2 and 5.3 FP8 keys; all 25 NVFP4 configurations retain
their previous whole-block or per-projection exclusion sets.

The executed boundary is documented by SGLang at immutable revision
[`49e384ce9d304648e9959666ecb8ce8cd98d0deb`](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py#L261-L295):
reuse layers bypass the entire indexer call. Its projection ownership is in
[`dsa_indexer.py`](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1352).
These are API/behavior references; no upstream implementation is copied here.

[`cpu-validation.json`](cpu-validation.json) records actual baseline/fixed Rust
SDK calls against identical existing table bytes. All 15 queried latencies and
all full-layer rooflines remain unchanged; reuse-layer rooflines change. An
independent GLM projection ledger and a synthetic 2 ms measured anchor test
verify the new arithmetic and extrapolation behavior. Neither fixture is a GPU
accuracy result. The two parity suites pass all 386 cases without golden changes;
46 DSA Rust tests pass, including FPM blend and invalid-fraction cases.

To reproduce the SDK receipt using either wheel:

```bash
python docs/perf_database/validation/aic-2004-sglang-engine/validate.py \
  --label baseline-or-fixed \
  --systems-root python/aisimulate/src/aisimulate_core/systems \
  --output sdk-receipt.json
```

Use independently built wheels and compare their native-library hashes. A
shared Cargo target can reuse an older-mtime archive's wrong native artifact;
the baseline receipt here uses a separate build directory and verified distinct
native bytes and behavior. The script calls Rust for every estimate and uses
`RustForwardPassPerfModel.best_available` for whole-model composition.

B200 decode recollection and bounded held-out validation are recorded in
[`gpu-decode-validation.json`](gpu-decode-validation.json), with the six original
rows in [`measured-decode.parquet`](measured-decode.parquet). Each case executes
one new token, with native input IDs, positions and total KV length verified;
the collector records 50 CUDA graph iterations under stock SGLang 0.5.14.
Both engines return all six measured coordinates exactly. The existing
21-full/57-reuse composition also matches all three measured-row sums.

With only the history-8192 full/skip pair retained, the two longer histories
produce these unweighted held-out errors:

| Component | Baseline MAPE | Fixed MAPE |
| --- | ---: | ---: |
| Full layer | 48.18% | 48.18% |
| Reuse layer | 91.35% | 1.99% |
| 21 full + 57 reuse layers | 18.64% | 35.41% |

The full-layer extrapolation still underestimates the long histories. Removing
the reuse overestimate removes an accidental cancellation, so the combined
attention error increases. This is a remaining limitation, not an accuracy
acceptance pass. The combined reference is a sum of module observations, not
an eight-GPU whole forward. These points share one campaign and synthetic
inputs; they do not establish checkpoint-value parity or full-matrix accuracy.
Historical GLM/SGLang whole-forward data uses a different framework revision,
model revision and speculative setup. Whole-model accuracy remains
`NOT_EVALUATED` for stock SGLang 0.5.14. The earlier six-point trial omitted
native input materialization and is excluded from this validation.

To isolate a new GPU decode slice without shared-source leakage:

```bash
python docs/perf_database/validation/aic-2004-sglang-engine/prepare_decode_validation.py \
  --source-parquet docs/perf_database/validation/aic-2004-sglang-engine/measured-decode.parquet \
  --systems-root python/aisimulate/src/aisimulate_core/systems \
  --output-dir /path/to/new/private-validation
python docs/perf_database/validation/aic-2004-sglang-engine/validate.py \
  --label baseline-or-fixed-anchor --context \
  --systems-root /path/to/new/private-validation/anchor/systems \
  --output sdk-anchor.json
```

Run the second command with both independently built wheels, and repeat using
the `exact/systems` root. The preparation script retains the full/skip pairs at
history lengths 8192, 131072 and 1048575 for exact lookup; only the 8192 pair is
available to the held-out estimates. It removes every other DSA generation
table from each private systems root, including other backends and versions.
Full, reuse and the `21 * full + 57 * reuse` attention composition must all be
reported: fixing one variant can remove an accidental cancellation with error
in the other. The composition is a sum of module measurements, not a measured
whole-model reference.

The projection-key change has an additional real-data check in
[`gpu-fp8-consumption.json`](gpu-fp8-consumption.json). Both binaries return
all 14 native GLM FP8 measurements exactly when explicitly given the FP8 key.
With only these FP8 rows available, the baseline canonical
`zai-org/GLM-5.2-FP8` model instead requests BF16 and fails all seven queries;
ordinary legacy data could silently satisfy that incorrect key. The fixed
canonical model needs no quantization override and matches all seven
`21 * full + 57 * reuse` attention sums, covering five prefill coordinates
(including one batch-32 point) and two decode histories. This proves precision
routing and table consumption, not whole-model accuracy. In particular, the
native prefill module measurements still include the compiler entry cost for
one standalone layer; its amortization across the serving model is unqualified.

To reproduce this isolated consumer check with either wheel:

```bash
python docs/perf_database/validation/aic-2004-sglang-engine/validate_fp8_consumption.py \
  --label baseline-or-fixed \
  --systems-root python/aisimulate/src/aisimulate_core/systems \
  --context-parquet docs/perf_database/validation/aic-2004-sglang-engine/measured-fp8-context.parquet \
  --generation-parquet docs/perf_database/validation/aic-2004-sglang-engine/measured-fp8-generation.parquet \
  --scratch-root /path/to/private-scratch \
  --output fp8-consumption.json
```

This creates and removes its own temporary systems copy, strips all other DSA
context/generation donors, and leaves the production database unchanged.

This change publishes bounded overlays and retains the packaged SGLang tables.
[`production-replacement-risk.json`](production-replacement-risk.json) records
the actual Rust source reports and existing table hashes/counts: B200 SGLang
0.5.14 has 95,203 context rows and 6,048 decode rows, with batches through 1024.
Each table has exactly one admitted primary and **zero fallback sources**.
The loader merges sources by coordinate, but there are no earlier SGLang or
authorized cross-backend DSA rows to fill the holes after whole-file replacement.

New measurements include a batch-32 full/skip pair at 128 new tokens and a
128-token prefix; they are not restricted entirely to batch 1. That bounded
sampling still does not qualify interpolation across the retired grid. The
separate `glm5_dsa_attn` table cannot fill monolithic DSA holes, and CP modeling
retains its legacy MQA/topK dependencies. Preserving overlays avoids silently
shrinking default coverage; it also means the packaged default measurements
retain the identified collector limitations. Default-data and whole-model
accuracy are not declared repaired by these validation results.
