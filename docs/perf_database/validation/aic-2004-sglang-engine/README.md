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

Checkpoint exclusion parsing also distinguished norm exclusions incorrectly:
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

B200 recollection and held-out validation are pending. They will distinguish
exact measured-coordinate lookups from extrapolation against withheld points.
Synthetic inputs do not establish checkpoint-value parity. The available
historical GLM/SGLang whole-forward data uses a different framework revision,
model revision and speculative setup; whole-model accuracy remains
`NOT_EVALUATED` for stock SGLang 0.5.14.

To isolate a new GPU decode slice without shared-source leakage:

```bash
python docs/perf_database/validation/aic-2004-sglang-engine/prepare_decode_validation.py \
  --source-parquet /path/to/collected/dsa_generation_module_perf.parquet \
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
