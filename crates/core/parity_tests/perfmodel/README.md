# Rust Engine Regression Tests (frozen goldens)

The migration harness that once compared the live Rust engine against the
live Python engine step. The Python step path is gone (dedup-plan Gate 3);
the golden fixtures captured from it while it was alive are now the
**permanent regression oracle** for the compiled engine.

## Pytest Suites

Run the engine-step golden checks:

```bash
uv run pytest -q -rx aic-core/rust/aiconfigurator-core/parity_tests/test_engine_step_parity.py
```

The suite compares the live Rust engine against **golden fixtures** for:

- `static`: `static_ctx`, `static_gen`, and `static_total` (plus the
  context/generation energy sums and power averages on the `POWER_CASES`,
  which sit on the power-carrying database identities)
- `mixed_step`: `estimate_mixed_step_latency_with_rust` for the same shape
- `cp_static_ctx`: the context phase through the cp-aware model builder
  (`cli_estimate` has no cp knob)
- `agg`: public `cli_estimate(mode="agg")`
- `disagg`: public `cli_estimate(mode="disagg")`

The case matrix: `SMOKE_CASES` x 4 surfaces, `POWER_CASES` (energy/power
coverage) x 4, `CP_CASES` (mixed only), `DSV4_CP_CASES` (cp_static_ctx +
mixed), `HYBRID_CASES` x 4 at a 1e-4 rtol, `SOL_CASES` (static+mixed) at
1e-4, and the two #1456 site-transfer tie-break anchors
(`TIE_AGG_CASES`/`TIE_DISAGG_CASES`) — plus the typed-error/provenance
contract tests and the anti-vacuous golden guards.
If an assertion fails, the message prints the golden value, Rust value,
absolute delta, percent delta, tolerance, and status for each metric.

`test_compile_engine_parity.py` covers the `compile_engine` -> `EngineHandle`
path specifically: op-transfer bincode round-trip fidelity, integration
checks against the frozen references, and the per-op FFI anchor
(`run_static_per_op` folded by name vs the frozen per-op
latency/energy/source dicts). Both suites run in the
`rust-engine-step-parity` CI job (`build-test.yml`).

Build the `aisimulate_core` extension first (the CI job does this with
`maturin develop --release`; from a clean checkout run
`cd aic-core && ../.venv/bin/maturin develop --release`), then return to the
repository root and run:

```bash
uv run pytest -q aic-core/rust/aiconfigurator-core/parity_tests/test_compile_engine_parity.py
```

## Golden Fixtures (captured at Gate 2, frozen at Gate 3)

- `goldens/engine_step.json` — every (case, surface) pair in
  `ENGINE_STEP_GOLDEN_MATRIX`, as `{"values": {...}}` or (error-symmetry
  cases) `{"error": ExceptionClassName}` records.
- `goldens/compile_engine.json` — the compile-engine subset references
  (static/mixed/decode per case + chunked-prefill, imbalance-scale, and
  WideEP references).
- `goldens/per_op.json` — the summary per-op dicts (latency + energy +
  source) for the compile-engine subset; the per-op op-list FFI anchor.

The FPM parity class (`TestRustEngineStepFpmParity`) follows the same
freeze-then-delete pattern with INLINE frozen references (`_FPM_*_FROZEN`):
its dataset is generated per-run from `_FPM_ROWS`, so it sits outside
`ENGINE_STEP_GOLDEN_MATRIX` and its Python-side values were frozen into the
test module at capture time.

The Python-era records were captured from the live Python engine step by the
retired `regenerate_goldens.py` (byte-reproducible: sorted keys, full float
repr, thread caps and capture HEAD in the header) and can never be
recaptured — the reference implementation is gone. **Never edit a frozen
value to silence a red test**: a Rust-vs-golden failure means the engine
drifted from the frozen reference; either the drift is a bug (fix it) or it
is a deliberate modeling change (pin the new values, below, and let the
golden diff carry the review).

### Post-freeze golden maintenance: `pin_goldens.py`

```bash
.venv/bin/python aic-core/rust/aiconfigurator-core/parity_tests/pin_goldens.py
```

- **Default (append-only)**: pins records for matrix entries the fixtures
  lack — i.e. parity cases added after the freeze. Values come from the live
  rust engine and are provenance-marked in the file's `post_freeze_pins` map,
  so python-era frozen values and rust-pinned values stay distinguishable.
- `--refresh KEY ...` / `--refresh-all`: recompute existing records after a
  **deliberate, reviewed** rust-side modeling change. The golden diff in the
  PR is the review artifact — reviewers see exactly which numbers moved and
  by how much.

The pin script keeps the retired capture script's guards: clean-tree
requirement, pinned thread caps, byte-reproducible output, and
all-payloads-before-any-write. `TestGoldenComparisonGuards` proves the
comparison itself still bites.

### MiMo-V2-Flash and Llama 4 attention layer types

The op-level SDK now reads which layers use full (global) attention as the
reference models define them. MiMo-V2-Flash marks a sliding-window layer with
`hybrid_layer_pattern[i] == 1`. Its packaged config has 9 full-attention
layers (0, 5, 11, ..., 47) and 39 sliding-window layers. The SDK used to model
39 full-attention layers and 9 sliding-window layers. In Llama 4, layers 3, 7,
11, ..., 47 use full attention (12 layers). The other 36 layers use chunked
attention. The SDK used to model every odd layer (24 layers) as full
attention.

Eight engine-step records were refreshed. The four
`mimo-v2-flash-b200-vllm-hybrid-attn-xshape` records failed their 1%
tolerance. The four `llama4-scout-b200-vllm-isl1024-osl2` records still passed
it, but they moved. Rule 1 in
`python/aisimulate/.claude/rules/rust-core/parity.md` asks a PR to refresh
every record that its change affects. The MiMo records were refreshed from
`38e518543f18e53c6c8a127ddb2c1d5830fa5a58` with a matching native build:

| Metric (ms) | Before | After |
| --- | ---: | ---: |
| Static context | 22.335 | 20.785 |
| Static generation | 4.416 | 4.421 |
| Static total | 26.751 | 25.206 |
| Mixed step | 23.678 | 22.132 |
| Aggregated TTFT | 69.838 | 68.098 |
| Aggregated TPOT | 23.678 | 22.132 |
| Aggregated request | 93.517 | 90.230 |
| Disaggregated TTFT | 44.224 | 41.153 |
| Disaggregated TPOT | 6.149 | 6.096 |
| Disaggregated request | 50.373 | 47.249 |

Only attention changes. At TP 8, both layer types have one KV head per GPU and
the same projection shapes. Every other operation therefore keeps its total.
Only the split of that total between the `global_*` and `swa_*` operations
changes. Every layer changes type: 39 layers move to the 128-token window, and
9 layers, with the dense layer 0, move to full attention. The net change is 30
window layers. Each layer type emits one attention operation, scaled by its
layer count. The two layer plans therefore give the cost of one layer of each
type:

- Static context: a layer costs 97.873 µs with full attention over 1,024
  tokens and 46.177 µs with the window, so the phase falls by
  30 × 51.696 µs = 1.551 ms.
- Static generation: in the decode step at batch 1, a window layer costs
  0.14 µs more (13.034 against 12.896 µs), so the phase rises by 0.004 ms.
- Mixed step: the 1,024-token prefill plus one decode token, −1.547 ms.
- Aggregated: TPOT is the mixed step. TTFT is (mixed step + 48 layers ×
  0.8 ms dispatch) × 1.125, the vLLM queueing factor at batch 2. Request
  latency is TTFT + TPOT.
- Disaggregated: TTFT is static context × 1.1 (prefill correction) × 1.8
  (TTFT correction). TPOT is the batch-4 decode step × 1.08 (decode
  correction). In that step, a window layer costs 1.650 µs less
  (13.311 against 14.961 µs), which is −0.0495 ms for 30 layers. Request
  latency is TTFT + TPOT.

The `llama4-scout-b200-vllm-isl1024-osl2` records were refreshed the same way.
"Recorded before" is the old fixture value. "Live at base" is the live engine
on the base revision, read with the same `pin_goldens.py --refresh` command.
"After" is the live engine with the layer types corrected, and it is the new
fixture value:

| Metric (ms) | Recorded before | Live at base | After |
| --- | ---: | ---: | ---: |
| Static context | 21.297 | 21.297 | 21.298 |
| Static generation | 4.619 | 4.619 | 4.615 |
| Static total | 25.916 | 25.916 | 25.913 |
| Mixed step | 22.020 | 22.061 | 22.059 |
| Aggregated TTFT | 67.973 | 68.019 | 68.016 |
| Aggregated TPOT | 22.020 | 22.061 | 22.059 |
| Aggregated request | 89.994 | 90.081 | 90.075 |
| Disaggregated TTFT | 42.169 | 42.169 | 42.170 |
| Disaggregated TPOT | 7.233 | 7.233 | 7.217 |
| Disaggregated request | 49.402 | 49.402 | 49.387 |

The move from "Live at base" to "After" comes from the layer-type correction.
The 24 odd layers used to be global. Now only layers 3, 7, ..., 47 are global,
so 12 layers change type. At ISL 1024, the 8,192-token chunk never limits
attention. The change therefore comes only from the attention lookup. The
window-8192 lookup costs a little more than the window-0 lookup in the context
phase and a little less in the decode step. Llama 4 gives a global layer and a
chunked layer the same projection and MoE shapes. Every other operation
therefore keeps its total. Only the split of that total between the `global_*`
and `swa_*` operations changes. The two layer plans give the cost of one layer
of each type:

- Static context: attention costs 38.411 µs per chunked layer and 38.283 µs per
  global layer, so the phase rises by 1.536 µs.
- Static generation: in the decode step at batch 1, attention costs 10.730 µs
  per chunked layer and 11.061 µs per global layer, so the phase falls by
  3.969 µs.
- Mixed step: the 1,024-token prefill plus one decode token, −2.437 µs.

The largest relative move is −0.22%, in disaggregated TPOT. Memory predictions
do not change, because HybridMoE KV-cache memory is the same for every layer
type.

The difference between "Recorded before" and "Live at base" is older than the
layer-type correction. The `post_freeze_pins` commits of the old Scout records
are not in this repository's history. The static and
disaggregated records were last pinned in PR #244. At the base, they match the
live values to the three decimals that they store. The mixed and aggregated
records still hold the values from the sync in PR #73. At the base, the live
engine gives a mixed step that is 0.186% higher than those records. This is the
largest difference, in `mixed_step`, which is also the aggregated TPOT. The
aggregated TTFT and request differ too, because they follow from the mixed
step. The difference stayed inside the 1% tolerance, so no PR refreshed these
records.

```bash
python/aisimulate/.venv/bin/python crates/core/parity_tests/perfmodel/pin_goldens.py \
  --refresh mimo-v2-flash-b200-vllm-hybrid-attn-xshape::static \
  mimo-v2-flash-b200-vllm-hybrid-attn-xshape::mixed \
  mimo-v2-flash-b200-vllm-hybrid-attn-xshape::agg \
  mimo-v2-flash-b200-vllm-hybrid-attn-xshape::disagg \
  llama4-scout-b200-vllm-isl1024-osl2::static \
  llama4-scout-b200-vllm-isl1024-osl2::mixed \
  llama4-scout-b200-vllm-isl1024-osl2::agg \
  llama4-scout-b200-vllm-isl1024-osl2::disagg
```

### GLM-5.3-Flash Ops data and the fleet `next` slot

Packaging the GLM-5.3-Flash Ops tables makes SGLang 0.5.20 and vLLM 0.31.0
the derived fleet `next` versions. The two Kimi-K3 B300 cases follow `next`,
so they now read the B300 SGLang 0.5.20 tables collected for GLM. The
change comes from `gemm_perf` and `custom_allreduce_perf`: with only the
GEMM table present, static context and generation change by +1.819 and
-0.607 ms; with only the all-reduce table, by -0.913 and -0.213 ms. With
only the 0.5.20 KDA, MoE, mHC or GLM attention table present, the records
are unchanged. Only the failing static and
disaggregated records were refreshed, from
`9a08d1557462c02a6cc8552360afee89538a0540` with a matching native build:

| Metric (ms), both cases | Before | After |
| --- | ---: | ---: |
| Static context | 87.881 | 88.761 |
| Static generation | 13.912 | 13.091 |
| Static total | 101.793 | 101.852 |
| Disaggregated TTFT | 174.004 | 175.747 |
| Disaggregated TPOT | 16.936 | 16.319 |
| Disaggregated request | 190.940 | 192.066 |

The aggregate references pass unchanged and the mixed records keep their
`PerfDataNotAvailableError`. The four Qwen3.8-Max GB300 cases pin SGLang
0.5.17 and vLLM 0.27.1, which are no longer queryable versions. They set
`allow_unlisted_version=True` and keep their references unchanged; the
parity tests and `pin_goldens.py` both admit that version through
`_surface_metrics`.

```bash
python/aisimulate/.venv/bin/python crates/core/parity_tests/perfmodel/pin_goldens.py \
  --refresh kimi-k3-b300-sglang-next-nospec::static \
  kimi-k3-b300-sglang-next-nospec::disagg \
  kimi-k3-b300-sglang-next-dspark-nextn7::static \
  kimi-k3-b300-sglang-next-dspark-nextn7::disagg
```

To reproduce the attribution, start from this PR with its native extension
built and with the pre-refresh `goldens/engine_step.json` from
`9a08d1557462c02a6cc8552360afee89538a0540` checked out. Move every
`data/b300_sxm/<family>/sglang/0.5.20` directory (`gemm`, `moe`, `kda`, `mhc`,
`comm`, `glm53_attention`) out of the systems tree. Then restore one family
at a time and run:

```bash
python/aisimulate/.venv/bin/python -m pytest -c python/aisimulate/pytest.ini -q -p no:timeout \
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py \
  -k 'kimi-k3-b300-sglang-next-nospec and (TestRustEngineStepStaticParity or TestRustEngineStepDisaggParity)'
```

With no 0.5.20 families present, the static and disaggregated records pass
against the old references. With only `gemm` or only `comm` present, both
records drift. `gemm` alone moves disaggregated TTFT/TPOT by +2.07%/−2.01%,
and `comm` alone moves them by −1.04%/−1.64%; the two together give the
refreshed values above. With only `kda`, `moe`, `mhc` or `glm53_attention`
present, both records pass.

### B60 graph-timing data promotion in PR #329

The `qwen3-30b-a3b-b60-vllm-scan-coverage` case follows the `current`
version slot, which moves from vLLM 0.26.0 to 0.28.0. Its static, mixed,
aggregate and disaggregated references were selectively refreshed from
`8f6faf712d0d6288b31aa41714a4ddb2fa590095` with a matching native build.
The aggregate reference also includes the new XPU backend's scheduling model.

| Metric (ms) | Before | After |
| --- | ---: | ---: |
| Static context | 164.960 | 133.227 |
| Static generation | 38.997 | 9.784 |
| Mixed step | 168.776 | 135.781 |
| Aggregate TTFT | 233.073 | 231.165 |
| Aggregate TPOT | 168.776 | 131.905 |
| Disaggregated TTFT | 326.621 | 263.788 |
| Disaggregated TPOT | 33.115 | 19.844 |

Selecting 0.26.0 explicitly on the PR reproduces the base revision's static
per-operation latencies. With 0.28.0, generation MoE changes from 21.219994
to 3.039994 ms, attention from 5.025209 to 2.485675 ms, and each MoE
dispatch phase from 3.638394 to 1.008275 ms. The new measurements account
for the static shift; these references do not independently establish
whole-model B60 silicon accuracy.

From a clean checkout of the source revision, with its native extension built
and development dependencies installed, regenerate only these four records:

```bash
PYTHONPATH=python/aisimulate/src:python/aisimulate \
python/aisimulate/.venv/bin/python crates/core/parity_tests/perfmodel/pin_goldens.py \
  --refresh qwen3-30b-a3b-b60-vllm-scan-coverage::static \
  qwen3-30b-a3b-b60-vllm-scan-coverage::mixed \
  qwen3-30b-a3b-b60-vllm-scan-coverage::agg \
  qwen3-30b-a3b-b60-vllm-scan-coverage::disagg
```

The fixture records the source revision in `post_freeze_pins`. Other cases,
the test matrix and tolerances are unchanged.

### FP8-block correction in PR #244

The selective refresh from `0a51476b6ab90bc0e475bd41d4b1c7abbef07b95`
covers 68 engine-step records, 19 compiled-engine references, and three
per-op cases. It follows removal of eager vLLM 0.24.0 FP8-block timings
and explicit reuse of the graph-timed 0.25.0 GEMM table. Declared reuse
also fills missing GEMM shapes for the other retained precisions.
Hand-built Rust engine/FPM fixtures that query this 0.24.0 FP8-block identity
must use `PerfDatabase::load_resolved` with shared-layer reuse enabled, as the
production engine does. Primary-only `PerfDatabase::load` intentionally cannot
answer those removed rows.

The three per-op cases change only 12 QKV/projection GEMM latency values;
other per-op latencies, energies, and source labels are unchanged. For
MiniMax-M2.5 (B200, ISL 1024, OSL 2), context QKV GEMM changes from
11.926155 to 0.948021 ms and generation QKV GEMM from 16.582272 to
0.565109 ms. Static, mixed-step, aggregated/disaggregated, chunked-prefill,
and imbalance-scale references inherit these data changes.

Only records implicated by the failed golden comparisons were refreshed,
using `pin_goldens.py --refresh`. Each refreshed record retains its source
commit in `post_freeze_pins`; test matrices and tolerances are unchanged.
These are prediction-regression baselines, not whole-model silicon validation.

### B200 SGLang dense-prefix correction in PR #303

Four existing B200 cases change because `deepseek_v32._dense_mlp_groups` now honors `first_k_dense_replace=3` and the checkpoint's packed-linear quantization exclusions. Previously those three dense layers were counted as MoE layers. DeepSeek-V3.2 now has 58 MoE plus three FP8-block dense layers (61 total); both GLM-5 cases have 75 MoE plus three BF16 dense layers (78 total). GLM-5.2 also has 75 MoE plus three dense layers, with its excluded packed gate/up and down projections using BF16 instead of inheriting the global NVFP4 GEMM mode.

The following deltas are `head 91687c4 - base 1267d0f`, in milliseconds, rounded to six decimal places. Full-precision reference values remain in `goldens/engine_step.json` and its base revision.

| Case ID | Mixed step | Static context | Static generation | Static total |
| --- | ---: | ---: | ---: | ---: |
| `deepseek-v32-b200-sglang-isl1024-osl2` | -1.384795 | -1.399000 | -0.150000 | -1.549000 |
| `glm5-b200-sglang-empirical` | -1.400422 | -1.427000 | -0.030000 | -1.457000 |
| `glm5-b200-sglang-isl16384-osl2` | -7.620506 | -7.620000 | -0.030000 | -7.650000 |
| `glm52-b200-sglang-isl1024-osl2` | -0.514094 | -0.533000 | -0.007000 | -0.540000 |

| Case | Aggregate TTFT / TPOT / request | Disaggregated TTFT / TPOT / request |
| --- | --- | --- |
| DeepSeek-V3.2 | -2.702024 / -1.385653 / -4.087678 | -2.768000 / -0.193000 / -2.961000 |
| GLM-5 empirical | `RuntimeError` unchanged | `RuntimeError` unchanged |
| GLM-5 ISL 16384 | `RuntimeError` unchanged | `RuntimeError` unchanged |
| GLM-5.2 | -1.002665 / -0.514187 / -1.516852 | -1.053000 / -0.054000 / -1.107000 |

The causal check kept Rust and the performance tables fixed and changed only dense-layer grouping in memory. Returning no dense groups reproduced all base records; restoring the current grouping reproduced all head records. An intermediate run retained three dense layers but forced their quantization to the global mode. For GLM-5.2, `mixed_step` decomposes into -0.520653620318299 ms from the dense/MoE layer correction plus +0.006560000280530 ms from honoring BF16 exclusions, giving -0.514093620037769 ms overall. Its rounded `static_total` similarly changes by -0.561 + 0.021 = -0.540 ms. The quantization-only delta is zero for the other three cases. These 16 surface comparisons cover 28 numeric metrics and preserve all four recorded `RuntimeError` outcomes; no golden values or tolerances were changed during this review.

To reproduce the current calculations without modifying the references, build the matching native extension and run from the repository root:

```bash
python/aisimulate/.venv/bin/python -m pytest -q -p no:timeout \
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py \
  -k 'deepseek-v32-b200-sglang-isl1024-osl2 or glm5-b200-sglang-empirical or glm5-b200-sglang-isl16384-osl2 or glm52-b200-sglang-isl1024-osl2'
```

The selective reference-generation command is `pin_goldens.py --refresh deepseek-v32-b200-sglang-isl1024-osl2 glm5-b200-sglang-empirical glm5-b200-sglang-isl16384-osl2 glm52-b200-sglang-isl1024-osl2` using the script in this directory and a clean, isolated checkout with a matching native build. It obtains each metric from the same `_surface_metrics(case, surface)` Rust-backed consumers as the tests. Keep error records unchanged when a surface still raises; these B200 regressions do not establish additional GPU accuracy coverage.

### GLM-5.3-Flash mixed-step attention phases in PR #323

`Glm53Attention` operations are named per layer (`attention_N`), so the
mixed-step attention filters did not recognize them. Context attention rode
the shared pass at the combined context-plus-decode shape, and generation
attention was omitted. The filters now classify these operations by
`is_context`. Only the two GLM SOL `mixed` records were refreshed with
`pin_goldens.py --refresh` (ISL 4096, prefix 2048, one decode request). Both
now include decode attention, and context attention is queried at the
prefill shape. `glm53flash-fp8-gb300-vllm-tp2-sol` changes from 37.865342 to
38.665549 ms. `glm53flash-nvfp4-gb300-sglang-tp4-sol` changes from 22.443503
to 22.869943 ms. Static records and all other models are unchanged.

### GLM-5.3-Flash router precision in PR #323

The GLM gate projection is a BF16 x BF16 tensor-core GEMM with FP32 logits in
both pinned runtimes (vLLM `GateLinear`, SGLang `MoEGate`). Its SOL previously
used scalar FP32 throughput with FP32 weight and input traffic. It now uses
BF16 tensor throughput, BF16 weight and input bytes, and FP32 output bytes.
The four GLM SOL records were refreshed with `pin_goldens.py --refresh`; the
offline checkpoint registration in the same PR does not change them.

| Case ID | Mixed step | Static context | Static generation | Static total |
| --- | --- | --- | --- | --- |
| `glm53flash-fp8-gb300-vllm-tp2-sol` | 38.665549 -> 36.240498 | 37.856 -> 35.432 | 1.553 -> 1.540 | 39.409 -> 36.972 |
| `glm53flash-nvfp4-gb300-sglang-tp4-sol` | 22.869943 -> 20.444891 | 22.435 -> 20.011 | 0.738 -> 0.726 | 23.173 -> 20.737 |

Values are in milliseconds. All other records are unchanged.

## Engine-Step Benchmark

[The engine-step benchmark](benchmark_engine_step.py) measures the current Rust
engine. Run it from the repository root with the matching native extension
installed:

```bash
python crates/core/parity_tests/perfmodel/benchmark_engine_step.py --warmup 5 --iterations 50
```

When `--case` is omitted, the benchmark runs all predefined cases.
Before each case starts, the script clears Python database/op/model caches and
Rust estimator/library caches. Use `--cache-mode cold` when every timed sample
should clear runtime caches first, and `--json` for machine-readable output.

The relative Rust-vs-Python CI perf gate (`test_engine_step_perf.py`)
retired with the Python step: its floors encoded "Rust must not lose to
Python", which the migration completed. If an absolute perf tripwire is
wanted, pin per-case wall-clock budgets from this benchmark on a quiet host.

### B200 power-import golden delta

The power import from AIConfigurator commit
`915f590680d8a79fe9c39f6f3a9ff13bc267fcce` adds measured B200 TensorRT-LLM
1.3.0rc20 power columns while preserving timing identities. The per-op goldens
for GPT-OSS-20B and Nemotron-Super-49B were regenerated with the repository
`pin_goldens.py --refresh` workflow at AISimulate commit
`36dcc8f3afe9e6e2e9de976737b6337fad8c4d74`, using that checkout's rebuilt native
extension. All 29 changed numeric fields are energy values moving from zero
to positive W-ms. Every latency and source tag is unchanged. For example,
GPT-OSS context attention changes from 0 to 443.84428875568517 W-ms and
Nemotron context all-reduce changes from 0 to 1241.060314309411 W-ms.
Both cases now require a nonzero energy comparison in `_POWER_SUBSET_IDS`.
These are regression expectations derived from imported operation measurements,
not independent silicon-accuracy qualification.

#### Recorded refresh commands

The following commands ran from the repository root at
`36dcc8f3afe9e6e2e9de976737b6337fad8c4d74`, after rebuilding the native extension
for that checkout. The pin script requires a clean input tree; reproduce a
historical refresh in an isolated checkout of that revision.

```bash
python/aisimulate/.venv/bin/python crates/core/parity_tests/perfmodel/pin_goldens.py \
  --refresh gpt-oss-20b-b200-trtllm-isl1024-osl2 \
  nemotron-nas-b200-trtllm-isl1024-osl2

python/aisimulate/.venv/bin/python -m pytest -q -p no:timeout \
  crates/core/parity_tests/perfmodel/test_compile_engine_parity.py
```

Recorded results: **2 per-op records pinned**, followed by **67 compile-engine
tests passed**. The refreshed goldens and nonzero-energy guards were committed
as `574c0d9ce64438c4a1c4e0fbbe58c5a9d4cd4de8`. An engine-step suite result was
not recorded with that historical refresh.

Both suites subsequently passed in the
[Engine Golden Regression job](https://github.com/ai-dynamo/aisimulate/actions/runs/35043503755/job/104628335804)
at `a83ad4f162669b318786cc279f320650ec09d5b1`, using a release native build:
**298 engine-step tests passed** and **67 compile-engine tests passed**. The
job ran these exact commands from the repository root:

```bash
python -m pytest -q -rx -n 4 -c python/aisimulate/pytest.ini \
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py

python -m pytest -q -rx -n 4 -c python/aisimulate/pytest.ini \
  crates/core/parity_tests/perfmodel/test_compile_engine_parity.py
```

For local reproduction, activate the repository environment with a native
extension built from matching source. On macOS, add `-p no:timeout` as described
in `AGENTS.md`. Later PR heads require their own CI results; the linked run
records the verified revision rather than certifying future changes.

The two identity-merged B200 attention tables include pinned upstream artifacts
in `power_upstream/*.parquet.source`. Focused tests in
`python/aisimulate/tests/unit/tools/test_power_data.py` check their upstream
SHA-256, every imported schema, identity and measurement, and the paired-zero
sentinel on local-only identities. These evidence files are excluded from
`*.parquet` runtime-table discovery.
