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
- `afd`: public `cli_estimate(mode="afd")` (ttft/tpot; the AFD session's
  per-op values cross the op-list evaluate FFI)

The case matrix: `SMOKE_CASES` x 4 surfaces, `POWER_CASES` (energy/power
coverage) x 4, `CP_CASES` (mixed only), `DSV4_CP_CASES` (cp_static_ctx +
mixed), `HYBRID_CASES` x 4 at a 1e-4 rtol, `SOL_CASES` (static+mixed) at
1e-4, the two #1456 site-transfer tie-break anchors
(`TIE_AGG_CASES`/`TIE_DISAGG_CASES`), and `AFD_CASES` — plus the
typed-error/provenance contract tests and the anti-vacuous golden guards.
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

Four existing B200 cases change because `deepseek_v32._dense_mlp_groups` now honors `first_k_dense_replace=3` and the checkpoint's packed-linear quantization exclusions. Previously those three dense layers were counted as MoE layers. DeepSeek-V3.2 now has 58 MoE plus three FP8-block dense layers (61 total); both GLM-5 cases have 75 MoE plus three BF16 dense layers (78 total). GLM-5.2 also has 75 MoE plus three dense layers, with its excluded packed gate/up and down projections using BF16 instead of inheriting the global NVFP4 GEMM mode. The VR200-only decode composition does not cause these B200 changes.

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

## Engine-Step Benchmark

Historical Python-vs-Rust speedup numbers (dated + commit-stamped) live in
[`perf-speedup-report.md`](../../perfmodel/docs/perf-speedup-report.md); they cannot be
regenerated (the Python arm is gone). The benchmark now times the rust
engine-step alone:

```bash
python aic-core/rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py --warmup 5 --iterations 50
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
