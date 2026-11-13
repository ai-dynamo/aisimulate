<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python performance-model API

The `aisim` wheel provides the `aisim_core` estimator namespace and
native extension. It does not require a separate core wheel or Dynamo.

Operation energy alone does not guarantee published power. Unified reporting
also applies the [modeled-power coverage gate](../power.md#publication-gate).

## Canonical estimator

```python
from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

config = ForwardPassPerfModelConfig(
    model="Qwen/Qwen3-32B", system="h200_sxm", backend="vllm",
    worker_type="aggregated", tp=2,
    estimation_mode="auto", fallback_policy="deny",
)
model = RustForwardPassPerfModel.best_available(config)
```

| Method | Contract |
| --- | --- |
| `estimate_forward_pass_time_ms(metrics, *, draft_length=None)` | One iteration's rank metrics, supplied as a dictionary or list of rank dictionaries; returns milliseconds, or `None` for an unready regression store. Empty work returns zero. A positive integer `draft_length` opts an ordinary STP FPM into the speculative decode approximation below. |
| `estimate_forward_pass_detailed(metrics, *, draft_length=None)` | With no draft length, returns the scalar latency plus native latency, correction, rank composition, and any direct lookup evidence. With a draft length, returns `latency_ms`, `draft_length`, `scaling_factor`, and the unmodified `stp_reference` detailed estimate. |
| `tune_with_fpms(iterations)` | Consume observed iterations for correction or regression. Preserve rank/iteration grouping and worker identity. |
| `static_phase_latency(batch_size=..., input_tokens=..., output_tokens=..., prefill=...)` | Native static phase integration before correction. Prefill is one pass; decode is total sequence decode, so `output_tokens=2` represents one decode iteration. |
| `static_phase_diagnostics(...)` | Operation diagnostics for supported native paths; details below. |
| `diagnostics()` | Readiness and requested/resolved estimator provenance. |
| `regression_store_diagnostics()` | Independent per-workload-store fit and retention state. |
| `estimate_cache_budget(config, budget)` | Class-level canonical resource-budget API; see [memory](../memory.md). |

Use [configuration](../configuration.md) for identity and controls,
[whole-forward FPM](../methods/whole-forward.md) for measured query domains,
and [online regression](../methods/online-regression.md) for telemetry semantics.
`best_available` validates configuration before selection; query-domain errors
do not silently change the chosen estimator. The raw PyO3 class and SDK wrapper
are different surfaces: new applications should use the SDK wrapper.

For an ordinary STP interpolation, regression or learned FPM, keyword-only
`draft_length=d` estimates one complete speculative decode iteration using
`(0.08*d + 1.25) * F_STP((d+1)*B, K_total)`. Supply logical decode request
counts and total past KV: the query widens the request count and preserves KV.
`d` must be a positive integer; booleans, non-integers and overflowing widths
are rejected. Only pure decode (with optional idle ranks) is supported;
prefill/mixed work, OP estimators and already-speculative estimators are rejected.
Omitting the argument preserves existing behavior. The option does not change
configuration, tuning or replay execution. See the
[whole-forward contract](../methods/whole-forward.md#estimate-speculative-decode-from-an-stp-fpm)
for reference coverage, detailed evidence and calibration limits. The raw
`aisim_core._native.RustForwardPassPerfModel` methods accept the same
keyword-only option with JSON-string `fpm_json` input; their detailed result
is a JSON string, while the SDK returns a dictionary.

<a id="ais-perf-model-loader"></a>

## `ais_perf_model` loader

`aisim_core.ais_perf_model` reads the
[unified perf-model input](../configuration.md#unified-perf-model-input) and
returns the mapping that `RustForwardPassPerfModel.best_available` takes. The
module is pure Python and imports only the standard library when it is
imported. It checks the shape of the value, paths, `from`, merging and fill.
It does not know the perf-model fields, their values or their defaults: the
constructor validates those when the model is built.

```python
from aisim_core.ais_perf_model import load, resolve
from aisim_core.sdk import RustForwardPassPerfModel

document = load("./perf-model.yaml")
print(document.worker_types)

entry = document.select("decode")
config = resolve(entry, worker_type="decode", defaults={"estimation_mode": "op_level"})
print(config)

model = RustForwardPassPerfModel.best_available(config)
```

With these two files in the working directory:

```yaml
# perf-model.yaml
- worker_type: prefill
  model: Qwen/Qwen3-32B
  system: h200_sxm
  backend: vllm
- worker_type: decode
  from: ./base.yaml
  tp: 2
```

```yaml
# base.yaml
model: Qwen/Qwen3-32B
system: h200_sxm
backend: vllm
tp: 1
```

the script prints the worker types and the decode config, and then builds
the model:

```text
('prefill', 'decode')
{'model': 'Qwen/Qwen3-32B', 'system': 'h200_sxm', 'backend': 'vllm', 'tp': 2, 'worker_type': 'decode', 'estimation_mode': 'op_level'}
```

The entry's `tp: 2` wins over the template's `tp: 1`. `defaults` adds
`estimation_mode` because neither file writes it.

The public names are the ones in `__all__`:

```python
def load(
    value: str | os.PathLike[str] | Mapping[str, Any] | Sequence[Any],
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> AisPerfModelDocument: ...

def expand_template(entry: Mapping[str, Any]) -> dict[str, Any]: ...

def resolve(
    entry: Mapping[str, Any],
    *,
    worker_type: str | None = None,
    fill: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] | None = None,
    workspace: str | os.PathLike[str] | None = None,
) -> dict[str, Any]: ...

def warn_deprecated(old: str | Sequence[str], new: str, *, stacklevel: int = 2) -> None: ...

@dataclass(frozen=True)
class AisPerfModelDocument:
    entries: tuple[dict[str, Any], ...]
    bare: bool
    source: str | None = None

    @property
    def worker_types(self) -> tuple[str, ...]: ...
    def select(self, worker_type: str) -> dict[str, Any]: ...
    def to_data(self) -> dict[str, Any] | list[dict[str, Any]]: ...

class AisPerfModelError(ValueError): ...
class AisPerfModelDeprecationWarning(FutureWarning): ...

WORKER_TYPES = ("prefill", "decode", "aggregated")
ESTIMATOR_KEYS = (
    "estimation_mode", "fallback_policy", "estimator_config", "systems_paths",
    "database_mode", "transfer_policy", "fpm_profile",
)
DEPRECATION_PREFIX = "ais-perf-model: "
```

| Name | Contract |
| --- | --- |
| `load` | Loads a value. A string that starts with `{` or `[` is inline JSON. Any other string, and a path object, is a file: `.json` is read as JSON, any other file as YAML. A mapping or a list is parsed data. Relative paths inside a file start at the file's folder. For inline JSON and parsed data they start at `base_dir`, which defaults to the working directory; a relative file path starts there too. Symbolic links are treated as [Relative paths](../configuration.md#relative-paths) says, and a link at the end of `base_dir` is followed. A key written twice in a file or in inline JSON is an error. |
| `AisPerfModelDocument` | A loaded value. `entries` holds the configs as written: relative paths are absolute and `null` keys are dropped, but `from` is not followed. `bare` is true when the value was one config and not a list. `source` is the file path, or `None` for inline JSON and parsed data. |
| `AisPerfModelDocument.worker_types` | The worker types the document names, in document order. |
| `AisPerfModelDocument.select` | A deep copy of the entry for `worker_type`. One config without `worker_type` serves every worker type and gets `worker_type` set in the copy. For a worker type the document does not have, the error lists the ones it has. |
| `AisPerfModelDocument.to_data` | The value in the shape it was written, with absolute paths. |
| `expand_template` | Merges the template file named by `from: <file>` into the entry. Keys of the entry win, `estimator_config` is merged mapping by mapping, and lists and scalars are replaced. The result has no `from`. An entry without `from`, or with a dataset or state snapshot `from`, is returned unchanged. |
| `resolve` | Returns the mapping for `best_available`: no `from`, no `null` keys at the top level, and only absolute paths. Pass the entry as written, from `select`, not a mapping that is already resolved. `worker_type` is set when the entry leaves it out and must match when the entry writes it. `fill` holds values the caller knows: a missing key is added, and a written key must be equal. `defaults` only add what is still missing. A dataset or a state snapshot is fixed: the result is its saved `config.json`, written keys and `fill` are only compared, and `defaults` are ignored. A dataset entry needs `workspace`, the folder in which the dataset is prepared once and reused after that. |
| `AisPerfModelError` | The value is wrong. The message says what to fix. `load`, `select`, `expand_template` and `resolve` raise it. |
| `warn_deprecated` | Emits one `AisPerfModelDeprecationWarning` that names the old field, or every old field of a sequence, and the replacement `new`. No other code emits this category. `stacklevel` counts as it does for `warnings.warn` called in place of this function. |
| `AisPerfModelDeprecationWarning` | The category of the warning for an old perf-model field. Callers can filter on it. |
| `WORKER_TYPES` | The worker types an entry can name. |
| `ESTIMATOR_KEYS` | The keys of an entry that set the estimator. |
| `DEPRECATION_PREFIX` | The text that every deprecation warning starts with. |

`fill` and `defaults` match keys by their exact spelling, so use the
perf-model field names, such as `tp`. A `null` inside `estimator_config` is a
value: it is kept and compared, and it stops `defaults` from filling that key.
The rules for entries, `from` and paths are in
[Unified perf-model input](../configuration.md#unified-perf-model-input).

## Stable Python facade

New Python code should import from the small facade:

```python
from aisim_core.sdk import (
    EngineHandle,
    ModelConfig,
    RuntimeConfig,
    RustForwardPassPerfModel,
    compile_engine,
    estimate_kv_cache,
    estimate_num_gpu_blocks,
)
```

The explicit module paths remain supported:

```python
from aisim_core.sdk.engine import EngineHandle, compile_engine
from aisim_core.sdk.rust_engine_step import RustForwardPassPerfModel
from aisim_core.sdk.memory import estimate_kv_cache, estimate_num_gpu_blocks
```

`aisim_core.sdk.__all__` is the supported high-level surface. The
facade resolves lazily, so importing it does not load the model registry,
performance database, or native engine until a name is used.

The top-level `aisim_core` module exposes the lower-level native
extension contract:

- `AicEngine`
- `RustForwardPassPerfModel` (the raw PyO3 class, distinct from the ergonomic
  SDK wrapper)
- `engine_spec_bincode_from_json`
- `_build_smoke`

The wheel includes `py.typed` and a stub for that native extension. The SDK
Python modules carry their own annotations.

### Context-attention kernel queries

`AicEngine.evaluate_context_attention_kernels_json` accepts `ops_json` (a JSON
array of serialized `ContextAttention` operations), `batch_size`, and `s`
(sequence length in tokens), with optional `prefix=0`,
`imbalance_correction_scale=1.0`, and `visual_block_upper_triangle=False`.
It evaluates only the attention kernels, excluding fused QK normalization,
RoPE, and KV-write work. Invalid JSON and other operation families raise
`ValueError`.

With `visual_block_upper_triangle=True`, the query prices only the additional
strict upper-triangle attention pairs inside a bidirectional visual block,
using the underlying causal kernel's database policy and provenance. This
mode requires `prefix=0`; a nonzero prefix raises `ValueError`. A block with
zero or one token has zero additional latency and energy, with `source="sol"`.
The flag is a runtime query option and does not change the serialized op or
engine-spec formats.

Results are `list[tuple[str, float, float, str]]`, containing
`(name, latency_ms, energy_wms, source)` with repeated operation names folded
together. Energy is in watt-milliseconds and is zero when power data is
unavailable. An empty operation list returns an empty list. `aisim_core.AicEngine` exposes this
method; `EngineHandle` provides an annotated SDK wrapper with the same query
options.

## Direct FPM query evidence

The same model returned by `RustForwardPassPerfModel.best_available(config)` exposes `estimate_forward_pass_detailed(metrics)` (also available on the Rust `ForwardPassPerfModel`). Its `latency_ms` equals `estimate_forward_pass_time_ms(metrics)` with the same query options. The schema below applies when `draft_length` is omitted; speculative queries retain this evidence under `stp_reference`. Four-element per-operation tuple APIs are unchanged.

```python
result = model.estimate_forward_pass_detailed({
    "scheduled_requests": {
        "num_prefill_requests": 1,
        "sum_prefill_tokens": 512,
        "sum_prefill_kv_tokens": 256,
    },
})
for rank in result["ranks"]:
    for query in rank["queries"]:
        print(query["resolution"], query["query"], query["support"])
```

Each direct lookup records `exact_lookup`, `within_curve_interpolation`, `cross_kv_interpolation`, `cross_batch_interpolation`, or `kv_first_interpolation`, its phase and model, requested coordinates, raw latency, and supporting measured coordinates, latencies and weights. Coordinates are iteration totals: `batch_size`, `total_prefill_tokens` (null for decode), and `total_kv_read_tokens`. Nested interpolation weights are multiplied for each supporting measurement. Prefill's `cross_kv_interpolation` identifies the usual token-first order; `kv_first_interpolation` identifies its KV-first fallback. Support always lists original measurements rather than intermediate estimates. This evidence preserves `source="silicon"`, whose meaning includes interpolation; it does not establish accuracy or observed CUDA graph dispatch.

Weights describe a raw lookup, before later composition. Each rank records prefill, decode and mixed-pass baseline latencies, plus `marginal_decode_ms = max(decode_ms - decode_baseline_ms, 0)`. A baseline lookup has `decode_baseline=true`: its query is the paired decode request, but its support contains each selected row's actual measured minimum-KV point. `max_rank` is the first zero-based input index attaining the positive native maximum. The result separately records `native_latency_ms`, `correction_factor`, and final `latency_ms`. Empty work has no query evidence. Operation graphs (op-level and DeepSeek-V4.1 stage graphs) instead record each rank's name-folded `operations`, with the [static phase diagnostics](#static-phase-diagnostics) fields, summing to the rank's uncorrected latency; see [per-iteration queries](../configuration.md#per-iteration-queries). Learned and regression estimators have no rank evidence; an unready regression retains `latency_ms=null`.

Replay and prediction capture this evidence with `ReplayOutputRequirements(capture_performance_diagnostics=True)`; CLI prediction enables it with `--detail source` or `--detail time`. Use `--format json` or inspect the saved `prediction.json` for the query records and measured support; the default source text output shows operation sources and fallbacks. Public `run_recommendation(..., output_requirements=ReplayOutputRequirements(capture_performance_diagnostics=True))` and `Sweeper(..., output_requirements=...)` forward the same request to each candidate's replay. Evidence survives saved results under `ReplayReport.metadata["fpm_query_evidence"]` and each candidate's `provenance.runner_metadata["fpm_query_evidence"]`. The native replay report carries the same top-level key; source details include `fpm_estimates` on whole-model operations. FPM power remains unavailable and SOL comparison has an explicit unavailable reason.

Captured replay records are grouped by timing provider and phase. Each `fpm_estimates` entry holds an `estimate`, an invocation `count`, and `latency_scale` for any synthetic replay speedup; raw support is never rescaled. Identical query coordinates within an immutable replay provider share one counted record. Counts describe timing invocations in the measurement epoch, not independent silicon observations or served requests. Distinct queries are retained without truncation, so capture memory grows with distinct coordinates; ordinary runs do not retain query histories. Aggregated topology has provider index 0; disaggregated topology orders prefill before decode.

## Static phase diagnostics

The canonical `ForwardPassPerfModel` returned by `best_available` exposes
`static_phase_diagnostics(batch_size, context_length, prefix, prefill)` in Rust; the Python
`RustForwardPassPerfModel` wrapper accepts the same named arguments (with `prefix=0`).
It returns name-folded operation latency/energy, source tags, executed MoE communication
measurement substitutions, and optional SOL latency/compute/memory evidence. Decode means one
step at `context_length + 1`; prefill removes the cached prefix. Values precede learned online
correction. Whole-model estimators reject operation decomposition; missing SOL implementations
carry explicit reasons without changing the selected latency estimate.

Replay requests this evidence through `ReplayOutputRequirements(capture_performance_diagnostics=True)`.
`TimingOperationEvidence.details` is optional; providers without it must retain `None` (Rust
struct literals must initialize the new field). The constructor keeps it absent by default.
The CLI's time/source reports sum the observed phase work, including repeated cached timing
queries, and preserve distinct fallback records while folding repeated operation names.
Identical substitutions are deduplicated, so record counts are not execution counts.
