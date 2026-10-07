<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Add a model architecture for SOL-assisted FPM

This guide is for developers adding or extending AISimulate's analytical model
support. Use it when FPM interpolation with `method: sol` needs an architecture
that AISimulate cannot yet describe correctly. The result is model/parser/operator
code and tests, not a collected timing dataset.

SOL-assisted FPM combines measured forward-pass timings with analytical
speed-of-light (SOL) cost estimates for supported workload shapes. To compute
those estimates, AISimulate needs a description of the model's attention, GEMM,
MoE and communication operations, their per-rank dimensions, and weight/KV-cache
memory requirements. This guide explains how to supply and validate that description.

For example, a new MoE architecture may need a different expert-operation graph
or KV-cache size calculation. Registering its architecture name is only the
first step: its operation descriptions and the Rust FPM SOL evaluator must also
agree with the target model and parallel configuration.

Ordinary [FPM self-service](../../../../docs/fpm-self-service/README.md) uses `method: direct` with a resource
profile and does not require this integration. If you only want to collect and
use forward timings, follow that workflow and its examples. For general model
registration, start with [How to add a new model](../add_a_new_model.md);
this page adds the requirements specific to SOL-assisted FPM.

## 1. Record the architecture and target deployment

Before changing code, inspect the pinned model and runtime. Retain these inputs
with the integration tests so the analytical description can be checked against
the implementation that will execute the model:

| Input | What to establish |
| --- | --- |
| Checkpoint | Model ID, immutable revision, local configuration and any quantization metadata. A config hash identifies metadata, not checkpoint weights. |
| Architecture | Layer types/counts, attention and KV heads, head dimensions, FFN/expert dimensions, routing, shared experts and architecture-specific fields. |
| Parallelism | The exact TP, PP, attention-DP, MoE-TP and EP layout, including how work, weights and cache are sharded or replicated per rank. |
| Memory | Weight precision, KV layout/precision, sliding windows or compression, and any recurrent state. A per-token slope is insufficient for a bounded or non-linear cache. |
| Runtime | GPU/interconnect, backend and exact version, attention/MoE kernels, effective precisions, scheduler limits and CUDA Graph policy. |

Reuse an existing analytical family only when these facts match its operation
and memory model. A matching architecture name or successful metadata parse does
not establish that the intended SOL query path works. The CPU example below uses
bundled Qwen3-0.6B metadata to demonstrate the checks; it does not validate another
model or replace model-specific tests.

## 2. Implement the model description

Follow [How to add a new model](../add_a_new_model.md) for the registry and native
operation contracts. For this registered-model/SOL route, review these concrete responsibilities:

| Responsibility | Source and required result |
| --- | --- |
| Parse the checkpoint | [`sdk/utils.py`](../../src/aisimulate_core/sdk/utils.py), especially `get_model_config_from_model_path()`. Confirm the parsed geometry and architecture-specific `extra_params`; add parsing only for fields the existing parser cannot represent correctly. |
| Select a family | [`sdk/common.py`](../../src/aisimulate_core/sdk/common.py), `ARCHITECTURE_TO_MODEL_FAMILY`. Reuse a family only when its operation pipeline, precision handling, and cache behavior match the model. A new architecture name alone does not require a dedicated class. |
| Construct a new family when needed | The [registry examples](../../src/aisimulate_core/sdk/models/README.md#adding-a-new-model) show `@register_model`, `create()`, and reusable blocks. Implement the actual `context_ops` and `generation_ops`; an empty placeholder class is insufficient. |
| Describe analytical work | Each operation needs correct dimensions, quantization, layer/repetition scaling, and per-rank parallelism. Preserve the context/generation attention and `logits_gemm` naming contracts. Inspect communication, overlap, and phase-specific behavior as well as GEMMs. |
| Describe memory | Operation `get_weights()` values provide the weight inventory. Model `get_kvcache_bytes_per_sequence()` and its inverse `get_kvcache_max_tokens()` must represent the real cache curve. Override both for non-linear/windowed/compressed layouts; do not extrapolate a one-token slope across a window boundary. |
| Support FPM analytical execution | Every retained operation must be supported by the Rust [`fpm_sol` evaluator](../../../../crates/core/src/perfmodel/operators/fpm_sol.rs) for the proposed deployment and shapes. Adding an ordinary native operation or a `SOL_FULL` diagnostic does not automatically add this support. |

Construct performance models through
`RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig)` and query
the returned model, as specified by the [canonical API](../../../../docs/core-api.md#choosing-a-forward-pass-api).
For registered FPM interpolation, select `estimation_mode: fpm_interpolation`,
`estimator_config.fpm_interpolation.method: sol`, and `fallback_policy: deny`.
The constructor requires a compatible registered class and a matching FPM pair.
Internally, the ordinary phase operations are retained as `sol_ops` inside
`FPMForwardOp`; their context weight inventory becomes `weight_bytes`. The model
class and cache methods remain available. Model authors should not duplicate
this phase replacement or implement a Python latency/SOL formula.

## 3. Run CPU checks before collecting timings

From the repository root, activate the installed [development environment](../../../../DEVELOPMENT.md)
and choose a fresh evidence directory:

```bash
source python/aisimulate/.venv/bin/activate
export FPM_CHECK_DIR="$(mktemp -d)"
```

The following example needs no model weights, GPU, or matching Qwen FPM pair. It
uses bundled model metadata and the H200 system/backend catalog. The memory
entry point still requires a resolvable catalog/database layout, even though
memory sizing does not query measured operation latencies. The example pins a
queryable backend version; select a version allowed by your installed
[`query_versions.yaml`](../../src/aisimulate_core/systems/query_versions.yaml)
when adapting it.

```bash
python - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path

from aisimulate_core.sdk import (
    ForwardPassPerfModelConfig,
    ModelConfig,
    RustForwardPassPerfModel,
    estimate_kv_cache,
)
from aisimulate_core.sdk.engine import build_ops_json
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.utils import get_model_config_from_model_path

out = Path(os.environ["FPM_CHECK_DIR"])
model_path, system, backend, version = "Qwen/Qwen3-0.6B", "h200_sxm", "vllm", "0.24.0"
topology = dict(tp_size=1, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=1)
options = ModelConfig(**topology)
info = get_model_config_from_model_path(model_path)
model = get_model(model_path, options, backend)

phase_ops = {"prefill": model.context_ops, "decode": model.generation_ops}
original = {phase: json.loads(build_ops_json(ops)) for phase, ops in phase_ops.items()}
assert all(original.values())
weights = sum(op.get_weights() for op in model.context_ops)
assert weights > 0

kv_bytes = {s: model.get_kvcache_bytes_per_sequence(s) for s in (1, 128, 8192)}
for s, budget in kv_bytes.items():
    assert budget > 0
    capacity = model.get_kvcache_max_tokens(budget)
    assert model.get_kvcache_bytes_per_sequence(capacity) <= budget
    assert model.get_kvcache_bytes_per_sequence(capacity + 1) > budget

precision = {k: getattr(model.config, k).name for k in (
    "gemm_quant_mode", "moe_quant_mode", "fmha_quant_mode", "kvcache_quant_mode", "comm_quant_mode"
)}
memory = estimate_kv_cache(
    model_path, system, backend, version, **topology, **precision,
    max_num_tokens=2048, max_batch_size=8,
    memory_fraction_kind="of_total", memory_fraction_value=0.9,
    allow_naive_fallback=False,
)
assert memory["source"] == "native" and memory["total_kv_size_tokens"] > 0

# These are op-level analytical diagnostics, not FPM interpolation.
sol = {}
for phase in phase_ops:
    config = ForwardPassPerfModelConfig(
        model=model_path, system=system, backend=backend, backend_version=version,
        worker_type=phase, tp=options.tp_size, pp=options.pp_size,
        attention_dp=options.attention_dp_size,
        moe_tp_size=options.moe_tp_size, moe_ep_size=options.moe_ep_size,
        **precision, estimation_mode="op_level", database_mode="SOL", fallback_policy="deny",
    )
    performance_model = RustForwardPassPerfModel.best_available(config)
    rows = performance_model.static_phase_diagnostics(
        batch_size=1, context_length=128, prefill=phase == "prefill",
    )
    assert rows and all(row["details"]["sol"] is not None for row in rows)
    assert all(math.isfinite(v) and v >= 0 for row in rows for v in row["details"]["sol"].values())
    assert sum(row["details"]["sol"]["latency_ms"] for row in rows) > 0
    sol[phase] = rows

report = {
    "model_path": model_path, "class": type(model).__name__, "family": model.model_family,
    "system": system, "backend": backend, "backend_version": version, "topology": topology,
    "geometry": {k: info[k] for k in ("architecture", "layers", "n", "n_kv", "d", "hidden_size")},
    "resolved_config_sha256": hashlib.sha256(
        json.dumps(info["raw_config"], sort_keys=True).encode()
    ).hexdigest(),
    "quantization": precision,
    "op_counts": {phase: len(ops) for phase, ops in phase_ops.items()},
    "weights_bytes_per_rank": weights, "kv_bytes_per_sequence_per_rank": kv_bytes,
    "memory": memory, "op_level_diagnostics": sol,
}
for name, value in (("integration.json", report), ("op-level.json", original)):
    (out / name).write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(f"Saved integration evidence to {out}")
PY
```

**Expected for this example:** `LLAMAModel` / `LLAMA`, 28 layers, 16 attention
heads, 8 KV heads, head dimension 128, and 14 operation entries per phase.
The modeled weight inventory is 1,503,133,696 bytes per rank. BF16 KV costs
114,688 bytes per token; an 8,192-token sequence uses 939,524,096 bytes per rank.
The diagnostic produces finite nonnegative per-operation SOL values, including
zero-cost communication at TP=PP=1. Prefill processes 128 tokens; decode uses
128 past tokens and one new token. These are analytical estimates, not measured
latency or peak runtime memory. They do not exercise registered FPM timing
transfer. `resolved_config_sha256` hashes parsed metadata, not checkpoint weights.

For your integration, replace the example model and deployment inputs together.
Compare the emitted operation dimensions and scale factors with the serving
architecture, and add independent expected-value tests for those facts. Exercise
each supported topology/precision and the relevant cache/window boundaries;
three positive sequence lengths and a successful import are not sufficient
model-specific evidence. See the existing [model configuration and memory tests](../../tests/unit/sdk/models/test_model_config.py)
for patterns. Include an unsupported architecture/topology case that must fail
explicitly, rather than falling back to another family or memory estimator.

## 4. Verify the FPM execution contract

For a registered-model integration, run its focused CPU tests together with the
existing [FPM graph and routing](../../tests/unit/sdk/test_fpm_forward.py),
[profile and canonical constructor](../../tests/unit/sdk/test_fpm_profile.py),
[operation serialization](../../tests/unit/sdk/test_opspec_coverage.py) and
[single-oracle](../../tests/cross_package/test_single_oracle_contract.py)
contract suites. Native [FPM SOL](../../../../crates/core/src/perfmodel/operators/fpm_sol.rs)
and [FPM forward](../../../../crates/core/src/perfmodel/operators/fpm_forward.rs)
tests cover analytical and transfer behavior. Follow the repository's
[test instructions](../../../../DEVELOPMENT.md#running-tests); on macOS, use
`-p no:timeout` as noted in the [repository guidance](../../../../AGENTS.md#cursor-cloud-specific-instructions).
Existing fixtures do not automatically cover a new model's operations or
demonstrate predictive accuracy.

There is currently no public model-readiness call that evaluates an arbitrary
retained graph through the exact FPM `sol_total`/`fpm_sol` path without a pair.
`static_phase_diagnostics()` above uses a different diagnostic path and integer
workload coordinates. FPM transfer maps iteration totals to potentially
fractional per-request coordinates. Also, exact FPM hits and some linear
interpolations can succeed without ever evaluating `sol_ops`.

Before declaring the new model ready for collection, add a model-specific CPU
regression that constructs the model through `best_available` with explicit
`fpm_interpolation`/`sol` and denied fallback, then queries deliberately synthetic
FPM cells at an off-site coordinate requiring SOL transfer. Cover both phases
and any relevant fractional coordinates. Use the synthetic pair layout in
[`test_fpm_forward.py`](../../tests/unit/sdk/test_fpm_forward.py), the canonical
constructor/query examples in [`test_fpm_profile.py`](../../tests/unit/sdk/test_fpm_profile.py), and the native
tests `prefill_off_site_kv_resolves_through_the_dsa_sol_transfer`,
`decode_pairless_batch_resolves_through_the_dsa_sol_transfer`, and
`unsupported_sol_family_is_lazy` in
[`fpm_forward.rs`](../../../../crates/core/src/perfmodel/operators/fpm_forward.rs)
as contract examples. Assert an independently justified result and a clear
failure for an unsupported shape or identity; an exact-hit-only fixture misses
this dependency. Keep synthetic timings in test fixtures, never publish them as
collected data. If an operation is unsupported, implement and numerically test
its Rust FPM SOL path before relying on sparse collection.

## 5. Review, install, and hand off to collection

Submit the model/parser/operation changes with the focused model tests, CPU
reports, resolved deployment inputs, and any explicit unsupported cases. Follow
the repository [review contract](../../../../REVIEW.md). After review, install
the approved AISimulate revision using the [development setup](../../../../DEVELOPMENT.md)
in the environment that will plan collection and run prediction. From that
checkout with its environment active:

```bash
git rev-parse HEAD
python -c 'import importlib.metadata; print(importlib.metadata.version("aisimulate"))'
aisimulate onboard --help
aisimulate predict --help
aisimulate recommend --help
```

Record the commit as well as the package version; several source revisions may
share a version. Rerun the CPU checks in a fresh process against the installed
integration. Preserve the reports with the pinned config files and test output.

The collection handoff consists of the canonical checkpoint identity/revision,
config files/hashes, reviewed AISimulate revision, exact runtime image/version,
hardware/topology/precision choices, runtime and collection limits, and the CPU evidence.
The metadata used for local construction must describe the same checkpoint
mounted in the collection runtime; a temporary local path is not a portable
replacement for the canonical identity in the published pair.

After integration, load a matching measured FPM pair and verify the explicit
`fpm_interpolation` / `method: sol` path with denied fallback. Use the
[self-service implementation reference](../../../../docs/fpm-self-service/implementation.md#plan-preview-and-explicitly-execute)
for supported collection configurations and
[dataset validation](../../../../docs/fpm-self-service/implementation.md#validate-and-install-the-fpm-profile).
A registered analytical class does not launch collection or establish runtime
compatibility. Keep CPU integration checks, measured coverage, completed
simulations and independent serving accuracy as separate results.
