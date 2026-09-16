<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prepare a model for FPM collection and simulation

FPM onboarding currently requires an AISimulate model class. Reuse a compatible
registered class or implement the missing model description before collecting
whole-forward timings. You do not need to collect per-operation GPU timings for
this workflow: the retained operations describe analytical work and memory.

This guide covers the model-integration stage of self-service onboarding. Its
CPU examples use the bundled **Qwen3-0.6B configuration as a procedural example**;
they do not integrate Inkling or establish coverage for another model. After
reviewing and installing your integration, continue with the
[FPM collection-to-prediction workflow](end-to-end-workflow.md).

## 1. Record the intended deployment

Keep these inputs with the integration issue and eventual collection artifacts:

| Input | What to inspect and record |
| --- | --- |
| Checkpoint | Canonical model ID, immutable revision, local `config.json` and optional quantization/processor metadata, and file hashes. AISimulate's config parser does not take a revision argument; use metadata from the pinned checkpoint in a local directory when an exact revision is needed. |
| Architecture | Layer types/counts, attention and KV heads, head dimension, FFN/expert dimensions, routing, shared experts, and any model-specific configuration fields. Compare these with the serving implementation. |
| Memory | Weight storage precision and replication/sharding; cache layout, quantization, sliding windows/compression, and fixed recurrent or decode state. |
| Runtime | GPU/system specification, backend and exact version, image digest, attention/MoE kernel choices, and effective GEMM/MoE/FMHA/KV/communication precision. Checkpoint weight precision alone does not specify all these values. |
| Topology and workload | TP, PP, attention DP, MoE TP/EP, CP, batch/concurrency range, input/output lengths, prefix lengths, and scheduler limits. Start with one topology and a bounded workload. |

The current collection workflow targets vLLM with `PP=CP=1`. Start with ordinary
autoregressive text inference. Encoder/multimodal models and MTP have explicit
FPM construction restrictions; a model class existing for another mode is not
evidence that the intended FPM deployment is supported.

## 2. Reuse or implement the model description

Follow [How to add a new model](../add_a_new_model.md) for the registry and native
operation contracts. For this FPM use case, review these concrete responsibilities:

| Responsibility | Source and required result |
| --- | --- |
| Parse the checkpoint | [`sdk/utils.py`](../../src/aiconfigurator_core/sdk/utils.py), especially `get_model_config_from_model_path()`. Confirm the parsed geometry and architecture-specific `extra_params`; add parsing only for fields the existing parser cannot represent correctly. |
| Select a family | [`sdk/common.py`](../../src/aiconfigurator_core/sdk/common.py), `ARCHITECTURE_TO_MODEL_FAMILY`. Reuse a family only when its operation pipeline, precision handling, and cache behavior match the model. A new architecture name alone does not require a dedicated class. |
| Construct a new family when needed | The [registry examples](../../src/aiconfigurator_core/sdk/models/README.md#adding-a-new-model) show `@register_model`, `create()`, and reusable blocks. Implement the actual `context_ops` and `generation_ops`; an empty placeholder class is insufficient. |
| Describe analytical work | Each operation needs correct dimensions, quantization, layer/repetition scaling, and per-rank parallelism. Preserve the context/generation attention and `logits_gemm` naming contracts. Inspect communication, overlap, and phase-specific behavior as well as GEMMs. |
| Describe memory | Operation `get_weights()` values provide the weight inventory. Model `get_kvcache_bytes_per_sequence()` and its inverse `get_kvcache_max_tokens()` must represent the real cache curve. Override both for non-linear/windowed/compressed layouts; do not extrapolate a one-token slope across a window boundary. |
| Support FPM analytical execution | Every retained operation must be supported by the Rust [`fpm_sol` evaluator](../../../../crates/core/src/perfmodel/operators/fpm_sol.rs) for the proposed deployment and shapes. Adding an ordinary native operation or a `SOL_FULL` diagnostic does not automatically add this support. |

With `ModelConfig(forward_model="fpm")`, `get_model()` first constructs the
registered class, then centrally replaces each ordinary target
phase with an `FPMForwardOp`. The original phase operations remain inside
`sol_ops`, and their context weight inventory becomes `weight_bytes`. The model
class and cache methods remain available. Do not implement another phase
replacement or a Python latency/SOL formula in the new class.

## 3. Run CPU checks before collecting timings

From the repository root, use the [development environment](../../../../DEVELOPMENT.md):

```bash
uv sync --project python/aisimulate --extra dev
source python/aisimulate/.venv/bin/activate
export FPM_CHECK_DIR="$(mktemp -d)"
```

The following example needs no model weights, GPU, or matching Qwen FPM pair. It
uses bundled model metadata and the H200 system/backend catalog. The memory
entry point still requires a resolvable catalog/database layout, even though
memory sizing does not query measured operation latencies. The example pins a
queryable backend version; select a version allowed by your installed
[`query_versions.yaml`](../../src/aiconfigurator_core/systems/query_versions.yaml)
when adapting it.

```bash
python - <<'PY'
import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path

from aisimulate_core.sdk import EngineHandle, ModelConfig, estimate_kv_cache
from aiconfigurator_core.sdk.engine import build_ops_json
from aiconfigurator_core.sdk.models import get_model
from aiconfigurator_core.sdk.utils import get_model_config_from_model_path

out = Path(os.environ["FPM_CHECK_DIR"])
model_path, system, backend, version = "Qwen/Qwen3-0.6B", "h200_sxm", "vllm", "0.24.0"
topology = dict(tp_size=1, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=1)
options = ModelConfig(**topology)
info = get_model_config_from_model_path(model_path)
model = get_model(model_path, replace(options), backend)
fpm = get_model(model_path, replace(options, forward_model="fpm"), backend)

phase_ops = {"prefill": model.context_ops, "decode": model.generation_ops}
original = {phase: json.loads(build_ops_json(ops)) for phase, ops in phase_ops.items()}
folded = {
    "prefill": json.loads(build_ops_json(fpm.context_ops)),
    "decode": json.loads(build_ops_json(fpm.generation_ops)),
}
weights = sum(op.get_weights() for op in model.context_ops)
assert weights > 0
for phase in phase_ops:
    assert original[phase] and len(folded[phase]) == 1
    whole = folded[phase][0]["FpmForward"]
    assert whole["sol_ops"] == original[phase]
    assert whole["weight_bytes"] == weights

kv_bytes = {s: model.get_kvcache_bytes_per_sequence(s) for s in (1, 128, 8192)}
for s, budget in kv_bytes.items():
    assert budget > 0 and fpm.get_kvcache_bytes_per_sequence(s) == budget
    capacity = model.get_kvcache_max_tokens(budget)
    assert model.get_kvcache_bytes_per_sequence(capacity) <= budget
    assert model.get_kvcache_bytes_per_sequence(capacity + 1) > budget

memory = estimate_kv_cache(
    model_path, system, backend, version, **topology,
    max_num_tokens=2048, max_batch_size=8,
    memory_fraction_kind="of_total", memory_fraction_value=0.9,
    allow_naive_fallback=False,
)
assert memory["source"] == "native" and memory["total_kv_size_tokens"] > 0

# This is the per-operation SOL_FULL diagnostic, not FPM interpolation.
engine = EngineHandle.compile(
    model_path, system, backend, backend_version=version, **topology,
    database_mode="SOL",
)
sol = {}
for phase, ops in phase_ops.items():
    rows = engine.evaluate_ops_sol_json(
        build_ops_json(ops), is_context=phase == "prefill",
        batch_size=1, s=128 if phase == "prefill" else 129,
    )
    assert rows and all(math.isfinite(v) and v >= 0 for row in rows for v in row[1:])
    assert sum(row[1] for row in rows) > 0
    sol[phase] = rows

report = {
    "model_path": model_path, "class": type(model).__name__, "family": model.model_family,
    "system": system, "backend": backend, "backend_version": version, "topology": topology,
    "geometry": {k: info[k] for k in ("architecture", "layers", "n", "n_kv", "d", "hidden_size")},
    "resolved_config_sha256": hashlib.sha256(
        json.dumps(info["raw_config"], sort_keys=True).encode()
    ).hexdigest(),
    "quantization": {k: getattr(model.config, k).name for k in (
        "gemm_quant_mode", "moe_quant_mode", "fmha_quant_mode", "kvcache_quant_mode", "comm_quant_mode"
    )},
    "op_counts": {phase: len(ops) for phase, ops in phase_ops.items()},
    "weights_bytes_per_rank": weights, "kv_bytes_per_sequence_per_rank": kv_bytes,
    "memory": memory, "sol_full_ms": sol,
}
for name, value in (("integration.json", report), ("op-level.json", original), ("fpm.json", folded)):
    (out / name).write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(f"Saved integration evidence to {out}")
PY
```

**Expected for this example:** `LLAMAModel` / `LLAMA`, 28 layers, 16 attention
heads, 8 KV heads, head dimension 128, and 14 operation entries per phase.
The modeled weight inventory is 1,503,133,696 bytes per rank. BF16 KV costs
114,688 bytes per token; an 8,192-token sequence uses 939,524,096 bytes per rank.
The FPM wrapper preserves that inventory and curve. The diagnostic produces
finite nonnegative per-operation SOL values, including zero-cost communication
at TP=PP=1. These are analytical estimates, not measured latency or peak runtime
memory. `resolved_config_sha256` hashes parsed metadata, not checkpoint weights.

For your integration, replace the example model and deployment inputs together.
Compare the emitted operation dimensions and scale factors with the serving
architecture, and add independent expected-value tests for those facts. Exercise
each supported topology/precision and the relevant cache/window boundaries;
three positive sequence lengths and a successful import are not sufficient
model-specific evidence. See the existing [model configuration and memory tests](../../tests/unit/sdk/models/test_model_config.py)
for patterns. Include an unsupported architecture/topology case that must fail
explicitly, rather than falling back to another family or memory estimator.

## 4. Verify the FPM execution contract

Run the existing CPU contract suites from the same checkout/environment:

```bash
python -m pytest -p no:timeout -c python/aisimulate/pytest.ini \
  python/aisimulate/tests/unit/sdk/test_fpm_forward.py \
  python/aisimulate/tests/unit/sdk/test_opspec_coverage.py \
  python/aisimulate/tests/cross_package/test_single_oracle_contract.py
cargo test -p aisimulate-core --lib perfmodel::operators::fpm_sol
cargo test -p aisimulate-core --lib perfmodel::operators::fpm_forward
```

`-p no:timeout` follows the repository's macOS test guidance. These tests exercise
the Python-to-Rust operation contract, FPM routing, and native analytical/transfer
behavior using existing models and synthetic fixtures. They do not automatically
cover your new model's operations or demonstrate predictive accuracy.

There is currently no public model-readiness call that evaluates an arbitrary
retained graph through the exact FPM `sol_total`/`fpm_sol` path without a pair.
`evaluate_ops_sol_json()` above uses a different diagnostic path and integer
workload coordinates. FPM transfer maps iteration totals to potentially
fractional per-request coordinates. Also, exact FPM hits and some linear
interpolations can succeed without ever evaluating `sol_ops`.

Before declaring the new model ready for collection, add a model-specific CPU
regression that serializes its real FPM graph and queries deliberately synthetic
FPM cells at an off-site coordinate requiring SOL transfer. Cover both phases
and any relevant fractional coordinates. Use the pair/engine setup in
[`test_fpm_forward.py`](../../tests/unit/sdk/test_fpm_forward.py) and the native
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
the approved AISimulate revision in the environment that will plan collection
and run prediction. For a source checkout at that revision:

```bash
git rev-parse HEAD
uv sync --project python/aisimulate --extra dev
source python/aisimulate/.venv/bin/activate
python -c 'import importlib.metadata; print(importlib.metadata.version("aisimulate"))'
aisimulate predict --help
aisimulate recommend --help
```

Record the commit as well as the package version; several source revisions may
share a version. Rerun the CPU checks in a fresh process against the installed
integration. Preserve the reports with the pinned config files and test output.

The collection handoff consists of the canonical checkpoint identity/revision,
config files/hashes, reviewed AISimulate revision, exact runtime image/version,
hardware/topology/precision choices, workload limits, and the CPU evidence.
The metadata used for local construction must describe the same checkpoint
mounted in the collection runtime; a temporary local path is not a portable
replacement for the canonical identity in the published pair.

Proceed to [freeze a bounded plan, smoke, collect, resume, and validate the pair](end-to-end-workflow.md#3-freeze-and-inspect-the-plan).
Only whole-forward silicon timings are collected for FPM. Then exercise the
ordinary standalone AISimulate/Mocker FPM `predict` and `recommend` paths on
covered candidates and save the matching inputs, data provenance, reports, and
reproduction commands. Model integration, collection completion, functional
prediction, and agreement with independent silicon measurements are separate
acceptance results; this guide establishes the procedure for the first stage.
