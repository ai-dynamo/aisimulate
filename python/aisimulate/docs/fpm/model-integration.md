<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prepare model metadata and choose an FPM execution route

FPM onboarding needs pinned model and deployment metadata, correct memory and
KV-cache accounting, and matching whole-forward timing data. The current
config/profile route supports ordinary FPM `predict` and `recommend` without
an op-level model class. Registered models can also use analytical SOL estimates
for timing transfer. Neither route requires per-operation GPU timing collection.

Start with the [FPM self-service workflow](../../../../docs/fpm-self-service.md#onboard-with-an-agent)
to inspect a model, choose a worker, derive and review its resource profile,
plan and collect timings, and validate replay. This guide records the metadata
shared by both routes, then describes the **optional registered-model/SOL
integration procedure** in sections 2–5. Its CPU example uses the bundled
**Qwen3-0.6B configuration**; it does not establish coverage for another model.

## Class-independent direct FPM

Supply a local model config to `aisimulate onboard init --model-config`, or
provide an existing `--fpm-profile`. Config-based setup derives supported
resource estimates and identifies missing bounds; review and edit the exact
values, effective precisions and provenance before accepting the profile.
Follow the self-service guide's [terminal or headless review flow](../../../../docs/fpm-self-service.md#3-derive-review-and-save-the-profile).
Profile resources must bound every rank of the selected topology. A profile is
a declaration, not proof of runtime compatibility or measured memory fit.
For full/sliding attention and supported convolution state, follow
[grouped cache review](../../../../docs/fpm-self-service.md#review-grouped-cache-resources):
runtime block sizes remain explicit inputs, and each group page includes every
group layer plus runtime padding on one rank. Use the
[canonical byte-budget API](../../../../docs/core-api.md#fpm-profile-cache-groups-and-byte-budgets)
for grouped resources; a scalar token capacity cannot represent window eviction
or transient prefill pages.

An existing launch configuration is optional evidence. The onboarding agent
proposes supported runtime settings from pinned checkpoint/runtime metadata and
available sidecars, asks only for unresolved facts, and presents the result for
review. The CLI does not import arbitrary launch arguments or inspect a remote
runtime automatically. Config intake proposes the current collector communication
identity `half`; unknown FMHA and KV precision remain explicit inputs. Packed
cache-page estimates remain distinct from runtime allocations including padding.

The saved request embeds the profile, and generated ordinary configurations use
`engine.fpm_profile`, `estimation_mode: fpm_interpolation`,
`estimator_config.fpm_interpolation.method: direct`, and `fallback_policy: deny`.
Direct interpolation uses measured whole-forward timings without constructing
an operation graph for timing or resources. Unsupported metadata and uncovered
queries fail explicitly. See [execution-route selection and interpolation rules](../../../../docs/fpm-self-service.md#choose-the-model-execution-route)
for exact-point, curve and two-sided interpolation coverage.

Each generated profile and plan selects one exact TP, DEP or TEP worker. To onboard
several configurations in one session, use the self-service guide's
[directory output](../../../../docs/fpm-self-service.md#onboard-multiple-parallel-configurations).
It reuses shared intake and creates a separate reviewed profile and request
for each tuple; rank-local byte bounds and cache groups are never transferred
between configurations. Follow each emitted plan command and collect or resume
each configuration independently. Follow the
[shared collection policy](../../../../docs/fpm-self-service.md#how-the-collection-grid-is-determined):
AISimulate sets runtime limits and the reviewed capture policy. New onboarding
uses `--prefill-cudagraph-policy runtime`, which leaves prefill compilation to
the pinned engine; explicit capture extension remains available. The collector
launches benchmark workers, and Dynamo uses initialized engine state, image
sampling defaults and feasibility checks to generate and time the exact grid.
Capture sizes and exact counts remain unresolved before engine initialization
in runtime mode. A complete generated grid does
not establish direct-FPM query coverage. Validation traffic is supplied
separately after a formal timing pair is verified. The
current [AgentX coverage check](../../../../docs/fpm-self-service.md#validate-fpm-query-coverage-with-agentx-replay)
uses cold aggregated replay, one client lane, HBM-only cache and no speculative
decoding. Missing timing stops replay and retains partial evidence; that evidence
cannot certify the rest of the trace. Coverage and measured accuracy are
separate results.

## 1. Record the intended deployment

Keep these inputs with the integration issue and eventual collection artifacts:

| Input | What to inspect and record |
| --- | --- |
| Checkpoint | Canonical model ID, immutable revision, local `config.json` and optional quantization/processor metadata, and file hashes. AISimulate's config parser does not take a revision argument; use metadata from the pinned checkpoint in a local directory when an exact revision is needed. |
| Architecture | Layer types/counts, attention and KV heads, head dimension, FFN/expert dimensions, routing, shared experts, and any model-specific configuration fields. Compare these with the serving implementation. |
| Memory | Weight storage precision and replication/sharding; cache layout, quantization, sliding windows/compression, and fixed recurrent or decode state. |
| Runtime | GPU/system specification, backend and exact version, image digest, attention/MoE kernel choices, and effective GEMM/MoE/FMHA/KV/communication precision. Checkpoint weight precision alone does not specify all these values. |
| Worker topology | Exact `(TP, PP, attention DP, MoE TP, MoE EP, CP)` tuple. Minimum collection GPUs are attention TP times attention DP. Total GPU allocation, node reservations and replica budgets are not onboarding intake requirements. |
| Runtime and collection bounds | Per-request context, per-attention-DP-rank scheduled-token and sequence limits, GPU memory fraction (new onboarding starts at 0.90), and runtime or explicit prefill CUDA graph policy. Review these independently of replay traffic; record actual captures after initialization. These settings do not prove memory fit or timing coverage. |
| Validation traffic | Select a local trace when validating the collected pair. Fixed input/output lengths, concurrency, TTFT and TPOT are optional synthetic-example inputs, not collection requirements. |

For example, MoE TP4 is `(4, 1, 1, 4, 1, 1)`, DEP8 is
`(1, 1, 8, 1, 8, 1)`, and TEP8 is `(8, 1, 1, 1, 8, 1)`. Equal GPU counts do
not make their resource bounds or timing cells interchangeable. Use the
[topology flags and collection limits](../../../../docs/fpm-self-service.md#create-the-request)
for the intended worker, and verify actual collection resources before execution.
The sequence limit does not reserve maximum context for every sequence.

The profile-based collection workflow targets vLLM text decoders with
`PP=CP=1` and linear or grouped cache storage. Grouped prediction and replay
currently require cold aggregated execution, HBM-only cache, no speculative
decoding and `prefix_caching: false`; generated grouped configurations preserve
that setting. A multimodal checkpoint can supply an
unambiguous `text_config` or flat decoder fields. Text-decoder timing and
config-derived estimates exclude encoders, projectors, preprocessing and other
non-text components. Observed runtime cache capacity accounts for every component
actually loaded by the worker; do not subtract guessed encoder allocations.
Preserve that scope in profile provenance. Unsupported cache semantics, encoder
pools, AFD, speculative decoding, wide EP and EPLB remain outside this route. A
registered class existing for another mode is not evidence of support for the
intended FPM deployment.

## 2. Registered-model route: reuse or implement the model description

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

For the config/profile route, continue the [self-service workflow](../../../../docs/fpm-self-service.md#plan-preview-and-explicitly-execute)
with the accepted profile. The [collection campaign example](end-to-end-workflow.md#3-freeze-and-inspect-the-plan)
also illustrates formal pair publication for a registered model. Only
whole-forward silicon timings are collected for FPM. Exercise ordinary `predict`
and `recommend` on covered candidates and retain the matching inputs, data
provenance, reports and reproduction commands. CPU integration checks,
collection completion, replay coverage, completed simulations and agreement
with independent silicon measurements are separate acceptance results.
