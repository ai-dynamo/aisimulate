<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python Replay API

The Python API compiles public configuration into a data-only runner contract.
Its `ReplaySpec` is not the native Rust JSON structure with the same name.
Use the compiler rather than translating fields between them by hand.

## Public configuration to runner

```python
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.replay import ReplayOutputRequirements

config = CorePredictionConfig.from_yaml("prediction.yaml")
spec = prediction_to_replay_spec(config)
factory = EngineReplayRunnerFactory()
factory.capabilities().require_compatible(spec)
runner = factory.create(worker_id=0)
try:
    report = runner.run(
        spec,
        output_requirements=ReplayOutputRequirements(
            include_raw_report=True,
            capture_per_request=True,
        ),
    )
finally:
    runner.close()
```

The configuration above is the same core YAML used by `predict --stack engine`.
`CorePredictionConfig` does not accept optional Router/Planner top-level sections;
the CLI resolves and compiles those through their selected configuration adapters.
The compiler signature is:

```python
prediction_to_replay_spec(
    config: CorePredictionConfig,
    *,
    adapter_specs: dict[str, AdapterReplaySpec] | None = None,
    afd_performance_model: AFDPerformanceModel | None = None,
    execution_mode: str = "offline",
) -> ReplaySpec
```

Compilation resolves concrete traffic, topology, capacity and per-role estimator
identities. `ForwardPassEstimatorSpec.config` is the sole estimator identity;
its convenience properties are projections. Do not independently edit a
model/backend/version property after materialization. Configuration and
capability validation precede native execution but do not guarantee measured
accuracy or availability of every timing query.

## Inputs and reports

The [runner ABI](../../adapters/runner-abi.md) defines `ReplaySpec`,
`BackendDeploymentSpec`, `RunnerFactory`, `RunnerCapabilities`, and `Runner`.
`spec.workload` contains concrete trace or synthetic inputs; `spec.concurrency`
is a closed-loop cap or `None` for open loop. `spec.goal` carries evaluation
requirements. Adapter data and runtime hooks are serialized values, not live
policy objects.

`ReplayReport.metrics` supplies numeric values used by scoring. Non-power
metrics must be finite numbers; only designated power fields can be null.
`metadata` retains provenance and optional native detail. Engine runner raw
reports are retained under `metadata["native_report"]` when requested; do not assume
that every optional runner exposes identical metadata. Agentic results retain
`agentic_qualification: functional_only` and phase/snapshot/profile evidence.
See [result definitions](../../getting-started/understand-results.md) before
comparing latency or throughput across runs.

`ReplayOutputRequirements` defaults to summary output. It can request a raw
report, per-request capture, memory diagnostics or performance diagnostics.
These controls retain additional evidence without changing the workload.
The built-in Engine runner supports `capture_telemetry=True` for offline
vLLM/SGLang aggregated and separate prefill/decode replay without encoder pools.
`telemetry_sample_interval_ms` defaults to `1000.0` and must be finite and
positive when enabled. Capture retains all-phase request, sample, and lifecycle
arrays under the native report's `telemetry` object even without separate
per-request capture. Its timestamps use absolute virtual milliseconds; existing
profiling summaries and per-request clock semantics are unchanged. See the
[telemetry contract](../../getting-started/understand-results.md#replay-telemetry).
Analytical EPD does not provide raw native
or per-request EPD reports. Large capture can require substantial memory.

## HTML dashboard

The CLI `--export-html` flag enables telemetry and writes a standalone
simulation dashboard. Python callers can render an already captured report:

```python
from pathlib import Path
from aisimulate.replay.dashboard import render_replay_dashboard, write_replay_dashboard

# Existing CLI output with prediction.json and the three telemetry JSONL files.
path = render_replay_dashboard(Path("./agentic-telemetry"))

# Or use native telemetry captured with ReplayOutputRequirements(capture_telemetry=True).
path = write_replay_dashboard(
    Path("./agentic-dashboard"),
    result.native,
    result.native["telemetry"],
    config=config.model_dump(mode="json", exclude_none=True),
)
```

Both helpers write `report.html` and return its path. Optional `config` supplies
display metadata; it does not change captured measurements. The helpers do not
rerun simulation or modify prediction summaries. They require native
`aisimulate.replay.telemetry.v1` data on its absolute virtual clock. Direct Python
callers own their resource budgets. See the
[dashboard views and semantics](../../getting-started/understand-results.md#replay-dashboard).

## Lifecycle and errors

One runner is reusable within one worker. The production Engine factory loads
`aisimulate._runtime` lazily and has no Dynamo dependency. A missing extension
raises `RunnerUnavailableError`; an invalid runtime or unsupported runtime
combination raises the runner's validation error. `close()` is currently a
no-op for this built-in runner; consumers must still call it for other runners.

Low-level calls do not automatically provide the CLI's subprocess host-memory
supervision. The caller owns resource budgets, output publication and failure
handling, especially for long agentic profiles. Use
[local resource controls](../../reference/local-resources.md) when executing
through the CLI or guarded Sweeper. The native JSON bridge and private runner
lowering helpers are implementation interfaces, not replacements for this API.

Source: [compiler](../../../python/aisimulate/src/aisimulate/compiler.py),
[runner](../../../python/aisimulate/src/aisimulate/runner.py), and
[contract definitions](../../../python/aisimulate/src/aisimulate/sweeper/replay.py).
