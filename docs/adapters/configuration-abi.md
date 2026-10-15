<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Configuration adapter ABI

Configuration compilation, SDK sweep providers and the legacy recipe importer
are separate interfaces. The public CLI invokes `SimulationConfigAdapter`;
Sweeper accepts `SweepConfigProvider` independently. External recipe ingestion
is documented under [AIC configuration import](../aic-backward-compatibility/configuration-import.md).

## SimulationConfigAdapter v3

[`config_adapter.py`](../../python/aisimulate/src/aisimulate/config_adapter.py)
defines `CONFIG_ADAPTER_API_VERSION = 3`. A selected adapter declares `name`,
`section`, and `config_adapter_api_version`. Discovery uses
`aisimulate.config_adapters`; names are `<stack>.<section>`. `section` must equal
the final name segment, be nonempty and contain no dot. The version must be an
actual integer equal to 3, not a boolean or a string.

```python
compile_prediction(
    config: Mapping[str, JSONValue], context: PredictionAdapterContext,
) -> AdapterReplaySpec

compile_recommendation(
    config: Mapping[str, JSONValue], context: RecommendationAdapterContext,
) -> AdapterSearchPlan

materialize_candidate(
    plan: AdapterSearchPlan,
    selection: Mapping[str, JSONValue],
    context: CandidateContext,
) -> AdapterReplaySpec
```

These are bound-method signatures. Every operation is required. Prediction
compiles one concrete section; recommendation prepares domains once, then
materializes one concrete selection at a time. AISimulate owns core engine and
traffic schemas. Each optional section owns separate typed prediction and
recommendation models; resolved runtime config is a third representation,
not the public search schema.

`PredictionAdapterContext` exposes concrete `engine`, `traffic`, `evaluation`.
`RecommendationAdapterContext` also exposes `optimization` and a `SweepContext`.
`CandidateContext` contains the resolved core sample, `BackendDeploymentSpec`
and optional concurrency. Read these contexts without mutating them or previously
returned plan/config values.

## Plans and runtime hooks

`AdapterSearchPlan` contains a `SearchSpaceFragment`, reusable JSON `state`,
JSON `diagnostics`, and `potential_runtime_hooks`. The fragment (API version 1)
contains per-branch choices, float ranges, logarithmic dimensions and conditional
children. Branches use native search names such as `agg` and `disagg`.
The core namespaces local names as `adapter::<adapter name>::<local parameter>`.

Declare all potential hooks during preparation so runner preflight can reject
an unsupported provider/kind/version before creating replay workers. Materialized
`AdapterReplaySpec` contains only concrete `config` and `runtime_hooks`. It must
not start replay or carry live policy objects. The selected runner composes all
hooks and executes exactly once.

Plans, payloads and diagnostics cross a strict JSON boundary: no arbitrary
objects, non-string dict keys or nonfinite numbers. `InfeasibleCandidate`
identifies a selection outside the provider's feasible domain. Invalid
configuration or ABI fails validation; do not turn unknown fields or unsupported
capabilities into silently ignored defaults.

## Discovery failures and compatibility

Direct injected adapters take precedence over installed entry points. Only
selected names are loaded; duplicate names, missing providers, wrong identity,
wrong version, missing required methods and construction failures produce
`ConfigAdapterResolutionError`. `CompiledSweepProvider` bridges a precompiled
v3 plan to the existing Sweeper-v1 protocol: preparation returns a deep copy of
the plan and materialization delegates to `materialize_candidate`.
That bridge does not make the two contracts interchangeable.

## SweepConfigProvider v1

[`sweeper/provider.py`](../../python/aisimulate/src/aisimulate/sweeper/provider.py)
retains `API_VERSION = 1` for SDK integrations. Providers declare `name` and
`api_version`, with two bound methods:

```python
generate_search_space(
    search_spec: Mapping[str, JSONValue], context: SweepContext,
) -> AdapterSearchPlan

materialize_replay(
    plan: AdapterSearchPlan,
    selection: Mapping[str, JSONValue],
    context: CandidateContext,
) -> AdapterReplaySpec
```

`SweepContext` gives isolated `core_search_space`, `workload`, `goal` and progress
preference. Providers run in the coordinator; spawned replay workers receive
only serializable ReplaySpec values. Discovery uses
`aisimulate.sweep_config_providers`, selected through SDK `adapters` keys;
direct constructor injection takes precedence. The name must match the selected
key and the integer version must match exactly. Structural name/method errors
are `TypeError`; an incompatible provider version raises `ValueError`.
The same JSON, context immutability and up-front hook declaration rules apply.
