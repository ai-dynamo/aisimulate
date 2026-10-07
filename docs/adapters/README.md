<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adapter contracts

AISimulate exposes separate boundaries for execution, configuration, output
and native Replay composition. An adapter's support at one boundary does not
establish support at the others. This directory is an ABI reference; usage of
the existing Dynamo integration is documented [with Replay](../replay/dynamo.md).

| Boundary | Owns | Discovery / version |
| --- | --- | --- |
| [Runner](runner-abi.md) | Execute one resolved ReplaySpec and report capabilities/metrics | `aisimulate.runner_factories`; Python replay-spec API 1 |
| [Simulation configuration](configuration-abi.md) | Compile prediction, search plan and candidate hooks | `aisimulate.config_adapters`; config API 3 |
| [SDK sweep provider](configuration-abi.md#sweepconfigprovider-v1) | SDK search-space preparation and materialization | `aisimulate.sweep_config_providers`; provider API 1 |
| [Recommendation output](output-abi.md) | Optional live callbacks and additional final artifacts | `aisimulate.output_adapters`; output API 1 |
| [Native composition](native-composition.md) | Placement and optional scaling in Rust Replayer | Rust traits in `aisimulate_core::replay`; crate-version compatibility |

Here ABI means the versioned cross-package data/interface contract. These
Python protocols and Rust traits are not a stable C binary ABI. Check the
installed package/crate pair rather than inferring compatibility from a shared
method name. Explicit version checks fail instead of silently translating an
unsupported contract.

AISimulate owns traffic, engine schemas, optimization, process supervision and
canonical output. Config adapters return data and hooks; a runner combines them
and starts execution once. The native composition depends on AISimulate's
neutral types, while AISimulate core does not import Dynamo's Router/Planner.
Output adapters do not influence simulation or ranking.

Discovery imports selected plugins only. Configuration/output names must be
unique and match their declared names. Built-in runner names take precedence;
for an optional stack, duplicate installed entry points fail. Zero-argument
constructors or preconstructed values are accepted where the resolver permits.
Provider objects stay in the coordinator process; replay workers receive a
serializable factory and data-only `ReplaySpec`, not a pickled provider.

The legacy [recipe configuration importer](../aic-backward-compatibility/configuration-import.md)
converts external recipes to AIC requests. It is neither `SimulationConfigAdapter`
nor `SweepConfigProvider` and does not add a replay framework.
