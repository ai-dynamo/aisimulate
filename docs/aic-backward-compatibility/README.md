<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIC Backward Compatibility

The `aisimulate` wheel includes the established `aiconfigurator` console command
for AIC workflows that have not moved to the unified simulation CLI. New
prediction and search integrations should start with `aisimulate predict` and
`aisimulate recommend`; they use serving traffic and scheduler-formed batches,
so migrating an AIC fixed-batch estimate is not a command-name substitution.

| Task | Documentation |
|---|---|
| Replace packages, imports and saved configurations | [Migration](migration.md) |
| Continue `default`, `estimate`, `recommend`, `exp`, `generate` or `support` | [Compatibility CLI](cli.md) |
| Configure AIC tasks and advanced search | [Advanced tuning](cli.md#advanced-tuning) |
| Import Dynamo recipes or InferenceX records into AIC estimates | [Configuration import](configuration-import.md) |
| Interpret the legacy matrix and curated model roster | [AIC support matrix](support-matrix.md) |

For AISimulate 0.13, install `aisimulate` instead of the standalone
`aiconfigurator` and `aiconfigurator-core` distributions. Python imports use
`aisimulate` and `aisimulate_core`; the old Python import namespaces are removed.
The compatibility executable does not imply a retained legacy import hook.
Follow the [installation guide](../getting-started/installation.md) in an
isolated environment and retain matching wheel/crate versions.

Deployment generation is also available as a current typed AISimulate API;
see [the candidate-to-artifact workflow](../sweeper/deployment-generation.md).
It is not restricted to legacy CLI callers. AIC's external-recipe importer,
the simulation config-adapter ABI and that deployment generator serve different
purposes.
