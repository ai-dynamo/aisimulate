# AISimulate Python source migration

AISimulate 0.13.0 removes the
`aiconfigurator` and `aiconfigurator_core` Python import namespaces. The legacy
`aiconfigurator` executable remains available, implemented by
`aisimulate.legacy_cli.entrypoint`. Published 0.12.0 wheels retain their existing
behavior; this change does not alter already released artifacts.

## Package layout

The source tree has two top-level packages:

| Package | Owns |
|---|---|
| `aisimulate` | Unified CLI, Replay, Sweeper, application SDK, configuration adapters, generator, legacy CLI, and the `_runtime` native extension |
| `aisimulate_core` | Estimator SDK, model configurations, performance data, and the `_native` binding facade |

Application SDK modules that expose core types delegate to the canonical core
module objects, preserving registry, exception, and cache identity. Core code
does not import the application orchestration or legacy CLI layers.

## Replacement imports

| Removed import | Replacement |
|---|---|
| `aiconfigurator_core` | `aisimulate_core` |
| `aiconfigurator_core.sdk` | `aisimulate_core.sdk` |
| `aiconfigurator.sdk.task_v2` | `aisimulate.sdk.task_v2` |
| `aiconfigurator.sdk.config_adapter` | `aisimulate.sdk.config_adapter` |
| `aiconfigurator.sdk.<core module>` | `aisimulate_core.sdk.<core module>` |
| `aiconfigurator.generator` | `aisimulate.generator` |
| `aiconfigurator.cli` | `aisimulate.legacy_cli` (legacy workflow internals) |

For new estimator integrations, prefer the supported
[core SDK facade](core-api.md#stable-python-facade):

```python
from aisimulate_core.sdk import EngineHandle, estimate_kv_cache
from aisimulate.sdk.task_v2 import Task
```

Resource lookup must use `importlib.resources.files("aisimulate_core")` for
`model_configs/` and `systems/`. The old `aic-core/` source symlinks are removed.
Do not infer resource locations from the legacy CLI package.

## Upgrade and qualification

Uninstall the former standalone distributions before installing the new wheel:

```bash
python3 -m pip uninstall -y aiconfigurator aiconfigurator-core
python3 -m pip install --upgrade aisimulate
aisimulate --help
aiconfigurator cli --help
```

Callers must update imports before using the new wheel. Old pickles that encode
removed module paths also require conversion in the old environment. There is
no automatic legacy import hook.

Release qualification requires exact-wheel Dynamo Router, Planner, Mocker, and deployment
adapter qualification before release. Removing import namespaces does not prove
those downstream consumers have migrated. Coordinate the wheel/crate minor
version with the release change; do not backport this break to 0.12.x.

Historical migration examples, upstream source/license attribution, and retained
legacy command names continue to identify AIConfigurator accurately. Existing
`aic_*` replay transport fields, native `AicEngine` type names, and FPM resource
labels are separate downstream contracts; their coordinated rename remains a
follow-up in AIC-1995 before the broader naming cleanup is complete.
