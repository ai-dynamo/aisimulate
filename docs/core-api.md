<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prediction and recommendation Python APIs

These application APIs compile public configuration into Replay and Sweeper runs.
For per-operation and forward-pass estimates, see the
[performance-model Python API](perf-model/api/python.md). For direct runner
control, see the [Replay Python API](replay/api/python.md).

## Table of contents

- [Python prediction API](#python-prediction-api)
- [Python recommendation API](#python-recommendation-api)

## Python prediction API

Use `aisimulate.predict.run_prediction` to compile and execute one public
`CorePredictionConfig`. It checks runner capabilities, creates and closes the
runner, and returns a `PredictionResult` with the same summary the CLI renders.

Use `prediction.yaml` from the [README prediction example](../README.md#predict-one-deployment).
In the activated AISimulate environment, save the following as
`predict_example.py` and run `python predict_example.py` from the directory
containing the YAML file.

```python
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.predict import run_prediction
from aisimulate.runner import EngineReplayRunnerFactory

if __name__ == "__main__":
    result = run_prediction(
        CorePredictionConfig.from_yaml("prediction.yaml"),
        stack="engine",
        runner_factory=EngineReplayRunnerFactory(),
    )
    print(result.summary)
```

See the [prediction configuration guide](reference/cli.md) for timing and workload options.

| Argument | Contract |
| --- | --- |
| `config` | Required validated `CorePredictionConfig`. |
| `stack` | Required keyword naming the execution stack; use `"engine"` with `EngineReplayRunnerFactory`. |
| `runner_factory` | Required keyword supplying a factory compatible with the compiled replay. |
| `adapter_configs` | Optional raw adapter blocks keyed by section name. |
| `providers` | Resolved config adapters keyed by `"<stack>.<section>"`; required for every supplied adapter block. Core config does not accept Router/Planner top-level sections. |
| `execution_mode` | Defaults to `"offline"`; the selected runner must support the requested mode. The built-in engine stack is offline. |
| `output_requirements` | Optional `ReplayOutputRequirements` from `aisimulate.sweeper.replay`. Omission enables raw-report capture except for analytical EPD. An explicit value replaces that default. |

`PredictionResult` contains:

- `summary`: merged prediction metrics, including normalized power fields.
- `native`: the native report with the summary merged in, or a summary fallback
  when the runner supplies no native report. Analytical EPD retains its metadata
  and approximation semantics instead of a token-replay report.
- `replay_spec`: the compiled specification that was executed.
- `report`: the runner's original `ReplayReport`, including metrics and metadata.

After creating a runner, the call closes it even when execution fails. Runner
execution exceptions are wrapped in `PredictionExecutionError`, preserving the
cause and any `fpm_query_coverage` attribute. `KeyboardInterrupt` and
`ResourceLimitError` propagate unchanged; configuration, compilation, capability
and runner-creation failures also propagate directly.

The call does not create output directories, save report files or print results.
The caller owns those actions and resource budgets. Passing the plain engine
factory does not enable the CLI's host-memory admission or subprocess supervision;
use the CLI for the automatic [local resource controls](reference/local-resources.md).

Source: [prediction entry point](../python/aisimulate/src/aisimulate/predict.py).

## Python recommendation API

Use `aisimulate.recommend.run_recommendation` to search a public
`CoreRecommendationConfig` and return a `SweepResult`. This entry point lowers
the public configuration into the [Sweeper SDK](sweeper/sdk.md); it does not
accept a `SmartSearchConfig` in place of `CoreRecommendationConfig`.

Use `recommendation.yaml` from the [README recommendation example](../README.md#recommend-a-deployment).
In the activated AISimulate environment, save the following as
`recommend_example.py` and run `python recommend_example.py` from the directory
containing the YAML file.

Keep the main guard because recommendation runs in supervised subprocesses.

```python
from aisimulate.config.cli import CoreRecommendationConfig
from aisimulate.recommend import run_recommendation
from aisimulate.runner import EngineReplayRunnerFactory

if __name__ == "__main__":
    result = run_recommendation(
        CoreRecommendationConfig.from_yaml("recommendation.yaml"),
        stack="engine",
        runner_factory=EngineReplayRunnerFactory(),
        show_progress=False,
    )
    print(result.counts)
    for candidate in result.selected_candidates:
        print(candidate.score, candidate.used_gpus, candidate.prediction_config)
```

| Argument | Contract |
| --- | --- |
| `config` | Required validated `CoreRecommendationConfig`. |
| `stack` | Required keyword naming the execution stack; use `"engine"` with `EngineReplayRunnerFactory`. |
| `runner_factory` | Required keyword supplying the candidate replay factory. |
| `adapter_configs` | Optional raw adapter blocks keyed by section name. |
| `providers` | Resolved config adapters keyed by `"<stack>.<section>"`; required for every supplied adapter block. |
| `output_configs` | Optional output-adapter configuration blocks for candidate and round callbacks. |
| `afd_performance_model` | Optional performance-model provider for analytical AFD search. |
| `show_progress` | Defaults to `True`; set `False` to disable progress display. |
| `output_requirements` | Optional `ReplayOutputRequirements` forwarded to each candidate replay. |

`SweepResult` contains:

- `candidates`: candidate records, including status, metrics, score, provenance,
  concrete prediction configuration and any rejection or failure reason.
- `counts`: totals for evaluated, feasible, failed, resource-limited and other
  candidate outcomes. A returned result does not imply that every candidate succeeded.
- `selected_candidate_ids` and `selected_candidates`: selected ledger IDs and
  corresponding `Candidate` objects in the same order. Each selected candidate's
  `prediction_config` can be used for a prediction.
- `execution_resources`: host-resource supervision evidence for this run.

Call `result.to_json()` to serialize the sweep result. Without output adapters,
this API does not publish the CLI's recommendation directory or YAML files.

Recommendation applies host-resource admission and subprocess supervision,
limits worker concurrency and cleans up its workers. Put calls behind
`if __name__ == "__main__":` in scripts; factories and providers must be
pickleable. See [local resource controls](reference/local-resources.md) for budgets,
resource-limited candidate results and bounded partial evidence.

Invalid core input fails schema validation. Supervised execution failures raise
`RuntimeError`; host-resource refusal or interruption raises `ResourceLimitError`,
and cancellation raises `KeyboardInterrupt`. Individual candidate failures are
recorded in the result rather than necessarily stopping the entire search.

Source: [recommendation entry point](../python/aisimulate/src/aisimulate/recommend.py)
and [result contract](../python/aisimulate/src/aisimulate/sweeper/result.py).
