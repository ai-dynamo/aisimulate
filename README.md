<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate

AISimulate predicts LLM serving behavior and searches for strong deployment
configurations offline, without bringing up a GPU serving cluster.

[Website](https://ai-dynamo.org/aisimulate/) ·
[E2E Accuracy Overview](https://ai-dynamo.org/aisimulate/e2e-accuracy/) ·
[FPM Accuracy Overview](https://ai-dynamo.org/aisimulate/fpm-accuracy/) ·
[FPE Support Matrix](https://ai-dynamo.org/aisimulate/fpe-support-matrix/) ·
[Legacy AIC Support Matrix](https://ai-dynamo.org/aisimulate/support-matrix/)

AISimulate is the successor to the
[AIConfigurator (AIC)](https://github.com/ai-dynamo/aiconfigurator)
repository. It brings the complete AIC application and estimator into one
standalone home with Dynamo-independent Replay and Sweeper capabilities.

The performance-modeling methodology is described in
[AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework
LLM Serving](https://arxiv.org/abs/2601.06288).

## Install

See the [installation guide](docs/installation.md) for published versions,
platform requirements, current-source setup, and internal nightlies. Documentation
on `main` can describe features newer than the latest published wheel.

### Engine-only

Install AISimulate by itself to use the built-in simulation engine without a
Dynamo dependency:

```bash
python3 -m pip install aisimulate
aisimulate --help
```

### With Dynamo

Install compatible AISimulate and Dynamo releases to enable the `dynamo`
runner and Dynamo-owned configuration adapters:

```bash
python3 -m pip install aisimulate ai-dynamo
aisimulate predict --help
```

AISimulate remains the CLI owner in both profiles. Select the integration at
runtime with `--stack dynamo`; installing Dynamo does not add another
simulation command.

**Planner needs additional dependencies.** The two-package installation above
supports basic Dynamo prediction, but does not install the complete Planner
environment. Before using a top-level `planner` section, install Dynamo's
`container/deps/requirements.planner.txt` from the same release tag or commit
as your Dynamo wheels. For example, after installing the Dynamo 1.5.0 RC9
artifacts and their compatible AISimulate wheel:

```bash
# Example for Dynamo 1.5.0 RC9; change this to your installed build's revision.
DYNAMO_REF=ffd7c1a90eb403c0d43911690c5c9b8457acd826
python3 -m pip install "grpcio-tools<=1.76.0" -r \
  "https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REF}/container/deps/requirements.planner.txt"
python3 -m pip check
```

The `grpcio-tools` cap matches RC9's
[common requirements](https://github.com/ai-dynamo/dynamo/blob/ffd7c1a90eb403c0d43911690c5c9b8457acd826/container/deps/requirements.common.txt).
It keeps the tooling compatible with Planner's `protobuf==6.33.6` pin.
When selecting another Dynamo revision, check its common requirements and
update this cap together with `DYNAMO_REF`.

For release candidates, use the exact release artifacts; a package version
alone may not identify the RC build. Alternatively, use the matching
`dynamo-planner` image, which includes the Planner prerequisites. See the
[Planner installation example](docs/cli/examples/dynamo-planner/README.md)
for a complete CPU-only prediction that loads Planner. `predict --help`
does not verify that optional adapters can load.

### Upgrade from standalone AIConfigurator

Remove the former standalone distributions first so that only AISimulate owns
the compatibility imports and command:

```bash
python3 -m pip uninstall -y aiconfigurator aiconfigurator-core
python3 -m pip install --upgrade aisimulate
```

## Predict one deployment

`predict` evaluates one pinned deployment configuration. Save this example as
`prediction.yaml`:

```yaml
engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  workers:
    aggregated: {}
```

### Engine-only prediction

```bash
aisimulate predict \
  --stack engine \
  --config prediction.yaml \
  --output-dir ./aisimulate-prediction
```

### Dynamo-integrated prediction

```bash
aisimulate predict \
  --stack dynamo \
  --config prediction.yaml \
  --output-dir ./aisimulate-dynamo-prediction
```

The CLI prints a concise summary and writes the selected runner's complete
report to `<output-dir>/prediction.json`. Add `--capture-per-request` to also
write `requests.jsonl`, or use `--format json` for machine-readable standard
output.

## Recommend a deployment

`recommend` searches the prediction schema plus search domains and an
optimization goal. Save this example as `recommendation.yaml`:

```yaml
engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: auto
  backend: {choices: [vllm, sglang]}
  workers:
    aggregated:
      parallelism: {preset: default}

optimization:
  target: throughput_per_gpu
  hardware: h200_sxm
  constraints:
    max_candidate_gpus: 8
```

### Engine-only recommendation

```bash
aisimulate recommend \
  --stack engine \
  --config recommendation.yaml \
  --output-dir ./aisimulate-recommendation
```

### Dynamo-integrated recommendation

```bash
aisimulate recommend \
  --stack dynamo \
  --config recommendation.yaml \
  --output-dir ./aisimulate-dynamo-recommendation
```

Each file under `<output-dir>/recommendations/` is a fully materialized,
concrete configuration. It contains no search domains and can be passed
directly back to `predict`:

```bash
aisimulate predict \
  --config ./aisimulate-recommendation/recommendations/0001.yaml \
  --output-dir ./aisimulate-best-prediction
```

Both commands support `--set PATH=YAML_VALUE`, `--output-dir`, `--overwrite`,
and `--format table|json`. See the [AISimulate CLI User Guide](docs/cli/user-guide.md) for the
complete schema, traffic models, search domains, presets, outputs, and error
contract.

## AIConfigurator compatibility CLI

The `aisimulate` wheel preserves the established `aiconfigurator` command for
workflows that have not yet moved to the unified CLI. AISimulate 0.13.0 keeps
this compatibility surface, while new prediction and search integrations
should start with `aisimulate predict` and `aisimulate recommend`.

```bash
# Check whether a model/system combination is supported.
aiconfigurator cli support \
  --model-path Qwen/Qwen3-32B-FP8 \
  --system h200_sxm

# Generate a starting deployment configuration.
aiconfigurator cli generate \
  --model-path Qwen/Qwen3-32B-FP8 \
  --total-gpus 8 \
  --system h200_sxm
```

The compatibility CLI preserves six workflows:

| Mode | Purpose |
|---|---|
| `default` | Compare aggregated and disaggregated candidates and select a strong starting point |
| `estimate` | Estimate one explicitly configured deployment |
| `recommend` | Find the minimum GPU count and configuration for a load target and SLA |
| `exp` | Run custom experiments from YAML |
| `generate` | Generate deployment artifacts without a parameter sweep |
| `support` | Check model and system coverage |

Read the [Legacy AIC CLI User Guide](docs/cli/legacy-aic-user-guide.md) for
command examples and the [AIC CLI and Python API overview](python/aisimulate/README.md)
for the complete compatibility surface. The
[AIC migration guide](docs/cli/migrate-from-aiconfigurator.md)
explains which AIC workflows map to `predict` or `recommend` and which ones
must continue using the compatibility command for now.

## AIConfigurator repository transition

The standalone [AIConfigurator repository](https://github.com/ai-dynamo/aiconfigurator)
will publish its final 0.12.0 `aiconfigurator` and `aiconfigurator-core`
artifacts and then be archived. AISimulate is the canonical home for ongoing
development, releases, issues, and pull requests; open all new issues and pull
requests in this repository.

The `aiconfigurator` compatibility command remains available from the
`aisimulate` wheel through 0.13.0. It is targeted for removal in AISimulate 0.14.0,
after every remaining AIC workflow has a verified replacement in the unified
`aisimulate` CLI. Until then, use the compatibility command for the workflows
identified in the migration guide.

## Experimental status and validation boundary

> [!WARNING]
> Replay and Sweeper are experimental surfaces intended for evaluation and
> feedback, not production capacity planning. Their APIs, schemas, search
> behavior, and output may change without a standard deprecation period.

AISimulate narrows a deployment search and identifies candidates; it does not
replace validation on the target hardware. Benchmark shortlisted
configurations on a real deployment before making production capacity or SLA
decisions.

## SDKs

Use the focused SDK documentation instead of treating CLI internals as public
APIs:

- [Estimator/FPE Python and Rust SDK](docs/core-api.md)
- [AIC-compatible modeled-power contract (semantics only)](docs/power-model.md)
- [FPM collection-to-prediction workflow](python/aisimulate/docs/fpm/end-to-end-workflow.md)
- [Replay SDK and artifact contract](crates/core/src/replay/README.md)
- [Sweeper SDK](docs/sweeper/overview.md)
- [AIConfigurator compatibility Python API](python/aisimulate/README.md#python-api)

## Support and accuracy

[Understand your prediction](docs/cli/understand-your-prediction.md) explains
report fields, latency populations, incomplete requests, and SLA interpretation.

Support coverage and accuracy are separate evidence. A supported cell means a
specific path can execute with the required data; it does not establish that
the resulting end-to-end prediction is accurate.

### Explicit CUDA graph reservation

KV-cache estimation and engine replay accept an optional rank-local
`cuda_graph_reserved_bytes` value. AISimulate subtracts this fixed runtime
reservation before allocating KV cache and preserves it when the native replay
runtime rematerializes capacity. For SGLang, the value is additional to the
graph/runtime headroom already encoded by `mem_fraction_static`. The default is
zero, so existing serialized callers do not change. See the
[core API contract](docs/core-api.md#kv-cache-capacity-reservation).

### FPE support matrix

The published [Forward Pass Engine (FPE) matrix](https://ai-dynamo.org/aisimulate/fpe-support-matrix/)
measures strict-native estimator coverage across a curated roster of current
models, GPU systems, backends, and backend versions. It probes native prefill,
decode-start, decode-end, and mixed forward-pass calls without fallback.
It does not certify the CLI, Replay,
Sweeper, serving orchestration, or prediction accuracy.

The matrix was introduced in
[AISimulate PR #41](https://github.com/ai-dynamo/aisimulate/pull/41). Nightly
CI refreshes the complete matrix at the nightly source SHA before release
artifacts advance to Artifactory.

### AIC CLI support matrix

The compatibility support matrix covers AIC command-based aggregated and
disaggregated workflows by model, system, backend, and backend version:

- [Interactive legacy AIC support matrix](https://ai-dynamo.org/aisimulate/support-matrix/)
- [AIC support-matrix data](python/aisimulate/src/aiconfigurator_core/systems/support_matrix/)
- [Curated model roster](python/aisimulate/docs/support-matrix/model-roster.md)

Check one exact cell from the installed package with:

```bash
aiconfigurator cli support \
  --model-path Qwen/Qwen3-32B-FP8 \
  --system h200_sxm \
  --backend vllm \
  --backend-version 0.14.0
```

### E2E accuracy overview

The published [E2E Accuracy Overview](https://ai-dynamo.org/aisimulate/e2e-accuracy/)
reports TTFT and TPOT error, curve-shape error, and prediction coverage against
matched measured-silicon operating points. It is evidence for the measured
configurations, not a universal support contract.

Forward-pass accuracy is tracked separately; see the
[prediction regression and accuracy design](python/aisimulate/docs/design/prediction_regression_gate_design.md)
and the current [silicon anchor set](python/aisimulate/tools/accuracy_tracking/silicon_refs.csv).

## Release artifacts and compatibility

Starting with 0.12.0, this repository owns the complete AIConfigurator product
surface—not only its native core. The application, CLI, generator, SDK,
Collector, tests, documentation, and development tooling live under
`python/aisimulate/`.

This repository produces exactly two release artifacts:

1. `aisimulate` Python wheel — the application, both console commands,
   estimator SDK, model/performance data, Replay, Sweeper, and unified native
   extension;
2. `aisimulate-core` Rust crate — the native estimator and simulation core for
   Rust consumers.

It does **not** publish an `aiconfigurator` or `aiconfigurator-core` wheel, a
Python `aisimulate-core` distribution, or an `aiconfigurator-core` crate. The
`aisimulate` wheel preserves the `aiconfigurator`, `aiconfigurator_core`, and
`aisimulate_core` Python import namespaces during the compatibility window.

The AISimulate wheel does not declare Dynamo as an installation dependency.
Dynamo-owned Router, Planner, runtime, transport, and live-Mocker integrations
consume AISimulate through optional adapters.

### Packaged legal-file copies

The root [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
are the canonical repository legal files. Because the Python wheel build is
rooted at `python/aisimulate/`, byte-identical copies are retained there so the
wheel can declare and distribute them. These copies do not create a separate
licensing boundary, and CI fails if either copy differs from its root original.
See the [artifact contract](docs/artifact-contract.md#packaged-license-files) for
the complete packaging contract.

## Develop from source

```bash
git clone https://github.com/ai-dynamo/aisimulate.git
cd aisimulate

uv venv .venv
source .venv/bin/activate
uv pip install -e ./python/aisimulate
```

Current performance profiles are checked-in Parquet files, so normal builds
and usage do not require Git LFS. Install Git LFS and run `git lfs pull` only
when working with retained legacy `*.txt` performance assets or their
compatibility tests.

## Accuracy evidence

The published [E2E Accuracy Overview](https://ai-dynamo.org/aisimulate/e2e-accuracy/)
reports matched client-observed TTFT and TPOT accuracy against measured silicon
operating points. It keeps accuracy, evidence coverage, and curve-shape error
separate and includes a machine-readable aggregate with exact snapshot digests.
See the [snapshot and regeneration details](pages/e2e-accuracy/README.md)
for evidence provenance and instructions to rebuild the report.

The checked-in snapshot excludes multi-node configurations and applies only to
the exact model, hardware, framework, topology, workload, and concurrency cells
that were measured. It is not a universal support or deployment-certification
claim. Forward-pass accuracy and strict-native estimator coverage remain
separate evidence lanes.

For a quick local validation subset:

```bash
cargo test --workspace
python -m pytest -c pytest.ini tests
python -m pytest -c python/aisimulate/pytest.ini python/aisimulate/tests -m "unit or build"
```

See the [CI guide](docs/ci.md) for the Fast/Full/Nightly hierarchy, code review,
complete test coverage, and release gates. Use [DEVELOPMENT.md](DEVELOPMENT.md)
for environment and local test details and [CONTRIBUTING.md](CONTRIBUTING.md)
before sending a change.

Direct Python `ReplaySpec.workload` synthetic workloads accept
`length_sampler: numpy_random_state` for InferenceX-compatible seeded token
lengths in Gym replay. The public prediction/recommendation YAML and CLI, and
the Sweeper `Workload` schema, do not expose this option and reject it.
The supported direct path must omit `source_type`; all workload-driver inputs
with `source_type` reject `length_sampler`. Materialized trace replay rejects
non-default samplers. The default `python_random`
preserves existing workloads. Both sample the full input vector before output
lengths; unknown sampler names are rejected. NumPy seeds must fit an unsigned
32-bit integer; the Python sampler retains unsigned 64-bit seed support.

### FPM accuracy overview

The [FPM Accuracy Overview](https://ai-dynamo.org/aisimulate/fpm-accuracy/?branch=main)
reports daily FPM (KV warmup on), FPM (KV warmup off), and online regression accuracy against
pinned Hugging Face measurements. It evaluates main and releases >= 0.12.0,
with MAPE, prediction coverage, and exact source provenance. Results stay in
GitHub Actions artifacts; the main Pages build publishes qualified aggregates.
See [evaluation and publication details](pages/fpm-accuracy/README.md).

Webpage sources live in [pages/](pages/README.md). Rust design documentation
remains under docs and is not deployed.
