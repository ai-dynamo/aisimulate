<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<!--
Maintainers and agents: CI executes the bash/yaml examples in this README.
- Keep each existing readme-check marker directly above its opening code fence,
  with no blank line between them. Keep IDs stable and unique.
- Use unindented triple-backtick fences labeled exactly bash or yaml. Do not
  change fence labels or formatting to bypass command coverage.
- When adding a block, copy an existing marker's format and use a new ID made
  of lowercase letters, digits, and hyphens. Add the matching entry to
  scripts/readme/readme_commands.json; remove both together when deleting a block.
- The manifest contains profiles, dependencies, timeouts, output assertions,
  and YAML filenames, not copies of commands. README blocks are the source.
- Keep dependencies before their consumers (source-install is bootstrapped
  first). Update manifest paths/assertions when changing outputs or examples.
- Validate structure before committing (requires psutil and PyYAML):
  python scripts/readme/check_readme_commands.py --validate
  This checks parsing/manifest consistency; CI also executes the commands.
See docs/ci.md and scripts/readme/check_readme_commands.py for execution details.
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
See the [migration guide](docs/MIGRATION.md) for command and API mappings.

The performance-modeling methodology is described in
[AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework
LLM Serving](https://arxiv.org/abs/2601.06288).

## Install

See the [installation guide](docs/installation.md) for published versions,
platform requirements, current-source setup, and internal nightlies. Documentation
on `main` can describe features newer than the latest published wheel.

### Engine-only

Install AISimulate by itself to use the built-in simulation engine without a
Dynamo dependency. Use a fresh virtual environment for each installation profile.
These commands install published releases; for the examples on `main`, use
[Develop from source](#develop-from-source):

<!-- readme-check: release-install -->
```bash
python3 -m pip install aisimulate
aisimulate --help
```

### With Dynamo

> **Compatibility warning:** Older Dynamo releases may be incompatible with
> current AISimulate source. Dynamo 1.5.0 uses removed AISimulate imports, and
> 1.4.2 does not register the `dynamo` stack. The pinned nightly pair below is
> validated together; it does not establish compatibility with AISimulate tip
> of tree. Do not upgrade AISimulate independently of Dynamo's declared dependency.

Install this pair on Linux with Python 3.12 in a **separate environment**:

<!-- readme-check: dynamo-install -->
```bash
python3 -m pip install --extra-index-url https://pypi.nvidia.com \
  "aisimulate==0.13.0.dev202609270000000058" "ai-dynamo==1.6.0.dev20260930"
python3 -m pip check
aisimulate predict --help
```

This pair exercises the nightly AISimulate wheel, not the current source checkout.
The daily README workflow records installed versions and runs both Dynamo
prediction and recommendation; `--help` alone does not validate the adapters.

**Planner needs additional dependencies.** Install the requirements from the
same Dynamo revision as the wheels before using a top-level `planner` section.
See the [Planner installation example](docs/cli/examples/dynamo-planner/README.md).
The root README examples use the replay adapter without Planner.

### Upgrade from standalone AIConfigurator

In an engine-only environment, remove the former distributions before upgrading:

<!-- readme-check: release-upgrade -->
```bash
python3 -m pip uninstall -y aiconfigurator aiconfigurator-core
python3 -m pip install --upgrade aisimulate
python3 -m pip check
```

## Predict one deployment

`predict` evaluates one pinned deployment configuration. Save this example as
`prediction.yaml`:

<!-- readme-check: prediction-config -->
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

<!-- readme-check: engine-predict -->
```bash
aisimulate predict \
  --stack engine \
  --config prediction.yaml \
  --output-dir ./aisimulate-prediction
```

### Python prediction API

Use the same `prediction.yaml` with the
[Python prediction API](docs/core-api.md#python-prediction-api).
See the API reference for the runnable example, arguments and results.

### Dynamo-integrated prediction

<!-- readme-check: dynamo-predict -->
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

For example, use two-way tensor parallelism, capture requests, and replace the
previous output with a JSON summary:

<!-- readme-check: engine-options -->
```bash
aisimulate predict \
  --stack engine \
  --config prediction.yaml \
  --set engine.workers.aggregated.parallelism.tensor=2 \
  --capture-per-request --format json --overwrite \
  --output-dir ./aisimulate-prediction > prediction-summary.json
```

For agentic trace replay, follow the [AgentX simulation quickstart](docs/agentx-quickstart.md).
It includes a Weka workload, a complete eight-GPU prefill/decode configuration,
KV cache warmup, and commands for running and inspecting the simulation.

## Recommend a deployment

`recommend` searches the prediction schema plus search domains and an
optimization goal. Save this example as `recommendation.yaml`:

<!-- readme-check: recommendation-config -->
```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 10}
  stop: {requests: 100}

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

optimizer:
  max_trials: 8
```

This quickstart limits the search to eight trials. Remove `optimizer.max_trials`
for the default 320-trial search, which can take substantially longer.

### Engine-only recommendation

<!-- readme-check: engine-recommend -->
```bash
aisimulate recommend \
  --stack engine \
  --config recommendation.yaml \
  --output-dir ./aisimulate-recommendation
```

### Python recommendation API

Use the same `recommendation.yaml` with the
[Python recommendation API](docs/core-api.md#python-recommendation-api).
See the API reference for the runnable example, arguments and results.

### Dynamo-integrated recommendation

<!-- readme-check: dynamo-recommend -->
```bash
aisimulate recommend \
  --stack dynamo \
  --config recommendation.yaml \
  --output-dir ./aisimulate-dynamo-recommendation
```

Each file under `<output-dir>/recommendations/` is a fully materialized,
concrete configuration. It contains no search domains and can be passed
directly back to `predict`:

<!-- readme-check: engine-roundtrip -->
```bash
aisimulate predict \
  --config ./aisimulate-recommendation/recommendations/0001.yaml \
  --output-dir ./aisimulate-best-prediction
```

For the Dynamo result, use the same stack on the round trip:

<!-- readme-check: dynamo-roundtrip -->
```bash
aisimulate predict \
  --stack dynamo \
  --config ./aisimulate-dynamo-recommendation/recommendations/0001.yaml \
  --output-dir ./aisimulate-dynamo-best-prediction
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

<!-- readme-check: legacy-cli -->
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
command examples and the [package overview](python/aisimulate/README.md)
for installation and current AISimulate workflows. The
[AIC migration guide](docs/MIGRATION.md)
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
- [Prediction and recommendation Python APIs](docs/core-api.md#python-prediction-api)
- [FPM self-service: onboard a model on target hardware](docs/fpm-self-service/README.md)
- [FPM self-service implementation and CLI reference](docs/fpm-self-service/implementation.md)
- [FPM self-service examples](docs/fpm-self-service/examples.md)
- [AIC-compatible modeled-power contract (semantics only)](docs/power-model.md)
- [Replay SDK and artifact contract](crates/core/src/replay/README.md)
- [Sweeper SDK](docs/sweeper/overview.md)
- [Legacy CLI reference](docs/cli/legacy-aic-user-guide.md)

### Whole-forward FPM data

Open-source whole-forward FPM datasets are hosted on
[Hugging Face](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset).
FPM prediction requires a Parquet dataset and its adjacent, same-stem
`.metadata.json` sidecar. See the
[core API guide](docs/core-api.md#external-whole-forward-fpm-data) for
configuration and compatibility fields.

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
- [AIC support-matrix data](python/aisimulate/src/aisimulate_core/systems/support_matrix/)
- [Curated model roster](python/aisimulate/docs/support-matrix/model-roster.md)

Check one exact cell from the installed package with (this cell currently reports
`NO`; exit status zero means the support query completed, not that it is supported):

<!-- readme-check: support-negative -->
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
`aisimulate` wheel exposes the `aisimulate` application and `aisimulate_core`
estimator packages. Only the legacy `aiconfigurator` executable remains; see
[Python source migration](docs/MIGRATION.md#python-imports-and-resources) for removed imports.

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

Install Python 3.12, `uv`, Rust/Cargo, and a C/C++ compiler first. From an empty
working directory (set `AISIMULATE_REF` to an exact commit for reproducibility):

<!-- readme-check: source-install -->
```bash
git clone https://github.com/ai-dynamo/aisimulate.git
cd aisimulate
git checkout "${AISIMULATE_REF:-main}"

uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -e ./python/aisimulate
python -m pip check
```

After pulling native Rust changes, rebuild the editable installation with
`uv pip install --reinstall-package aisimulate -e ./python/aisimulate`.
A missing `_runtime` attribute such as `SglangPrefillAttentionSequence` usually
means the Python source and compiled extension are from different revisions.
Run the command in the environment that owns your `aiconfigurator` executable.

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

For complete local validation, start in the repository root with the source
virtual environment active. Keep this environment engine-only; Dynamo's pytest
plugins and optional modules can interfere with isolated package tests.

<!-- readme-check: test-install -->
```bash
uv pip install -e './python/aisimulate[dev]' 'maturin>=1.12,<2' 'pip-licenses==5.5.5'
uv pip install -r scripts/fpm_accuracy/requirements.txt
python -m pip check
git fetch origin d066e918705b98e2d55eed55743ce8d225f129ea
```

These tests need network access for model metadata. Use valid Hugging Face
credentials, or anonymous access to public models. For anonymous metadata access,
set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`; custom token/cache locations use
`HF_TOKEN_PATH`, `HF_HOME`, and `XDG_CACHE_HOME` (including `~/` paths).
The per-process Git settings
below disable signing only for temporary test repositories. On macOS, append
`-p no:timeout` to pytest to avoid SIGALRM crash dialogs.

<!-- readme-check: rust-tests -->
```bash
cargo test --workspace
```

<!-- readme-check: root-tests -->
```bash
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=commit.gpgsign GIT_CONFIG_VALUE_0=false \
  PYTHONPATH="$PWD/python/aisimulate:$PWD/python/aisimulate/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m pytest -c pytest.ini tests
```

<!-- readme-check: package-tests -->
```bash
(cd python/aisimulate && \
  GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=commit.gpgsign GIT_CONFIG_VALUE_0=false \
  PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m pytest -c pytest.ini tests -m "unit or build")
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
with `source_type` reject `length_sampler` at the common runner entrypoint,
including AFD and AFD+PD. Materialized trace replay rejects
non-default samplers. The default `python_random`
preserves existing workloads. Both sample the full input vector before output
lengths; unknown sampler names are rejected. NumPy seeds must fit an unsigned
32-bit integer; the Python sampler retains unsigned 64-bit seed support.
Each replay initializes its own seeded sampler, so independent runs reproduce
the same request lengths.

### FPM accuracy overview

The [FPM Accuracy Overview](https://ai-dynamo.org/aisimulate/fpm-accuracy/?branch=main)
reports daily FPM (KV warmup on), FPM (KV warmup off), and online regression accuracy against
pinned Hugging Face measurements. It evaluates main and releases >= 0.12.0,
with MAPE, prediction coverage, and exact source provenance. Results stay in
GitHub Actions artifacts; the main Pages build publishes qualified aggregates.
See [evaluation and publication details](pages/fpm-accuracy/README.md).

Webpage sources live in [pages/](pages/README.md). Rust design documentation
remains under docs and is not deployed.
