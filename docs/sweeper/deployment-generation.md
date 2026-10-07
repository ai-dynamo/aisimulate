<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Generate and validate a deployment

Use this workflow to select a configuration, predict it, generate serving
artifacts, and compare with a real benchmark. Run the simulation and generator
steps from an AISimulate source checkout using the
[installation guide](../getting-started/installation.md#use-current-source).
The GPU deployment steps run on a separate serving host.

The unified `recommend` command writes concrete **prediction configurations**.
Its numbered YAML files are not Kubernetes manifests. The current generator API is `aisimulate.generator`; the same wheel also
provides the `aiconfigurator` compatibility command. No separate standalone
AIConfigurator installation is required.

## 1. Pin the example

This recipe uses one aggregated worker, Qwen3-32B-FP8, H200 hardware,
1 or 2 GPUs, 1,024 input tokens, 128 output tokens, concurrency 4, and prefix
caching disabled. It intentionally uses a small search budget to demonstrate
the workflow; it does not establish the best possible deployment.

| Version/identity | Choice |
|---|---|
| AISimulate | Your source checkout; retain `git rev-parse HEAD` and `uv.lock`. |
| Prediction backend/data version | vLLM `0.24.0`. |
| Generated backend configuration | vLLM `0.24.0`, taken from the selected candidate. |
| Dynamo deployment target | `dynamo-j2`, Dynamo `1.3.0`. |
| Serving image | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0`; record the pulled digest. |

The Dynamo/backend pair is recorded in the generator's
[version matrix](../../python/aisimulate/src/aisimulate/generator/facts/runtimes/dynamo.yaml).
These pins make the intended environment explicit; local artifact generation
is not a GPU deployment qualification. Before serving, verify the image's
actual versions and the target machine's driver/GPU requirements using the
[Dynamo 1.3.0 release](https://github.com/ai-dynamo/dynamo/releases/tag/v1.3.0).
For another backend or release, update the prediction-data version, generated
configuration version, and runtime image together. Do not assume a template
version override makes a different runtime compatible.

## 2. Recommend and predict

From the AISimulate repository root with its virtual environment active:

```bash
mkdir -p deployment-study
git rev-parse HEAD > deployment-study/aisimulate-source-sha.txt
uv pip freeze --python python/aisimulate/.venv/bin/python > deployment-study/python-packages.txt
aisimulate recommend --stack engine \
  --config examples/cli/dynamo-deployment-recommend.yaml \
  --output-dir deployment-study/recommendation

aisimulate predict --stack engine \
  --config deployment-study/recommendation/recommendations/0001.yaml \
  --capture-per-request --output-dir deployment-study/prediction
```

The [example YAML](../../examples/cli/dynamo-deployment-recommend.yaml)
fixes concurrency and compares two TP shapes under a two-GPU limit. Inspect
`recommendation.json` for failures and selected candidate IDs. If no configuration
is selected and there are zero resource-limited candidates, `recommend` exits `1`
and produces no numbered YAML. Any resource-limited candidate makes the exit
status `3`, including when the result is empty or some selected YAML files remain
available. Inspect the ledger and [resource diagnostics](../reference/local-resources.md)
before proceeding. Use a fresh output directory for
another study, or apply the CLI's explicit `--overwrite` policy.

## 3. Render the selected candidate

Save the following as `deployment-study/render.py`, and run it from the
repository root with `python deployment-study/render.py`. It reads the first
scalar selection and its matching workload from the durable ledger. For a
Pareto run, choose an ID from `views.pareto_front` in `recommendation.json`
and pass it explicitly, for example
`python deployment-study/render.py --candidate-id candidate-000001` (replace
the illustrative ID with your choice). Unknown and infeasible IDs are rejected.

```python
import argparse
import json
from dataclasses import replace
from pathlib import Path

from aisimulate.generator.api import generate_from_request
from aisimulate.generator.request import from_sweeper_candidate

parser = argparse.ArgumentParser(description="Render a selected recommendation")
parser.add_argument("--candidate-id", help="Explicit candidate ID; required for Pareto results")
args = parser.parse_args()
study = Path("deployment-study")
result = json.loads((study / "recommendation/recommendation.json").read_text())
candidate_id = args.candidate_id
if candidate_id is None:
    selected = result["views"]["top_n"]
    if not selected:
        parser.error("No scalar selection: choose a Pareto candidate with --candidate-id")
    candidate_id = selected[0]
candidate = next((row for row in result["candidates"] if row["candidate_id"] == candidate_id), None)
if candidate is None:
    parser.error(f"Unknown candidate ID: {candidate_id}")
if candidate["status"] != "feasible":
    parser.error("The selected candidate must be feasible")
request = from_sweeper_candidate(
    candidate,
    workload=candidate["provenance"]["workload"],
    deployment_target="dynamo-j2",
    output_dir=str(study / "generated"),
    generator_overrides={
        "ServiceConfig": {
            "model_path": "/workspace/models/Qwen3-32B-FP8",
            "served_model_name": "Qwen/Qwen3-32B-FP8",
            "head_node_ip": "127.0.0.1",
        },
    },
)
request = replace(request, backend=replace(request.backend, dynamo_version="1.3.0"))
artifacts = generate_from_request(request)
(study / "selected-candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
print("Selected:", candidate_id)
print("Generated:", sorted(artifacts))
```

The bridge preserves evaluated parallelism, worker counts, token limits, and
supported cache settings. Environment overrides must not silently replace
those choices. It rejects combinations it cannot lower, including analytical
AFD/EPD and heterogeneous P/D hardware. AFD predictions instead produce their
own analytical artifacts; see [AFD topology](../replay/engine/analytical.md#afd).

The renderer currently reports that vLLM `0.24.0` uses the closest prior
`cli_args.0.20.1.j2` flag template. Keep that warning with the generated artifacts
and verify the emitted flags against the actual `0.24.0` runtime.

Inspect the returned filenames and generated scripts. For this vLLM example,
`run_0.sh` launches the frontend and worker processes; `bench_run.sh` runs an
AIPerf sweep. Run `bash -n` on both before copying them to the serving host.
If you deliberately use the compatibility CLI's `generate` command instead,
it creates a starting configuration without replay/SLA optimization; it does
not recreate the selected recommendation. See the
[legacy CLI guide](../aic-backward-compatibility/cli.md).

## 4. Launch on the serving host

The generated scripts use Dynamo's default service discovery. On the serving
host, start the dependencies from the pinned Dynamo source checkout:

```bash
git clone --branch v1.3.0 --depth 1 https://github.com/ai-dynamo/dynamo.git dynamo-1.3.0
docker compose -f dynamo-1.3.0/dev/docker-compose.yml up -d
```

The release's [development Compose file](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/dev/docker-compose.yml)
starts etcd and NATS. This example uses the same host network for those services
and the serving container. The pinned
[Dynamo quickstart](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/docs/getting-started/quickstart.mdx)
also documents a file-discovery alternative, but switching to it requires
matching frontend and worker arguments.

Download the exact model revision you intend to benchmark into
`/absolute/model-cache/Qwen3-32B-FP8` on the serving host. Retain its revision
with the study and use that same model configuration for prediction. Copy
`deployment-study` to the serving host, then adapt the two absolute host paths:

```bash
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
docker image inspect nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0 \
  --format '{{json .RepoDigests}}'
docker run --rm -it --gpus all --network host --ipc host \
  -v /absolute/deployment-study:/workspace/study \
  -v /absolute/model-cache:/workspace/models:ro \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0 bash
```

Inside the container, check the versions before launch:

```bash
python -c 'from importlib.metadata import version; print("Dynamo", version("ai-dynamo")); print("vLLM", version("vllm"))'
cd /workspace/study/generated
bash run_0.sh
```

Keep this process running. In another shell on the same host, check the
frontend at `http://127.0.0.1:8000/v1/models` before benchmarking. Model loading
and frontend readiness do not establish SLA performance.

## 5. Benchmark and compare

In a second shell with AIPerf installed, access the generated directory and
model/tokenizer files. The runtime image may already contain AIPerf; otherwise
use the installation instructions for the AIPerf release you select and record
its version. Set the benchmark to the example's fixed concurrency:

```bash
AICONFIGURATOR_BENCH_CONCURRENCY=4 \
  AICONFIGURATOR_BENCH_MULTI_ROUND=10 \
  AICONFIGURATOR_BENCH_ENDPOINT_URL=http://127.0.0.1:8000 \
  BENCH_ARTIFACT_DIR=/workspace/study/benchmark \
  bash /workspace/study/generated/bench_run.sh
```

Inspect `bench_run.sh` first: it records token lengths and standard deviations,
request count, prefix length, warmup count, endpoint type, and tokenizer. The
overrides above request 40 measured requests (`4 × 10`) plus 8 warmup requests. Its
warmup requests are additional to measured requests. Match these settings to
the replay workload; record any warmup, tokenizer, startup, or cache-history
differences rather than interpreting them as estimator error. Retain AIPerf's
raw outputs and check completed-request counts before comparing latency.

Compare like-for-like TTFT, per-request TPOT, throughput, and GPU normalization.
AIPerf's per-request average ITL corresponds to replay TPOT; replay's
individual-token-gap ITL distribution is a different population. See
[Understand your prediction](../getting-started/understand-results.md)
for definitions, incomplete requests, and goodput/SLA interpretation.

For a concurrency curve, change both the replay load and benchmark concurrency
for each point. A candidate that wins at one concurrency may lose at another.
A sampled Pareto front does not bound all possible hardware performance.

### Multi-node and Kubernetes variants

This worked example is single-node. For multi-node deployment, generate the
intended topology, inspect every emitted `run_N.sh`, configure the real head
node address, and follow the selected Dynamo release's networking and service
discovery instructions. A GPU budget alone does not choose placement or prove
that every GPU in the budget is used.

For Kubernetes, install the matching Dynamo platform/CRDs first. Set
`K8sConfig.k8s_namespace`, model cache/PVC, and image explicitly, then inspect
`k8s_deploy.yaml` against that release's CRD. The
[target reference](#deployment-targets) describes the available targets
and overrides. Do not apply a numbered `recommendations/*.yaml` as a manifest.

## Deployment targets

The typed generator lowers a request through deployment defaults, rule plugins,
backend parameter mappings and versioned templates. Template versions,
prediction data versions and runtime image versions are separate identities;
validate them together. Cluster-specific namespace, mounts, model path and
image settings belong to generator overrides.

| Target | Artifacts and scope |
|---|---|
| `dynamo-j2`, `dynamo-python` | Dynamo worker launch scripts, backend flags/configuration and deployment resources |
| `llm-d-helm`, `llm-d-kustomize` | llm-d deployment configuration with explicitly selected runtime images |
| `fpm` | Reusable vLLM resource workload, `fpm_env.sh` and `run.sh` |
| `slurm` | Single-node Dynamo service/benchmark bundle for Slurm with Pyxis/Enroot |

The candidate bridge preserves evaluated worker counts, GPU use, backend/data
identity, scheduler limits and supported adapter state. It rejects unknown
adapter mappings and GPU-count mismatches. Canonical result views remain
unchanged by artifact rendering. Compatibility CLI generation flags are in
[the AIC CLI reference](../aic-backward-compatibility/cli.md#deployment-target-selection).

## FPM resource workload

The FPM target supports exactly one aggregated vLLM worker replica, which may
span nodes. It emits only `k8s_deploy.yaml`, `fpm_env.sh` and `run.sh`. A single
node uses a Pod; multiple nodes use LeaderWorkerSet by default or Grove when
`K8sConfig.fpm_orchestrator: grove` is selected. Runtime collection phase
(`--benchmark-mode agg|prefill|decode`) does not change this topology.

The manifest reserves resources and mounts. `fpm_env.sh` exports resolved
identity and node/rank discovery; `run.sh` launches the engine in the foreground.
The collector stages and starts its runtime on every Pod, enforces completion,
collects evidence, terminates engines and cleans up. Launching `run.sh` alone
is not a complete measurement round. Each run reloads the model; reusing a Pod
does not reuse an in-memory engine. `/results` defaults to Pod-local `emptyDir`,
so retrieve results before deleting the workload or configure a persistent mount.

The complete resource, overlay, multinode and completion contracts belong to
[FPM self-service](../perf-model/fpm-self-service/implementation.md), with a
[generator input example](../aic-backward-compatibility/cli.md#fpm-v1-resource-workload-and-run-script).

## Validate generated arguments

Run the repository validator inside the matching backend runtime:

```bash
python python/aisimulate/tools/generator_validator/validator.py \
  --backend vllm --path deployment-study/generated/k8s_deploy.yaml
```

Pass the manifest file for this single-candidate output. The validator's
directory mode expects an AIC results tree with both `agg/top1/k8s_deploy.yaml`
and `disagg/top1/k8s_deploy.yaml`.

The validator supports Dynamo outputs. It loads the installed backend's argument
schema (vLLM `EngineArgs`, SGLang `ServerArgs`, or TensorRT-LLM `TorchLlmArgs`);
it does not validate llm-d Helm values, GPU capacity, endpoint readiness or
performance. Select `--backend sglang` or `trtllm` for those runtimes. For
TensorRT-LLM launch scripts, stage emitted engine YAML in the script's
`/workspace/engine_configs/` location before execution.

## Slurm target


`--deployment-target slurm` generates a self-contained Dynamo deployment bundle
for a Slurm cluster with Pyxis/Enroot. It supports vLLM, SGLang and TRT-LLM in
aggregated or disaggregated P/D mode, including multiple worker replicas on one
NVIDIA GPU node. The sum of all workers' resolved GPU counts must fit
`NodeConfig.num_gpus_per_node`. Multinode workers, dedicated encode pools and
Dynamo Planner are rejected rather than silently changing the topology.

Set worker topology and engine configuration through the structured generator
configuration. For Slurm, `Workers.<role>.extra_cli_args` cannot override parallel
sizes, GPU placement, or launch options, including their backend aliases and
abbreviations. Config-file and engine-overlay arguments (`--config`,
`--disagg-config`, `--extra-engine-args`, `--override-engine-args`, and `--trtllm.*`) are also reserved
so they cannot change the GPU requirements after allocation. Ordinary tuning
arguments remain available. Backend topology environment overrides are rejected in
`SlurmConfig.env`: `DYN_TRTLLM_OVERRIDE_ENGINE_ARGS` for TRT-LLM;
`DYN_SGL_DISAGG_CONFIG` and `DYN_SGL_DISAGG_CONFIG_KEY` for SGLang; and
`VLLM_DP_SIZE`, `VLLM_DP_RANK`, `VLLM_DP_RANK_LOCAL`, `VLLM_DP_MASTER_IP`, and
`VLLM_DP_MASTER_PORT` for vLLM. Workers neutralize inherited values for these
variables without changing other environment settings or structured DP configurations.

Worker roles are also reserved, including vLLM's legacy `--is-prefill-worker`,
`--is-decode-worker`, `--multimodal-encode-worker`, and `--multimodal-decode-worker`
flags and SGLang's `--multimodal-encode-worker` flag. Their negated forms cannot
override generated roles either. `SlurmConfig.env` rejects
`DYN_VLLM_DISAGGREGATION_MODE`, `DYN_VLLM_IS_PREFILL_WORKER`,
`DYN_VLLM_IS_DECODE_WORKER`, `DYN_VLLM_MULTIMODAL_ENCODE_WORKER`,
`DYN_VLLM_MULTIMODAL_DECODE_WORKER`, `DYN_TRTLLM_DISAGGREGATION_MODE`, and
`DYN_SGL_MULTIMODAL_ENCODE_WORKER` for their corresponding backends. Workers remove
these inherited role defaults so the generated agg/prefill/decode role remains
authoritative, including the legacy vLLM role syntax used before Dynamo 1.0.

The bundle contains:

- `deploy.sbatch`: load the model, verify inference, then keep the Dynamo service
  running until cancelled or its time limit.
- `benchmark.sbatch`: start the same service, wait for every worker and the model
  endpoint, run a chat smoke request, run AIPerf, and stop all owned processes.
- `submit.sh`: validate with `sbatch --test-only`, submit once and save `job.id`.
- `environment.sh`: Pyxis image and mounts; cluster paths never become package defaults.
- `deployment.json` and `slurm_runtime.py`: resolved launch commands and a standard-library
  supervisor for discovery services, worker GPU placement, readiness and cleanup.
- `bench_run.sh`: the existing shared AIPerf benchmark template, using a local
  service endpoint instead of Kubernetes DNS.
- TRT-LLM role engine YAML files when applicable.

The container must contain the matching Dynamo/backend release, `etcd` and
`nats-server`; benchmark jobs also require `aiperf` on PATH. The generated bundle
is mounted at `/work`. Mount the complete model cache root read-only when a
checkpoint snapshot contains symlinks to sibling blobs. Every worker receives a
slice of Slurm's actual `CUDA_VISIBLE_DEVICES`, including nonzero IDs or GPU UUIDs.
Each job starts its own etcd/NATS with separate ports and data directories.
Model discovery can precede frontend route registration. Readiness therefore
requires a successful inference request, retrying temporary HTTP 404/503 responses
and transport errors such as disconnects, connection refusal and timeouts within
the startup deadline. Other HTTP errors fail immediately; malformed inference
responses and responses without generated text fail the job.

Automated tests cover emitted artifacts for vLLM, SGLang and TRT-LLM, including
static file snapshots and shell syntax. CPU supervisor tests exercise real local
HTTP requests, socket binding and child-process cleanup with simulated service
launches. This coverage does not qualify Slurm/Pyxis integration, GPU execution,
model capacity or performance estimates. Validate the chosen container, model and
topology on the target cluster with the generated smoke and benchmark job, and
retain its logs, package versions and AIPerf reports.

Example unified generator input (`slurm.yaml`):

```yaml
ServiceConfig:
  model_path: /models/Qwen3-8B
  served_model_name: qwen3
  port: 8000
DynConfig:
  mode: agg
NodeConfig:
  num_gpus_per_node: 8
WorkerConfig:
  agg_workers: 1
Workers:
  agg:
    tensor_parallel_size: 2
    max_batch_size: 16
    max_seq_len: 4096
SlaConfig:
  isl: 512
  osl: 128
SlurmConfig:
  account: YOUR_ACCOUNT
  partition: batch
  job_name: qwen3
  time: "01:00:00"
  cpus_per_task: 32
  memory: 256G
  container_image: /shared/images/dynamo-vllm.sqsh
  container_mounts:
    - /shared/models/Qwen3-8B:/models/Qwen3-8B:ro
  env:
    HF_HUB_OFFLINE: "1"
  startup_timeout: 1800
  benchmark_timeout: 1800
  benchmark_concurrency: [1, 2, 4, 8]
  benchmark_rounds: 20
```

```bash
python -m aisimulate.generator.main render-artifacts \
  --backend vllm --version 0.20.1 --deployment-target slurm \
  --config slurm.yaml --output ./slurm-bundle
# Copy the bundle to a fresh directory on the cluster, then run there:
bash submit.sh benchmark --test-only
bash submit.sh benchmark
# Alternative for a persistent service: bash submit.sh serve
```

The same `SlurmConfig` section is accepted through `--generator-config`, dotted
`--generator-set SlurmConfig.*` overrides, SDK bridges and typed requests. Existing
Dynamo Kubernetes, llm-d, FPM and sflow outputs are unaffected by choosing those
targets. Slurm does not emit Kubernetes resources or require nv-sflow.

The AISimulate package also exposes generation through its compatibility CLI:

```bash
aiconfigurator cli generate \
  --model-path Qwen/Qwen3-8B --system b200_sxm --backend vllm --total-gpus 2 \
  --dynamo-version 1.2.0 --config-template-version 0.20.1 \
  --deployment-target slurm --generator-config slurm.yaml --save-dir ./results
```

Copy the generated model directory containing `submit.sh` to the cluster.
This command uses naive sizing plus explicit overrides; it does not perform an
SLA optimization search. Set `rule: benchmark` in the generator input to preserve
the benchmark batch-size rules. Executable lookup and child processes both use
`SlurmConfig.env.PATH`, so an independently installed AIPerf client can be mounted
read-only and added to that path without changing the backend Python environment.

Account, partition and container image must be supplied. Defaults are a one-hour
limit, 16 CPUs, 64 GiB memory, 30-minute startup and benchmark deadlines, and
concurrencies 1/2/4/8 with 20 requests per concurrency slot. ISL/OSL, tokenizer,
endpoint type and other workload controls come from the existing `BenchConfig`
and `SlaConfig`. Backend template versions must match the container's backend;
they are independent of the Slurm scheduler configuration.

Set the Dynamo version to match the image as well. Dynamo 1.3 rejects the old
`nvext.ignore_eos` request field. For Dynamo 1.3 and newer, both benchmark templates
emit only the supported root-level `ignore_eos` field; older versions retain the
legacy request form.

Backend sizing rules still apply. In particular, the vLLM `benchmark` rule sets
decode `max_num_tokens` to `max_batch_size`. Inspect `deployment.json` and any
generated engine YAML for the resolved worker settings before submission. Check
that the selected container can load the actual checkpoint and complete inference
and benchmarking with those settings; artifact generation alone does not validate
capacity or backend runtime compatibility.

Logs, smoke responses, actual package versions and AIPerf reports are retained
under `results/<job-id>/`. `result.json` records failure causes and cleanup.
Benchmark failures, incomplete request counts, nonzero request errors and premature
worker exits produce a nonzero job exit. Serving
jobs retain resources until cancelled with `scancel` or the Slurm time limit.

`submit.sh` leaves a submission lock even if the scheduler response is lost.
After a disconnected submission, inspect `job.id`, `job.id.tmp`, `squeue` and
`sacct` before doing anything else. Use a fresh bundle directory for a deliberate
new run; do not remove the lock and blindly resubmit. Generation itself never
contacts the cluster or submits a job.
