<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo deployment with AISimulate

Use this workflow to select a configuration, predict it, generate serving
artifacts, and compare with a real benchmark. Run the simulation and generator
steps from an AISimulate source checkout using the
[installation guide](../../../docs/installation.md#use-current-source).
The GPU deployment steps run on a separate serving host.

The unified `recommend` command writes concrete **prediction configurations**.
Its numbered YAML files are not Kubernetes manifests. The generator remains
available through the `aiconfigurator` namespace shipped in the same wheel;
there is no separate standalone AIConfigurator installation in this workflow.

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
[version matrix](../src/aiconfigurator/generator/facts/runtimes/dynamo.yaml).
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

The [example YAML](../../../examples/cli/dynamo-deployment-recommend.yaml)
fixes concurrency and compares two TP shapes under a two-GPU limit. Inspect
`recommendation.json` for failures and selected candidate IDs. If no configuration
is selected and there are zero resource-limited candidates, `recommend` exits `1`
and produces no numbered YAML. Any resource-limited candidate makes the exit
status `3`, including when the result is empty or some selected YAML files remain
available. Inspect the ledger and [resource diagnostics](../../../docs/local-resources.md)
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

from aiconfigurator.generator.api import generate_from_request
from aiconfigurator.generator.request import from_sweeper_candidate

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
own analytical artifacts; see [AFD topology](../../../docs/sweeper/afd-topology.md).

The renderer currently reports that vLLM `0.24.0` uses the closest prior
`cli_args.0.20.1.j2` flag template. Keep that warning with the generated artifacts
and verify the emitted flags against the actual `0.24.0` runtime.

Inspect the returned filenames and generated scripts. For this vLLM example,
`run_0.sh` launches the frontend and worker processes; `bench_run.sh` runs an
AIPerf sweep. Run `bash -n` on both before copying them to the serving host.
If you deliberately use the compatibility CLI's `generate` command instead,
it creates a starting configuration without replay/SLA optimization; it does
not recreate the selected recommendation. See the
[legacy CLI guide](../../../docs/cli/legacy-aic-user-guide.md).

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
[Understand your prediction](../../../docs/cli/understand-your-prediction.md)
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
[generator overview](generator_overview.md) describes the available targets
and overrides. Do not apply a numbered `recommendations/*.yaml` as a manifest.

## 6. FPM V1 Resource Workload Workflow

`--deployment-target fpm` is a separate target for a vLLM single aggregated-worker deployment with exactly one worker replica. It emits a Pod for a single-node topology. A multinode topology emits a `LeaderWorkerSet` by default, or a Grove `PodCliqueSet` when `K8sConfig.fpm_orchestrator: grove` is selected. Router/planner configurations and invalid FPM topologies fail closed. It does not change the output of `dynamo-j2`, `dynamo-python`, or llm-d targets.

FPM V1 emits exactly three artifacts:

```text
artifacts/
├── k8s_deploy.yaml   # keepalive Pod, LeaderWorkerSet, or PodCliqueSet
├── fpm_env.sh        # contract FPM_* exports: topology, rank/leader discovery, benchmark identity
└── run.sh            # sources fpm_env.sh, then launches the complete vLLM command in the foreground
```

The workload requests the generated image, per-node GPU limit, preserved custom resources, volumes, and mounts, but contains no engine arguments or engine/FPM environment variables. Add tokenized launch arguments through `Workers.agg.extra_cli_args: list[str]` and concrete `{name, value}` environment entries through `K8sConfig.extra_env`; the generator places both in `run.sh`. `--benchmark-mode` is required and accepts `agg`, `prefill`, or `decode`; it selects the runtime collection phase without changing the required single aggregated-worker topology. `K8sConfig.fpm_shared_memory_size`, `K8sConfig.fpm_resource_labels`, and `K8sConfig.worker_extra_pod_spec.mainContainer.resources` configure generated shared memory, workload/Pod labels, and non-GPU resource requests or limits. By default the FPM workload's main container renders **Guaranteed QoS**: CPU and memory are set with requests equal to limits, scaled per GPU (14 CPU cores and 64Gi of memory per GPU). Without such quota the Pod runs BestEffort for CPU, and launch-bound FPM measurements disperse across restarts with neighbour interference; requests==limits removes that spread. The defaults are skipped for any key (`cpu` or `memory`) already present in `worker_extra_pod_spec.mainContainer.resources` requests or limits, so explicit values always win. Grove output does not select a scheduler or queue by default; clusters that use KAI Scheduler can set `worker_extra_pod_spec.schedulerName: kai-scheduler` and `fpm_resource_labels.kai.scheduler/queue` explicitly. `valueFrom`, `envFrom`, and Secret-derived environment values are not supported in V1.

For a single-node Pod, an agent can create the resource once and execute the generated scripts in it:

```bash
kubectl apply -f artifacts/k8s_deploy.yaml
kubectl wait --for=condition=Ready pod/<pod> --timeout=10m
kubectl exec <pod> -- mkdir -p /tmp/fpm-bench
kubectl cp artifacts/fpm_env.sh <pod>:/tmp/fpm-bench/fpm_env.sh
kubectl cp artifacts/run.sh <pod>:/tmp/fpm-bench/run.sh
kubectl exec <pod> -- bash /tmp/fpm-bench/run.sh
```

This standalone flow demonstrates deployment only: `run.sh` just launches the engine, so completion gating and engine termination require the collector's staged runtime (`fpm_exec.sh`) and this sequence alone is not a complete measurement round.

For a multinode workload, the collector stages the complete runtime bundle on every Pod and starts the same scripts concurrently across them. `fpm_env.sh` derives rank and leader address from the selected controller's LWS or Grove environment (failing closed with exit 2), and `run.sh` appends the required model- or data-parallel coordination arguments. A multinode `--dump-config-to` path must contain `{node_rank}`; this substitution applies only to that option. Waiting for local result files and enforcing the schema-v2 completion contract belong to the collector's staged in-pod runtime, which consumes the `FPM_*` facts exported by `fpm_env.sh`. Strict plan/workload/repeat validation, result collection/aggregation/evidence, exit coordination, and cleanup remain collector responsibilities.

Each collection still starts a new engine and reloads the model. The collector's staged runtime stops result-producing engines after their local completion gate, refuses to overwrite existing benchmark outputs, and coordinates headless followers and final cleanup. By default `/results` is backed by Pod-local `emptyDir`, and those results disappear when the Pod is deleted; a matching user-provided volume and mount are preserved. Persistent engines and reuse of a GPU-resident model are outside the V1 scope. A target cluster needs the controller selected by `K8sConfig.fpm_orchestrator`; any scheduler and queue required by that cluster must be supplied through the generic Pod-spec and label overlays. Multinode GB200 configurations that enable MNNVL also include the generated `ComputeDomain` in `k8s_deploy.yaml`; single-node output never emits one. See the [Legacy AIC CLI User Guide](../../../docs/cli/legacy-aic-user-guide.md#fpm-v1-resource-workload-and-run-script) for the full input example.

The current vLLM template matrix tops out at `0.20.1`; reference `0.24.0`-only flags may be passed through, but their runtime compatibility is not yet validated by the generator.
