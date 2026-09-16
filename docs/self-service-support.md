<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Onboard a model for FPM simulation on a target hardware platform

`aisimulate onboard` guides onboarding a new model for FPM simulation on your designated hardware platform. It records the model, runtime, target GPU system and allocation, plans one pure tensor-parallel worker, and produces ordinary `predict` and `recommend` configurations that read your collected FPM data. It builds on AISimulate's existing per-worker FPM support and packaged collector. No other draft PR needs to be merged first.

Planning works before the model has an AISimulate model class or measured FPM timings. A valid request records your choices; model integration, runtime compatibility, and data readiness remain **unchecked**, and accuracy is **not assessed**. This setup does not implement an Inkling model class, run preflight checks, provision GPU resources, or establish measured accuracy. Those steps belong to the broader integration project.

## Create the request

Use an environment installed from this checkout; see [development setup](../DEVELOPMENT.md). Guided setup requires a terminal and starts only when explicitly requested:

```bash
aisimulate onboard init --interactive --output support-request.yaml
```

Enter the actual model identifier or checkpoint path, pinned model revision, dense/MoE kind, pinned vLLM version, GPU system, allocation, and interconnect. Then choose a TP size and pilot workload. Supplied options skip their prompts. Enter accepts displayed defaults; invalid values can be corrected; Ctrl-C or end-of-input cancels without saving. Existing files require `--overwrite`.

Both guided and scripted setup use onboarding. `--profile onboarding` is an optional spelling of the same behavior. For automation, supply the identity flags directly:

```bash
aisimulate onboard init \
  --model /models/your-pinned-checkpoint \
  --model-revision YOUR_IMMUTABLE_REVISION \
  --model-kind dense \
  --framework-version YOUR_PINNED_VLLM_VERSION \
  --gpu h200_sxm --gpu-count 4 --interconnect nvswitch \
  --tensor-parallel 2 \
  --output support-request.yaml
```

Replace the model and runtime placeholders with the actual inputs. The example does not identify an Inkling checkpoint or claim that four H200s can run your model. Framework support currently selects vLLM. Optional tokenizer, chat-template, and AISimulate revisions are recorded only when supplied.

The default pilot uses 1,024 input tokens, 128 output tokens, concurrency 1, four requests, a 16,384-token context limit, TTFT target 1,000 ms, and TPOT target 100 ms. Scripted setup defaults to TP1 unless `--tensor-parallel` is supplied. These are planning defaults, not measured model capacity or latency. Change them with the corresponding flags shown by `aisimulate onboard init --help`.

One node is the default; GPUs per node then equals `--gpu-count`. For multiple nodes, supply both `--node-count` and `--gpus-per-node`; their product must equal the total allocation. Each pure-TP worker must fit on one node. Advanced flags include `--request-count`, `--max-candidates`, `--objective`, `--seed`, and `--sm`.

## Plan, preview, and explicitly execute

```bash
aisimulate onboard plan \
  --config support-request.yaml --output-dir ./aisimulate-support

aisimulate onboard collect-fpm \
  --config ./aisimulate-support/request.yaml \
  --output-dir ./aisimulate-support
```

The first command saves the request, `support-plan.json`, `commands.json`, `predict/pilot.yaml`, `recommend/pilot.yaml`, and a local `systems/` directory. The second prints the collector command without launching it. Generated command vectors and printed next commands use absolute output paths and preserve spaces or shell punctuation. Use a separate output directory for each request. On an existing plan, `--overwrite` can repair missing generated files for the identical request; it rejects changed inputs and preserves existing collected data.

The search uses one selected TP size. By default it evaluates a single worker, so recommendation is not a broad deployment search. `--max-candidates 2` additionally considers the largest count of identical workers that fits the allocation, when that differs from one worker. Each choice gets an independent recommendation config pinned to that replica count with a one-trial budget. The single worker keeps `recommend/pilot.yaml`; the second choice uses `recommend/replicas-N.yaml`, where `N` is its replica count. The plan reports the actual candidate count and lists both config and result paths. Dense collection uses the `tp` preset; MoE uses `pure_tp`; the selected TP size remains exact.

Before execution, prepare the real checkpoint and the pinned runtime using the existing [FPM collection guide](../python/aisimulate/docs/fpm/end-to-end-workflow.md). The packaged collector invokes a Generator-resolved Dynamo/vLLM deployment and needs the corresponding GPU resources, deployment configuration, permissions, and model access. Invoking its command locally does not create that environment. `commands.json` publishes the guarded `aisimulate onboard collect-fpm --execute` command for collection, alongside a read-only collector planning command.

After those prerequisites are ready, explicitly launch collection from that environment:

```bash
aisimulate onboard collect-fpm \
  --config ./aisimulate-support/request.yaml \
  --output-dir ./aisimulate-support --execute
```

Execution requires a matching saved plan. The request records model and runtime revisions; this setup does not download a pinned checkpoint or verify the installed runtime against them. Keep the actual checkpoint and runtime consistent with the request before collecting or predicting.

Set deployment options directly on `onboard collect-fpm`: `--dynamo-version VERSION`, `--image IMAGE`, `--namespace NAME`, `--model-cache NAME[:MOUNT[:SUBPATH]]`, `--transport nvlink|ib|efa`, and `--image-pull-secret NAME`. The mount, when supplied, is an absolute container path. Prefer an immutable image digest. Supply the same options when previewing, executing, and resuming; deployment settings are part of the collector's frozen-plan identity, so changed settings require a new output directory. Arbitrary collector arguments and engine overrides are not accepted by this command.

For a diagnostic run, add `--execute --smoke`; `--limit N` also requires `--smoke`. Diagnostic smoke and limited runs do not publish formal FPM data. Existing campaign data, raw artifacts, or checkpoints require explicit `--resume` and a readable matching collector checkpoint; otherwise choose a new output directory. A custom `--checkpoint-dir`, if needed, must remain inside the plan's `fpm-checkpoint/` directory. Selecting an empty checkpoint directory does not allow reuse of existing campaign artifacts. Smoke and formal campaigns have separate checkpoints and artifact directories, so an existing smoke run does not prevent the first formal run, or vice versa. The collector verifies the resumed checkpoint's frozen-plan identity.

Planning and collection reject concurrent onboarding operations. The persistent `.support.lock` file uses an OS advisory lock; ownership is released when the process exits, including after an abrupt termination. Leave the file in place. Request validation, saved onboarding-plan checks, and collector input resolution exit 2. Failures after collector execution starts, including a frozen checkpoint identity mismatch, exit 1 with a concise message. Interruption exits 130.

The collector narrows initial prefill sampling with the pilot's input-token and concurrency bounds. Decode uses the collector's existing profile; a four-request synthetic pilot does not imply four timing samples or a short decode campaign. Inspect the generated command and collector plan before committing GPU time. Successful formal collection publishes the FPM Parquet file and metadata pair into the plan's local systems data directory; diagnostic success alone does not provide that pair.

## Run the generated ordinary configurations

After model integration and formal data collection are complete:

```bash
aisimulate predict \
  --config ./aisimulate-support/predict/pilot.yaml \
  --output-dir ./aisimulate-support/predict-results/pilot

aisimulate recommend \
  --config ./aisimulate-support/recommend/pilot.yaml \
  --output-dir ./aisimulate-support/recommend-results/pilot
```

With two candidates, run both replica counts. This loop derives the candidate names from the validated saved request and constructs fixed `aisimulate recommend` commands:

```bash
python3 - <<'PY'
import shlex
import subprocess
from pathlib import Path

from aisimulate.support.plan import check_plan
from aisimulate.support.schema import SupportRequest

root = Path("./aisimulate-support").resolve()
request = SupportRequest.from_yaml(root / "request.yaml")
check_plan(request, root)
max_replicas = request.identity.node_count * (request.identity.gpus_per_node // request.search.tensor_parallel)
replicas = list(dict.fromkeys((1, max_replicas)))[:request.search.max_candidates]
statuses = []
for count in replicas:
    name = "pilot" if count == 1 else f"replicas-{count}"
    command = [
        "aisimulate", "recommend",
        "--config", str(root / f"recommend/{name}.yaml"),
        "--output-dir", str(root / f"recommend-results/{name}"),
        "--format", "json",
    ]
    result = subprocess.run(command, check=False)
    statuses.append(result.returncode)
    print(f"exit {result.returncode}: {shlex.join(command)}", flush=True)
raise SystemExit(1 if any(statuses) else 0)
PY
```

The loop reports each command's exit status and attempts all commands, including when the pilot finds no feasible candidate. It exits 1 after all attempts if any command returned a nonzero status, otherwise 0. Successful outputs remain usable even when the loop exits 1; inspect each result before comparing candidates.

Run either the individual recommendation command or the loop against fresh result directories. The loop also handles a one-candidate plan and does not execute entries from `commands.json`. Results remain separate under `recommend-results/pilot` and, when present, `recommend-results/replicas-N`; compare their objective and latency results for the same workload. This plan does not produce a combined ranking or search additional TP sizes or scheduler settings.

The generated configurations select `engine.workers.aggregated.timing.forward_model: fpm` and set `engine.systems_path` to the plan's absolute local systems directory. The same root supplies hardware and collected FPM data. Recommendation preserves it in exported prediction configs. Moving the plan to another machine requires updating absolute paths or regenerating it there.

Ordinary `predict` and `recommend` commands retain their existing behavior and defaults. Their generated configs can be loaded and edited through the public configuration schema. Missing model integration or data may still prevent execution; a successful simulation is not an accuracy result. Compare its output with an independent run of the same model, runtime, topology, and workload to assess accuracy.

## Hand off a new model architecture

Give an engineer the saved request, the pinned model configuration, and the plan. Follow [How to Add a New Model](../python/aisimulate/docs/add_a_new_model.md) to integrate the architecture and model class, including its operation graph, memory accounting, and KV-cache behavior. FPM supplies measured whole-forward timings; it does not replace the model structure needed by simulation.

The model guide also describes legacy per-operation silicon profiling. Collecting those per-operation timings is not mandatory for this FPM workflow. Integrate the model structure first, collect the matching whole-forward FPM data through the existing collector, and then verify prediction and recommendation on the target deployment.
