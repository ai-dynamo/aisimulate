# Migrate from AIConfigurator

Install AISimulate first, then migrate each CLI workflow when its replacement fits your needs.

The `aisimulate` package includes all six `aiconfigurator cli` commands. You can keep using those
commands while adopting `aisimulate predict` for serving prediction and `aisimulate recommend`
for configuration search. The new commands use YAML inputs and different search semantics.

> [!WARNING]
> **Experimental.** The AISimulate recommendation schema and search behavior may change without a
> standard deprecation period. Validate selected configurations on your target hardware.

**Contents**

1. [Install AISimulate](#1-install-aisimulate)
2. [AIC to AISimulate command mapping](#2-aic-to-aisimulate-command-mapping)
3. [General migration examples](#3-general-migration-examples)
4. [Advanced migration examples](#4-advanced-migration-examples)
5. [Remaining feature and performance gaps](#5-remaining-feature-and-performance-gaps)
6. [Reference](#6-reference)

## 1. Install AISimulate

In the Python environment where you use AIC, replace the standalone distributions:

```bash
python3 -m pip uninstall -y aiconfigurator aiconfigurator-core
python3 -m pip install --upgrade aisimulate
aiconfigurator cli --help
aisimulate --help
```

Both commands now come from AISimulate. Existing AIC flags and experiment YAML continue to use
`aiconfigurator`; there is no automatic converter to the new CLI input format. See the
[Legacy AIC CLI User Guide](legacy-aic-user-guide.md) for the six-command reference.

## 2. AIC to AISimulate command mapping

The rows follow the legacy guide's command order. “Keep AIC” means use the compatibility command
installed above.

| AIC command | Path to use | Key difference |
|---|---|---|
| `generate` | Keep AIC `generate`. | [Deployment files](#deployment-artifacts) still require AIC or the generator SDK. |
| `estimate` | `aisimulate predict` for serving prediction. [Example](#migrate-one-concrete-deployment). | Keep AIC for batch/static estimates, diagnostics, and [power reports](#power-and-energy-analysis). |
| `support` | Keep AIC `support`. | No unified support-query command. |
| `recommend` | [Keep AIC for minimum-GPU sizing](#keep-minimum-gpu-sizing-on-the-compatibility-cli). | AISimulate `recommend` offers [search under a specified load](#search-under-a-request-rate), with a different objective. |
| `default` | `aisimulate recommend`. [Example](#search-with-a-fixed-gpu-budget). | Supply traffic, a GPU ceiling, and a search objective. |
| `exp` | Keep AIC for existing experiment files. | Translate individual experiments to `predict` or `recommend`; no equivalent file orchestration. |

## 3. General migration examples

These examples use the offline `engine` stack and pin vLLM performance-data version 0.24.0.
The output excerpts were captured with AISimulate 0.12.0 built from this repository on
2026-09-14. They are simulation results and may change with the software or performance data.
Save the YAML files as named and run commands from that directory, using fresh output directories.
For Dynamo Router/Planner integration, see
[execution stacks](user-guide.md#choose-an-execution-stack).

### Migrate one concrete deployment

**Before — AIC estimates a batch at a fixed parallel configuration:**

```bash
aiconfigurator cli estimate \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system h200_sxm --backend vllm --backend-version 0.24.0 \
  --estimate-mode agg --tp-size 2 --batch-size 64 \
  --isl 1024 --osl 128
```

**After — predict serving behavior on two H200 GPUs.** Save as `prediction.yaml`:

```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 64}
  stop: {requests: 100}
engine:
  mode: aggregated
  model: meta-llama/Meta-Llama-3.1-8B
  hardware: h200_sxm
  backend: vllm
  backend_version: "0.24.0"
  workers:
    aggregated:
      parallelism: {tensor: 2, replicas: 1}
```

```bash
aisimulate predict --config prediction.yaml --output-dir ./prediction-output
```

**Result to inspect:** `prediction-output/prediction.json` contains completed-request counts, TTFT,
inter-token latency, and output throughput. Recorded report excerpt (rounded):

```json
{
  "completed_requests": 100,
  "mean_ttft_ms": 365.35,
  "mean_itl_ms": 8.54,
  "output_throughput_tok_s": 5114.43
}
```

**What changed:** AIC's `--batch-size 64` fixes an estimator batch. AISimulate's `concurrency: 64`
keeps up to 64 requests in flight while the scheduler forms batches. Here TP=2 and one replica
use two GPUs, but the latency and throughput results describe serving traffic. Use AIC when you
need its original batch-level result.

### Search with a fixed GPU budget

**Before — AIC searches within an eight-GPU budget:**

```bash
aiconfigurator cli default \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system h200_sxm --backend vllm --backend-version 0.24.0 --serving-mode agg \
  --total-gpus 8 --isl 1024 --osl 128 \
  --ttft 800 --tpot 30 --strict-sla
```

**After — search for throughput with 32 requests in flight.** Save as `budget-search.yaml`:

```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 32}
  stop: {requests: 320}
engine:
  mode: aggregated
  model: meta-llama/Meta-Llama-3.1-8B
  hardware: h200_sxm
  backend: vllm
  backend_version: "0.24.0"
  workers:
    aggregated:
      parallelism: {preset: default}
evaluation:
  sla: {ttft_ms: 800, itl_ms: 30}
optimization:
  target: throughput
  strict_sla: true
  constraints: {max_candidate_gpus: 8}
optimizer: {algorithm: random, max_trials: 8, parallelism: 1, seed: 11}
```

```bash
aisimulate recommend --config budget-search.yaml --output-dir ./budget-search
```

**Result to inspect:** the terminal lists selected configurations. Recorded output excerpt (first two results):

```text
AISimulate recommendations
1: score=8518 used_gpus=8 config=budget-search/recommendations/0001.yaml
2: score=6753 used_gpus=6 config=budget-search/recommendations/0002.yaml
```

The score for `target: throughput` is output tokens/s; `used_gpus` is the candidate's GPU count.
`budget-search/recommendation.json` records candidate metrics and rejection reasons. Selected
prediction inputs are saved under `budget-search/recommendations/`, in rank order. Evaluate the
first result with:

```bash
aisimulate predict \
  --config ./budget-search/recommendations/0001.yaml \
  --output-dir ./budget-selected
```

Check `budget-selected/prediction.json` for latency and throughput under the saved workload.
If the search finds no feasible candidate, it writes `recommendation.json`, exits with status 1,
and produces no selected YAML; inspect that report before running the prediction command.

**What changed:** eight GPUs is a ceiling, so the winner may use fewer. This example evaluates
eight random trials to keep the walkthrough bounded; increase the trial budget for your search.
It ranks configurations under the supplied traffic and does not reproduce AIC's full capacity
sweep. With `strict_sla: true`, candidates must pass the configured aggregate-mean latency bounds.

### Search under a request rate

**Before — AIC sizes the minimum GPUs for four requests/s:**

```bash
aiconfigurator cli recommend \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system h200_sxm --backend vllm --backend-version 0.24.0 \
  --target-request-rate 4 --isl 1024 --osl 128 \
  --ttft 800 --tpot 30 --strict-sla
```

In the recorded AIC run, the top aggregated sizing result used **one GPU**: TP=1 and one replica.

**After — if your goal is to rank configurations at that load,** reuse `budget-search.yaml` and
override its traffic and objective:

```bash
aisimulate recommend --config budget-search.yaml \
  --set 'traffic.load={type: constant_rate, requests_per_second: 4}' \
  --set optimization.target=goodput_per_gpu \
  --output-dir ./rate-search
```

**Result to inspect:** `rate-search/recommendation.json` and its selected prediction YAML.
Recorded terminal excerpt (first two results):

```text
AISimulate recommendations
1: score=255.4 used_gpus=2 config=rate-search/recommendations/0001.yaml
2: score=254.7 used_gpus=2 config=rate-search/recommendations/0002.yaml
```

The top result here uses **two GPUs**. Its score is SLA-qualified output tokens/s/GPU.
The input/output lengths, eight-GPU ceiling, latency bounds, and trial budget come from
`budget-search.yaml`. To search at a fixed concurrency
instead, use `--set 'traffic.load={type: concurrency, concurrency: 32}'`.

**What changed:** offered request rate and in-flight concurrency describe traffic; they do not
ask AISimulate for the smallest fleet that can serve it. Ranking by `goodput_per_gpu` can select
more GPUs than the smallest SLA-compliant configuration. If minimum GPU or replica count is your
required result, keep the AIC command above.

## 4. Advanced migration examples

- [Regular prefill/decode disaggregation](#predict-regular-prefilldecode-disaggregation)
- [Parallelism and throughput tradeoffs](#search-parallelism-and-throughput-tradeoffs)
- [Traces and multi-turn sessions](#replay-traces-and-multi-turn-sessions)
- [Cache capacity and host offload](#model-cache-capacity-and-host-offload)
- [Dynamo routing and planning](#include-dynamo-routing-and-planning)
- [Op-level and FPM timing](#select-op-level-or-whole-forward-fpm-timing)
- [Analytical EPD](#predict-and-search-analytical-epd)
- [Heterogeneous P/D hardware](#migrate-heterogeneous-pd-hardware-with-sweeper)
- [AFD](#afd-translation)

The first examples reuse `prediction.yaml` and `budget-search.yaml` from the general examples;
run them from the directory containing those files. Commands with checked-in configuration paths
run from the repository root. Feature guides describe model, data, and topology restrictions.

### Predict regular prefill/decode disaggregation

**Before — AIC estimates separate prefill and decode workers:**

```bash
aiconfigurator cli estimate \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system h200_sxm --backend vllm --backend-version 0.24.0 \
  --estimate-mode disagg --isl 1024 --osl 128 \
  --prefill-tp-size 1 --prefill-batch-size 1 --prefill-num-workers 1 \
  --decode-tp-size 1 --decode-batch-size 64 --decode-num-workers 1
```

**After — predict serving traffic with one H200 GPU per role.** Reuse the earlier `prediction.yaml`,
set `engine.mode: disaggregated`, and replace its aggregated worker with prefill and decode workers:

```bash
aisimulate predict --config prediction.yaml \
  --set engine.mode=disaggregated \
  --set 'engine.workers={prefill: {parallelism: {tensor: 1, replicas: 1}}, decode: {parallelism: {tensor: 1, replicas: 1}}}' \
  --output-dir ./pd-prediction
```

**Result to inspect:** `pd-prediction/prediction.json` contains the serving latency and throughput
for the two-GPU deployment.

**What changed:** AIC fixes the prefill and decode batch sizes. AISimulate reuses the earlier
64-request concurrency and lets each role's scheduler form batches. Both roles share the configured
model, hardware, backend, and backend version; their parallelism, scheduler, and KV-cache settings
can differ. `engine.kv_transfer` controls transfer bandwidth and which prompt KV bytes are charged.
For search, use the same two roles with recommendation domains; see the
[engine fields](user-guide.md#engine-fields). Different P/D hardware requires the
[Sweeper SDK path](#migrate-heterogeneous-pd-hardware-with-sweeper) below.

### Search parallelism and throughput tradeoffs

**Before — AIC searches parallel configurations within an eight-GPU budget:**

```bash
aiconfigurator cli default \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system h200_sxm --backend vllm --backend-version 0.24.0 \
  --serving-mode agg --total-gpus 8 --isl 1024 --osl 128 \
  --ttft 800 --tpot 30 --strict-sla
```

**After — search explicit TP choices and return throughput tradeoffs.** Reuse `budget-search.yaml`:

```bash
aisimulate recommend --config budget-search.yaml \
  --set engine.workers.aggregated.parallelism.preset=false \
  --set 'engine.workers.aggregated.parallelism.tensor={choices: [1, 2, 4]}' \
  --set optimization.target=pareto \
  --output-dir ./pareto-search
```

**Result to inspect:** `pareto-search/recommendation.json` records the nondominated
`throughput_per_gpu` versus `throughput_per_user` frontier and its selected prediction YAML.

**What changed:** the AIC command uses its capacity sweep and default parallelism domain. The
AISimulate command uses the earlier 32-request concurrency, restricts TP to 1/2/4, and returns a
Pareto front instead of a scalar ranking. AISimulate also exposes attention DP, MoE TP/EP, replicas,
and supported backend choices; one run uses one hardware SKU. See
[parallelism presets](user-guide.md#parallelism-preset-behavior) and
[optimization goals](user-guide.md#optimization-goal).

### Replay traces and multi-turn sessions

**AIC counterpart:** the AIC CLI has no corresponding trace/session replay command. This is an
additional AISimulate serving-simulation capability.

**AISimulate example — inspect individual requests in a prediction:**

```bash
aisimulate predict --config prediction.yaml --capture-per-request --output-dir ./request-details
```

**Result to inspect:** `request-details/requests.jsonl` contains individual request records alongside
the aggregate `prediction.json`.

**What changes for an AIC user:** the command above keeps the earlier synthetic workload. Replace
its traffic using the [trace example and format table](user-guide.md#trace-source), or the
[synthetic-session example](user-guide.md#synthetic-session-source), to replay recorded requests or
multi-turn sessions. Trace inputs include Mooncake, Dynamo, and agentic formats; format-specific
topology restrictions apply. Sessions preserve turn order and can model shared-prefix groups.
Their shared-prefix ratio is a workload model, not a direct translation of AIC's exact `--prefix`
token count. Analytical EPD does not support traces, sessions, or per-request capture; AFD requires
fixed synthetic requests.

### Model cache capacity and host offload

**Before — AIC estimates a fixed cached prefix and GPU-memory allocation:**

```bash
aiconfigurator cli estimate \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system h200_sxm --backend vllm --backend-version 0.24.0 \
  --estimate-mode agg --tp-size 2 --batch-size 64 --isl 1024 --osl 128 \
  --prefix 512 --free-gpu-memory-fraction 0.8
```

**After — configure serving-cache capacity and add host offload.** Reuse `prediction.yaml`:

```bash
aisimulate predict --config prediction.yaml \
  --set engine.workers.aggregated.kv_cache.capacity.memory_fraction=0.8 \
  --set 'engine.workers.aggregated.kv_cache.host_offload={num_host_blocks: 4096}' \
  --output-dir ./cache-prediction
```

**Result to inspect:** `cache-prediction/prediction.json` reports the serving result with the
configured GPU/host KV capacity. Whether offload is exercised depends on cache pressure and reuse.

**What changed:** AIC's `--prefix 512` assumes 512 tokens are already cached. AISimulate models prefix
reuse from the workload and cache state; the command above does not recreate that fixed hit count.
Host offload is an additional serving feature with no matching AIC CLI flag. This vLLM example uses
prefix caching and attention DP=1, as required by the
[host-offload contract](user-guide.md#native-vllm-host-offload-prediction). Host capacity and bandwidth
stay fixed during recommendation, which also requires fixed parallelism. Other `kv_cache` controls
include block size, fixed GPU capacity, and CUDA-graph memory reservation.

### Include Dynamo routing and planning

**AIC counterpart:** the AIC CLI can size configurations and generate deployment files; it has no
corresponding Router/Planner replay command.

**AISimulate example — run with the Dynamo adapter.** Install `ai-dynamo` and save the
[complete Dynamo prediction example](user-guide.md#complete-dynamo-prediction-example) as
`dynamo-prediction.yaml`, then run:

```bash
aisimulate predict --stack dynamo --config dynamo-prediction.yaml --output-dir ./dynamo-prediction
```

**Result to inspect:** `dynamo-prediction/prediction.json` contains the serving metrics for the
configured engine and Router. That linked example uses round-robin routing with Planner disabled.

**What changes for an AIC user:** `--stack dynamo` accepts Router policy and Planner scaling
configuration in addition to the core serving inputs. For a search that includes Planner settings,
use the [Dynamo search example](user-guide.md#dynamo-scalar-recommendation-example). Planner runtime
scaling limits and the search's candidate GPU budget are separate controls.

### Select op-level or whole-forward FPM timing

**Before — AIC estimates a batch using collected whole-forward profiles:**

```bash
AIC_ALLOW_UNLISTED_VERSIONS=1 aiconfigurator cli estimate \
  --model-path MiniMaxAI/MiniMax-M2.7 \
  --system h200_sxm --backend vllm --backend-version 0.25.1 \
  --forward-model fpm --estimate-mode agg \
  --tp-size 4 --moe-tp-size 4 --moe-ep-size 1 \
  --fmha-quant-mode bfloat16 --batch-size 4 --isl 1024 --osl 32
```

**After — use FPM timing in a serving prediction for the same model, hardware, and parallelism:**

```bash
AIC_ALLOW_UNLISTED_VERSIONS=1 aisimulate predict \
  --config tests/e2e/configs/unified_cli/predict/fpm/01-minimax-m27-h200-tp4-fpm.yaml \
  --set engine.workers.aggregated.timing.forward_model=fpm \
  --output-dir ./minimax-fpm
```

**Result to inspect:** `minimax-fpm/prediction.json` contains the serving prediction using
whole-forward profiles.

**What changed:** AIC's `--forward-model` becomes a per-role
`engine.workers.<role>.timing.forward_model` setting. `op_level` remains the default. The AISimulate
fixture uses four in-flight requests rather than fixing every scheduler batch to four. The AIC
command pins FMHA precision to match the collected profile. Both commands pin vLLM 0.25.1, which
requires the shown unlisted-version override. FPM requires
`timing.type: default` and matching profile coverage; a missing profile fails explicitly. See the
[FPM guide](../../python/aisimulate/docs/fpm/README.md).

### Predict and search analytical EPD

**Before — AIC estimates a dedicated image-encoder pool plus an aggregated language worker:**

```bash
aiconfigurator cli estimate \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --system h200_sxm --backend sglang --backend-version 0.5.14 \
  --estimate-mode agg --tp-size 1 --batch-size 8 --isl 128 --osl 32 \
  --image-height 448 --image-width 448 --num-images 1 \
  --enable-epd --encoder-tp 1 --encoder-batch-size 2 --encoder-num-workers 1
```

**After — predict the same E+agg layout or search encoder and language-worker configurations:**

```bash
aisimulate predict --config examples/cli/epd-predict-aggregated.yaml --output-dir ./epd-prediction
aisimulate recommend --config examples/cli/epd-recommend.yaml --output-dir ./epd-search
```

**Result to inspect:** `epd-prediction/prediction.json` identifies the `analytical_epd_overlay`;
`epd-search/recommendations/` contains selected encoder/language-worker prediction configurations.

**What changed:** encoder flags move to `engine.workers.encoder`, and image inputs move to
`traffic.source.images`. The prediction uses eight in-flight requests; the search example also
explores E+P+D. These paths require fixed synthetic images and concurrency. They model encoder
capacity and latency without event-level encoder queueing or embedding transfer. See
[EPD inputs and limits](../sweeper/epd.md#unified-cli).

### Migrate heterogeneous P/D hardware with Sweeper

**Before — AIC searches H200 prefill with GB200 decode:**

```bash
aiconfigurator cli default \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system h200_sxm --decode-system gb200 --backend vllm --backend-version 0.24.0 \
  --serving-mode disagg --total-gpus 8 --isl 1024 --osl 128 \
  --ttft 800 --tpot 30 --strict-sla
```

**After — use the Sweeper SDK for a bounded search over those two hardware roles.** Save as
`heterogeneous-pd.py`:

```python
from pathlib import Path

from aisimulate import EngineReplayRunnerFactory
from aisimulate.sweeper import SmartSearchConfig, Sweeper

if __name__ == "__main__":
    config = SmartSearchConfig.model_validate({
        "search_space": {
            "model_name": "meta-llama/Meta-Llama-3.1-8B",
            "hardware_sku": "h200_sxm",
            "prefill_hardware_sku": "h200_sxm",
            "decode_hardware_sku": "gb200",
            "backend": ["vllm"], "backend_version": "0.24.0",
            "deployment_mode": ["disagg"], "gpu_budget": 8,
        },
        "workload": {"isl": 1024, "osl": 128, "request_rate": 4, "num_request_ratio": 10},
        "goal": {"target": "throughput", "strict_sla": True,
                 "sla": {"ttft_ms": 800, "itl_ms": 30}},
        "sweep": {"algorithm": "random", "max_trials": 4, "parallel_evals": 1},
    })
    result = Sweeper(runner_factory=EngineReplayRunnerFactory()).run(config)
    Path("heterogeneous-pd-results.json").write_text(result.to_json())
```

```bash
python3 heterogeneous-pd.py
```

**Result to inspect:** `heterogeneous-pd-results.json` contains candidate configurations and metrics
for the two hardware roles within the shared eight-GPU budget.

**What changed:** AIC `--system` / `--decode-system` (or experiment fields `prefill_system_name` /
`decode_system_name`) become `search_space.prefill_hardware_sku` / `decode_hardware_sku`.
`hardware_sku` supplies the fallback for an omitted role. This SDK example ranks throughput under
four requests/s and four random suggestions, so it does not reproduce AIC's capacity sweep.
Both roles must share the model and backend/version. The unified CLI still uses a shared hardware
SKU; see the [SDK hardware/backend restrictions](../sweeper/configuration.md#backend-fields),
including the Dynamo Router AIC load-model restriction.

### AFD translation

**Before — AIC runs a bounded search for decode-side AFD with a regular prefill companion:**

```bash
aiconfigurator cli default \
  --model-path Qwen/Qwen3-32B --system h200_sxm --backend trtllm \
  --serving-mode afd --total-gpus 32 --isl 1024 --osl 128 \
  --afd-max-a-batch-size 128 --afd-max-candidates 32 --afd-candidate-overflow truncate \
  --ttft 800 --tpot 30 --strict-sla
```

**After — search the same topology intent with the analytical AFD engine.** Save as
`afd-recommendation.yaml`:

<!-- afd-migration-contract-start -->
```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: constant_rate
    requests_per_second: 4
  stop:
    requests_per_load_unit: 10

engine:
  mode: afd
  model: Qwen/Qwen3-32B
  hardware: h200_sxm
  backend: trtllm
  afd:
    phase: decode
    combined_with_pd: true
    a_batch_size: 128

evaluation:
  sla:
    ttft_ms: 800
    itl_ms: 30

optimization:
  target: throughput_per_gpu
  strict_sla: true
  constraints:
    max_candidate_gpus: 32
```
<!-- afd-migration-contract-end -->

```bash
aisimulate recommend --config afd-recommendation.yaml --output-dir ./afd-search
aisimulate predict --config ./afd-search/recommendations/0001.yaml --output-dir ./afd-selected
```

**Result to inspect:** `afd-search/recommendations/0001.yaml` describes a concrete topology;
`afd-selected/afd-replay-spec.json` and `afd-selected/afd-qualification.json` record the analytical
inputs and GPU accounting.

**What changed:** `engine.afd` makes the decode-side AFD and prefill-companion topology explicit.
The GPU budget includes attention, FFN, and companion pools. The search uses an explicit request
rate and throughput/GPU objective. The AIC example caps attention batch size at 128 and explicitly
truncates enumeration to 32 topologies; the AISimulate YAML fixes attention batch size at 128.
These searches do not cover identical operating points. Native
AFD deployment generation remains unavailable. Use fixed-length synthetic traffic with an
absolute load; see [AFD topology and limits](../sweeper/afd-topology.md).

## 5. Remaining feature and performance gaps

These gaps concern the unified `aisimulate predict` and `aisimulate recommend` commands. The
AISimulate package still includes the compatibility AIC CLI and SDKs, so a feature can be available
in the package without a unified-CLI replacement.

### Recommendation runtime

**Bayesian recommendation can take much longer than AIC sizing.** Recorded September 11–12, 2026
baseline measurements on an Apple M3 Pro used Python 3.12.11, native release builds, and shared
B200/vLLM 0.24.0 performance data. Medians from three fresh-process runs were:

| Case | AIC `recommend` | AISimulate Bayesian `recommend` | AISimulate / AIC runtime |
|---|---:|---:|---:|
| Llama 3.1 8B; ISL 1,024 / OSL 128; concurrency 10; 32 AISimulate suggestions | 8.34 s | 48.81 s | 5.9× |
| Llama 3.1 70B; ISL 8,192 / OSL 512; concurrency 32; 64 AISimulate suggestions | 7.24 s | 148.32 s | 20.5× |

These are CLI wall-clock times, including startup, search, replay, and output writes; they are not
GPU inference latency. AIC performed minimum-GPU agg/disagg sizing under an SLA. AISimulate searched
aggregated throughput/GPU with GPU ceilings of 8 and 32 and no SLA filter. The workloads therefore
produce different answers. These dated source snapshots do not measure the current checkout or
establish a universal speed ratio; the 70B runs also used a shared workstation.
Profiling identified Bayesian suggestion generation as a major cost in addition to replay.

A proposed optimizer change reduced the recorded Bayesian medians to **42.97 s** and **145.89 s**,
leaving about **5.2×** and **20.2×** gaps. The small 70B reduction was within run-to-run variation.
Random search was much faster in the recorded optimized variant: **3.39 s** for 8B with four workers
and **7.74 s** for 70B with sixteen workers. Its best throughput/GPU score was about **35%** and
**17%** lower than Bayesian, respectively. Default-budget Bayesian searches (320 suggestions) were
not timed. The general examples above explicitly use eight random trials; their runtime and search
quality should not be confused with the Bayesian cases.

The [benchmark procedure](https://github.com/ai-dynamo/aisimulate/blob/dc797c4f8ee79138cea359f881a363cb1b0b6330/docs/cli/runtime-benchmark.md),
[8B evidence](https://github.com/ai-dynamo/aisimulate/blob/dc797c4f8ee79138cea359f881a363cb1b0b6330/docs/cli/benchmarks/recommend-runtime-2026-09-11.json),
and [70B evidence](https://github.com/ai-dynamo/aisimulate/blob/dc797c4f8ee79138cea359f881a363cb1b0b6330/docs/cli/benchmarks/recommend-runtime-70b-2026-09-12.json)
pin the source revisions, commands, timings, and scores. Choose algorithm, trial budget, and worker
count explicitly; reducing search time can reduce recommendation quality. Use AIC sizing when its
answer fits your task and fast iteration is important.

### Keep minimum-GPU sizing on the compatibility CLI

Use `aiconfigurator cli recommend` with `--target-request-rate` or `--target-concurrency` for
AIC's minimum-GPU and replica-sizing result. The [request-rate example](#search-under-a-request-rate)
shows the command and the alternative AISimulate ranking workflow.

Legacy `default` also routes to sizing when a load target is supplied without `--total-gpus`.
If both are supplied, it uses the GPU budget and warns that the load target is ignored.

### Static estimates and diagnostics

Keep `estimate` for a fixed batch or single pass, including per-operation reports. For example,
inspect one decode pass and its memory, timing, energy, and data sources:

```bash
aiconfigurator cli estimate \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system h200_sxm --backend vllm --backend-version 0.24.0 \
  --estimate-mode static_gen --batch-size 64 --tp-size 2 \
  --isl 1024 --osl 128 --detail memory,time,energy,source
```

The result is a per-operation estimator breakdown. See
[estimate modes and outputs](legacy-aic-user-guide.md#estimate-mode).

### Power and energy analysis

AIC-style modeled power analysis remains available through the compatibility
`aiconfigurator cli estimate` command and estimator SDK bundled with AISimulate. Unified
`aisimulate predict` and `aisimulate recommend` do not provide an equivalent complete power report
or power/energy optimization objective.

For modeled power and per-operation energy of a decode pass, run:

```bash
aiconfigurator cli estimate \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system h200_sxm --backend vllm --backend-version 0.24.0 \
  --estimate-mode static_gen --batch-size 64 --tp-size 2 \
  --isl 1024 --osl 128 \
  --detail energy
```

**Result to inspect:** the terminal prints a `Power (per GPU)` summary and the per-operation energy
breakdown. AIC reports `power_w` only when `power_coverage` reaches 90%;
otherwise power is reported as unavailable. These are modeled GPU values and depend on energy-data
coverage, rather than measurements of whole-node or datacenter consumption.

The unified EPD path can preserve limited encoder-power metadata in `predict --format json` and
the `summary` in `prediction.json`: `encoder_power_w` appears only when encoder energy data is
available, alongside `encoder_power_coverage`. With no data, coverage is zero and the wattage field
is omitted. The normal terminal summary does not display these power fields. Recommendation
artifacts can also retain this metadata for each candidate. These fields do not provide a power
report for the full encoder-plus-language deployment or replace AIC's power analysis. Use the
compatibility command or SDK when power is a required analysis result.

### Deployment artifacts

Keep `generate` when you need deployment files:

```bash
aiconfigurator cli generate \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system h200_sxm --backend vllm --total-gpus 8 \
  --deployment-target dynamo-j2 --save-dir ./deployment
```

This writes a basic deployment configuration under `deployment/` without search or SLA
optimization. AISimulate's `recommendations/*.yaml` files are inputs to `predict`, not launch
manifests. For programmatic generation from supported agg/disagg candidates, see the
[generator SDK](../../python/aisimulate/docs/generator_overview.md). Generation does not support
analytical EPD/AFD or heterogeneous P/D hardware.

### Experiment files and support queries

Existing named experiments still run with AIC; `experiments.yaml` below is your existing file:

```bash
aiconfigurator cli exp --yaml-path experiments.yaml --save-dir ./aic-experiments
aiconfigurator cli support --model-path meta-llama/Meta-Llama-3.1-8B --system all --backend all
```

The first command saves experiment results; the second prints AIC's agg/disagg support matrix.
Translate each experiment separately before moving it to the unified CLI. Estimator data coverage
alone does not establish support for an entire CLI workflow.

### Estimator controls and speculative decoding

Backend version and op-level/FPM selection have unified mappings. The following controls still
require AIC or the estimator SDK:

- Performance-data selection: database modes such as `HYBRID` and `SOL`, transfer policy, and custom
  system definitions. See [database modes](legacy-aic-user-guide.md#database-mode) and
  [system paths](legacy-aic-user-guide.md#systems-paths).
- Explicit GEMM, KV-cache, FMHA, MoE, and communication quantization overrides, attention/MoE kernel
  selectors. See [advanced AIC tuning](../../python/aisimulate/docs/advanced_tuning.md).
- Exact cached-prefix input (`--prefix N`): no direct unified-CLI mapping. AISimulate supports
  prefix-cache simulation, but `kv_cache.prefix_caching: true` enables reuse rather than specifying
  a fixed cached-token count. Session shared-prefix settings describe workload sharing. Keep AIC
  when you need its assumption that exactly `N` prompt tokens are already cached.
- Speculative decoding, including MTP, n-gram, EAGLE-3, DFlash, DSpark, and standalone draft models.
  These cost models are available through compatibility `estimate` and SDK/Task interfaces, subject
  to scheme-specific limits; the unified CLI has no speculative configuration. The caller supplies
  accepted-token assumptions. See [the estimate example and restrictions](../../python/aisimulate/src/aiconfigurator_core/sdk/speculation/README.md#estimate-command).

### Legacy search domains and topology coverage

The default unified parallelism preset currently uses worker sizes of 1/2/4/8/16 GPUs and PP=1.
Explicit parallelism configurations are a separate path, and CP has no unified configuration field.
The existing TP/DP/MoE/replica search does not reproduce every AIC PP/CP, batch/context, or exhaustive
search domain. Retain the AIC sweep when those exact domains are required; see the
[default search projection](../sweeper/architecture.md#parallelism-search-projection).

Backend and model support also depend on topology and available performance data. Analytical
EPD/AFD, heterogeneous P/D through the SDK, and native host offload each have the restrictions linked
in the advanced examples. Those features do not establish support for every combination, and a
schema accepting a parallelism value does not qualify its serving behavior on real hardware.

## 6. Reference

### Common input mapping

Use these mappings when translating a workflow; choose traffic and search objectives explicitly.

| AIC input | AISimulate input | Translation note |
|---|---|---|
| `--model-path`, `--system`, `--backend` | `engine.model`, `engine.hardware`, `engine.backend` | Concrete model, hardware, and backend; optional `engine.backend_version` pins the version. |
| `--isl`, `--osl` | `traffic.source.input_tokens`, `traffic.source.output_tokens` | Synthetic request lengths. |
| `--total-gpus` | `optimization.constraints.max_candidate_gpus` | Search ceiling; prediction GPU use follows worker parallelism and replicas. |
| `--target-request-rate` | `traffic.load: {type: constant_rate, requests_per_second: N}` | Offered traffic. |
| `--target-concurrency` | `traffic.load: {type: concurrency, concurrency: N}` | In-flight request cap. |
| `--ttft`, `--tpot` | `evaluation.sla.ttft_ms`, `evaluation.sla.itl_ms` | Goodput uses request-level latency; strict filtering uses aggregate means, including mean TPOT. |
| `--request-latency` | `evaluation.sla.e2e_ms` | Use instead of TTFT/ITL bounds. |
| `--strict-sla` | `optimization.strict_sla: true` | Reject aggregate-mean latency violations before ranking. |

See the [AISimulate input reference](user-guide.md#configuration-model) for full schemas and
[result reference](user-guide.md#outputs) for files and metric units. Analytical EPD uses
aggregate-only SLA semantics, described in its feature guide.

### Repository and release transition

AISimulate is the home for ongoing development, issues, and releases. The standalone
AIConfigurator repository is scheduled to archive after its final 0.12.0 release. AISimulate 0.12.0
keeps the AIC compatibility command; removal is targeted for 0.13.0 after all remaining workflows
have verified unified-CLI replacements. See the [release transition policy](../../README.md#aiconfigurator-repository-transition)
and [repository history](../repository-history.md).
