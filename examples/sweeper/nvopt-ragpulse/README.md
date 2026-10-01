<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVOpt: three RAGPulse search-space designs

These experiment configurations are frozen against Dynamo
`c7241c2f153efba10b57c38c2144b70d82194a4d` and AISimulate nightly
`0.13.0.dev202609300000000061`. The local broad sweep was stopped by request;
these files do not resume it or launch another local sweep. The environment
image is separate from the experiment data and execution harness. The
subsequently authorized remote harness lives under [runtime/](runtime/).

Published environment: `nvcr.io/nvidian/dynamo-dev/aisimulate:hzhou-0930-02`.
Use the immutable digest in [container/push-result.json](container/push-result.json).
The [container source bundle](container/README.md) includes the Dockerfile,
144 hashed dependency wheels' metadata, build provenance and validation evidence.
It runs on CPUs with `JAX_ENABLE_X64=true`; no physical GPU is needed.

The [completed results](results/README.md) contain all three
full-budget studies, independent winner validation, timing and serving figures,
and coverage references. Numerical runtime pins remain unchanged.

| Experiment | Configuration | Search scope |
|---|---|---|
| 1. Static engine | [01-static.yaml](01-static.yaml) | Aggregated/disaggregated engine mapping and scheduler; round-robin placement, no Planner |
| 2. Engine + KV Router | [02-kv-router.yaml](02-kv-router.yaml) | The entire experiment 1 space plus KV Router knobs |
| 3. Engine + KV Router + Planner | [03-kv-router-planner.yaml](03-kv-router-planner.yaml) | The entire experiment 2 space plus Planner knobs; four-day predictor history |

These are three independent broad Bayesian searches, **256 native suggestions
each**, seed `20260929`, 32 concurrent evaluations, and a 7,200-second native
evaluation deadline. There is no four-hour whole-study deadline. Failures,
cache hits and projected duplicate configurations consume suggestions; 256
suggestions do not mean 256 unique successful replays or exhaustive coverage.
Native early projection-stall termination must be reported with the actual
budget consumed. Engine parameters remain searchable in experiments 2 and 3;
the winner of an earlier experiment is not fixed as their engine configuration.

## Shared engine space and measurement

- Model: `deepseek-ai/DeepSeek-V3`; hardware: `h200_sxm`.
- Backends: vLLM `0.24.0`, SGLang `0.5.14`; aggregated and disaggregated serving.
- Maximum initial deployment: 256 simulated GPUs. No artificial minimum GPU
  floor or preferred 100-GPU filter. The optimizer can choose fewer GPUs.
- Parallelism uses the pinned native `default` legal mapping menu, including
  independent prefill/decode attention DP. Observed menu DP sizes are 1, 8, 16;
  these are legal coupled TP/DP/MoE/replica mappings, not independent Cartesian
  TP/DP/EP dimensions. PP remains 1. Initial replica counts are optimized; in
  experiment 3 Planner subsequently changes replicas within the GPU budget,
  not the per-worker TP/DP mapping.
- Op-level timing with fallback denied; BF16 attention, FP8 KV; KV blocks of
  64 tokens, prefix caching and chunked prefill enabled, memory fraction 0.9.
- Disaggregated transfer: 35,136 bytes/token, 64 GB/s, `full_prompt` timing.
- Context length 8,192. Pinned Dynamo's SGLang parser does not accept the
  corresponding `max_model_len` argument: the runner must omit it as in the
  local baseline. The evaluation trace maximum ISL+OSL is 7,599, so that
  omission does not truncate evaluation requests.

| Scheduler knob | Aggregated | Prefill | Decode |
|---|---|---|---|
| Max batched tokens | 1024, 2048, 4096, 8192, 16384, 32768 | same | 2048, 4096, 8192, 16384 |
| Max sequences | 64, 128, 256, 512, 1024 | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 | 64, 128, 256, 512, 1024 |

All experiments evaluate the same 512-copy, fifth-day trace: 537,600 requests,
1,476,413,952 input tokens and 160,730,624 output tokens. Arrivals remain
open-loop over 24 hours; replay may drain until virtual hour 25. Incomplete
request/token cohorts are infeasible, never fabricated zero-score trials.

The objective is **SLA-compliant output tokens / 86,400 / average GPUs**.
Use TTFT <= 1,000 ms and **per-request mean ITL <= 50 ms**, not the maximum
single-token gap. Average GPUs are `native_gpu_hours / native_duration_hours`;
for static deployments this is exactly the fixed GPU count. For Planner this
uses provisioned GPU time over the native replay, including drain. Normalize
the compliant-token rate to 86,400 seconds but preserve native duration and
GPU-hours. Report native GPU-hours as well; do not charge Planner for its
256-GPU ceiling or initial allocation as though they were used all day.
There is no minimum SLA-pass-fraction filter: always show coverage and total
goodput next to efficiency so a small but overloaded deployment is visible.

## Router: defaults plus three points per numeric knob

The default domains below come from the pinned Dynamo
[Router adapter](https://github.com/ai-dynamo/dynamo/blob/c7241c2f153efba10b57c38c2144b70d82194a4d/components/src/dynamo/router/simulation/config.py).
Experiments 2 and 3 fix `policy: kv_router`; experiment 1 supplies the
round-robin comparison. Keep both default load-model choices, `none` and `ais`.

| Knob | Native default search values | Added values |
|---|---|---|
| `overlap_score_credit` | 0, 0.5, 1 | **0.25, 0.75, 1.5** |
| `prefill_load_scale` | 0, 0.25, 0.5, 1, 2, 4, 8, 16, 32 | **0.125, 64, 128** |
| `temperature` | 0, 0.2, 0.5, 1 | **0.1, 0.75, 2** |

The numeric domains bracket the existing settings while retaining every
default value. This gives 1,008 nominal Router combinations per compatible
engine mapping (2 load models × 6 credits × 12 scales × 7 temperatures).
Admission-control thresholds are excluded: the pinned replay adapter explicitly
does not support them. They are not silently accepted no-op knobs.

## Planner: defaults plus three expanded groups

Experiment 3 fixes `policy: enabled`. It retains all six **enabled** native
scaling presets; the `disabled` preset belongs to experiment 2 and is excluded
so experiment 3 cannot silently turn Planner off. Source:
[Planner presets](https://github.com/ai-dynamo/dynamo/blob/c7241c2f153efba10b57c38c2144b70d82194a4d/components/src/dynamo/planner/simulation/presets.py).

| Group | Native default candidates retained | Added combinations |
|---|---|---|
| Scaling policy | `throughput_180_5`, `throughput_600_5`, `load_180_5`, `load_180_10`, `hybrid_180_5`, `hybrid_600_5` | **hybrid (60s, 5s), (300s, 10s), (900s, 10s)** |
| FPM sampling | `small` (32,4), `default` (64,16), `large` (128,16), `fine` (128,64) | **(256 samples,16 buckets), (256 samples,64 buckets)** |
| Load sensitivity | `aggressive` (70,3), `default` (80,5), `conservative` (90,8) | **(60,2), (85,10), (95,12)**: down-sensitivity %, minimum observations |

Custom entries are self-contained mappings accepted by the public schema,
not invented preset names or a Cartesian product of mutually incompatible
preset and independent fields. Bucket sizes remain perfect squares.
FPM sampling matters only for throughput-enabled policies, and load
sensitivity only for load-enabled policies; native materialization/cache
handles those conditional identities. Do not interpret their full Cartesian
product as an equal number of distinct replays.

Keep the predictor default menu unchanged: `constant_last`, `arima_raw`,
`arima_log1p`, Prophet windows 20/50 with raw/log1p, and Kalman default/reactive
with raw/log1p: **11 candidates**. Predictor selection is a separate native
pre-search per throughput interval (60, 180, 300, 600, 900 seconds), not another
11-fold Vizier dimension. Fit and select only on days 1–4, recording its
separate timing, losses and selected family. Native fallback/no-winner must
remain explicit in the report.

The runtime ceiling is **256 GPUs**, with at least one aggregated worker or
one worker in each prefill/decode role. Preserve the default
`max_throughput_scaling_replicas: 8`: this is a **per-observation change limit**,
not an eight-worker total cap. Initial replicas come from the engine search.
Worker startup delay remains the baseline default (zero); these experiments
do not establish accuracy for real model-loading/cold-start delays.

## Four-day history, fifth-day evaluation

The source is [RAGPulse revision 7da286be](https://github.com/flashserve/RAGPulse/tree/7da286becf0f049b2bcb1e5a11d9ba8eb638eff4).
Use source time `[0, 345600)` only as Planner history and `[345600, 432000)`
only for evaluation. Both use the same 512-copy transform and seed
`20260928`, disjoint hash namespaces per copy, and deterministic per-session
phase shifts within ±30 seconds tapered at day boundaries. Time is not
compressed. Session IDs stay out of replay rows to preserve open-loop arrivals.

RAGPulse supplies component identifiers, not native token-block hashes. The
existing conversion packs ordered opaque component slices into 64-token blocks
and chains each block to its preceding prefix. Unlocated residual tokens are
row-specific. Router reuse results are consequently **RAGPulse-derived
opaque-component workload predictions**, not measured original token reuse.
Keep this qualification in plots and final findings.

The history trace contains 2,024,448 scaled requests. Before remote execution,
materialize or supply it separately, verify its four-day provenance/counts,
and record its SHA256. A saved 180-second observation file is not valid for
the other cadence choices. Preserve idle/zero-request intervals.
Days 1–4 train/bootstrap only the load predictor: they do not warm engine KV
caches or count toward GPU-hours, goodput, or evaluation requests. Start all
three evaluation caches cold. Each Planner candidate receives a fresh
predictor instance bootstrapped with its matching-cadence historical data.

## Execution boundary and review checks

The YAML files are public recommendation **search-space declarations**. Use the
experiment harness rather than plain `aisimulate recommend` to enforce the
fixed-day and history protocol. [experiment-contract.yaml](experiment-contract.yaml)
records these execution requirements, implemented under [runtime/](runtime/):

1. Preserve the local baseline's fixed-day scoring and precision/canonical
   lowering bridge. The pinned stock Dynamo factory does not advertise the
   two precision controls even though its lowering/runtime implements them.
2. During Planner `compile_recommendation`, replace only its
   `RecommendationAdapterContext.sweep.workload` trace with the four-day
   history. The native adapter otherwise selects its predictor from the
   evaluation workload, which would leak day 5 into predictor selection.
3. During Planner hook materialization, add
   `planner_config.load_predictor_warmup_trace` pointing to the history trace.
   The native runtime supports it; the public Planner recommendation schema
   currently does not expose it. Do not add an unrecognized YAML field or
   mistake predictor selection for per-candidate warmup.
4. Keep the simulated GPU ceiling independent of CPU-host memory limits;
   require full cohort completion; retain native rejection/cache/timeout
   receipts. The 7,200-second deadline applies to native worker waves and can
   replace the pool; it does not guarantee every candidate finishes.

The native replay calculations and optimizer selection are unchanged. The
bridge retains lightweight native summaries during search, records requested/effective specs
and timings, and checks the frozen native binary and AIS package identity.
Planner predictor evaluations are independent tasks dispatched to eight spawn
workers using the native history aggregation, common warmup and forecast-loss
functions. An all-11-preset, two-cadence fixture matched serial losses and
winner selection exactly; fallback and tie ordering were also checked. This
changes preparation scheduling, not the predictor search menu.

Example command **inside a compute allocation**, with immutable source, data,
and writable results/temp mounts:

```bash
python /workspace/bundle/runtime/run_sweep.py \
  --scenario 1 \
  --config /workspace/bundle/01-static.yaml \
  --output /results/formal-v1 --memory-limit-gib 320 \
  --pool-cleanup-grace-seconds 60
```

Use scenario 2/3 and their matching YAML for the other studies. The output
directory must be new. `--parallelism` and `--candidate-timeout` are explicit
resource overrides, recorded in run metadata; they do not reduce the trial
budget or workload. Archive each interrupted run before changing these limits.
`--replay-spec /path/to/requested-spec.json` independently reruns a selected
native candidate without compiling a new optimizer study. Resource supervision
records RSS/headroom, process exits and termination. A successful launch or a
passing fixture is not evidence that the full campaign has completed.

The custom harness calls Sweeper directly to retain audited optimizer receipts.
Its effective resource policy is recorded in `supervisor.json`: the initial
remote runs enforce a 320 GiB process-tree RSS guard, 16 GiB host-available
headroom and 96 GiB per-worker virtual-address limit, inside externally limited
40-CPU / 384-GiB containers. It does **not** invoke the stock CLI resource
supervisor, so the YAML's `reserve_memory_gb` and initialization timeout must
not be reported as additional enforced limits. Predictor preparation is timed
separately and is outside the per-evaluation deadline. Running immutable
bundles are never edited to change these settings retrospectively.

The 32-worker campaign exposed a native pool-cleanup limit: the pinned
nightly waits only two consecutive 2-second joins for its executor manager.
A controlled fixture with 32 workers retaining about 216.67 GiB reproduced
`executor cleanup did not complete; refusing replacement`. Independent
30- and 60-second join budgets completed cleanup in about 4.69 seconds;
all owned worker identities disappeared, the manager stopped, and a new
pool returned results. This measures retained worker memory, without
attributing its contents to a particular cache or performance-data table.

The execution-only recovery in commit `38a89d02` adds
`--pool-cleanup-grace-seconds`. Use 60 for this campaign. Omitting the flag
preserves the native default. It changes the native manager/worker join
waits while retaining ownership checks and escalation; the separately bound
descendant-process grace stays at 2 seconds. Requested and effective values
are recorded in `run-start.json` and `supervisor.json`.

Recovery bundles copy the original `71fbad95` Static/Router or `9ebfc7de`
Planner bundle and replace only `runtime/run_sweep.py`; retain a complete
file-hash comparison. The image and numerical engine remain frozen. Archive
failed runs and restart a full 256-suggestion study in a new output directory.
Do not silently reuse their optimizer observations or describe this as resume.
Include failed-run and recovery time in campaign cost, even when plotting the
new study separately. Fixture success and actual formal pool replacement
are separate validation checkpoints.

Run the lightweight checker **inside the pinned environment**:

```bash
python examples/sweeper/nvopt-ragpulse/validate_design.py
```

It validates core/adapter schemas, default preservation, domain extensions,
cross-scenario nesting, and predictor cadence coverage. It never trains a
predictor, builds an estimator, or launches a simulation. Runtime smoke-test
evidence belongs to the separately published container manifest.

## Larger CPU execution profile

The stopped workstation run used eight concurrent evaluations. The reviewed
remote profile now uses **32**, with all model, traffic, Router/Planner domains,
SLA, seed, 256-suggestion budget and per-evaluation timeout unchanged. The image
already supports this; no runtime rebuild is required.

| Profile | Concurrent evaluations | Application CPU limit | Requested job/container memory | AIS execution memory budget |
|---|---:|---:|---:|---:|
| Default remote | **32** | **40** | **384 GiB** | **320 GB** |
| Optional larger-memory node | 64 | 72 | 640 GiB | 576 GB |

Eight local workers together with the optimizer peaked near 59.4 GiB. Budget
roughly 7–8 GiB per replay worker before extra Planner/optimizer overhead; CPU
count alone is not enough to select parallelism. The larger profile requires
changing `optimizer.parallelism`, `execution.resources.cpu_limit` and
`execution.resources.memory_limit_gb` together in a separate copy of the three
YAMLs and their experiment contract, then requesting the matching external
memory allocation. Neither profile is a
measured speedup or a guarantee that every configuration fits.

Prefer one study per allocation with **48 hours requested**, subject to the
scheduler's current QoS, partition, reboot and account restrictions. Three
independent allocations can run the three studies concurrently if resources
are actually granted. Slurm wall time is an external limit, separate from the
native optimizer's suggestion budget; preserve partial results on termination.
Do not assume preemption/requeue resumes the optimizer without an explicitly
implemented resume path.

On an oversubscribed CPU pool, the scheduler may expose the whole shared node
even when fewer CPUs are requested. Keep the explicit application/container
CPU and memory limits; shared CPU count does not imply dedicated cores.
The published image is **linux/amd64**, so use x86 nodes; ARM/Grace nodes need
a separately built and validated image.

Larger batches reduce sequential optimizer feedback opportunities at a fixed
256-suggestion budget. They also still wait for slow evaluations in each native
wave. Use 32 as the speed/adaptivity compromise; 64 is an explicit alternate
execution identity, not a promise of eightfold speedup over the local run.
Record parallelism in each result because changing it can change suggestions
even with the same seed.

Use disk-backed temporary/output volumes and **zero physical GPUs**. Mount data
read-only under `/data/ragpulse`. The source bundle and result volumes stay
separate from the environment image.

For each trial record the actual engine/Router/Planner configuration, native
acceptance, simulation wall time, optimizer timing, goodput/GPU and SLA
coverage. Plot cumulative best against elapsed wall time together with trial
durations. For final winners also plot hourly offered traffic, goodput,
latencies, allocated GPUs, cache reuse and Planner scale decisions; separately
replay the selected configuration before calling it validated. Day 5 is also
used for optimizer feedback: a fresh winner replay checks reproducibility, not
generalization to another held-out day or accuracy on physical GPUs.

## Replacing Vizier with other optimizers

Vizier selects candidates; the replay simulator does not depend on it. A new
optimizer can reuse the same simulation and scoring through either route:

1. **Keep AISimulate Sweeper.** Replace `AuditedSamplerFactory` in
   [run_sweep.py](runtime/run_sweep.py), passed through
   `Sweeper(..., sampler_factory=...)`. Implement the
   [BranchSampler contract](../../../python/aisimulate/src/aisimulate/sweeper/sampler.py):
   `branch`, `suggest(count)`, `observe(suggestion, metrics)` and
   `observe_infeasible(suggestion, reason)`. Return `Suggestion` objects with
   a concrete `parallel_config` and knob `selection`. Remove the
   `SeededBayesianBranchSampler` class assertion and adapt the audit wrapper
   to your trial handles, retaining unique IDs for accounting. This route
   retains the compiled branch domains and supported-mapping checks; changing
   only `optimizer.algorithm` in YAML is not a custom-optimizer plugin.
2. **Own the search loop and search space.** Bypass recommendation/Sweeper
   compilation and generate explicit engine, Router and Planner settings.
   Use the canonical
   [prediction compiler](../../../python/aisimulate/src/aisimulate/compiler.py)
   and adapter materialization to build a concrete `ReplaySpec`, including
   the required runtime hooks. Set its goal target to `goodput_per_gpu` and
   preserve the scenario's trace, SLA and Planner history. Reuse
   [ScenarioRunnerFactory](runtime/scenario_runner.py): create a runner per
   worker with `factory.create(worker_id)`, evaluate via `runner.run(spec)`,
   and close it when finished. Feed
   `report.metadata["scenario"]["derived"]["fixed_day_goodput_per_gpu"]`
   back to the optimizer. The existing `--replay-spec` path in `run_sweep.py`
   is a working example of evaluation without a new optimizer study.

**The current presets and parallelization search machinery are optional.**
With your own driver, you can discard the preset menus, preset expansion,
legal-mapping catalog, TP/DP/MoE feature encoding and nearest-mapping projection,
and define your own variables and constraints. Planner predictor preset
pre-search can also be replaced by an explicitly configured predictor or your
own history-only selection. CPU evaluation batching/concurrency can be managed
by the new driver as well. None of these search policies is required by replay;
the final worker mappings must still be explicit and satisfy backend, memory
and GPU-budget constraints.

Keep the frozen evaluation/scoring contract for comparable results. An external
driver owns trial budgets, deduplication, timeout/resource supervision and
result logging; preserve failed/incomplete outcomes rather than assigning fake
zero scores. The existing campaign helpers can be reused for those duties.
