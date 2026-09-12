# Migration runtime benchmark

This benchmark measures host CPU execution time. It does not measure GPU inference speed or
establish prediction accuracy on silicon. See [the migration guide](migrate-from-aiconfigurator.md)
for the dated results and their interpretation.

## Evidence formats

The controllers write detailed local `protocol: 1` results, including full reports and machine paths.
The dated JSON files linked from the guide are **compact publication bundles**, not verbatim
controller output. Each `publication` field names its format, hashes the original input file(s),
and describes the projection. Retained timing samples are unchanged. Replay publication groups
variant metadata and omits full reports/stdout/paths; recommendation publication additionally keeps
a union of concrete candidate configurations and per-sample score maps. Reproduction writes the
full controller format, not this presentation schema. Original replay native hashes were captured
after measurement; the recommendation run uses the newer before/after attestation checks.

## Recommend boundaries and reproduction

`scripts/benchmark_recommend_runtime.py` launches the installed `aiconfigurator` and `aisimulate`
entrypoint scripts with each variant's Python interpreter and pinned source paths. Wall time runs
from process launch through exit, including imports, optimization, worker startup, prediction, and
report/output writes. Commands run serially, with case order reversed on alternating rounds.
Unlike the estimate/replay controller, this controller has no unrecorded warm-up pass; native import
preflight and earlier diagnostics warmed filesystem caches. No OS cache purge or core pinning is
performed. Common BLAS/OpenMP thread pools are limited to one thread. Builds run outside timings.

The dated record uses standalone AIC `f254959eb89e2f206b8f9a77051644d7c1cbdb89`, recommendation base
`46ca8915a3ba9b5b17c2c925644127a6ed9de869`, and optimized sampler
`1024438228eed7a8ef6c108e012d26f1d408d4dc`. This recommendation base **already has the replay cache**,
so the before/after comparison isolates trial-axis padding. Both AISimulate variants load the same
release native binary, verified by SHA-256. Later documentation/ownership commits do not change
measured production code. The artifact records exact source identities, dependencies, native hashes,
per-run times/counts/scores, candidate configurations, and before/after prediction checks.

AIC receives:

```bash
aiconfigurator cli recommend \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --system b200_sxm --backend vllm --backend-version 0.24.0 \
  --target-concurrency 10 --isl 1024 --osl 128 --ttft 2000 --tpot 30 \
  --systems-paths "$AISIM_SHARED_SYSTEMS"
```

AISimulate receives `aisimulate recommend --config CONFIG.json --format json --output-dir OUTPUT`.
The harness writes this configuration, changing only the algorithm/parallelism for the named cases:

```yaml
engine:
  mode: aggregated
  model: meta-llama/Meta-Llama-3.1-8B
  hardware: b200_sxm
  backend: vllm
  backend_version: 0.24.0
  workers:
    aggregated:
      parallelism:
        preset: default
optimization:
  target: throughput_per_gpu
  constraints:
    max_candidate_gpus: 8
optimizer:
  algorithm: bayesian
  max_trials: 32
  parallelism: 16
  seed: 42
```

Omitted traffic resolves to 1,024 input tokens, 128 output tokens, concurrency 10, and 100 requests
per candidate. The AISimulate case has no SLA constraint and searches only aggregated deployments;
AIC searches aggregated/disaggregated sizing for a load target and SLA. AIC's result is not expected
to match AISimulate's. Full default 320-suggestion searches, other objectives, and disaggregated
AISimulate searches remain unmeasured. Random search is an explicit algorithm change with the
quality tradeoff shown in the migration guide; neither defaults nor Bayesian budgets were reduced
by the production patch.

First complete the [installation setup below](#reproduce-compatibility-estimates-and-replay), then
create a separate recommendation base and variant file:

```bash
git worktree add --detach "$AISIM_BENCHMARK_ROOT/recommend-base" 46ca8915a3ba9b5b17c2c925644127a6ed9de869
uv sync --project "$AISIM_BENCHMARK_ROOT/recommend-base/python/aisimulate" --extra dev --python 3.12
python3 - <<'PYCODE'
import json
import os
from pathlib import Path

work = Path(os.environ["AISIM_BENCHMARK_ROOT"])
variants = json.loads((work / "variants.json").read_text())
base = next(v for v in variants if v["label"] == "base")
root = work / "recommend-base"
base.update(root=str(root), python=str(root / "python/aisimulate/.venv/bin/python"),
            sources=[str(root / "python/aisimulate/src")])
(work / "recommend-variants.json").write_text(json.dumps(variants, indent=2))
(work / "head-variant.json").write_text(json.dumps([v for v in variants if v["label"] == "head"]))
PYCODE
export AISIM_SHARED_SYSTEMS="$PWD/python/aisimulate/src/aiconfigurator_core/systems"
python3 scripts/benchmark_recommend_runtime.py \
  --variants "$AISIM_BENCHMARK_ROOT/recommend-variants.json" \
  --systems-path "$AISIM_SHARED_SYSTEMS" \
  --rounds 3 --trials 32 --parallelism 16 \
  --output-dir "$AISIM_BENCHMARK_ROOT/recommend-results"
python3 scripts/benchmark_recommend_runtime.py \
  --variants "$AISIM_BENCHMARK_ROOT/head-variant.json" \
  --systems-path "$AISIM_SHARED_SYSTEMS" \
  --rounds 3 --trials 32 --parallelism 4 --algorithms random \
  --output-dir "$AISIM_BENCHMARK_ROOT/random-four-workers"
```

Output directories must be new. The controller checks bundled AISimulate systems-tree hashes
against the explicit AIC tree, rejects unsuccessful/timed-out searches or missing recommendations,
and rechecks source/native identity after timing. It retains complete recommendation reports locally.
The committed [compact evidence](benchmarks/recommend-runtime-2026-09-11.json) keeps timing samples,
configuration/score maps, and comparisons, excluding machine-specific absolute paths. Candidates
are matched by their concrete `prediction_config`, since completion order and candidate IDs can
vary. Shared candidate metrics use the same exclusions/tolerance as the replay comparison below.
Candidate-set or best-score equality is observed evidence for the measured cases, not a required
contract of JAX padding across all workloads.

The separate 64-suggestion prototype diagnostic wraps sampler initialization, `suggest`, `observe`,
search-space enumeration, and candidate materialization with `time.perf_counter` timers around the
public CLI operation. The evidence retains those phase samples. Its total starts before importing
the CLI in an already-started interpreter; it is not a fresh-process timing sample and is not used
in the repeated CLI ratios. The padding prototype imports JAX earlier, so import versus sampler-init
allocation differs; `suggest` remains the useful dominant boundary. Fitting, acquisition optimization,
and JAX compilation are combined in that boundary, not individually attributed.

## Compatibility estimate and replay boundaries

The fixed case uses `meta-llama/Meta-Llama-3.1-8B`, B200, vLLM `0.24.0`, TP4/PP1/attention-DP1,
MoE-TP4/EP1, BF16 model defaults, 1,024 input tokens, and 128 output tokens. The analytical AIC
estimate uses batch 16. The serving replay uses concurrency 16 with either 100 or 1,000 completed
requests. These inputs deliberately bound local memory and execution time.

- **Matched AIC CLI:** `aiconfigurator cli estimate --estimate-mode agg`, called through each
  installed distribution's public `main` with the same arguments. The controller times the entire
  child process, including imports, shared-data bootstrap, estimation, output, and small harness
  overhead. Log timestamps are excluded from the result comparison.
- **Matched SDK:** `InferenceSession.run_static_latency_only` for `static_ctx` and `static_gen`.
  OSL is 2 for a single decode step, stride is 1, and the batch is 16. Session setup and each phase's
  first query are recorded separately. Ten untimed repeats precede 100 timed warm calls per phase.
  This measures the Python SDK/Rust boundary, not a pure Rust function.
- **Unified CLI:** `aisimulate predict --stack engine` through the public `main`, including YAML
  parsing and JSON output in a temporary directory. This performs a serving replay, which has
  different semantics from one AIC analytical estimate.
- **Warm runner:** three samples from `EngineReplayRunner.run` after one initial run whose elapsed
  sample is discarded. All four calls contribute to `worker_operation_s`; only the last three contribute to
  `warm_replay_s`. Each call creates a fresh timing provider and its cache, so this includes
  provider construction and reporting in an
  already imported process with warm data access. A separate native `wall_time_ms` measurement
  includes native validation, worker/runtime construction, event processing, and statistics
  collection. It excludes AIC timing-provider compilation and Python orchestration.

The controller runs one child at a time and reverses variant order on alternating rounds. One full
unrecorded pass primes installation/import and filesystem caches. Five recorded rounds are the
default. "Fresh process" therefore does **not** mean a purged OS page cache or first-ever package
import. Native builds must be optimized release builds; compilation is outside the measurements.

All variants use one explicit AISimulate systems tree through the SDK data-root setter and
`AICONFIGURATOR_SYSTEMS_PATH`; the AIC command also receives `--systems-paths`. This bootstrap is
necessary because the two repositories' bundled data can differ. The controller hashes that tree,
records source revisions and loaded module paths, and checks predictions before reporting ratios.
The benchmark model configuration is identical in the two measured repositories. Common Python
dependencies also match. This is a controlled runtime comparison, not a comparison of arbitrary
separately installed wheels with different data.

Replay comparisons exclude only `wall_time_ms`, `processed_tokens_per_s`, and
`processed_output_tokens_per_s`, following the native canonical-result exclusions. Counts and
fields must match exactly. Real-valued metrics use relative tolerance `1e-12` and absolute tolerance
`1e-9` for the unchanged runtime's floating-point aggregation-order differences. These are
correctness comparisons between builds, not silicon validation.

## Reproduce compatibility estimates and replay

From the AISimulate checkout containing this change, with `git`, `uv`, a Rust toolchain, and Python
3.12 available, create isolated installations from clean committed sources. The two pinned
revisions are the original replay-cache baseline snapshot; the current checkout supplies the
optimized variant. Keep outputs outside every variant checkout: the controllers reject modified
and untracked source before timing. They attest the resolved loaded native extension path and
SHA-256 separately because native build products are Git-ignored. Hashes identify the binary that
ran; they do not independently prove how it was compiled. Use release builds from the pinned source.

```bash
export AISIM_BENCHMARK_ROOT="$(mktemp -d)"
git clone https://github.com/ai-dynamo/aiconfigurator "$AISIM_BENCHMARK_ROOT/aic"
git -C "$AISIM_BENCHMARK_ROOT/aic" checkout f254959eb89e2f206b8f9a77051644d7c1cbdb89
git worktree add --detach "$AISIM_BENCHMARK_ROOT/base" 68d3e1b727d41cd6bf1336637a8392ed8e4e0ddb
uv sync --project "$AISIM_BENCHMARK_ROOT/aic" --extra dev --python 3.12
uv sync --project "$AISIM_BENCHMARK_ROOT/base/python/aisimulate" --extra dev --python 3.12
uv sync --project python/aisimulate --extra dev --python 3.12 --reinstall-package aisimulate
python3 - <<'PY'
import json
import os
from pathlib import Path

work = Path(os.environ["AISIM_BENCHMARK_ROOT"])
head = Path.cwd()
variants = []
for label, root, kind in (("aic", work / "aic", "aiconfigurator"),
                          ("base", work / "base", "aisimulate"),
                          ("head", head, "aisimulate")):
    package = root if kind == "aiconfigurator" else root / "python/aisimulate"
    sources = [package / "src"]
    if kind == "aiconfigurator":
        sources.append(root / "aic-core/src")
    variants.append({"label": label, "kind": kind, "root": str(root),
                     "python": str(package / ".venv/bin/python"),
                     "sources": [str(path) for path in sources]})
(work / "variants.json").write_text(json.dumps(variants, indent=2))
PY
python3 scripts/benchmark_migration_runtime.py \
  --variants "$AISIM_BENCHMARK_ROOT/variants.json" \
  --systems-path "$AISIM_BENCHMARK_ROOT/base/python/aisimulate/src/aiconfigurator_core/systems" \
  --rounds 5 \
  --output "$AISIM_BENCHMARK_ROOT/results.json"
```

The JSON retains raw samples, warm-call distributions, shared-data hashes, dependency versions,
source identities, and result checks. A failed command, timeout, or prediction mismatch fails the
benchmark instead of becoming a timing sample. Compare medians for each boundary separately;
`process_wall_s` for an SDK or warm-runner worker includes its repetitions and must not be reported
as CLI latency.

Run CPU-heavy builds and unrelated local jobs separately when collecting performance evidence.
The controller limits common native thread pools to one thread, but does not pin cores or purge
caches. Repeat on the deployment's actual host for capacity planning.

## Replay cache boundary and remaining work

The native replay timing provider memoizes successful scalar latency predictions by their exact
estimator coordinates, up to 1,024 entries per provider. The provider already fixes model,
hardware, backend, data, quantization, and topology. Prefill, mean-context decode, and exact-total
FPM decode have distinct keys. FPM decode keys retain the existing subtraction of current tokens
and physical-capacity cap. Failures and non-finite or negative latencies are not retained. Cache
entries disappear with the provider; candidates never reuse another candidate's timing results.

This reduces repeated estimator calls during a replay. It does not remove initial model/data
loading, Python imports, provider construction, scheduler events, or report generation. Traces
with few repeated coordinates may see little benefit; eviction bounds memory instead of promising
a hit for every request length. This change applies to the built-in engine stack's native AIC
provider. The separate Dynamo adapter and analytical AFD/EPD paths need their own measurements.

Rust execution, lazy database loading, existing lookup indexes, and lower-level caches already
reduce estimator work. The advisory forward-performance gate covers the warm SDK boundary; it
does not prove startup or complete recommendation throughput. Additional profiling can target
imports, source-resolution/setup, unique timing queries, and search orchestration. Default-budget searches,
MoE models, FPM speed, trace workloads, and additional hosts are outside this fixed benchmark.
