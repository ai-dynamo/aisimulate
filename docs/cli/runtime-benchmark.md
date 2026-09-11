# Migration runtime benchmark

This benchmark measures host CPU execution time. It does not measure GPU inference speed or
establish prediction accuracy on silicon. See [the migration guide](migrate-from-aiconfigurator.md)
for the dated results and their interpretation.

## Measurement boundaries

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
- **Warm runner:** three calls to `EngineReplayRunner.run` after one untimed run. Each call creates
  a fresh timing provider and its cache, so this includes provider construction and reporting in an
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

## Reproduce

From the AISimulate checkout containing this change, with `git`, `uv`, a Rust toolchain, and Python
3.12 available, create isolated installations. The two pinned revisions are the baseline snapshot;
the current checkout supplies the optimized variant.

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

## Optimization boundary and remaining work

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
imports, source-resolution/setup, unique timing queries, and search orchestration. Full searches,
MoE models, FPM speed, trace workloads, and additional hosts are outside this fixed benchmark.
