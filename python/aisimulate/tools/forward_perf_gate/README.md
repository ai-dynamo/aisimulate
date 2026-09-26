# Forward-prediction performance gate

This tool measures the CPU time for
`InferenceSession.run_static_latency_only`. It compares a pull request head
with its merge base on the same worker. The check is advisory: regressions make
the check red, but repository rules do not require it.

The workflow starts on every trusted `pull-request/*` push. A small
GitHub-hosted selection job checks the complete PR change set against
`scripts/select_forward_perf.py` before starting the benchmark runner. This
also works when the bot creates a branch with an empty push commit list.
Unrelated PRs, including gate documentation-only changes, produce an explicit skip.
Manual dispatch forces a comparison only when the trusted copy matches the current
PR head; an older trusted copy fails selection instead of benchmarking an older revision.

Each revision runs the matrix in one process for the availability pass and one
new process for each measured round. Within that process, cases are grouped by
model and database mode. Before each group, the worker releases the prior
objects, clears the prediction caches, and builds one model, database view,
session, and Rust engine. It initializes each measured phase with one
unrecorded `(batch_size=2, ISL=2048, prefix=0)` query that is outside the benchmark
matrix, with OSL 8 for context and OSL 256 for generation. These coordinates
are fixed within the protocol; changing them requires a protocol-version change.
Priming stride comes from the shared, hashed case request.
It then runs every target point in the group. The first prediction for
each distinct target is the `cold` sample: a new query against a steady-state
engine. Ten unrecorded repeats of that target follow, then 100 repeats produce
the `warm` median. Thus `cold` means a cold query, not a cold process or empty
session. The existing
`benchmark_engine_step.py --cache-mode cold` has a different boundary because
it clears the engine-handle cache before each call.

If the off-matrix priming query fails, the worker records `PRIMING_FAILED` for
only that phase and continues with other phases and groups. A missing-data
priming failure is invalid and blocks the comparison, as do other priming failures.

The harness treats `clear_caches()` as the complete cache-isolation contract.
It calls the public database eviction interface before each model/database-mode
group. When the system, backend, or version changes, it also evicts the outgoing
database before preparing the next group, including after setup or priming failures.
It does not know about or manage individual SDK caches.

The worker uses a versioned JSON protocol so each Git revision can adapt its
own internal SDK interface. It accepts either one `case` or an ordered `cases`
array. Protocol v2 enables shared-layer data reuse, including replacement
measurements declared in `reuse.yaml`. Protocol v1 disabled this reuse, so its
timings cannot be compared with v2. The workflow explicitly skips comparisons
between different protocol versions. Rebase PRs onto the protocol-v2 change to
resume comparisons after it merges. Run one request with:

```bash
python tools/forward_perf_gate/worker.py --request request.json --pretty
```

The matrix has 64 cases and reports 128 cold/warm comparisons. The first 36
requests retain the original Qwen3-32B and Qwen3-235B-A22B matrix on
B200/vLLM 0.24.0: SILICON and EMPIRICAL modes, each with the nine points from
the prediction regression grid. Their IDs, values, and relative order are unchanged.

The next 20 cases use SILICON mode and vLLM 0.24.0:

| Model | System | TP | Attention DP | MoE TP | EP |
|---|---|---:|---:|---:|---:|
| `deepseek-ai/DeepSeek-V3.2` | B200 | 8 | 1 | 1 | 8 |
| `deepseek-ai/DeepSeek-V4-Flash` | B200 | 8 | 1 | 1 | 8 |
| `Qwen/Qwen3.5-397B-A17B` | B200 | 8 | 1 | 1 | 8 |
| `openai/gpt-oss-120b` | B200 | 8 | 1 | 1 | 8 |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` | H100 | 8 | 1 | 1 | 8 |

Each model has two context cases (batch 1, ISL 1,024 and 32,768) and two
generation cases (batch 1 and 128, ISL 1,024). PP is 1 throughout the matrix.

The final eight cases also use SILICON mode:

- Both original Qwen models on B200/vLLM 0.24.0 retain their original parallel
  layouts, with context batch 1, ISL 8,192, and cached prefixes 4,096 and 7,168.
- DeepSeek V3.2 on B200/vLLM 0.24.0 uses attention DP=8, TP=1, MoE TP=1,
  and EP=8 for generation at batch 32/ISL 1,024 and batch 8/ISL 32,768.
- Qwen3.5 on B200/SGLang 0.5.14 uses TP=8, attention DP=1, MoE TP=1, and
  EP=8 for context batch 1/ISL 8,192 and generation batch 32/ISL 1,024.

SGLang retains its original 0.5.14 pin and parallel layout.

Context OSL is 8, generation OSL is 256, and stride is 32. New profile labels
include the system, backend/version, and parallel layout to keep report cells
separate. Worker groups already include those configuration fields. The prefix
suffix separates otherwise equal configurations from the original profiles so
their cases do not change the original cache-preparation groups. The worker
uses pinned requested versions with shared-layer data reuse enabled on both revisions.

Before rollout, validate all 64 cases against two separate installations of
the same revision for three five-round comparisons. Require no missing data
or invalid comparisons, investigate any case flagged in at least two runs,
and compare runtime with the original matrix. Missing data or more than five
additional benchmark minutes blocks rollout; do not remove cases or change
thresholds to hide a blocker.

The normal CI comparison uses the base revision's controller and matrix.
When a PR changes the gate's Python files or the shared prediction grid, the same
job also runs the PR's controller against the already-built base and head
installations. This validates new cases in CI before merge, with separate results under `head-controller/`
in the artifact and a separate summary. Both runs use the same measurement
method and retain their own regression checks. After merge, the expanded matrix
becomes the normal comparison when a PR's merge base includes it.

The default comparison requires four of five paired rounds to exceed both a
10% relative threshold and a 2 us absolute threshold. Other round counts use
an 80% quorum. Every selected case must have data on both revisions. A data miss
on either or both revisions is invalid and blocks the comparison, during both
availability and measured rounds. Malformed worker responses and incomplete
runs also block the comparison. Case names and errors remain visible in the summary.
The worker treats missing silicon data, unavailable empirical data, missing
system FLOPS, and unavailable SOL models as data misses.
The controller alternates which revision runs first and reverses the case order
on alternating rounds. Raw results are checkpointed after the paired
availability pass and after each paired measured round.

The controller starts each revision's worker from its matching source checkout
and installed wheel. For a local one-point check:

```bash
python tools/forward_perf_gate/run.py \
  --base-python /path/to/base-venv/bin/python \
  --base-worker /path/to/base/tools/forward_perf_gate/worker.py \
  --base-revision BASE_SHA \
  --head-python /path/to/head-venv/bin/python \
  --head-worker /path/to/head/tools/forward_perf_gate/worker.py \
  --head-revision HEAD_SHA \
  --output-dir forward-perf-results \
  --smoke
```

Smoke mode selects one case and defaults to one round, one warmup call, and
three measured calls. Explicit `--rounds`, `--warmup`, or `--iterations`
values override their individual smoke defaults.

Land the worker and protocol before enabling the workflow. This ensures the
merge-base and head revisions both have a revision-local adapter.
