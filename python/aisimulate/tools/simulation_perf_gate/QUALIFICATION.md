# Simulation performance validation

## Current timing-only gate (protocol v4)

The current gate runs five timing pairs for each of 12 cases. It checks completed
work, input/model identity, and valid measurements. It has no equivalence phase,
per-request capture, request-record artifacts, or large-cache control replay.
Simulated-result differences do not change the timing verdict.

### CI workload-size trial

[Run 36936991854](https://github.com/ai-dynamo/aisimulate/actions/runs/36936991854)
built revisions `8f0eed87` and `b5285ec9` separately on
`prod-aisimulate-default-amd64-v1`, using Python 3.12.15 and Rust 1.96.1.
The base controller tested original counts; the head controller tested half
counts for all ten synthetic cases. Both complete AgentX plays were unchanged.
Both matrices passed all 12 cases and five measurement rounds.

This protocol-v3 trial still included the old equivalence checks. Its records
provide separate workload evidence; those checks are removed from the v4 gate.
The table uses the lower of the base/head native medians. Cache counts are
four-turn sessions; all other synthetic counts are requests.

| Case | Unit | Original count | Half count | Original minimum median (ms) | Half minimum median (ms) | Frozen count |
|---|---|---:|---:|---:|---:|---:|
| dense-vllm | requests | 131072 | 65536 | 5130.6 | 2823.1 | 65536 |
| dense-sglang | requests | 16384 | 8192 | 1586.4 | 966.6 | 16384 |
| dense-trtllm | requests | 131072 | 65536 | 5937.1 | 3264.5 | 65536 |
| moe-long-prefill | requests | 16384 | 8192 | 2820.8 | 1628.5 | 16384 |
| moe-long-decode | requests | 4096 | 2048 | 3137.3 | 1746.6 | 4096 |
| cache-pressure-vllm | sessions | 4096 | 2048 | 2417.7 | 1464.5 | 4096 |
| cache-pressure-sglang | sessions | 2048 | 1024 | 1848.8 | 1108.0 | 2048 |
| mla-multiworker-dp | requests | 32768 | 16384 | 2565.1 | 1589.4 | 32768 |
| pd-vllm | requests | 65536 | 32768 | 3927.1 | 2244.4 | 32768 |
| pd-sglang | requests | 8192 | 4096 | 1194.1 | 789.2 | 8192 |

Only dense vLLM, dense TRT-LLM, and P/D vLLM retain half counts. All other counts
are restored. Concurrency, token lengths, models, backend settings, cache
capacity, session structure, and worker topology are unchanged. Neither side
resizes a workload during a comparison.

The half-count trial retained cache reuse (25.56% vLLM; 18.47% SGLang) and
additional prefill under pressure (9,543,602 and 5,810,555 tokens). Every P/D
request activated its decode destination: 32,768 vLLM requests, 4,096 SGLang
requests, and all 129 AgentX P/D requests. The shortened P/D vLLM case used its
one prefill and both decode workers. The MLA trial reached two logical workers,
eight DP ranks, and 16 schedulers on both revisions. Both complete AgentX cases produced
114,540 output tokens. The three original SGLang workloads were already below
two seconds; none was reduced.

### Normal CI before/after

Both runs used the ordinary base/head path on
`prod-aisimulate-default-amd64-v1`, with two distinct compatible revisions,
separate release wheels and environments, 12 cases, and five timing pairs.
Each head adds only a package README comment to its base, so the native work is
unchanged within each comparison while source and wheel checksums differ.
Neither run used self-comparison, qualification, or a second controller run.

- Before: [run 36940629153](https://github.com/ai-dynamo/aisimulate/actions/runs/36940629153),
  protocol v2, base `84e028e8`, head `cf2001bf`. The retained serial workflow
  only separated build/install timing and pinned the original Rust version.
- After: [run 36943181943](https://github.com/ai-dynamo/aisimulate/actions/runs/36943181943),
  protocol v4, base `49280bbe`, head `f3ae975b`. This uses parallel builds,
  the timing-only gate, and the three frozen count reductions.

Both used Python 3.12.15, Rust 1.99.0, maturin 1.15.0, uv 0.11.3, the repository's
CI container setting, and `maturin build --release --locked`. Both passed all 12
cases. Final wheel revisions, wheel/requirements checksums, build settings, and
model/data trees were checked again from the downloaded artifacts.

| Critical-path time | Before | After |
|---|---:|---:|
| Runner queue | 32 s | 114 s |
| Selection, setup, artifacts, cleanup, and scheduling gaps | 87 s | 157 s |
| Build | 462 s, serial | 221 s, parallel |
| Install and verification | 92 s | 88 s |
| Benchmark step | 964 s | 585 s |
| **Total** | **1637 s (27m 17s)** | **1165 s (19m 25s)** |

For parallel jobs, the table follows the head build, which finished last; the
base/head build steps took 212/221 seconds. Their runner queues were 24/81
seconds. Selection and comparison queues were 8/25 seconds. The other-time row
is the remaining workflow interval, so the columns add to total elapsed time.
The before-run queues were 8 seconds for selection and 24 seconds for comparison.
Controller-recorded benchmark times were 963.179 and 584.740 seconds.

The observed normal-CI saving was **472 seconds (7m 52s), or 28.8%**. The benchmark
step saved 379 seconds (39.3%). This is one before/after pair; queue and setup
time can vary. No qualification-job duration contributes to this result.

All three shortened cases remained above two seconds in the final normal run:

| Shortened case | Base native median (ms) | Head native median (ms) |
|---|---:|---:|
| dense-vllm | 2845.7 | 2842.2 |
| dense-trtllm | 3296.7 | 3230.0 |
| pd-vllm | 2167.4 | 2171.5 |

The final result contains only measurement rounds and no request-record
artifacts. Both builds finished before comparison began. Evidence is retained in
`build/simulation-perf-v3/ci-{serial-baseline,final-normal,final-wheels}/`,
`{serial-baseline,final-normal}-timings.txt`, `run-{36940629153,36943181943}/`,
and `final-audit.txt`.

Main still lacks the benchmark adapter. The primary PR's
[automatic run 36939124432](https://github.com/ai-dynamo/aisimulate/actions/runs/36939124432)
reported the missing base and skipped both builds and comparison. The separate
compatible-revision runs above validate the normal path without hiding that skip.

### Focused checks and slowdown controls

348 benchmark/workflow tests pass. They cover five timing pairs with alternating
order, strict protocol and input handling, incomplete work, invalid model
provenance, non-finite diagnostics, artifact revision/checksum checks, revision
preparation, and all three self-comparison invocations. Python lint/format,
workflow lint, packaged legal checks, and strict CODEOWNERS coverage pass.

A local run with original counts passed all 12 cases and five timing pairs in
321.6 seconds, excluding builds. It used one existing installation on both
sides; it is not a normal-CI timing comparison.

Each retained shorter workload also ran a real CPU-contention control. The head
worker started a competing process on its assigned CPU; reported timings were
not modified. All three controls detected a regression in five of five rounds:

| Shortened case | Base native median (ms) | Contended head median (ms) | Verdict |
|---|---:|---:|---|
| dense-vllm | 1337.7 | 2665.7 | PERFORMANCE_REGRESSION |
| dense-trtllm | 1496.1 | 2944.7 | PERFORMANCE_REGRESSION |
| pd-vllm | 1078.7 | 2175.5 | PERFORMANCE_REGRESSION |

Local evidence is retained in `build/simulation-perf-v3/`: `timing-v4-full/`,
`controls/short-slow/`, `controls/pd-short-slow/`, and `final-tests.log`.
The two-second count decision uses CI medians, not these faster local timings.

## Historical local qualification (protocols v1/v2)

The records below describe earlier versions with equivalence checks. They do
not describe the current timing-only gate. Old artifacts have not been rewritten.

The full-suite measurements below were collected with protocol v1. The protocol
v2 review checks are recorded at the end; they do not replace CI qualification.

All 12 cases passed three complete same-revision comparisons using separate
release-wheel installations. These are local results, not acceptance on the
`prod-aisimulate-default-amd64-v1` CI runner. Automatic selection follows the
trusted PR workflow; these measurements do not establish CI runner acceptance.

Host: Intel(R) Core(TM) Ultra 9 285K; CPU 0; Python 3.12.12;
rustc 1.96.1 (31fca3adb 2026-06-26). Each worker used the pinned single-thread environment.
Both environments had the same 79 installed package versions.

| Comparison | Elapsed, including equivalence and artifacts | Result | Lowest synthetic replay median |
|---|---:|---|---:|
| 1 | 589.5 s | 12/12 PASS | 2137.5 ms |
| 2 | 589.4 s | 12/12 PASS | 2137.0 ms |
| 3 | 588.5 s | 12/12 PASS | 2104.5 ms |

Builds are excluded. Every comparison used one detailed equivalence pass and
five paired measurement rounds. There were no false regressions, behavior
changes, or invalid comparisons. Both installations completed the full input.

### Frozen workload sizes

Native replay values below span the six base/head medians from the three runs.
Concurrency, request lengths, and cache capacity remained fixed while sizing.

| Case | Requests | Sessions | Native replay median range |
|---|---:|---:|---:|
| dense-vllm | 131072 | — | 3049.6–3224.0 ms |
| dense-sglang | 16384 | — | 2883.4–2906.8 ms |
| dense-trtllm | 131072 | — | 3279.5–3391.8 ms |
| moe-long-prefill | 16384 | — | 2399.7–2425.6 ms |
| moe-long-decode | 4096 | — | 2252.1–2265.7 ms |
| cache-pressure-vllm | 16384 | 4096 | 2330.8–2374.8 ms |
| cache-pressure-sglang | 8192 | 2048 | 2331.8–2363.3 ms |
| mla-multiworker-dp | 32768 | — | 2104.5–2151.1 ms |
| pd-vllm | 65536 | — | 2606.0–2618.2 ms |
| pd-sglang | 8192 | — | 2237.7–2259.2 ms |
| agentx-vllm-aggregated | 129 | — | 2508.6–2539.3 ms |
| agentx-sglang-disaggregated | 129 | — | 2605.2–2632.8 ms |

Sizing started at 256 requests or 64 four-turn sessions and used doubling.
An initial calibration comparison found dense-vllm below two seconds; its
count was doubled from 65,536 to 131,072 before these three accepted runs.
The AgentX play stayed intact. No cases or thresholds were removed or relaxed.

### Coverage and controls

- `cache-pressure-vllm`: 25.64% cache reuse and 19,243,570 additional committed prefill tokens versus the large-cache control.
- `cache-pressure-sglang`: 18.34% cache reuse and 11,609,009 additional committed prefill tokens versus the large-cache control.
- `pd-vllm`: all 65,536 requests reached destination activation.
- `pd-sglang`: all 8,192 requests reached destination activation.
- `agentx-sglang-disaggregated`: all 129 requests reached destination activation.

The slowdown control ran a competing process on the head worker's CPU during
the real AgentX replay. It returned `PERFORMANCE_REGRESSION`: 5/5
rounds exceeded both thresholds; median native replay increased from
2508.8 to 5073.9 ms. Simulated results matched.

The behavior control changed the head worker's scheduler limit from 256 to 32
sequences for dense-vllm. Both sides completed, and the result was
`BEHAVIOR_CHANGED`. The comparator did not call it a confirmed speed regression.

### Reproduction and provenance

Source base: `e95d37863fb30232e089b21f058bc28d1dfa1729` plus this change.
Release wheel SHA-256: `7888b7e27057566daa75327bd22bcce13f2aa381e987aa4ea65dd25e6d55550d`.
Case-set SHA-256: `aaff48234f3efc590cb57a81a6c73a94796b75ba223b022c262e15d968657783`.
Model/data Git tree: `9e8a83f44b4cd8313d119ae67a6bb6a627c54441`.

Local evidence is retained under `build/simulation-perf/`: `qualification-{1,2,3}`
contain raw JSON, reports, inputs, compressed request records, and worker logs;
`control/` contains the control wrappers and results; `provenance.json`, wheel
hashes, and package manifests identify the tested installations.
The final comparator was also applied to the saved results.

Validation: 686 affected Python/native and workflow tests passed. Ruff lint
and format, Rust format, actionlint, strict CODEOWNERS coverage and generated
file checks, packaged legal-file checks, and diff whitespace checks passed.

Repeat qualification and controls on the CI runner. See [README.md](README.md)
for the protocol, commands, and manual workflow dispatch. The local pass does
not establish CI runner acceptance.

### Protocol v2 review validation

The fixed comparison fields were checked against all retained request records
and summaries from the three local qualifications and both controls. All 38
case comparisons kept their expected classifications: 36 `PASS`, one
`PERFORMANCE_REGRESSION`, and one `BEHAVIOR_CHANGED`. The audit converted the
protocol tag in memory to exercise v2 validation on the old evidence; production
comparisons still reject protocol mismatches. These are not new measurements.

A live check used the updated controller and worker with the two existing
release-wheel installations. Dense vLLM, MLA multiworker DP, P/D SGLang, and
AgentX P/D SGLang all passed one equivalence pass and one paired measurement
round. Total elapsed time was 79.8 seconds. This is a smoke check, not the
five-round full-suite qualification or CI runner acceptance.

Validation: 324 focused benchmark and workflow tests passed. Ruff lint/format,
actionlint, and diff whitespace checks passed. Evidence remains under
`build/simulation-perf/review-{tests.log,recheck.json,recheck.log,smoke/}`.
Qualification on the CI runner remains pending for the final implementation
with unchanged workloads and thresholds.
