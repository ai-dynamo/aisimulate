# Local simulation-performance qualification

All 12 cases passed three complete same-revision comparisons using separate
release-wheel installations. These are local results, not acceptance on the
`prod-aisimulate-default-amd64-v1` CI runner. Automatic CI remains disabled.

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

## Frozen workload sizes

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

## Coverage and controls

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

## Reproduction and provenance

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

Before enabling `SIMULATION_PERF_ENABLED`, repeat qualification and controls
on the CI runner. See [README.md](README.md) for the protocol, commands, and
manual workflow dispatch. The local pass does not remove this rollout gate.
