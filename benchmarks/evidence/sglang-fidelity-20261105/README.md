# SGLang HiCache offline correctness and CPU profile

The integrated SGLang fidelity implementation completes a synthetic 48,000-request
offline soak across eight seeds, three HiCache write policies, and overlap enabled
and disabled. Every request completes with its requested output count. Debug and
release runs agree exactly on the recorded completion, output, first-admission
reuse, and terminal-time totals in [measurements.json](measurements.json).

The workload uses 32 GPU pages of 64 tokens, host capacities of 8–72 pages, twelve
prefix families across three salts, chunked prefill, capacity pressure, and both
zero-duration and bandwidth-limited transfers. The normal test suite also replays
24 smaller cells twice and compares full reports, per-request records, and
observer logs. These are simulated buffers and fixed model timings; no model
weights or inference hardware are required. This validates simulator behavior,
not agreement with measured serving latency.

## Timing and profile

Source: `27390abfb9c099effe4b88f8b64ed527e47d2fd9`, Rust 1.98.1, release
optimization level 3, one codegen unit, thin LTO, and debug information level 1.
Measurements ran on one physical core of an Intel Xeon Platinum 8559C; the workload
was pinned to logical CPU 2 and no task benchmark/build used its SMT sibling 6.
Builds and tests were stopped during measurements.

Six trials per A/A arm, in balanced ABBA order with 200 ms between trials, gave
medians of **2361.78 ms** and **2360.93 ms**. The ratio was **0.9996**, with a 95%
bootstrap interval of **0.9872–1.0051** (10,000 resamples, seed 1) and two-sided
Mann–Whitney p=0.575. The largest interval departure from 1.0 is about 1.3%.
All twelve runs had identical recorded simulation outcomes. The harness's
`elapsed_ms` excludes the wait for a periodic host service before each trial;
external process wall time includes it and is not used here.

A separate build of the same source forced frame pointers and unwind tables for
`perf` sampling at 999 Hz. Decoding with `--no-inline` retained all 7,743,853,968
sampled workload cycles; folded library shares agree with `perf report` and no
samples were lost. Earlier captures with incomplete callchains were discarded
from the analysis. The timestamp-loop subtree contains 93.68% of the workload
cycles and excludes setup and final report construction.

Within that subtree, self costs were approximately:

| Function | Sampled cycles |
| --- | ---: |
| libc `memmove` | 14.25% |
| `SglangCore::schedule_pass` | 14.10% |
| `EngineComponent::drive_ready` | 4.58% |
| libc `malloc` | 4.45% |
| `AggRuntime::drain_current_timestamp` | 4.23% |
| libc `_int_malloc` | 3.81% |

The inspected flamegraph attributes much of the copying to replay/engine pass
results flowing through `drive_ready` and `complete_pass`. Allocation costs are
spread across scheduler and replay work. This is a useful direction for a focused
result-representation experiment; it does not establish a safe local fix.
No performance patch was accepted and no A/B speedup is claimed. Simulator CPU
runtime is distinct from the virtual request latencies it predicts.

## Reproduce

From the repository root, build and freeze the executable reported by Cargo:

```sh
CARGO_PROFILE_RELEASE_DEBUG=1 \
  cargo test --release -p aisim-core --test sglang_fidelity --no-run
cp target/release/deps/sglang_fidelity-<cargo-reported-hash> ./sglang-soak-baseline
sha256sum ./sglang-soak-baseline
taskset -c 2 env AISIM_SOAK_REQUESTS=1000 AISIM_SOAK_SEEDS=8 \
  AISIM_SOAK_RESULT=baseline.json ./sglang-soak-baseline \
  extended_offline_soak_profile --exact --ignored --nocapture
```

Choose an available physical core and exclude competing work on it and its SMT
sibling. Repeat the unchanged binary as two arms in ABBA order before comparing
any candidate. Use the harness's internal `elapsed_ms`; retain all outcome fields
and require their agreement. The JSON records every A/A elapsed value, binary
hashes, build flags, profile checks, and workload parameters.

For callchains, rebuild separately with
`RUSTFLAGS="${RUSTFLAGS:-} -Cforce-frame-pointers=yes -Cforce-unwind-tables=yes"`,
freeze that executable, and run the same workload under:

```sh
perf record -F 999 -e cycles:u --call-graph dwarf,32768 -o soak.perf -- \
  taskset -c 2 env AISIM_SOAK_REQUESTS=1000 AISIM_SOAK_SEEDS=8 \
  ./sglang-soak-profile extended_offline_soak_profile --exact --ignored --nocapture
perf report -i soak.perf --comms extended_offlin --percentage relative \
  --no-children --sort dso --stdio
perf script -i soak.perf --comms extended_offlin --no-inline > soak.script
```

Check decoded cycle totals and library shares before interpreting a flamegraph.
Keep the profiling build separate from unprofiled timing comparisons.
