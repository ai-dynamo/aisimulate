# Unchanged consumers after construction verification reuse

The optimized native build preserves all results from the previous seeded
builds. The original Gym and Dynamo adapter sources were not modified.

| Check | Optimized candidate | Previous candidate | Merged #658 control |
|---|---:|---:|---:|
| Unchanged Gym suite | 691 passed | 691 passed | 691 passed |
| Real-native Gym scenarios | 13 passed | 13 passed | 13 passed |
| Unchanged Dynamo adapter/query tests | 31 passed | 31 passed | 31 passed |
| Real-native Dynamo cases | 14 passed, 3 existing failures | 14 passed, 3 existing failures | 11 passed, 3 existing failures |

The optimized extension SHA-256 is
`6a12ba8d1af12a1930d8a030a400b22aa2aacd439f458002488adb71a2f42d42`.
[optimized-origins.json](optimized-origins.json) records loaded module paths,
module hashes and installed dependencies. The exact source and build inputs are
in [candidate-optimized-source.json](../candidate-optimized-source.json) and
[candidate-optimized-build.json](../candidate-optimized-build.json). The old
observed working-tree HEAD alone does not identify the assembled implementation.
The [optimization proof](../construction-receipts-native-delta.json) and
[Rust checks](../optimized-rust-checks.json) separately record the source changes,
429 passing FPM tests and 26 passing public API tests.

All [13 Gym receipts](gym/receipt-comparison.json) are byte-identical to the
previous `73bc10ea` and `7fe75475` builds and merged control `9f4819e2` (native
`d88bc557`). They cover caller-managed C-only initialization, B+C at 18/20 seed
rows, frozen/online B, hybrid/selector variants and original-A/A+C with manual
mixed-store append. Wrapper schedules and prediction-before-admission behavior
remain unchanged.

[Dynamo comparisons](dynamo/receipt-comparison.json) show all 14 successful
cases equal both previous candidates after excluding only measured grouping
duration. All 11 shared successful control cases also match after excluding
grouping duration and the candidate's additive execution-readiness record.
Raw receipts retain these fields. The candidate additionally exercises explicit
frozen/online bundles and fixed snapshot transport through the unchanged adapter.

The same three failures remain: attention-DP-2 decode and aggregated capacity
construct a rank with zero decode requests and positive decode KV tokens, and
an aggregated learned adapter reports sufficient data when its seed lacks the
locally mixed store. Both failures reproduce on the control; the candidate's
native execution readiness additionally identifies the missing store. The
native harness intentionally exits nonzero rather than hiding these cases.

[Source verification](source-verification.json) rechecks all 422 unique pinned
consumer files after the run. Gym is pinned at
`a8b30ca6e1237f653e17835da76fa0f3eed839a8`, with relevant files verified identical
to merged `9f4819e2b77afb4798bc8133c42ccc1d1b045e5c`; Dynamo is pinned at
`318ab0390d8e3e1fa7517ea6a1ebf46a7854c236`. External source and original legal
files stay in the recorded temporary directories. All four self-authored
harnesses are byte-identical to the previous run; [summary.json](summary.json)
records their digests. [Input references](input-references.json) bind the reused
source manifests and dedicated self-authored two-store fixture. The fixture's
seed bytes are identical across candidates and control.

[The original report](../consumers/README.md) documents the complete case scope,
control results and fixture provenance; [the preceding build report](../consumers-final/README.md)
retains the `73bc10ea` results. These are synthetic functional compatibility
checks, not serving-accuracy or host-cost qualification. The ordinary unit
suites retain their original fakes; the separate native harnesses use the actual
SDK and extension. No production Dynamo service or distributed runtime was
started. Dynamo snapshot coverage here is fixed-query transport; complete
continuation and relocation have separate native/SDK tests. External C
collection, grouping and initial fitting remain caller responsibilities.

The checks ran alongside other correctness tests, so their elapsed times are
not performance evidence. No dependencies, native binaries or consumer sources
were changed during this run. Exact commands and exit codes are in
[commands-candidate.json](commands-candidate.json); raw logs and per-scenario
receipts are retained here. Reproduce with the recorded temporary sources:

```sh
python3 benchmarks/evidence/gbdt_seeded_revision_20261103/consumers-optimized/run_consumers.py candidate \
  --sdk-root /Users/yueqiz/Documents/ChatGPT/AISimulate/work/gbdt-implementation \
  --native-sha256 6a12ba8d1af12a1930d8a030a400b22aa2aacd439f458002488adb71a2f42d42
```
