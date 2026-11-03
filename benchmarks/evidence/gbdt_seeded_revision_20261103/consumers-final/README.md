# Final native build: unchanged consumer compatibility

The final native build preserves the results of the earlier seeded revision:
691 unchanged Gym tests, all 13 real-native Gym scenarios and 31 unchanged
Dynamo unit tests pass. The Dynamo real-native harness passes 14 cases and
retains the same three failures reproduced on the merged #658 control.

| Check | Final candidate | Previous seeded candidate | Merged #658 control |
|---|---:|---:|---:|
| Unchanged Gym suite | 691 passed | 691 passed | 691 passed |
| Real-native Gym scenarios | 13 passed | 13 passed | 13 passed |
| Unchanged Dynamo adapter/query tests | 31 passed | 31 passed | 31 passed |
| Real-native Dynamo cases | 14 passed, 3 existing failures | 14 passed, 3 existing failures | 11 passed, 3 existing failures |

The final native SHA-256 is
`73bc10ea8bbe3f2e7936d56d0e7ec25be77f792e34c32cf59386e0adcf5d491c`.
The previous seeded build is
`7fe75475aca65819a9174942ded464ff49060f082357753bcde5286a2028a0d0`.
The control is merged prerequisite
`9f4819e2b77afb4798bc8133c42ccc1d1b045e5c`, with native SHA-256
`d88bc55746664e144057e57e17ee5357c08667ea52861627a7ad6a9949f74cd4`.
[final-origins.json](final-origins.json) records actual module paths, module and
native digests, and installed package versions. The observed working-tree HEAD
is recorded for context; the assembled implementation is identified by the
source proof and binary digest, not that older HEAD alone.

All [13 Gym receipts](gym/receipt-comparison.json) are byte-identical to both
the previous seeded candidate and merged control. The real-native scenarios
exercise external C-only initialization, B+C at 18/20 seed rows, frozen and
online B, hybrid and selector variants, and original-A/A+C construction with
manual mixed-store append. Gym's original wrappers, scheduling and
prediction-before-admission order are unchanged.

[Dynamo receipt comparisons](dynamo/receipt-comparison.json) show all 14
successful final cases equal the previous seeded candidate after excluding only
measured grouping duration. All 11 shared successful control cases also match,
excluding grouping duration and the candidate's additive native execution
readiness. Raw results retain those fields. The three additional candidate
cases cover explicit frozen/online bundles and fixed snapshot transport through
the unchanged adapter.

The preserved failures are `regression_attention_dp2_capacity_decode`,
`regression_attention_dp2_capacity_aggregated` and
`learned_required_store_guard`. The two capacity cases construct a synthetic
rank with zero decode requests and positive decode KV tokens; both versions
reject that query. The learned case intentionally has pure-prefill/pure-decode
stores without a locally mixed store: the unchanged adapter reports sufficient
data, but the mixed query has no estimate. Native execution readiness correctly
reports false on the candidate. These failures are unchanged from the earlier
seeded run and reproduce on the merged control. The harness deliberately exits
nonzero to preserve them.

[Source verification](source-verification.json) rechecks all 422 unique archived
consumer source files after this final run. Gym uses unchanged
`a8b30ca6e1237f653e17835da76fa0f3eed839a8` files verified identical at merged
`9f4819e2`; Dynamo uses unchanged
`318ab0390d8e3e1fa7517ea6a1ebf46a7854c236`. External source and original legal
files remain in the recorded temporary directories. [Input references](input-references.json)
bind the existing source manifests and dedicated self-authored aggregated
fixture, avoiding duplicate input copies. The same two-store checkpoint is
used for both versions; the control uses the legacy checkpoint API and the
candidate uses its validated bundle. No product example or external dataset was
substituted.

[The previous report](../consumers/README.md) contains the original control
results, full case scope and fixture provenance. This rerun does not claim GPU
serving accuracy, real service execution, distributed transport or resource
feasibility. Upstream unit tests retain their original engine fakes; the separate
real-native harnesses exercise the actual candidate SDK and extension. Dynamo
snapshot coverage here is a fixed query; snapshot continuation and relocation
have separate native/SDK coverage. External C collection and initial fitting
remain the responsibility of Gym or the calling application.

Reproduce from the implementation checkout with the recorded temporary source
trees available:

```sh
python3 benchmarks/evidence/gbdt_seeded_revision_20261103/consumers-final/run_consumers.py candidate \
  --sdk-root /Users/yueqiz/Documents/ChatGPT/AISimulate/work/gbdt-implementation \
  --native-sha256 73bc10ea8bbe3f2e7936d56d0e7ec25be77f792e34c32cf59386e0adcf5d491c
```

[commands-candidate.json](commands-candidate.json) records exact subprocess
commands, thread limits and exit codes; [summary.json](summary.json) binds the
self-authored harness digests. The four raw logs and per-scenario receipts are
retained beside this report. No dependency installation or native rebuild took
place during these checks.
