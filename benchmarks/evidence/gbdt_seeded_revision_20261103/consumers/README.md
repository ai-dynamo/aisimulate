# Unchanged Gym and Dynamo compatibility after the seeded revision

The revised package passes the unchanged Gym suite and every real-native Gym
scenario exercised here. All 13 scenario receipts are byte-identical to the
merged #658 control. The prior four A/A+C constructor failures are resolved.
The unchanged Dynamo adapter has no new failure in these checks: all 11 shared
successful cases match the control, and three additional bundle/snapshot cases
pass. Three existing Dynamo integration gaps reproduce on both versions.

## Builds and unchanged consumers

The control is merged prerequisite `9f4819e2b77afb4798bc8133c42ccc1d1b045e5c`,
with native SHA-256
`d88bc55746664e144057e57e17ee5357c08667ea52861627a7ad6a9949f74cd4`.
The candidate is the assembled seeded revision, with native SHA-256
`7fe75475aca65819a9174942ded464ff49060f082357753bcde5286a2028a0d0`.
[candidate-build.json](../candidate-build.json) records its actual dirty source
file hashes and build identity; the observed old HEAD alone does not identify
the tested implementation. [baseline-build.json](../baseline-build.json) records
the control. The harnesses verify the loaded SDK/native paths and binary hashes.
No dependencies or native binaries were changed during these consumer checks.

Gym source and tests come from the unchanged archive at
`a8b30ca6e1237f653e17835da76fa0f3eed839a8`; relevant Gym, dashboard and scenario
fixture files are identical at merged `9f4819e2`. Dynamo uses the unchanged
[318ab039 adapter and engine-query implementation](https://github.com/ai-dynamo/dynamo/tree/318ab0390d8e3e1fa7517ea6a1ebf46a7854c236/components/src/dynamo/planner/core/perf_model).
[source-integrity.json](source-integrity.json) verifies all 422 unique archived
consumer/source files. External Dynamo code and its original legal files remain
under `/private/tmp`; this evidence directory contains only the source manifest
and self-authored harnesses.

## Results

| Check | Revised candidate | Merged #658 control |
|---|---:|---:|
| Unchanged Gym suite | 691 passed | 691 passed |
| Real-native Gym scenarios | 13 passed | 13 passed |
| Unchanged Dynamo adapter/query unit tests | 31 passed | 31 passed |
| Real-native Dynamo cases | 14 passed, 3 existing failures | 11 passed, 3 existing failures |

Gym's real-native matrix exercises external C-only initialization, B+C with
18/20 seed rows, frozen B, online B, hybrid and selector variants, and original-A
plus A+C construction/manual mixed-store append. It preserves the wrapper's
worker/store isolation, prediction-before-admission order and update schedule.
The adapter code is unmodified. [The 13 receipt comparisons](gym/candidate-control-comparison.json)
include predictions and lifecycle diagnostics; all match exactly. The ordinary
691-test suite includes mocks and policy tests and is not by itself the native
compatibility evidence.

Dynamo's native cases exercise default regression for all three roles, deferred
capability publication, complete-rank attention-DP-2 updates, query and capacity
paths, capability-triggered reconstruction, and frozen/online legacy checkpoint
pass-through. Candidate-only cases cover frozen and online explicit bundles and
a complete fixed snapshot passed through the unchanged adapter. The independent
native harness has no mocks. The 31 upstream unit tests retain their existing
engine fakes and use a no-op import for unavailable Dynamo runtime logging.

[Dynamo comparisons](dynamo/candidate-control-comparison.json) show exact equality
for all shared successful predictions and lifecycle details after excluding
only measured grouping duration and the additive execution-readiness record.
Raw reports retain both fields. The candidate's additional execution decision
is not substituted for the unchanged adapter's own decision.

The three preserved failures are:

1. Decode and aggregated capacity at attention DP 2: Dynamo constructs a
   synthetic rank with zero decode requests and positive decode KV tokens. Both
   versions reject that invalid query, and capacity returns `None`. Complete-rank
   live tuning, direct prediction and reconstruction pass.
2. Aggregated learned readiness with only pure-prefill/pure-decode stores:
   Dynamo reports sufficient data although a mixed query has no estimate. The
   candidate additionally reports `execution.ready: false`. This is an existing
   consumer coverage issue, not a native training responsibility.

## Synthetic fixture and scope

The Dynamo fixture preserves the original self-authored aggregated two-store
checkpoint, identity and source provenance. [prepare_fixture.py](dynamo/prepare_fixture.py)
exports those exact checkpoint bytes through the revised public producer, so
the bundle honestly records its offline `training_recipe`. The product example
now uses a decode-only role; it cannot replace this deliberately incomplete
aggregated fixture. [fixture-provenance.json](dynamo/fixture-provenance.json)
records the original immutable revision and every seed digest. Both versions
use identical seed bytes; the control accesses them through the legacy checkpoint
API. No third-party dataset or measured hardware target is introduced.

These checks establish functional compatibility for the enumerated paths. They
do not qualify GPU serving accuracy, real production services, distributed
transport, rank synchronization or resource feasibility. Dynamo snapshot
coverage is a fixed query; full continuation and relocation are covered by the
separate native/SDK snapshot tests. External C collection, grouping and initial
training remain owned by Gym or the calling application.

## Reproduction and raw evidence

[commands-baseline.json](commands-baseline.json) and
[commands-candidate.json](commands-candidate.json) retain commands, paths, thread
limits and exit codes. [run_consumers.py](run_consumers.py) runs the same sources
against an explicitly selected SDK root and expected native hash; the unchanged
source trees must be available at their recorded temporary paths. The dedicated
fixture is included, so it need not be regenerated for a rerun.

Logs are `gym-unchanged-*.log`, `gym-real-native-*.log`, `dynamo-upstream-*.log`
and `dynamo-real-native-*.log`. Detailed native results are under
`gym/real-native/{baseline,candidate}` and `dynamo/{baseline,candidate}`.
The Dynamo harness intentionally exits nonzero for the three preserved failures.
An initial attempt using the previous bundle schema is retained under
[stale-curated-bundle-attempt](dynamo/stale-curated-bundle-attempt/README.md);
the final results use the dedicated verified fixture.
