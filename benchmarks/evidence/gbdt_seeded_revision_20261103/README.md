# Seeded GBDT revision evidence

A later review fixed bulk original-A manifest handling and inherited onboarding
controls in PR1, with a relocation regression in PR3. See the
[follow-up verification](../gbdt_review_fixes_20261103/README.md) for the rebuilt
`d7c69b78` extension and fresh checks. The campaigns below retain their original
build identities; their results have not been relabeled as runs of the follow-up.

This directory records qualification of the revised seeded workflow based on
merged prerequisite `9f4819e2b77afb4798bc8133c42ccc1d1b045e5c` (PR #658).
It is separate from the earlier native-initialization experiment and from the
compatibility audit of the unrewritten stack. Neither historical report qualifies
this revision. Historical evidence is preserved separately as described below.

## Revised four-PR delivery

| Branch | Proposed title | Commit | Scope |
| --- | --- | --- | --- |
| `codex/gbdt-source-bundles` | `feat: add pinned learned bundles and seeded workflows` | `98a2998fc38263fed96bd71dd779717c95c3da40` | Explicit bundle production/export/inspection/install, exact source identity and provenance, optional catalogue discovery and qualified prior, legacy raw-source compatibility and seeded onboarding integration. |
| `codex/gbdt-native-bootstrap` | `feat: expose learned execution readiness to consumers` | `bc35afdfb4159000dd616132cac085972c527916` | Additive readiness/dependency information and explicit learned caller coverage checks, preserving seeded numerical behavior and existing diagnostics. The branch name is historical; native initial training is removed. |
| `codex/gbdt-complete-snapshots` | `feat: preserve complete learned snapshots and continuation` | `3630fed8e30ec24edee78f569b9d7276e60ddb4a` | Complete seeded state/source/prior closure, fixed/continued restore and relocation, old snapshot compatibility, and caller consumption. |
| `codex/gbdt-adoption-examples` | `feat: add reproducible opt-in learned examples` | Identified by branch/scope; no self-referential commit hash. | Decode-only B→externally supplied C→B+C example, trainer-free packaged verification, supported optional trainer, installed-package checks and application ownership guidance. |

The default remains `auto`. Applications own telemetry, rank synchronization,
sampling, deduplication, chronology, initial fitting, stream offsets and deployment
promotion. AISimulate provides seeded updates and model snapshots. Existing Gym
protocols and Dynamo caller choices do not require migration merely to preserve
their behavior. The optional onboarding path keeps all six stages, generated
interpolation choices and separate memory, coverage and serving-accuracy evidence.

## Qualification status

Core runtime, legacy, consumer, real-data and host-cost evidence accompanies PR3.
PR4 adds example/installed-wheel evidence, consolidated adoption guidance and
the final reference ledger; it is not a prerequisite for the earlier feature
PRs' runtime qualification.

Pre-review PR1–3 runtime qualification was completed on native 6a12ba8d, with the exact build
and earlier campaign identities distinguished below. Final PR4 source generation
pins the producing PR3 revision and passes its public lifecycle/verification
checks. The recorded pre-review application wheel also passes the installed end-to-end workflow,
with a native extension byte-identical to qualified build 6a12ba8d.

The assembled candidate native
`7fe75475aca65819a9174942ded464ff49060f082357753bcde5286a2028a0d0`
has the recorded results below. This is a working-tree build described by
[candidate-build.json](candidate-build.json), not a clean build of its observed
old HEAD. Counts identify their individual scopes and must not be summed as a
disjoint test total. They do not qualify each PR independently.

| Check | Recorded result | Evidence |
| --- | --- | --- |
| Native FPM and public API | 424 and 26 passed | [FPM](fpm-rust.log), [public API](public-api-rust.log) |
| Current SDK tests | 922 passed | [SDK](sdk-final.log) |
| Archived prerequisite SDK tests, unchanged | 192 passed | [Legacy SDK](legacy/unchanged-sdk-tests.log) |
| Caller/sweeper checks | 1,079 passed, 5 integration-CI skips | [Callers](callers-elevated.log) |
| Native engine/compile parity | 389 passed; no golden changes against merged prerequisite | [Parity](parity.log) |
| Snapshot regressions | 57 passed | [Snapshots](snapshot-regressions.log) |
| Actual old snapshot/online artifacts | Six old model variants relocate and continue identically; old online checkpoint continuation passes | [Cross-version](legacy/cross-version-results.json) |
| Onboarding plan rerun after pin-consumer fixes | 158 passed | [Plan](support-plan-dependencies-final.log) |
| Unchanged Gym | 691 tests and 13 real-native scenarios pass on both builds; all 13 receipts identical | [Consumers](consumers/README.md) |
| Unchanged Dynamo | 31 unit tests pass on both; 11 shared native cases match, three candidate bundle/snapshot cases pass; three pre-existing failures reproduce on both | [Consumers](consumers/README.md) |
| Full36 C/BC and HF smoke accuracy | All 74 cells complete; 6,407,633 records have identical prediction bits/availability and admission payloads; no prediction/tuning errors | [Accuracy](accuracy/README.md) |
| Dashboard browser fixtures | Both accuracy and scenario checks pass | [Accuracy browser](browser-accuracy.log), [scenario browser](browser-scenarios.log) |
| PR4 source preparation on 7fe75475 | Ten B→C→B+C lifecycle checks, complete decode readiness, trainer-free verification and optimized-Python tamper rejection pass; retained as preparation evidence | [Examples](examples/README.md) |

The exact accuracy result preserves the baseline's caller-managed initial fit,
coverage limitations and numerical results. It is not a new accuracy claim. All
36 cases, including thirteen M3 cases, remain present. Synthetic example results
are functional evidence and do not qualify actual model/hardware deployment.

The dependency-verifier build is
`73bc10ea8bbe3f2e7936d56d0e7ec25be77f792e34c32cf59386e0adcf5d491c`, recorded in
[candidate-final-source.json](candidate-final-source.json). The
[native source-delta proof](final-verifier-native-delta.json) identifies only two
added explicit dependency-verifier methods in runtime Rust; all other recorded
runtime Rust/Cargo bytes match the accuracy-tested build. This is source evidence,
not a second full36 measurement or a performance result.

| Dependency-verifier build follow-up | Result | Evidence |
| --- | --- | --- |
| SDK | 922 passed again | [Final SDK](final-verifier-sdk.log) |
| Learned onboarding and dependency checks | 22 passed, including added/removed prior dependencies and structured incomplete assessment | [Final support](final-verifier-support.log) |
| Complete onboarding support suite | 1,916 passed, including the 22 learned-support tests | [Onboarding](onboarding-final.log) |
| Unchanged consumer suite and native cases | Gym 691/13 and Dynamo 31/14 pass; all prior successful receipts match; the same three Dynamo failures remain | [Final consumers](consumers-final/README.md) |
| Static checks | 36 changed Python files, Rust formatting and packaged legal checks pass | [Static checks](final-static-checks.json) |
| Pre-optimization synthetic host costs | Five pairs per case; warm queries/updates preserved; duplicated cold-boundary verification identified for correction | [Host costs](host-costs/README.md) |

The directory-addition failure in
[learned-onboarding-final.log](learned-onboarding-final.log) is retained as the
reason for the dependency-verification fix and is superseded by the successful
22-test final-build rerun. Earlier plan failures remain recorded beside the
successful 158-test rerun. The first matched host-cost measurements identified
duplicate source verification. Those 73bc10ea measurements remain pre-optimization
evidence and are not relabeled as final costs.

The optimized native build is
`6a12ba8d1af12a1930d8a030a400b22aa2aacd439f458002488adb71a2f42d42`, recorded in
[candidate-optimized-source.json](candidate-optimized-source.json). The
[construction source delta](construction-receipts-native-delta.json) records the
narrow reuse of path/digest/length receipts from bytes actually read and verified
during the same construction or restoration. It preserves independent identity,
manifest, payload and extra-dependency checks; it adds no global cache or deferred
source pin. The original 7fe75475 full36 accuracy campaign retains its actual
native identity. Source-delta evidence and the following reruns bridge the later
boundary-only changes; they do not constitute another full36 measurement.

| Optimized-build follow-up | Result | Evidence |
| --- | --- | --- |
| Native FPM and public API | 429 and 26 passed, including receipt and tamper regressions | [Rust checks](optimized-rust-checks.json) |
| Current SDK | 922 passed | [SDK](optimized-validation/sdk.json) |
| Learned onboarding/dependency checks | 22 passed | [Support](optimized-validation/learned-support.json) |
| Snapshot regressions | 57 passed again | [Snapshots](optimized-validation/snapshot.json) |
| Unchanged prerequisite SDK | 192 passed again | [Legacy SDK](optimized-validation/unchanged-sdk.json) |
| Actual old snapshot/online artifacts | Six schema-2 variants relocate and continue identically; old version-5 checkpoint continuation passes | [Snapshots](optimized-validation/old-artifacts-results.json), [continuation](optimized-validation/old-continuation-results.json) |
| Unchanged consumers | Gym 691/13 and Dynamo 31/14 pass; successful receipts preserved and the same three existing Dynamo failures remain | [Consumers](consumers-optimized/summary.json) |
| Final matched host costs | Five fresh pairs per case; redundant scans removed; remaining complete-snapshot save/restore cost and warm-query variation reported | [Final costs](host-costs-optimized/README.md) |

The [PR3 source proof](pr3-source-proof.json) verifies all 268 native build-input
bytes against the optimized tested build. Its matched cost run records complete
snapshot restore overhead of 0.41–0.87 ms median and save overhead of 2.87–14.03 ms
for the tested closures. Warm SDK paired medians range 0.985–1.040; retained
training-state and bare checkpoint bytes match baseline. These synthetic costs
do not establish serving accuracy or a universal size/performance guarantee.

Final PR4 source regeneration uses producing PR3
`3e6e82c1b67e4448da01d60ae9d06893d1f8489f` and native 6a12ba8d. All ten lifecycle
checks and seven packaged verification checks pass, including complete decode
readiness, trainer-free verification and optimized-Python tamper rejection.
The [byte comparison](examples/final-artifact-determinism.json) confirms unchanged
bundle/seed/input bytes; final provenance is recorded separately. The final
[installed-wheel workflow](examples/installed-wheel.json) passes outside the
checkout, including source/snapshot APIs, the B+C example, trainer-free
verification and tamper rejection. Wheel SHA-256:
`f8cc4bd57e8d8a9c08af28d345dd129bdf2eea548308e3cc6dc6c9146c62495e`.
Its native extension exactly matches the qualified 6a12ba8d binary. The
[example report](examples/README.md) retains build/command receipts and the
non-product collection setup correction.

Only fresh evidence is included in this revised delivery. The earlier stack is
preserved in local `codex/archive-gbdt-*` refs, including
`codex/archive-gbdt-adoption-examples` at
`3cb6679d065f1ccfe3d4bbdd0e321a5a78238957`. Historical evidence directories are
also retained separately under `work/gbdt-historical-evidence-20261103` in the
local repository workspace. Their build identities and results remain historical
and must not be relabeled as tests of this revision.

[evidence-relocation.json](evidence-relocation.json) records the archive commits,
verbatim copies of the two required harnesses and full36 specification, and the
host-harness source provenance. Existing command/result paths remain unchanged
as execution facts; current reproduction commands use the copies in this
directory. Uncommitted local artifacts are pinned by exact hashes without
inventing an archive commit for them. Original measured inputs and the verified
HF cache remain external; all required hashes are retained.
