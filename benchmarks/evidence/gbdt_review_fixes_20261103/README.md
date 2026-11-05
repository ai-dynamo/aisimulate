# GBDT review regression fixes

The review diff retains summaries and selected verification receipts. The complete
raw results, logs, harnesses and source inventories remain available at the
[immutable evidence commit](https://github.com/ai-dynamo/aisimulate/tree/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_review_fixes_20261103).
Archived links below point to those exact bytes; no measurements were rerun or
relabelled during this reduction. To reproduce an archived campaign, check out
`3630fed8e30ec24edee78f569b9d7276e60ddb4a` in a separate worktree and follow
its recorded commands and input hashes. This preserves the original relative
layout required by its scripts. Private evaluation inputs remain external.

This follow-up fixes two regressions in the local four-PR stack. The pre-review
head was `497480757978d631433b26c5daa39b85e462a339`, on merged #658
`9f4819e2b77afb4798bc8133c42ccc1d1b045e5c`. Both production fixes belong in PR1;
the complete-snapshot regression is in PR3. The four dependent branches are
restacked locally. Earlier commits and qualification retain their own identities.

## Fixes

1. **Original-A manifests larger than 16 MiB.** Legacy mixed-fallback construction
   and portable remapping now stream the bulk manifest, retaining only dependency
   references and computing its digest and length from the same input stream.
   Small metadata keeps its 16 MiB limit. Duplicate keys, malformed input, changed
   tails, and dependency mismatches remain errors. A real observation population
   exceeding the limit now constructs successfully, records the correct digest
   and length, and survives complete-snapshot relocation after original inputs
   are deleted. See [the detailed Rust report](RUST-ORIGINAL-A.md).
2. **Inherited onboarding timing and default mode.** Selection accepts an omitted
   worker timing block; pinning creates that block only when writing resolved
   controls. Replay accepts the omitted default aggregated mode. Serving
   preparation handles inherited estimator controls, including the distinction
   between `None` (inherit) and an explicit empty mapping (override). The raw
   engine mapping is preserved so absent snapshot controls do not become explicit
   defaults. Tests omit timing and mode independently and exercise both bundle
   and relocated-snapshot replay/serving paths.

Auto's estimator order, learned prediction/update arithmetic, Gym/Dynamo adapters,
configuration schemas, and caller ownership of live feedback are unchanged.

## Verification

| Check | Result | Evidence |
| --- | --- | --- |
| Python negative oracle | Three new omission combinations fail with the reported `KeyError`; existing form passes | [Before](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_review_fixes_20261103/python-before.json), [log](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_review_fixes_20261103/python-before.log) |
| Rust negative oracle | Valid >16 MiB manifest fails legacy construction before the fix | [Before](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_review_fixes_20261103/rust-original-a-before.json), [log](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_review_fixes_20261103/rust-original-a-before.log) |
| Isolated revised PR1 | 10 matching original-A tests pass | [PR1](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_review_fixes_20261103/pr1-original-a.json) |
| Assembled native FPM | 435 pass | [Rust](fpm-rust.json), [log](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_review_fixes_20261103/fpm-rust.log) |
| Public Rust consumer | 26 pass | [Public API](public-api-rust.json), [log](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_review_fixes_20261103/public-api-rust.log) |
| Rebuilt-native Python integration | 343 pass, including all 29 learned-onboarding cases plus ordinary onboarding, bundle, mixed fallback, and snapshot suites | [Integration](python-integrated.json), [log](python-integrated.log) |
| Formatting and packaged legal files | Pass | [Static checks](static-checks.json) |

The [release extension build](native-build.json) is identified by
[native-identity.json](native-identity.json):
`d7c69b783e5b424b3bc6cd47bd96f18d8a8d810a21b3100ed1b5db550c25cf03`.
The Python integration run uses the worktree-local environment and that rebuilt
extension. [Source inputs](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_review_fixes_20261103/source-inputs.json) record all source bytes and their
narrow delta from the earlier `6a12ba8d` build; the [PR3 source proof](pr3-source-proof.json)
binds the restacked source to the tested files. The initial 29-case Python check
on the previous binary is retained separately and is superseded by the integrated
343-case run for final-build coverage.

The source projection performs linear input scanning plus object-key duplicate
checks. It retains dependency references, open-object keys, nesting state and the
parser's largest scalar buffer, without retaining a JSON observation population.
Prediction and update paths gain no work. This is a cold-path complexity and
retention assessment, not a new timing or memory benchmark. Existing full36
accuracy, consumer, host-cost and installed-wheel reports remain measurements of
their recorded pre-review builds; they were not rerun or relabeled for this fix.
The synthetic example's recorded producer revision and artifact hashes remain
unchanged.

The initial Rust test compilation mistake is retained in the focused report.
The raw JSON digest fixture intentionally includes trailing whitespace, documented
in [the whitespace check](https://github.com/ai-dynamo/aisimulate/blob/3630fed8e30ec24edee78f569b9d7276e60ddb4a/benchmarks/evidence/gbdt_review_fixes_20261103/intentional-fixture-whitespace.json). Configured Ruff
hooks run separately for production files so historical executed evidence scripts
keep their pinned bytes; no persistent hook configuration is changed.
