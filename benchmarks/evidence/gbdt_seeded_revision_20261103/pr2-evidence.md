# Execution-readiness qualification

The stage receipts retain their original build identity and hashes. This cleanup
changes only evidence presentation; it does not rerun or relabel any tests.

| Check | Recorded result | Archived log |
| --- | --- | --- |
| pr2-fpm-rust | 414 passed | [pr2-fpm-rust.log](https://github.com/ai-dynamo/aisimulate/blob/bc35afdfb4159000dd616132cac085972c527916/benchmarks/evidence/gbdt_seeded_revision_20261103/pr2-fpm-rust.log) |
| pr2-public-api-rust | 26 passed | [pr2-public-api-rust.log](https://github.com/ai-dynamo/aisimulate/blob/bc35afdfb4159000dd616132cac085972c527916/benchmarks/evidence/gbdt_seeded_revision_20261103/pr2-public-api-rust.log) |
| pr2-resolver-pre-fix-oracle | Source-pin race regression fails before the fix | [pr2-resolver-pre-fix-oracle.log](https://github.com/ai-dynamo/aisimulate/blob/bc35afdfb4159000dd616132cac085972c527916/benchmarks/evidence/gbdt_seeded_revision_20261103/pr2-resolver-pre-fix-oracle.log) |
| pr2-resolver | 95 passed | [pr2-resolver.log](https://github.com/ai-dynamo/aisimulate/blob/bc35afdfb4159000dd616132cac085972c527916/benchmarks/evidence/gbdt_seeded_revision_20261103/pr2-resolver.log) |
| pr2-support-plan | 158 passed | [pr2-support-plan.log](https://github.com/ai-dynamo/aisimulate/blob/bc35afdfb4159000dd616132cac085972c527916/benchmarks/evidence/gbdt_seeded_revision_20261103/pr2-support-plan.log) |

[pr2-checks.json](pr2-checks.json) and
[pr2-support-plan.json](pr2-support-plan.json) preserve the executed receipts.
Their relative log filenames resolve within the immutable commit linked above;
each archived file still matches the digest recorded in its receipt. Reproduce
the original stage from that checkout with its recorded inputs and commands.
Later assembled-stack qualification is recorded separately; these counts are
not claims about independently rerun cleanup commits.
