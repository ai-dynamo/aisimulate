<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CI qualification and rollout

`Full CI Success` aggregates validation before protected artifact staging.
The staging job depends on this result. Main and release validation can finish
while the `automated-release` environment waits for a human; successful
validation does not mean artifacts have been uploaded.

## Required checks

`.github/required-main-checks.json` adds a separate rule for the default branch:
`Fast CI Success`, `Full CI Success`, and `codeowners`, bound to the GitHub
Actions application. It requires the branch to be current and leaves the
existing review and CODEOWNER rules intact. Do not replace the existing
ruleset with this payload.

After the workflow changes land and their main-branch validation succeeds, a
repository administrator can activate it:

```bash
gh api repos/ai-dynamo/aisimulate/rulesets --method POST \
  --input .github/required-main-checks.json
gh api repos/ai-dynamo/aisimulate/rules/branches/main
```

Inspect existing rules first and update an existing rule with the same name
instead of creating duplicates. Confirm all three required checks and strict
branch currency in the effective branch rules. Maintainer access alone did not
allow creation during the September 11, 2026 rollout (HTTP 404); the checked-in
payload has not activated enforcement. Retain the trusted-copy Full CI backstop
until the effective rules confirm enforcement.

## Native build environment

`scripts/ci_install_build_tools.sh` skips apt when `cc`, `c++`, and `make`
already exist. Existing images get three bounded bootstrap attempts with apt
transport retries and refreshed indexes between attempts. Package signature and
checksum checks remain enabled. The comparison workflow uses the helper from
the workflow commit even when testing a historical checkout.

`.github/ci-image/Dockerfile` prepares build tools once in the existing runner
image. The runner-image owner should build and smoke-test both CPU architectures
before changing `CI_JOB_CONTAINER_IMAGE`:

```bash
bash scripts/build_ci_image.sh
```

Export `AISIM_BASE_IMAGE_BY_DIGEST` with the current runner image's immutable
digest and `AISIM_BUILD_IMAGE_TAG` with an authorized image destination. The
wrapper rejects missing, mutable, and malformed base-image references before
invoking Docker. Check its runner user, entrypoint, both architectures, native
compilation, and a full validation run. Then set `CI_JOB_CONTAINER_IMAGE` to the
new multi-architecture image digest. Keep the old value for rollback. This
repository change does not publish an image or change the shared ECR image.

## Native numerical stability

`scripts/check_prediction_numerics.py` runs eight frozen public native-engine
queries in the Engine Golden Regression job: dense Qwen3-32B and MoE
MiniMax-M2.5, prefill and decode, short and long sequences. The manifest records
the source commit that produced its expected values; qualification requires
that full SHA to resolve to a commit in the checkout. Two percent relative and
0.0001 ms absolute tolerances allow small numerical variation. Missing,
duplicate, failed, nonfinite, nonpositive, and out-of-tolerance results fail.

These values characterize the native model at the recorded commit. They do
not measure prediction accuracy against hardware. The broad old/new modeling
report retains its advisory numerical-drift policy. An intentional model or
data change needs an explained before/after review; never refresh this baseline
only to turn a failed job green. `native-prediction-numerics` retains the exact
manifest digest and observed values.

## Installed public CLI

Every platform-wheel job checks that loaded package and native-extension files
match the wheel bytes, then runs `recommend` and consumes its generated YAML
through `predict` in an unrelated temporary directory with isolated Python.
The small existing fixed-timing fixture requires all six requests to complete.
The uploaded `installed-cli-*` evidence includes the wheel hash and counts.
This checks packaging and the public configuration round trip; numerical
accuracy has separate evidence.

## Related CI work

- [#145](https://github.com/ai-dynamo/aisimulate/pull/145) owns test sharding,
  build reuse, and the integration modules omitted by the previous marker gate.
- [#147](https://github.com/ai-dynamo/aisimulate/pull/147) owns exact-wheel FPE
  nightly qualification and the real installed generator smoke test.
- [#71](https://github.com/ai-dynamo/aisimulate/pull/71) and
  [#75](https://github.com/ai-dynamo/aisimulate/pull/75) provide the native
  benchmark and advisory comparison workflow. Land the harness before its
  workflow; revalidate against current main and retain noisy-result evidence.
- [#174](https://github.com/ai-dynamo/aisimulate/pull/174) owns the reported
  large-sweep resource regression. Its Linux native evidence and prerequisite
  integrations remain required. Dynamo-native evidence belongs in downstream
  compatibility qualification; the AISimulate wheel stays standalone.
