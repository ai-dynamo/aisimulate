<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Maintaining the required CI ruleset

Edit [required-main-checks.json](required-main-checks.json) through a reviewed PR,
then apply that file to GitHub. The file is the intended configuration; GitHub's
live ruleset is what enforces it. Committing or merging the file does not update
GitHub settings automatically.

Repository ruleset
[23671922](https://github.com/ai-dynamo/aisimulate/rules/23671922) is the managed
target. Do not infer its current state from this document or the JSON. Run the
verifier with an account that can see bypass settings, and retain the resulting
evidence for each change.

## Edit once, then apply

1. Review changes to the JSON through the normal PR process. Before adding or
   renaming a required check, verify its actual workflow result name and GitHub
   Actions application binding. In the PR template, name the person or account
   that will apply the change and the issue or PR that will hold the evidence.
   The apply owner must have Admin or `edit repository rules` access; a
   CODEOWNER approval alone does not establish that permission.
2. Mention or request review from the named apply owner. That owner acknowledges
   the handoff in the PR before merge. Do not merge a ruleset-file change with
   those fields unanswered: merging the file does not schedule or perform the
   live update.
3. After merge, the apply owner checks out the exact merged `main` commit and
   records its SHA. First save a read-only snapshot and preview the difference:

   ```bash
   ci_sha="$(git rev-parse HEAD)"
   test "${ci_sha}" = "$(gh api repos/ai-dynamo/aisimulate/commits/main --jq .sha)"
   python3 scripts/check_required_main_checks.py \
     --repository ai-dynamo/aisimulate \
     --output "main-rules-before-${ci_sha}.json"
   python3 scripts/sync_required_main_checks.py --ruleset-id 23671922
   ```

4. After reviewing the diff, the apply owner applies the same file and captures
   a second verifier snapshot:

   ```bash
   python3 scripts/sync_required_main_checks.py --ruleset-id 23671922 \
     --apply --backup "main-ci-rules-before-${ci_sha}.json"
   python3 scripts/check_required_main_checks.py \
     --repository ai-dynamo/aisimulate \
     --output "main-rules-after-${ci_sha}.json"
   ```

   Use a new backup path for each update. The command saves the complete live
   definition before changing anything and verifies GitHub's configuration
   after the update. It updates the existing ruleset; it never creates another
   ruleset or changes organization rules. Do not simultaneously edit the same
   ruleset in the UI: GitHub provides no atomic compare-and-update guarantee.
5. Link the merged commit SHA, ruleset URL, backup, before/after snapshots, and
   command outcomes in the merged PR or its tracking issue. Mark the handoff
   complete only after the post-apply verifier passes. If a command fails, leave
   the handoff open and record the failure for the apply owner to resolve.

The helpers require Python 3.11+ and an authenticated GitHub CLI (`gh`). Without
`--apply`, it uses only GET requests. Exit 0 means the complete managed
configuration matches, exit 1 means drift, and exit 2 means verification or
application could not complete. Check ordering is ignored. Hidden bypass
settings, configured bypass actors, a different target, and unrelated rules in
the selected ruleset stop the operation. A failed update/read-back can leave an
applied change; inspect GitHub before retrying, using the saved definition for
an administrator-reviewed recovery if needed.

## Who applies it

GitHub requires **Admin** access or a custom role with **edit repository rules**.
The Maintain role alone is insufficient. REST API credentials also need the
appropriate permission; fine-grained tokens and GitHub Apps need repository
**Administration: write**. GitHub can hide bypass settings from accounts
without ruleset write access, so even a complete comparison may require the
administrator's account. See [GitHub's ruleset API documentation](https://docs.github.com/en/rest/repos/rules#update-a-repository-ruleset).

An administrator must perform each apply unless the organization provisions an
authorized automation identity. This helper does not grant permissions or
install credentials, and no workflow automatically applies changes. A later
automation would require administrator setup, protected credentials and a
reviewed trigger; ordinary PR jobs should not receive those credentials.

Emergency UI edits must be reconciled back into the JSON before the next apply.
The helper verifies this repository ruleset only; use the [CI guide](../docs/ci.md#required-checks-and-release-approval)
for effective branch protections and rollout evidence. A configuration match
does not mean an individual PR is approved or ready to merge.
