<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Maintaining the required CI ruleset

Edit [required-main-checks.json](required-main-checks.json) through a reviewed PR,
then apply that file to GitHub. The file is the intended configuration; GitHub's
live ruleset is what enforces it. Committing or merging the file does not update
GitHub settings automatically.

As of September 18, 2026, repository ruleset
[23671922](https://github.com/ai-dynamo/aisimulate/rules/23671922) is active for
the default branch. Its visible CI requirements match the file. Check live
settings each time; this snapshot does not establish hidden bypass settings or
controlled-PR enforcement evidence. Existing organization rules independently
require human/CODEOWNER approval and resolved conversations.

## Edit once, then apply

1. Review and merge changes to the JSON through the normal PR process. Before
   adding or renaming a required check, verify its actual workflow result name
   and GitHub Actions application binding.
2. An administrator checks out the reviewed commit and previews the difference:

   ```bash
   python3 scripts/sync_required_main_checks.py --ruleset-id 23671922
   ```

3. After reviewing the diff, that administrator applies the same file:

   ```bash
   python3 scripts/sync_required_main_checks.py --ruleset-id 23671922 \
     --apply --backup main-ci-rules-before.json
   ```

   Use a new backup path for each update. The command saves the complete live
   definition before changing anything and verifies GitHub's configuration
   after the update. It updates the existing ruleset; it never creates another
   ruleset or changes organization rules. Do not simultaneously edit the same
   ruleset in the UI: GitHub provides no atomic compare-and-update guarantee.

The helper requires Python 3.10+ and an authenticated GitHub CLI (`gh`). Without
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
