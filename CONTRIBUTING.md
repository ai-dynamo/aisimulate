<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to AISimulate

Thank you for your interest in contributing to AISimulate!

For the project overview, repository layout, setup, and development commands,
see the [AISimulate README](README.md). For focused AIConfigurator development
instructions, see the [application README](python/aisimulate/README.md).

## Quick Links

- [CI guide: workflows, code review, and required checks](docs/ci.md)
- [Good first issues](https://github.com/ai-dynamo/aisimulate/labels/good-first-issue)
- [Help wanted](https://github.com/ai-dynamo/aisimulate/labels/help-wanted)
- [Open an issue](https://github.com/ai-dynamo/aisimulate/issues/new)
- [Slack](https://ai-dynamo.org/slack)
- [Office Hours](https://www.youtube.com/playlist?list=PL5B692fm6--tgryKu94h2Zb7jTFM3Go4X)
- [Community Meetings](https://docs.google.com/document/d/1uR8xD_hlYGwV6QspvSc36k1H-wo1BUcVmFbHH9xlXd8/view) ([YouTube](https://www.youtube.com/@ai-dynamo-community)) -- Weekly (Wed 10:30 AM PT) development community meetings

## Preparing a pull request

Use the root [PR template](.github/pull_request_template.md) to describe the
behavior change, validation, and review handoff. Choose low, medium, or high risk
using the [review contract](REVIEW.md#risk-tiered-review-and-ci), name the
responsible CODEOWNER team/reviewer, and identify the expert decision for
high-risk work. Keep unfinished work as a draft: Fast CI still runs, and marking
the PR ready starts eligible automatic CodeRabbit review without a
`review-ready` label. Remove title/label exclusions when requesting review;
a skipped CodeRabbit check does not qualify the PR for maintainer Full CI
admission. Medium/high risk also requires same-commit Codex review.

Follow the [handoff and finding policy](REVIEW.md#review-handoff-and-finding-disposition)
when requesting review or responding to findings. Keep blocking conversations
visible, link agreed follow-up issues, and refresh review/CI evidence after each
push. Every tier needs applicable CODEOWNER approval; high risk also needs the
relevant expert. The [CI guide](docs/ci.md#code-review-and-pr-admission) describes
Full CI admission and the separate repository-enforcement requirements.

## Developer Certificate of Origin

AISimulate is an open source project released under the Apache 2.0 license
(see either [the Apache site](https://www.apache.org/licenses/LICENSE-2.0) or
the [LICENSE file](./LICENSE)). The Apache 2.0 license allows you to freely use,
modify, distribute, and sell your own products that include Apache 2.0 licensed
software.

We respect the intellectual property rights of others and want to make sure all
incoming contributions are correctly attributed and licensed. A Developer
Certificate of Origin (DCO) is a lightweight mechanism to do that.

The DCO is a declaration attached to every contribution made by every
developer. In the commit message of the contribution, the developer simply
adds a `Signed-off-by` statement and thereby agrees to the DCO, which you can
find below or at [DeveloperCertificate.org](https://developercertificate.org/).

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

We require that every contribution to AISimulate is signed with a Developer
Certificate of Origin. Additionally, please use your real name. We do not
accept anonymous contributors or those using pseudonyms.

Each commit must include a DCO sign-off that looks like this:

```
Signed-off-by: Jane Smith <jane.smith@email.com>
```

You may type this line yourself when writing your commit messages. If your
`user.name` and `user.email` are set in your Git configuration, you can use
`-s` or `--signoff` to add the `Signed-off-by` line to the end of the commit
message.

For example:

```bash
git commit -s -m "Describe your change"
```

By contributing, you agree that your contributions will be licensed under the
[Apache 2.0 License](https://github.com/ai-dynamo/aisimulate/blob/main/LICENSE).
