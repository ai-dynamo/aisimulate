<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate Dynamo policy adapter

This optional package connects AISimulate's placement interface to native
Dynamo KV selection and session affinity. AISimulate's base package has no
Dynamo dependency. The adapter uses the existing public APIs from merged
Dynamo commit `d9eb42db1168131fdae318eef77255637e4d3495`; Dynamo PR #15149 is
not required.

Build and install this wheel together with the base AISimulate wheel from the
same checkout. The Python versions must match exactly, and both native modules
must report the same core source digest and serialized replay contract. The
loader rejects a mismatched pair, including different development builds with
the same version. It never substitutes round-robin for missing or incompatible
Dynamo routing.

The native crate shares the root Cargo lockfile and only imports Dynamo's router and
its dependencies. It does not require the full `ai-dynamo` Python distribution,
Dynamo Mocker or a running Dynamo service. Existing explicit `--stack dynamo`
continues to select the full Dynamo provider when installed.

See the repository's `docs/agentx-quickstart.md` for supported configurations,
installation commands and acceptance scope. The new package is source-only
until matching release wheels are published; a source commit is not evidence
of package availability.
