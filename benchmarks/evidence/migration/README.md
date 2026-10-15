<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Migration execution evidence

[examples.json](examples.json) retains the fixed-revision command results from
the migration qualification. The payload is unchanged. Its source revisions
and result scope are evidence for that run, not proof that the current checkout
or an arbitrary wheel passes the same commands.

Use the [current AIC migration guide](../../../docs/aic-backward-compatibility/migration.md)
for supported replacements and the [CI guide](../../../docs/ci/README.md) for
current validation.
