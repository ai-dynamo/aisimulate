<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Dynamo policy plugin

This private Rust crate is compiled into the optional `aisimulate-dynamo-policy`
Python wheel. It consumes the unmodified, merged ai-dynamo/dynamo revision
`d9eb42db1168131fdae318eef77255637e4d3495` through Cargo's immutable Git dependency.
It does not depend on Dynamo PR #15149, a development AISimulate commit from
Dynamo, or a local dependency override.

AISimulate's canonical replay bridge constructs workloads, performance models,
engines and reports. The plugin supplies a `ReplayComposition` whose policies use
Dynamo's public `SelectionServiceBuilder`, `SelectionCore`, native worker-selection
policy, native `SessionAffinity`, and native booking lifecycle. Affinity holds stay tentative until the engine accepts
dispatch; siblings waiting on initialization remain queued and can be cancelled.
Replay's physical
engine KV stores/removals feed the native event-driven index. `session` and
`sibling_group` affinity alter the key supplied to the native affinity table; they
do not implement an independent worker selector. Sibling keys include play/root
identity and parent conversation identity; roots use a separate key namespace.

Every placement owns an isolated paused Tokio runtime. A blocking clock guard
prevents Tokio's idle auto-advance; only replay timestamps advance virtual time.
Replay terminal callbacks free native reservations. Native production request
expiry is disabled via the public host lifecycle interface, while native affinity
idle TTL remains enabled and follows the replay clock.

The public `SelectionCore` uses the native stochastic selector. Explicit selector
seeds, authored DP pins, custom policy classes, remote indexers, and dynamic
scaling are rejected. Workload seeds and automatically selected worker/DP affinity
are supported. The Python adapter validates its remaining workload capability
boundaries before simulation.

## Attribution

`src/events.rs` contains a modified protocol conversion derived from:

- Repository: <https://github.com/ai-dynamo/dynamo>
- Revision: `d9eb42db1168131fdae318eef77255637e4d3495`
- Original file: `lib/mocker/src/engine_observations.rs`
- Copyright: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
- License: Apache License 2.0, upstream `LICENSE` at that revision.

The conversion preserves upstream copyright/license attribution; it adds checked
conversion errors and the AISimulate observation batch boundary. The rest of this
crate is independently authored adapter glue calling public interfaces, without
copying Dynamo's selection, affinity, hashing, or cache algorithms. The repository
root `THIRD_PARTY_NOTICES.md` is the canonical attribution inventory and must be
included with the Python distribution.
