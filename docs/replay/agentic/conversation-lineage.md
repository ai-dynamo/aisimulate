# Agentic conversation lineage

The 0.13.0 replay identity adds an optional `lineage` object with schema
`aisimulate.agentic.conversation-lineage.v1`, `root_conversation_id`, and optional
`parent_conversation_id`. The existing `root_id` and `parent_id` remain request
IDs. A later conversation turn can have no request-level spawn dependency while
still retaining its conversation parent.

Lineage is compiled from all spawn edges in the validated source graph, before
snapshot history is removed. All turns in a conversation share its lineage.
Snapshot instances namespace every lineage reference with the same play instance
as the request, including references to historical conversations. Recycling a
lane therefore does not reuse the prior play's conversation or affinity keys.

A conversation without a spawn parent is a root and names itself as
`root_conversation_id`. Multiple distinct spawn parents, self-parenting, or a
cycle between conversations leave lineage absent for that conversation and its
descendants. Such request DAGs remain valid replay inputs; an adapter requiring
conversation-tree routing must reject unavailable or unsupported lineage instead
of choosing a parent by request order.

Old JSON without `lineage` still deserializes and round-trips without the field.
New Rust struct literals must set `lineage` explicitly. Consumers must check the
schema before relying on ancestry. No router policy is enabled by carrying these
identities: a versioned consumer must invoke its native policy and independently
qualify actual placement, cache behavior, and supported topologies.

Placement policies receive optional default no-op `dispatch_committed` and
`dispatch_aborted` callbacks. A successful dispatch means the engine accepted
ownership; for P/D decode this is destination reservation, before transfer or
token generation. A policy can hold a tentative binding during selection and
commit it only after acceptance. `advance_clock(now_ms)` runs during semantic
settlement and can release policy waiters after a commit or abort.
`next_wakeup_ms()` must identify concrete future policy work. TTL housekeeping
that cannot release a request should run lazily and must not extend the run.

This contract uses AISimulate's existing graph edges and snapshot namespaces. It
does not alter graph digests, sampling, request dependencies, cache allocation,
warmup behavior, or the methodology qualification status of replay.
