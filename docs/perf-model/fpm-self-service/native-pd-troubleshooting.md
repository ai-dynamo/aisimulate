<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native P/D capture: troubleshooting and completion checks

Use this reference during onboarding [stage 5](implementation.md#5-collect-and-verify-data).
Start with the [P/D setup reference](native-pd-setup.md) for runtime pins and
role configuration. These checks also apply to an existing deployment; use your
own workload, cluster tools and artifact names.

## Diagnose the pilot before increasing load

| Symptom | Check and recovery |
| --- | --- |
| HTTP success but no useful output | Inspect the finished choice, completion-token count and client errors. Require multiple generated tokens and busy native rows from both roles. HTTP 200 can accompany an aborted request with zero output. |
| Decode cannot finish the bootstrap handshake | In the pinned SGLang pairing, worker `--host` also controls the prefill bootstrap listener. The advertised node address must be reachable from decode. Binding only to `127.0.0.1` while advertising a node address breaks the handshake. Use the appropriate reachable interface, distinct bootstrap ports and normal network access controls. |
| Prefill OOM despite `--max-prefill-tokens` | Inspect the **resolved** `--chunked-prefill-size`. In this SGLang revision, `--max-prefill-tokens` is not a strict chunk limit. The reference run used a 4,096-token chunk with a 262,144-token context limit. Changing chunk size changes the observed batch population; save it with the new capture. |
| SGLang decode expires in `WaitingForInput` | The pinned runtime's `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` bounds elapsed waiting after metadata dispatch, including prefill queueing/computation and KV transfer. It is not an inter-chunk inactivity timer. The reference run set `1800` in decode's environment; the prefill bootstrap deadline is separate. Check load and transfer health before changing it. |
| Mooncake reports a full transfer queue | The pinned Mooncake `0.3.13.post1` limit counts queued **transfer pieces per peer**, not requests or tokens. Cached-prefix transfers can have many pieces even with small prefill chunks. The reference run set `MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535` on both roles. A larger queue can consume more CPU metadata and increase residence time; it still needs a successful pilot and completed workload. |
| Requests stall or queue continuously | Reduce the offered load to a sustainable pilot, preserving the intended range of batch sizes and contexts. Record the changed rate/concurrency and rerun in a fresh capture. This reduces queue-induced instability; do not smooth the surface by deleting valid slow observations. |
| Client reports prepared data but sends nothing | Verify preparation exits successfully and the first actual requests are sent. A progress counter reaching its total does not prove preparation or worker-pool shutdown completed. Preserve tokenizer, dependency and workload-client identities separately from GPU qualification. |
| A role has no native rows | Verify native instrumentation, role-specific capture environment, writable mounted paths, expected producer identity and actual traffic. Frontend registration alone does not verify the producer. For SGLang, inspect the native counter correction and resolved scheduling settings. |

The numeric settings above describe the [tested P/D configurations](native-pd-setup.md#tested-identities).
They are not defaults for another model, runtime or cluster. Retain each failed
attempt, diagnose the cause and record any configuration change before retrying.

### Check the actual worker process

Save the launch plan **and** the effective environment/arguments supplied to each
worker process. A setting in a scheduler wrapper, frontend or parent shell may
never reach the worker. In particular, verify the waiting timeout on decode and
the transfer-queue setting on both SGLang roles. Check initialized worker logs
for chunk size, scheduler mode, memory/cache settings and native metrics. Keep
planned values distinct from observed values; incomplete observations stay
unresolved.

## Verify requests, transfer and native labels

Before the workload, send two distinct short prompts with multiple output tokens.
Where supported, fixed-length smoke can use `ignore_eos: true` and 16 requested
output tokens; require exactly 16 completion tokens for that test. Otherwise
record the intended termination rule and verify it. Check the finished response,
integer prompt/completion/total usage, client errors and both roles' busy records.
Correlate transfer logs or equivalent runtime evidence with those requests; two
registered workers do not prove that decode received real prefill KV.

Extend the pilot to repeated prefixes, chunked prefill when enabled, concurrent
work and decode after an idle interval. These cases can expose counter/timer
errors that a short uncached request misses. The reference campaign's accepted
smoke does not independently qualify label alignment for every such case.

Preserve native durations in seconds and their producer conventions. In the
pinned runtimes, vLLM normally measures host time between busy output-update
arrivals; after initialization or an empty output, the next busy row uses its
matching schedule timestamp. SGLang reports completed CUDA timer intervals with
overlap disabled. These labels have different boundaries and neither measures
end-to-end request latency. Do not replace them with client token gaps.

vLLM decode context counts prior computed tokens; SGLang ordinary decode lengths
include the current token. Preserve backend identity instead of applying a shared
one-token correction. Serving role also differs from batch class: keep vLLM
decode-admission and mixed rows under their original decode identity and report
trainer exclusions. The tested SGLang recipe supplies pure-prefill and pure-decode
observations only; it establishes no mixed-work coverage.

## Drain, close and account for the capture

1. Stop new sends, allow outstanding requests to finish, and preserve separate
   warmup, measurement and drain timestamps and client completion counts.
2. Observe idle on every expected producer. Close or rotate capture files through
   the deployment owner's procedure; stop only workers owned by the collection.
3. Inspect both wrapper and inner engine shutdown. Record signals, forced kills,
   writer-close errors, transport errors and rotation/pruning. A process-group
   leader exiting zero does not prove all descendants flushed successfully.
4. Read every retained gzip/JSONL file to completion, verify stable hashes after
   close and run the [capture validator](e2e-gbdt.md#2-check-and-prepare-the-data).
   Check expected workers, restarts, duplicates, schema, counters and labels.
5. Review phase selection against the complete raw stream, then prepare separate
   role exports using the chapter's selection helper. Preserve the raw envelopes,
   selected ranges, hashes and export receipts together.

Forced worker-group termination or an unresolved flush failure leaves completion
unestablished. An inner-engine kill warning needs correlation with the last busy
records, subsequent idle records, drained requests and relay/writer shutdown;
neither the warning alone nor an outer exit code resolves the tail.

The reference plan used a 128 MiB segment threshold and retained 64 segments per
role. Size retention for **your** volume and preserve files before pruning. All
retained files being readable proves storage integrity, not that every produced
measurement reached the sink. Nonconsecutive native counters alone also do not
prove loss, and continuous counters do not rule out events discarded before
counter assignment. Publisher losslessness remains unproven for these recipes;
it needs producer/relay/writer accounting and verified drain behavior in addition
to valid files and completed requests.

### Keep a reviewable completion record

| Keep | What it establishes |
| --- | --- |
| Runtime/model/tokenizer pins; per-role launch and observed settings | Which deployment produced the measurements. |
| Workload/client identity, phase boundaries, request results and exit status | What traffic ran and whether it completed under the chosen acceptance criteria. |
| Smoke and transfer evidence, all native segments, shutdown/rotation logs | Observed worker activity and the scope of retained capture. |
| Expected stream identities, hashes, validator and selection/export receipts | Which persisted records passed checks and entered training. |
| Separate coverage and accuracy reports | What the trained model can predict and how closely it matches held-out measurements and serving runs. |

Do not infer a smoke cutoff from the largest counter in a few displayed samples.
Review the entire excluded prefix, including engine-startup records where present.
Keep workload warmup and drain identifiable and state whether they enter training.
If other traffic shares the workers, establish a defensible selection boundary or
collect in an isolated deployment; relay arrival time alone is not the native
timer's measurement interval.

When these checks pass, continue with [training and accuracy qualification](e2e-gbdt.md#3-train-and-qualify-accuracy).
Capture completion, publisher losslessness, model accuracy and Replay/resource
qualification remain separate results.
