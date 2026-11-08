---
description: >
  Collector failure doctrine: observe, don't predict; the escalation decision tree.
paths:
  - "collector/**"
  - "tests/unit/collector/**"
---

# Collector Failure Handling

**Observe, don't predict.** A failing case costs seconds and is recorded
automatically with a classification. A prediction rule costs maintenance
forever and rots when the framework fixes the bug. There is no expected-failure
layer and no auto-skip: a group that fails as a whole is fixed, or left out in
`collector/skips.py` with its reason, never silently tolerated. `overview.md`
defines the failure principle and the per-consumer-key accounting; this page is
the procedure.

## What happens automatically

- Every worker failure is recorded in `errors_<module>.json` and
  `collection_summary_<backend>.json` with a classification: `kernel_limit`
  when the message starts with `FIXME(kernel-limit)`, otherwise `unexpected`.
  Each record carries the case parameters, the exception and a `(model, dtype)`
  group; the summary aggregates `errors_by_group`.
- A CUDA-fatal error resets the worker after its task is recorded.
- The SDK tolerates missing points: it interpolates, extrapolates, reuses
  sibling-version rows, or falls back to empirical estimates. A handful of
  failed cases does not invalidate a (system, backend, version) dataset.

## Decision tree for a failing case

```text
case failed
├─ Recorded and classified in the failure log?          → yes: DONE (the default)
├─ Would it succeed on the wrong kernel?                 → collector/skips.py entry
├─ Does it hang or kill the node?                        → collector/skips.py entry
├─ Does a whole block fail, or can the backend not       → collector/skips.py entry
│    serve the model?
├─ (op × backend) never debugged?                        → registry unverified=True
├─ Not validated on this SM only?                        → registry unverified_sms=(sm,)
├─ Out of memory?                                        → unclassified until it reproduces on
│                                                          a clean GPU; if real, the worker
│                                                          refuses it before allocating
├─ Crash inside framework code?                          → serving-parity audit first; a guard
│                                                          or a framework bug report only after
│                                                          the audit finds no collector-side
│                                                          divergence
└─ Proven collector bug?                                 → fix the code, never via a skip
```

`layer_permissions.md` holds the rules behind each branch: "Leaving cases
out", "Capacity" and "Metadata and input parity". Scattered framework gaps
have no home in `collector/skips.py`: they fail, the log groups them by
(model, dtype), and the next pin bump re-tests them for free.

## Triage thresholds

- Scattered failures up to about 5% of a table are acceptable when explained
  and unclustered.
- Count failures per consumer key bucket (`overview.md`). A bucket below about
  20% success is an unavailable key: report it in the run summary and the PR.
  It needs action only when a supported model queries that key or the bucket
  was available in the previous run.
- Investigate when scattered failures pass about 5% of a table, or cluster by
  op, backend, model, shape family or SM.
- About a third failing, or a whole family, is a systemic collector problem:
  stop collecting and fix the collector.

Never hide failures with broad skips, retries, generic OOM labels, reduced
coverage, synthetic rows or weakened benchmarks.

## Consuming failure records

- Compare with the previous run of the same (backend, version); only new
  failure groups deserve attention.
- Re-check a failure with a targeted run (`--ops`, `--model-path`,
  `--case-filter`), never a full collection.
- Failure records are observations. Do not machine-translate them into
  `collector/skips.py` entries; each escalation above needs its evidence.
