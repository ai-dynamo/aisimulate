<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Recommendation output ABI

[`output_adapter.py`](../../python/aisimulate/src/aisimulate/output_adapter.py)
defines `RecommendationOutputAdapter` and `OUTPUT_ADAPTER_API_VERSION = 1`.
An output adapter adds artifacts or live observations to a recommendation;
it does not change replay, candidate scores or saved prediction inputs.

## Identity and required writer

The entry-point group is `aisimulate.output_adapters`. A selected `--output`
name also identifies its top-level configuration section. AISimulate extracts
explicitly selected output sections before simulation-schema validation.
An adapter declares the same `name`, integer `api_version = 1`, and:

```python
write(
    config: Mapping[str, Any],
    *,
    result: SweepResult,
    output_dir: Path,
) -> Sequence[str | Path]
```

This is a bound-method signature. The supplied directory is prepared and
canonical recommendation output has already been written. Return relative
paths that exist after the call; a bare string/bytes result, absolute path,
empty path or `..` component fails validation. Adapters share the output
directory and must not overwrite canonical files. The returned path check is
not a sandbox for arbitrary filesystem writes.

Writer exceptions or invalid returned paths become `OutputAdapterExecutionError`
and fail the command. A failed additional writer does not imply that canonical
files were never written; inspect the output directory when retrying.
`KeyboardInterrupt` propagates unchanged.

## Optional live subscription

A callable `subscribe(config)` may return `RecommendationOutputCallbacks` or
`None`. The callbacks are:

```python
on_candidate(record: CandidateRecord) -> None
on_round(round_no: int, candidates: list[Candidate]) -> None
on_complete() -> None
on_failure(error: Exception) -> None
```

Subscription runs in the supervised recommendation worker. Progress callbacks
receive a deep copy of their event data, and every callback executes
synchronously on the search path. Return promptly; failure aborts the
recommendation through `OutputAdapterExecutionError`. A subscription returning
another type is invalid.
Callbacks and final `write()` may run on different adapter instances, so their
contract does not provide shared in-memory state.

`on_complete` runs after the search and every progress callback has finished, but
before the worker returns its successful result to the parent process.
`on_failure` runs before an exception from the search or another callback leaves
the worker. Every registered failure callback is attempted in subscription order
even when an earlier callback fails. Failure-callback errors retain their adapter
and event attribution and are reported after all callbacks have run. The original
search or callback exception remains the primary error and carries any additional
failure-callback diagnostics. These terminal callbacks let the worker-local adapter
stop and join live publishers before the separate final writer publishes
authoritative output. An adapter that does not start background work can omit them.

## Discovery and deployment limits

Selected adapters only are imported. Direct injection wins over entry points;
missing/duplicate names, incompatible versions, bad declared names, missing
`write`, or a non-callable `subscribe` raise `OutputAdapterResolutionError`.
Loading/constructing a selected plugin can also fail at resolution.

A deployment-output adapter must preserve candidate identity and reject shapes
it cannot render. Being able to write an artifact does not qualify a Runner
backend, model, topology or live serving deployment. See
[deployment generation](../sweeper/deployment-generation.md) for the existing
candidate-to-artifact workflow and its restrictions.
