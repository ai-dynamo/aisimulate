# Slurm collection completion hook

`slurm_collection_hook.py` is an external campaign observer/submitter. It uses
existing `collect.py --resume` jobs and their unmodified checkpoints. It does
not weaken atomic-write protection, classify case failures, replace missing
rows, or change the performance-data contract.

Run with `--config <campaign-config.json> --apply --watch`. The configuration
contains the complete `expected_tasks` mapping (all required op families),
`unresolved_scopes` for work not yet enumerated, and one or more `campaigns`:

```json
{
  "user": "YOUR_USER",
  "max_active_jobs": 20,
  "max_infra_attempts_per_revision": 2,
  "expected_tasks": {"gemm": 149776, "moe": 3069},
  "unresolved_scopes": ["Add every remaining required family before completion"],
  "campaigns": [{
    "root": "/YOUR/CAMPAIGN",
    "plan": "gemm-shards.json",
    "runner_revision": "YOUR_AUDITED_RUNNER_REVISION",
    "slurm": {
      "account": "YOUR_ACCOUNT",
      "partition": "batch",
      "image": "/YOUR/PINNED_RUNTIME.sqsh"
    }
  }]
}
```

The example is intentionally **not** a complete all-model plan. The hook must
not finish merely because GEMM is done while other declared families remain.
An unresolved scope prevents completion. Each plan has `smokes` and `shards`
lists, with immutable `id`, `op`, `mode`, `planned_tasks`, documented CLI
`filters`, and an optional `smoke_dependency`. GEMM full shards include a
`case_set_sha256` over sorted physical case strings. Other families should
supply `expected_case_ids_sha256` over the actual stock collector IDs.

The task runner stores `results/<id>/job-<jobid>/status.json` and a quiescent
`data/` snapshot. The hook verifies the status against the plan, exact successful
checkpoint IDs, runtime/SM identity, finalized table metadata, row versions,
finite latencies, and GEMM physical output keys. Missing/failed/duplicate cases
and unvalidated data cannot report completion. The global `hook_status.json`
separates active, validated, missing-family, and fix-required states.

Infrastructure retries are bounded **per runner revision**. Case failures wait
for a collector fix; they are not blindly retried. Fix-required states do not
terminate a watch process or falsely mark the campaign complete. Other ready
shards continue. Update the audited runner revision only after fixing the
underlying problem, not to defeat the retry limit.

## Filesystem and runtime prerequisites

- The collector requires a filesystem supporting atomic no-replace rename.
  Run its output, checkpoint and temporary files on node-local POSIX storage
  when a shared filesystem rejects this primitive. Do not replace the
  collector's safety guard with a non-atomic operation.
- After all writers exit, copy the unchanged snapshot to shared storage. Keep
  original failed attempts. A transferred unresolved transaction must pass the
  stock recovery checks before it can be resumed; do not edit its ledger.
- Install `pyarrow` in the isolated runtime for official parquet finalization.
  Keep the GPU framework's existing NumPy/Torch/CUDA versions unchanged.
- Provide a real Git executable and the actual source checkout for
  `collector_ref`; never fabricate a commit in metadata. Verify runtime and
  source identity before timing.
- `--gpu-freq=1965` plus sampled clocks is a frequency request, **not proof of a
  hard clock lock**. The current campaign explicitly allows this requested-and-
  monitored mode; preserve that qualification in campaign evidence.

The hook uses `squeue`/`sbatch` under the same authorized user and only submits
its explicitly declared task-owned shards. Use a scheduler-managed CPU job for
persistent watching; do not rely on an interactive terminal staying connected.

## Reviewed failures and orderly completion

The optional per-op `resume_with_recorded_failures` campaign mapping and matching
plan `reviewed_failure_continuation` mapping are **review records, not selectors**.
They permit the unchanged CLI's `--resume` to finish unattempted cases after an
observed failure group has been investigated. Prior failed IDs remain failed;
no failing point is retried, removed, relabeled, or given a synthetic latency.
An all-attempted shard with failures is reported as
`complete_with_recorded_failures`, not validated or globally complete.

For the B200 0.25.0 campaign, DSA early stops were reviewed against the installed
vLLM `dd10e03f95f94edbea1975c67ace3a35ec9a8a40` source:
`vllm/v1/attention/backends/mla/flashmla_sparse.py:836` requires the BF16 prefill
padding to be divisible by the requested number of heads. Non-divisor head
counts remain in the frozen sweep and produce real recorded failures. The
review allows remaining cases to execute; it does not substitute another backend.

The canonical `run_shard.py` also detects a separate teardown problem: Python
multiprocessing workers can remain alive after every task is accounted and the
official parquet/sidecar transaction has committed. After a grace period it
reaps only its own matching child workers (exact parent PID and UID), recording
the cleanup. It never uses a global process-name kill or cleans workers while
cases or publication transactions remain unfinished.
