<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Manual FPE snapshot: release/0.12.0

This is a one-time strict-native FPE coverage snapshot of release commit
`1f728534910187ecb2f021ef5e4bd4949bd555d0`. It uses an AISimulate 0.12.0 wheel
built from that commit's unmodified source archive. The release's curated
inventory supplies the models, systems, backends, and versions.

The newer FPE probe harness and qualifier come from
`58256c63de898b57c207d8829e5bb0cb3a5c7c3c`. They add installed-wheel isolation,
explicit topology-rejection classification, attention-backend identity, and
complete-report qualification. They do not replace the release's Python SDK,
native estimator, model definitions, or performance tables. `run_shard.py`
verifies the release archive identity and every installed package file against
the one shared wheel before probing it. The release wheel commit and harness
commit are recorded separately; this is manual evidence, not an Actions run.

Eight existing Brev CPU workers executed the 28 system/backend shards, with
8 vCPUs and 32 GiB RAM per worker and at most 8 probe threads per shard.
Native compilation ran once with 6 Cargo jobs on a CPU worker. The local Mac
only edited/orchestrated and transferred files; it ran no native build or
matrix probes. No workers were created, stopped, or deleted.

## Evidence

Qualification passed all 28 shards and all four required probes. The 888,543
raw probe results contain no unexpected `BUILD_FAILED` or `QUERY_FAILED`
results. The generated matrix covers 3,975 model/system/backend/version entries
across 75 models and 10 systems: 2,595 `PASS`, 1,323 `FAIL`, and 57
`HW_INCOMPATIBLE`. Expected coverage gaps remain visible as failures; successful
qualification means the reports are complete and the required probes pass.

- `manifest.json`: source/tooling identities, archive digest, complete
  qualification, and a readable per-system coverage summary.
- `snapshot.zip`: qualified split web CSVs and their index, the qualifier's
  report, raw-report checksum ledger, and completed worker execution records.
- `raw-reports.json`: checksums, sizes, and metrics for all raw probe JSONs.
- `execution.json`: worker/shard start, completion, return code, and elapsed time.
- `shards.json`: the release inventory's complete system/backend shard list.
- `installed-requirements.txt`: the installed release environment.

Raw reports, logs, the source archives, and the exact wheel are retained on
Brev under `/home/ubuntu/aisim-fpe-release012-01a0a2c1`; worker 26 holds the
combined reports in `collection/`. The checksum ledger makes those retained
reports auditable without checking gigabytes of raw probes into Git.

The publisher validates the ZIP digest, complete qualification, CSV identity,
and release ancestry. It labels this dataset **Qualified manual snapshot**.
A successful automatic run at the same or a newer release commit supersedes
it. Main's data stays independent.

## Reproduction

Run on Linux CPU workers. These steps build/probe native code and should not
be run on a laptop during ordinary Pages work.

1. Export `source.tar` with `git archive --format=tar` at release source
   `1f728534910187ecb2f021ef5e4bd4949bd555d0`. Export `tooling.tar` at
   `58256c63de898b57c207d8829e5bb0cb3a5c7c3c`, containing the four
   `python/aisimulate/tools/support_matrix/` files (`fpe_support_matrix.py`,
   `generate_fpe_support_matrix.py`, `build_fpe_support_matrix.py`,
   `qualify_fpe_support_matrix.py`) and `.github/fpe-required-probes.json`.
   Verify both archive identities with `git get-tar-commit-id`.
2. Extract into `source/` and `tooling/`. Using Python 3.12.13, maturin 1.15.0,
   and Rust 1.98.0, run `python -m maturin build --release --locked
   --auditwheel skip --out <work-root>/wheels` from `source/python/aisimulate`.
   This worker-local wheel has a Linux tag; it is qualification input and is
   not a published release wheel.
3. Run `uv sync --project source/python/aisimulate --python 3.12 --locked
   --no-install-project --no-dev`, then install the built wheel with
   `uv pip install --python source/python/aisimulate/.venv/bin/python
   --no-deps wheels/*.whl`. Verify dependencies with `uv pip check`.
4. Copy the release's `tools/support_matrix/support_matrix.py` into the
   corresponding tooling directory. Create empty `__init__.py` files in its
   `tools/` and `tools/support_matrix/` directories. Copy `run_shard.py` and
   `worker_run.py` and `package_snapshot.py` from this directory into the work root.
5. Discover the complete shard list with the release `SupportMatrix` inventory
   and save it as `shards.json`. For every system/backend pair, run
   `source/python/aisimulate/.venv/bin/python run_shard.py --system <system>
   --backend <backend> --max-workers 8 --output-dir results/<system>/<backend>`.
   Do not apply model/version/topology limits to the full run.
   To reproduce the execution records, divide `shards.json` into disjoint
   per-worker `assignment.json` lists and run `worker_run.py` with the release
   environment; it invokes the same command and records `progress.json`.
6. Collect every shard's JSON and metrics under `collection/`, along with the
   completed execution records. Run `package_snapshot.py` from the installed
   release environment. It invokes the complete-report qualifier with the
   unchanged four required probes, then builds the deterministic split matrix.

The web rows' commands use `python run_shard.py` in this prepared work root,
with the installed release environment activated. The packaging script updates
only that command prefix so reproduction uses the same verified release wheel.

Wheel digest for this run:
`a06fb0c31d65956bb70effb6c3a4e9777f327db20c2ee918de12f2099869f4b9`.
A rebuild may have a different wheel digest; a new run must consistently use
and record its newly built wheel across every shard and qualification input.
No missing native data or unexpected failure may be relabeled merely to pass
qualification. FPE coverage does not establish deployment or prediction accuracy.
