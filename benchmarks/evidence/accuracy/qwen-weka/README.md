<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen Weka collection

## 2026-11-02 pilot outcome

Collection stopped after native replay qualification failed; the six full
profiles and twelve-run publication did not start. The silicon pilot completed
its 600-second admission window and original grace period. Accounting retains
422 requests: 421 complete records and one cancelled credit without an
individual record. That cancellation has unknown timings and lineage.

The evaluated op-data wheel came from AISim repository commit
`77da57c599994055c04b36027d39d34ac11918e4`, SHA256
`b90910c00462abe235e0089920dbdb57d77b7042a5efcbe1c43d71634dd41237`.
It selected vLLM database 0.31.0 and the reviewed context-attention artifact
`d1188fe9cdb287bb5c87a9094026a17fe50d2401266a71a657facadbfec45700`.
The native telemetry ended with `server_unsettled` after the profile cutoff.
No numerical accuracy result was qualified or published.

Pilot silicon and all 36 feature states are stored as an explicitly unscored,
failed qualification record in the private HF review candidate at silicon
dataset revision `35829eb29e4a3949a60602b9a156adb6b76a9681`. The active catalog
is unchanged. GPU allocations consumed 2810 of 28800 seconds (0.78 node-hours),
including the two client-setup failures; 25990 seconds remain in this campaign's
budget. Collection requires a new successful qualification before resuming.

This campaign replaces active finite Qwen coverage only after six measured
profiles and the complete twelve-run GLM/Qwen replay campaign qualify. Until
then the existing preview remains active. The finite campaign is retained as
`archive/finite-v1`; legacy prompt/cache/repeat URLs select that archive.

## Protocol

- One eight-B300 node, TP8, Qwen3.5-397B-A17B BF16, vLLM 0.31.0.
- AgentX harness 1.0.12; Weka revision
  `23f152f6f0f9399a85901b89a6458def0ef16729`.
- Native context limit 262144; whole-play filtering selects 220 of 393 plays.
  Do not truncate inputs. Preserve the original eligible-play order.
- Seed 42, recorded initial trajectory cuts, live warmup, sequential recycling,
  original trace delays, ten-second global and 300-second tree idle caps.
- An unscored 600-second 14-lane pilot, then one 3600-second profile at each of
  1, 4, 8, 10, 13 and 14 lanes. Use fresh servers, 30-second response grace and
  the harness's ten-second cancellation drain.
- Prefix caching and hybrid KV/recurrent-state handling enabled, synchronous
  continuous batching, chunked prefill, 8192 batched tokens, 64 sequences.
- No speculation, offload, KVCR, Planner, disaggregation, additional parallelism,
  cache-bust, shuffled recycling, quantized weights/KV, or multimodal inputs.

The pinned harness normally requires a profile of at least 900 seconds and
`first_turn_prefix` cache-bust. The requested ten-minute pilot and cache-bust-off
baseline therefore use its explicit `--unsafe-override` option. Retain these
warnings and `submission_valid=false`; do not describe these measurements as
valid official leaderboard submissions. The harness controller and corpus
loader remain unchanged.

## Collection

`prepare.sbatch` installs an isolated client and pins/downloads the corpus on a
CPU node. `preflight.sbatch` resolves the official harness configuration and
checks its actual whole-play filter without sending requests. `collect.sbatch`
freezes the collection code per job and runs `collect.py` on one GPU node.
The receipt probe reads actual per-rank server/cache configuration after startup.
Raw requests, streams, full configuration, snapshots and one-second server
metrics remain private.

Set `CAMPAIGN_ROOT` to private cluster storage. Before **each** allocation,
subtract all prior GPU job elapsed time from 28800 seconds; set
`REMAINING_GPU_SECONDS` and the Slurm `--time` to that remaining budget. CPU
preparation does not consume node-hours. Failed startup allocations do.
Do not restart failed qualification or exceed the aggregate budget.

The collector pauses after the pilot. Create `PILOT_QUALIFIED.json` only after
checking the full measured window, eligible corpus, recorded selections,
request/stream accounting, effective feature settings and a functional native
replay. A successful client exit alone does not qualify it. Without qualification
the collector stops before consuming the time needed for six full profiles.
Before intake, verify frozen executable source hashes against the signed source
commit. Retain that comparison as `collector-source.json` beside `execution.json`;
normalization carries this receipt into HF when present. Prediction provenance
records its separately evaluated wheel and commit.

## Normalization and HF

`normalize.py --prepare-corpus` verifies the original checksum and emits the
eligible corpus. Native namespaces depend on **filtered** row ordinals; retain
its generated namespace map. `--run` normalizes an actual completed profile,
preserves every request, imports recorded cuts, and uses the actual cache receipt.
Use an op-based replay template with timing corrections disabled.
Run this CPU utility with `uv run --with blake3==1.0.8 python normalize.py ...`.
Run the synthetic producer tests with the same dependency plus pytest:
`uv run --with blake3==1.0.8 --with pytest python -m pytest -c /dev/null test_normalize.py`.
Supply selected intervals with `--silicon`, generated by the worker's
`ci.agentx_silicon` adapter. After all six profiles finish, `--manifest` freezes
their identities and artifacts for reviewed intake. It refuses incomplete runs.
Raw streaming records retain output token IDs privately. Numerical comparisons
use only conserved chunk counts and recorded timestamps; missing chunk accounting
stays unavailable. Preemption counts identify their covered observation intervals;
missing scrapes do not become zeros or prove a feature was disabled.
Raw context-overflow records omitted from the harness's ordinary metric export
remain in measured accounting with explicit exclusions. Spawn/join counters
retain their harness-export scope and definitions; request-derived counts name
their preparation/profile scope and missing observations separately.

Silicon feature records use `aisimulate.benchmark.features.v1`, linked through
`runs[].artifacts.feature_metadata` with path, SHA256 and byte count. Stable IDs
in `scripts/e2e_accuracy/benchmark_features.py` identify feature-matrix concepts;
the taxonomy pins the matrix source commit and TOML checksum. These identifiers
are independent of display labels. State is enabled, disabled, not applicable or
unknown; observed use is separately counted. An enabled feature does not prove
it was exercised. Unsupported event instrumentation remains explicitly unknown.
Evidence hashes identify retained artifacts; JSON pointers locate settings.

The dataset worker imports an explicitly reviewed six-profile manifest alongside
the six GLM points. It opens a private HF candidate, never promotes it. A daily
campaign uses one resolved dataset revision and each evaluated branch's own
wheel. Its prediction settings never overwrite measured metadata. Public detail
payloads contain only the validated numerical/feature projection, without raw
prompts, token arrays, logs, credentials or local paths. vLLM running/waiting
gauges can be compared directly; cache/prompt counters with different meanings
remain unavailable for comparison, while the original observations are retained.
