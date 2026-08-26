# Engine-only replay golden-point seeds

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Use these configurations as preflight and qualification seeds for AI Simulate's built-in
`RoundRobinComposition`. The long-corpus results are qualified only for the pinned AI
Simulate revision, trace, runner, and configuration recorded below; they are not universal
capacities or portable performance results.

The small seeds come from committed tests. The 5,000-row golden points were measured with
the skill's committed Rust runner. Re-run them on the pinned baseline, then use exactly the
same inputs and configuration on the candidate. Never tune revisions separately.

## Expected results at a glance

These are the currently committed preflight expectations:

| Seed | Expected result |
| --- | --- |
| Aggregated vLLM, one worker | 1 request completes; 4 input / 2 output tokens; virtual duration 14 ms; first token 12 ms; terminal time 14 ms |
| Aggregated vLLM, two workers | Four requests map to sorted decode worker IDs `[0, 0, 1, 1]` in both canonical repetitions |
| Aggregated attention-DP identity | One logical worker exposes rank identities `(0, 0)` and `(0, 1)` |
| Disaggregated vLLM | Request completes on prefill/decode worker 0 with every handoff lifecycle timestamp present and zero route-overlap tokens |
| Disaggregated SGLang | Same lifecycle completeness and zero route-overlap requirement under SGLang timing |
| Disaggregated TRT-LLM | Explicit `UNSUPPORTED` error; an aggregated substitute does not pass this row |

The authoritative long-corpus status is different:

| 5,000-row requirement | Current status in AI Simulate |
| --- | --- |
| Canonical 5,000-row input trace | **REPRODUCIBLE FROM PINNED UPSTREAM; NOT COMMITTED LOCALLY** |
| Qualified engine-only configurations | **QUALIFIED ON `fb7b0d56de035d66ca9fcae7ff424055dcbf37cd`** |
| Expected pressure/reuse/worker/handoff counters | **QUALIFIED; SEE TABLES BELOW** |
| Baseline canonical digests | **QUALIFIED IN TWO FRESH PROCESSES PER ROW** |
| Paired performance results | **OUTSIDE THIS SKILL; REQUIRES A FUTURE PURPOSE-BUILT HARNESS** |

## Reconstruct the canonical 5,000-row trace

The historical campaign used the first 5,000 rows of Mooncake's 23,608-row arXiv trace.
Pin the upstream Git commit rather than downloading from the moving `main` branch:

- repository: `https://github.com/kvcache-ai/Mooncake`;
- commit: `dedfbde5daa3d5bc020adfd1d6e01c1d544f2098`;
- source path: `FAST25-release/arxiv-trace/mooncake_trace.jsonl`;
- source rows: 23,608;
- full-source SHA-256:
  `b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711`;
- slice rule: lines 1 through 5,000, equivalent to zero-based rows 0 through 4,999,
  preserving source arrival order; and
- slice SHA-256:
  `3892ae19ae480b643155f0c6b9d798591cbe2e73bec6a0fa5ae3d3bc0332fb8a`.

Reconstruct and verify it with:

```bash
set -euo pipefail

MOONCAKE_COMMIT=dedfbde5daa3d5bc020adfd1d6e01c1d544f2098
MOONCAKE_DIR=/tmp/aisimulate-replay-parity-mooncake
MOONCAKE_FULL="$MOONCAKE_DIR/mooncake_trace.jsonl"
MOONCAKE_5000="$MOONCAKE_DIR/mooncake_trace.rows-000000-004999.jsonl"

mkdir -p "$MOONCAKE_DIR"
curl -L \
  "https://raw.githubusercontent.com/kvcache-ai/Mooncake/$MOONCAKE_COMMIT/FAST25-release/arxiv-trace/mooncake_trace.jsonl" \
  -o "$MOONCAKE_FULL"

printf '%s  %s\n' \
  'b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711' \
  "$MOONCAKE_FULL" \
  | shasum -a 256 -c -
test "$(wc -l < "$MOONCAKE_FULL" | tr -d ' ')" = 23608

head -n 5000 "$MOONCAKE_FULL" > "$MOONCAKE_5000"
printf '%s  %s\n' \
  '3892ae19ae480b643155f0c6b9d798591cbe2e73bec6a0fa5ae3d3bc0332fb8a' \
  "$MOONCAKE_5000" \
  | shasum -a 256 -c -
test "$(wc -l < "$MOONCAKE_5000" | tr -d ' ')" = 5000
```

Stop if the full-file checksum, row count, slice checksum, or slice row count differs. Do
not fall back to the moving `main` URL, accept a checksum mismatch, or construct 5,000 rows
by duplicating a shorter fixture.

This pinned source and slice were independently verified while authoring this reference.
The commit-qualified raw URL produced both checksums above.

## Build and run the qualification runner

The skill includes a standalone Rust runner at
`scripts/replay-parity-runner`. It reads one frozen JSON config, verifies the trace SHA,
forces `RoundRobinComposition` with no scaling, enables `CanonicalV1`, and writes:

- `canonical.jsonl`, one canonical record;
- `qualification.jsonl`, the counters and canonical SHA; and
- optional `report.json` when `write_full_report` is enabled.

Do not assume the runner exists inside a historical baseline checkout. Use the skill's
builder to copy the exact same runner source and lockfile into isolated build directories,
then point each generated manifest at that checkout's `crates/core`:

```bash
set -euo pipefail

SKILL_ROOT=python/aisimulate/.agents/skills/aisimulate-replay-parity
BUILDER="$SKILL_ROOT/scripts/build_runner.py"
BASELINE_CHECKOUT=/path/to/baseline-worktree
CANDIDATE_CHECKOUT=/path/to/candidate-worktree
CAMPAIGN_ROOT=/tmp/aisimulate-engine-parity

mkdir -p "$CAMPAIGN_ROOT"
python3 "$BUILDER" \
  --checkout "$BASELINE_CHECKOUT" \
  --output-dir "$CAMPAIGN_ROOT/baseline-runner" \
  --toolchain 1.93.1 \
  > "$CAMPAIGN_ROOT/baseline-build.json"
python3 "$BUILDER" \
  --checkout "$CANDIDATE_CHECKOUT" \
  --output-dir "$CAMPAIGN_ROOT/candidate-runner" \
  --toolchain 1.93.1 \
  > "$CAMPAIGN_ROOT/candidate-build.json"
```

The two build records must report the same `runner_source_sha256`; they separately record
the checkout revision, the same revision embedded into the binary, and the built binary
SHA. The runner does not accept a user-supplied source revision. A runner compilation
failure on an older revision is a compatibility blocker to resolve explicitly, not
permission to use a different runner. The builder rejects dirty checkouts by default. Use
`--allow-dirty-checkout` only for a documented diagnostic build, never for an authoritative
campaign artifact.

`--toolchain 1.93.1` uses `rustup run 1.93.1 cargo ...`. Omit `--toolchain` to invoke the
selected `--cargo` executable directly, including a Homebrew Cargo installation; the
builder never passes rustup's `+toolchain` shorthand to Cargo itself.

For initial golden qualification against the current checkout, use
`$CAMPAIGN_ROOT/baseline-runner/target/release/aisimulate-replay-parity-runner`. After
reconstructing the trace at the documented `/tmp` path, run every row twice in separate
processes:

```bash
set -euo pipefail

RUNNER_DIR=python/aisimulate/.agents/skills/aisimulate-replay-parity/scripts/replay-parity-runner
RUNNER="$CAMPAIGN_ROOT/baseline-runner/target/release/aisimulate-replay-parity-runner"
CONFIG_DIR="$RUNNER_DIR/configs"
OUTPUT_ROOT=/tmp/aisimulate-engine-golden

mkdir -p "$OUTPUT_ROOT"
for ROW in \
  vllm-aggregated vllm-disaggregated \
  sglang-aggregated sglang-disaggregated \
  trtllm-aggregated
do
  "$RUNNER" "$CONFIG_DIR/$ROW.json" "$OUTPUT_ROOT/$ROW-run1"
  "$RUNNER" "$CONFIG_DIR/$ROW.json" "$OUTPUT_ROOT/$ROW-run2"
  cmp "$OUTPUT_ROOT/$ROW-run1/canonical.jsonl" \
      "$OUTPUT_ROOT/$ROW-run2/canonical.jsonl"
done
```

Each output directory must be absent or empty. Golden configs contain partial expected
results and fail immediately on counter or digest drift. For a new comparison, reuse the
same config bytes for baseline and candidate; each runner reports its build-embedded
checkout revision. Preserve all semantic fields and baseline expectations when running the
candidate. To inspect an intentional mismatch, rerun into a new directory with
`expected: {}` and `write_full_report: true`. Do not copy Dynamo's KV-aware expected
counters into an engine-only result.

## Committed fixed-timing seeds

The primary source is `crates/core/tests/engine.rs`. These tests are deterministic semantic
seeds, not timing benchmarks.

### Aggregated single-worker seed

`built_in_aggregated_replay_produces_a_deterministic_report` uses:

- backend: vLLM defaults;
- topology: aggregated, one worker, zero startup delay;
- placement/scaling: round-robin / none;
- DP/TP: 1 / 1;
- GPU blocks: 16;
- block size: 4;
- max sequences: 4;
- max batched tokens: 64;
- timing: fixed 10 ms prefill and 2 ms decode;
- request: arrival 0 ms, 4 input tokens, 2 output tokens; and
- per-request capture enabled with canonical request identities.

Expected semantic output:

| Signal | Expected value |
| --- | --- |
| Completed requests | 1 |
| Input / output tokens | 4 / 2 |
| Virtual duration | 14 ms |
| Decode GPUs per worker | 1 |
| First token | 12 ms |
| Terminal time | 14 ms |

Use this seed to verify runner wiring, canonical determinism, report construction, and the
closed volatile-field exclusion list.

### Aggregated multi-worker round-robin seed

`multi_worker_round_robin_uses_each_logical_worker_deterministically` starts from the same
engine configuration with:

- two aggregated workers;
- fixed 1 ms prefill and 1 ms decode timing; and
- four simultaneous requests, each with 4 input tokens and 1 output token.

The sorted decode worker identities must be `[0, 0, 1, 1]` on both canonical repetitions.
Use this seed to verify that the campaign metadata and evidence expose all configured
workers. Do not infer long-corpus pressure or fairness from four requests.

### Aggregated attention-DP identity seed

`attention_dp_offline_fpm_preserves_logical_worker_and_rank_identity` uses one aggregated
logical worker with DP=2, fixed 100 ms prefill/decode timing, and one request preassigned to
each rank. Its focused test composition captures FPM identities and expects:

```text
[(worker 0, rank 0), (worker 0, rank 1)]
```

This seed validates grouped-rank identity and is not part of the default built-in
no-scaling campaign because the focused test injects a capture scaling policy. For an
authoritative attention-DP row, retain the built-in round-robin/no-scaling composition and
capture equivalent rank/barrier evidence without changing semantics.

### vLLM disaggregated seed

`native_vllm_disaggregated_replay_completes_deterministically` uses:

- one prefill and one decode worker, both role DP=1 and TP=1;
- backend-native block size with 32 GPU blocks, max sequences 4, and max batched tokens 64
  for each role;
- fixed prefill-role timing of 3 ms prefill / 1 ms decode;
- fixed decode-role timing of 3 ms prefill / 2 ms decode;
- 1 ms configured handoff latency; and
- one request with 4 input tokens and 2 output tokens.

Both canonical repetitions must complete the request and expose prefill/decode worker 0
plus non-null prefill admission, source-held, destination-reserved,
destination-activated, decode-admission, and source-released timestamps. The seed expects
zero route-overlap tokens because the built-in composition is engine-only round-robin.

### SGLang disaggregated seed

`native_sglang_disaggregated_replay_completes_deterministically` uses the same topology,
capacity shape, request, and handoff delay with:

- SGLang backend defaults;
- fixed prefill-role timing of 4 ms prefill / 1 ms decode; and
- fixed decode-role timing of 4 ms prefill / 2 ms decode.

It requires the same handoff lifecycle fields and zero route-overlap observations. Use it
to catch backend-specific handoff-order and completion-visibility drift.

### Explicit unsupported seed

`native_trtllm_disaggregated_replay_is_an_explicit_error` requires TRT-LLM disaggregated
replay to fail with an explicit unsupported error. Do not replace this with an aggregated
run and claim disaggregated coverage.

## Committed trace preflight fixtures

These checksums describe the files on main when this reference was authored. Recompute and
record the checksum from each pinned revision; a change requires inspection rather than an
automatic checksum update.

| Format and role | Path | Rows | SHA-256 |
| --- | --- | ---: | --- |
| Mooncake trace timestamps | `tests/e2e/configs/unified_cli/fixtures/traces/mooncake.jsonl` | 2 | `eb40484ba40d9ada82ebe85d5c9ad5c7f2be060e76769e6f0d15f0c625b40267` |
| Mooncake delta concurrency | `tests/e2e/configs/unified_cli/fixtures/traces/mooncake-delta.jsonl` | 2 | `b620a5dd131929f47dfacecc5f8d269c0596cb8f4671845b0e0fcc47a2a36e5c` |
| Agentic Mooncake | `tests/e2e/configs/unified_cli/fixtures/traces/agentic-mooncake.jsonl` | 4 | `a38695cbe4f2e4204343e4d4bc3bf64be99594aeabd69b43b2a1ee46f6934fee` |
| Sweeper search smoke | `tests/sweeper/data/mooncake_tiny.jsonl` | 16 | `a3f74a57f7bfd98301eb0eb57f4b5ac04fc5eea400d901d77edd8c1fcb3b737b` |

The corresponding unified engine CLI seeds are:

- `tests/e2e/configs/unified_cli/predict/engine/04-trace-mooncake-speedup.yaml`;
- `tests/e2e/configs/unified_cli/predict/engine/05-trace-mooncake-delta-concurrency.yaml`;
  and
- `tests/e2e/configs/unified_cli/predict/engine/06-trace-agentic-mooncake.yaml`.

They use a tiny model, fixed engine timing, one aggregated vLLM worker, and small fixed KV
capacity. Use them to verify trace parsing, speedup, concurrency, and agentic lowering.
They do not exercise the authoritative multi-worker/backend/topology matrix.

## Qualified internal-polynomial long-corpus golden points

These results were qualified on 2026-08-26 against AI Simulate
`fb7b0d56de035d66ca9fcae7ff424055dcbf37cd`, using Rust/Cargo 1.93.1 in release mode,
the pinned 5,000-row trace above, trace block size 512, arrival speedup 4, model/decode
speedups 1, DP=1, TP=1, round-robin placement, and no scaling. Qualification ran on an
Apple M3 Pro host; all values below are semantic virtual-time results except where noted.
The runner-source SHA-256 reported by `build_runner.py` was
`2985f08ebf09fc95f0c131b4973ef0cb245c602fe1e2c944bc26c4277f2f1f54`.

### Frozen configurations

| Row | Workers | Engine blocks / block size | Max seqs / batch tokens | Disaggregated transfer |
| --- | --- | --- | --- | --- |
| vLLM aggregated | 4 aggregated | 6,144 / 64 | 16 / 8,192 | N/A |
| vLLM disaggregated | 2 prefill + 2 decode | 9,000 / 64 per role | 16 / 8,192 | 1 byte/token, 100 GB/s, full prompt |
| SGLang aggregated | 4 aggregated | 1,536 / 512 | 256 / 32,768 | N/A |
| SGLang disaggregated | 2 prefill + 2 decode | 12,500 / 512 per role | 256 / 32,768 | 262,144 bytes/token, 100 GB/s, full prompt |
| TRT-LLM aggregated | 4 aggregated | 3,867 / 32 | 16 / 8,192 | N/A |

The exact executable inputs are committed under
`scripts/replay-parity-runner/configs/`. Their SHA-256 values are:

| Row | Config SHA-256 |
| --- | --- |
| vLLM aggregated | `e728259700d541bbbb654ec5b22dfe3e296ed8bcf02f2787e8712cb29ef9e437` |
| vLLM disaggregated | `6e32d682c1916a14a7ebe7d70dc1acf98bceef05e2aa5ca26e77f74ec88cb484` |
| SGLang aggregated | `c3633206f9c229b0ae0c07c38f7596c1d9371510d69c2100f0800bc5d668c8c7` |
| SGLang disaggregated | `dd751fc51710a1d61f9ce4a53176ccbefa49c32116ab898f2e6bd213ecd86088` |
| TRT-LLM aggregated | `4298fd49e83a2245f98d326355fab7c5f5ab5f7e95d68fb427e83e10e5839ca1` |

### Expected counters and canonical digests

Every row must report:

- 5,000 completed requests and no rejected, canceled, failed, or stranded requests;
- 46,542,297 input tokens;
- 922,544 requested and emitted output tokens;
- zero short-output requests;
- every configured logical worker observed; and
- every pressure record readmitted.

Additional row-specific golden values are:

| Row | Pressure | Requests with reuse | Worker / handoff evidence | Virtual duration (ms) | Canonical SHA-256 |
| --- | ---: | ---: | --- | ---: | --- |
| vLLM aggregated | 2 preemptions | 4,910 | decode workers 0–3 | 588,733.1598880008 | `9932920c5cfa60df59ac89ea7439b50cbb47e757cdb2f5e8e4204c50458669e9` |
| vLLM disaggregated | 1 preemption | 4,991 | prefill/decode workers 0–1; 5,000 complete handoffs | 678,340.9482381875 | `124439d723a989e76ac1d3e7cc1df1479b2d7df73affa3bd6f14d7d5e24f76b4` |
| SGLang aggregated | 2 retractions | 4,825 | decode workers 0–3 | 336,898.96313401073 | `f4babef8ce600873345afdcbc2d430fa1312ec2f7922f2eae8d44fb1758dfb11` |
| SGLang disaggregated | 1 retraction | 4,991 | prefill/decode workers 0–1; 5,000 complete handoffs | 353,777.14366691856 | `d90439168ab1165f206c6023bb559d73b9f15d9fdabe7f931b6357b2d3b50c16` |
| TRT-LLM aggregated | 0, by guaranteed-no-evict policy | 4,260 | decode workers 0–3 | 1,232,603.1052780284 | `fbe366f4729baccb75c3b267d9c2e311eb2789c6ad2bf2e5878aabac9c20f398` |

All 5,000 placements were immediate in the built-in round-robin composition. The
disaggregated rows emitted one immediate prefill and one immediate decode placement per
request. Canonical SHA-256 is over the single canonical JSON line including its trailing
newline. Two fresh processes produced the same digest and counters for every row.

### Capacity boundary observations

Use these points as drift detectors, not as permission to retune the candidate:

| Row | Lower observation | Frozen point | Upper observation |
| --- | --- | --- | --- |
| vLLM aggregated | 4,096 blocks → 20 preemptions | 6,144 → 2 | 8,192 → 0 |
| vLLM disaggregated | 8,192 blocks/role → 4 preemptions | 9,000 → 1 | 10,000 → 0 |
| SGLang aggregated | 512 pages → 20 retractions; 1,024 → 2 | 1,536 → 2 | 2,048 → 0 |
| SGLang disaggregated | 12,000 pages/role → 6 retractions | 12,500 → 1 | 13,000 → 0 |
| TRT-LLM aggregated | 3,866 blocks → 27 missing output tokens despite 5,000 terminal completions | 3,867 → all output tokens | 4,096 → all output tokens |

The TRT-LLM boundary is why completed-request count alone is insufficient: qualification
must also require exact authored output-token completion.

### Performance is outside the current skill

Do not treat the runner's `reported_wall_time_ms` or `measured_process_ms` as a portable
golden. The initial runs enabled canonical capture on an interactive desktop and did not
use the required isolated `replay_execution_ms` boundary. The current skill intentionally
does not execute a performance gate. A future purpose-built performance harness must add
that boundary and paired orchestration before reporting performance. Static golden points
cover deterministic semantics, lifecycle counters, capacity boundaries, and canonical
digests only.

## Requalify on a new baseline

For every row:

1. Start from the committed frozen config and run it twice in fresh processes.
2. If the canonical digest or counters drift, determine whether the change is intentional
   before searching for a new capacity.
3. When requalification is necessary, change one capacity or concurrency dimension at a
   time on the pinned baseline.
4. Freeze the nearest stable configuration with one to three applicable pressure events,
   complete output tokens, and no terminal failures.
5. Run the candidate with the exact frozen configuration; never search again on the
   candidate.
6. Record lower and upper capacity observations so later campaigns can detect fixture
   drift.

A changed counter is not automatically a product failure, but it requires explanation and
baseline requalification before performance collection. Do not add guessed or single-run
values.
