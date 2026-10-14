# Fixed-length DSpark: implementation and evidence

Base revision: `cb71f236a7a4d658f6be90ada3c37241d209fadb`.
Phase 1 implements op-level costs and Replay. Phase 2 is a placeholder;
explicit DSpark FPM requests are rejected. No FPM accuracy is claimed.

## Implemented contract

- Canonical Rust/Python `best_available` accepts typed `Dspark` identity.
  Resolved draft configuration and revision survive serialization and caches.
- Fixed K drafts, K+1 verification queries, variable accepted drafts plus one
  bonus. Adaptive verification and simultaneous `nextn` are rejected.
- V4 context projection/KV preparation, parallel draft backbone, logits,
  Markov sampling and target verification use the existing op graph.
  A typed sequence cap carries the draft window through compiled execution.
- Draft weights/KV participate in automatic capacity. Shared target weights
  are counted once. Accepted tokens determine progress, not current-round cost.
- Aggregated and P/D vLLM/SGLang Replay accepts DSpark, including AgentX.
  Recommendations preserve the configuration in saved prediction YAML.
- Automatic P/D transfer includes prepared draft KV once, including when the
  scalar target cache reports a destination prefix hit. Explicit byte overrides
  remain caller-owned totals.

See the existing [speculation guide](../../../docs/replay/engine/speculation.md),
[API contract](../../../docs/perf-model/configuration.md), and
[transfer contract](../../../docs/replay/engine/kv-transfer.md).

## Measurement eligibility

| Priority | Combination | Evidence | Status |
| --- | --- | --- | --- |
| P0 | V4-Pro-0813 / vLLM / 8 B200, TP8 EP1 | AgentX IDs 442741, 442744, 442734, 442737, 442740, 442739; concurrency 1/4/6/10/14/16 | Reference only. K=6 and synthetic AL=3.77 observed; exact adaptive-off execution and workload identity remain unresolved. |
| P1 | V4-Pro-0813 / SGLang / 8 B200, TP8 EP1 | AgentX IDs 442271, 442275, 442272, 442273, 442276; concurrency 1–5 | Reference only. Draft width 6 observed; acceptance, ragged verification and workload reconciliation remain unresolved. |
| Diagnostic | V4-Flash-0731 / vLLM / 4 B200, TP4 EP4 | Playground K=3/5/7 static waves | Available, but `round_ms_*` measures SSE chunk gaps. No independent draft/verify kernel timing; not scored as per-token or operation latency. |
| P/D | V4-Flash-0731 / vLLM / 8 B200, 1P1D TP4 EP4 | Controlled collection manifest and driver | Runtime/checkpoint inspected. GPU collection blocked on available 8-GPU capacity. |

`references.json` records API metadata, artifact IDs, archive/log hashes and
observed launch fields. The source runs are
[vLLM attempt 2](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35434571418/attempts/2)
and [SGLang attempt 1](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34603861241/attempts/1).
Sibling results include offload variants; only the eleven offload-disabled
IDs above belong to the initial matrix.

Eligible paired measured points: **0**. TTFT/TPOT/throughput error, MAPE and
signed bias are **unavailable**, not zero. No numerical pass is declared.
Do not infer fixed DSpark from the public `draft_model` or `mtp` labels.
Recollect a combination if fixed verification or exact corpus identity cannot
be established. Kimi-K3 remains deferred.

## Controlled collection

`runtime-receipt.json` records the successfully inspected vLLM 0.29.0 image,
immutable image/tag identity and available Flash checkpoint. The inspected
image supports `enable_adaptive_verification=false`. The Pro checkpoint was
not found at its expected shared-cache path.
The pinned v0.29.0 configuration source maps synthetic length 3.77 to 2.77
accepted drafts using rates `[1, 1, 0.77, 0, 0, 0]`, matching Replay's
adjacent-integer assumption. This does not qualify the older AgentX producer.

`collect_pd.py` launches one TP4/EP4 producer and consumer, K=6, synthetic
committed length 3.77, FP8 KV, eager execution, prefix caching, 8 GiB KV per GPU,
four sequences per worker, and Mooncake TCP. It records the exact commands,
model-config hash, logs, all input/output token IDs and request metrics.
The controlled corpus has three turns, a 50 ms tool delay, 128 initial tokens,
32 additional input tokens per later turn and 64 output tokens per turn.
One session warms the servers before measured session concurrency 1 and 4.

TTFT starts before the producer HTTP request. TPOT is `(last token arrival -
first token arrival)/(output token count - 1)`, never a mean streaming chunk
gap. This small controlled corpus is separate from AgentX accuracy evidence.
The driver and connector handshake have not yet executed on GPUs.

Target: `nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster`, namespace
`spec-decoding-eval`. Resources: 8 B200, 32 CPU, 512 GiB RAM, read-only
`shared-model-cache` and the task-owned `aisim-dspark-results-20261014` PVC.
KAI queue `dynamo`; no dedicated-node tolerations or unrelated workloads changed.
After 18 minutes without a GPU assignment, the task's inspection/collection Jobs
and ConfigMap were deleted. The results PVC was retained. `collection-status.json`
records the scheduler's resource-shortage events and resume action. No GPU
measurement was produced, and no task GPU allocation remains.

```sh
PY=python/aisimulate/.venv/bin/python
EVIDENCE=benchmarks/evidence/dspark-fixed-20261014
$PY "$EVIDENCE/collect_reference.py"
kubectl --context nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster \
  -n spec-decoding-eval create configmap aisim-dspark-scripts-20261014 \
  --from-file=collect_pd.py="$EVIDENCE/collect_pd.py" --dry-run=client -o yaml | \
kubectl --context nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster \
  -n spec-decoding-eval apply -f -
kubectl --context nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster \
  -n spec-decoding-eval apply -f "$EVIDENCE/collect-pd.yaml"
```

Before scoring the collected run, replay the recorded token IDs, warmup and
closed-loop session load. Resolve actual cache capacity and transfer behavior
from server logs. Pin the serving revision separately from the op database
version. The existing B200 vLLM op data is 0.24.0; it is not vLLM 0.29.0 data.

## Verification

- Rust library: 1920 passed, 1 ignored.
- Python DSpark/ngram/speculation suite: 313 passed.
- Final DSpark feature suite: 51 passed, including Pro TP8/EP1 vLLM/SGLang
  construction and memory, and automatic draft-resolution round trips.
- Final DSpark/ngram and affected Sweeper regression subset: 163 passed.
- Earlier config/runner/Sweeper regression suite: 780 passed.
- Final release parity: 386 passed in 197.59 seconds.
- K=3/5/6/7, zero/full/fractional acceptance, repeatable seeds, short outputs,
  context limits, KV exhaustion, cancellation and rejection rollback covered.
  The state-pressure test also exercises DSpark K=6/7 preemption and completion.
- Aggregated/P/D AgentX with snapshot/warmup, saved recommendation round trips,
  real op-cost P/D with automatic transfer, draft memory, and window caps covered.
- Actual op-level P/D smoke completed one request with all 19 output tokens.
  This is functional evidence, not a silicon accuracy result.
- Ruff, Rust formatting, packaged legal-file checks and explicit ownership
  coverage passed. CODEOWNERS tests: 107 passed.

```sh
cargo test -p aisimulate-core --lib --no-default-features
python/aisimulate/.venv/bin/pytest -p no:timeout -q \
  tests/test_dspark_cli.py tests/test_ngram_cli.py \
  python/aisimulate/tests/unit/sdk/speculation
python/aisimulate/.venv/bin/pytest -p no:timeout -q \
  crates/core/parity_tests/perfmodel/test_compile_engine_parity.py \
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py
```

## PR integration checks

The draft PR rebases only the DSpark implementation/evidence commits onto
`9666978555325f9caa4b816e8509527313f8e330`. It excludes the unrelated commits
from the original development base. Conflict resolution preserves the newer
KV memory module, uses engine-spec schema 29, and adds draft KV to the existing
once-per-handoff `state_bytes` field.

- Rebased Rust library: 1975 passed, 1 ignored, including a regression for
  window-capped draft transfer alongside other request state.
- Ruff, Rust formatting, packaged legal-file equality and CODEOWNERS regeneration
  passed.
- The host measurements and earlier parity counts below describe the recorded
  pre-rebase implementation; they are not new measurements of current main.

## Host cost

`compare_host.py` measures complete release CLI wall time, with one discarded
warmup and alternating base/head repetitions. It fixes thread counts,
checks completed request/token counts, records native binary/config/lockfile
identity, and records peak RSS on macOS. The AgentX fixture is the repository's
129-request, 114540-output-token play. A 128-request non-speculative control
checks ordinary execution. Candidate DSpark is a separate feature-cost cell;
its reduced forward count is not a baseline implementation speedup.

```sh
git worktree add --detach /tmp/aisim-dspark-base-cb71f236 cb71f236
uv sync --project /tmp/aisim-dspark-base-cb71f236/python/aisimulate \
  --extra dev --python "$PWD/python/aisimulate/.venv/bin/python"
# Align the optional HTTP packages present in the measured candidate environment.
uv pip install --python /tmp/aisim-dspark-base-cb71f236/python/aisimulate/.venv/bin/python \
  certifi==2026.7.22 charset-normalizer==3.5.2 idna==3.20 requests==2.34.2 urllib3==2.8.0
# Build the candidate from its Python package directory, which owns the module name.
(cd python/aisimulate && VIRTUAL_ENV="$PWD/.venv" uv tool run maturin develop --release)
python/aisimulate/.venv/bin/python \
  benchmarks/evidence/dspark-fixed-20261014/compare_host.py \
  --base /tmp/aisim-dspark-base-cb71f236 --head "$PWD" \
  --output /tmp/aisim-dspark-host --repeats 5
```

The initial three-repetition run (`host-initial.json`) showed noisy control
timing, including a 4.19 s baseline sample. A second five-repetition run
(`host-stable.json`) used identical production code, binaries and workloads:

| Full CLI case | Baseline median (range), s | Candidate median (range), s | Median RSS, MiB (base / head) |
| --- | --- | --- | --- |
| Non-speculative control | 2.470 (2.408–2.684) | 2.432 (2.339–2.573) | 197.27 / 197.58 |
| Non-speculative AgentX | 3.848 (3.773–4.246) | 3.957 (3.670–4.182) | 196.80 / 197.00 |
| DSpark AgentX | Unsupported | 4.011 (3.667–4.188) | — / 197.70 |

Baseline is `cb71f236`; candidate is signed commit `9d841505`. Loaded native
hashes match their release artifacts; common dependency versions and lockfile
hashes match. A later audit found five extra HTTP packages in the candidate
environment, so these two runs are preliminary. The baseline environment was
then aligned for the final comparison. Candidate changes during the second run
were ownership/evidence only. The preliminary samples vary: initial
control median changed +7.8%, then -1.6%; AgentX changed -2.2%, then +2.8%.
These short CLI samples do not establish performance at larger resident state.
`host-semantic-comparison.json` confirms equal non-speculative completion/token
counts and TTFT/TPOT/E2E predictions; AgentX lifecycle digests also match.

Final dependency-matched results (`host-matched.json`, five repetitions):

| Full CLI case | Baseline median (range), s | Candidate median (range), s | Median RSS, MiB (base / head) |
| --- | --- | --- | --- |
| Non-speculative control | 2.523 (2.432–2.659) | 2.529 (2.408–2.728) | 197.25 / 198.00 |
| Non-speculative AgentX | 3.736 (3.681–3.942) | 3.896 (3.729–3.972) | 199.39 / 198.12 |
| DSpark AgentX | Unsupported | 3.470 (3.418–3.583) | — / 199.27 |

The final medians change +0.24% for the control and +4.29% for AgentX
(+0.160 s). The ranges overlap, but these samples do not rule out a small
host-cost regression. No numerical performance threshold was specified.
The harness now rejects mismatched installed dependency sets before timing.

Implementation cost: resolution and draft-capacity inversion occur during model
construction. The scheduler reuses the existing sampler and reservation paths;
there is no new per-round checkpoint read or draft-model construction.
Ordinary mixed-step composition adds draft-name checks within its existing
operation traversal; it adds no resident-state scan or new retained cache.

## Remaining causes and coverage gaps

1. V4 draft SWA uses a window-capped HCA proxy. Wide-query target verification
   uses token-basis attention; matched query-width measurements are absent.
2. V4 context preparation follows pinned vLLM source. SGLang uses that cost
   approximation until its exact implementation is measured. Norm/KV-store
   work uses bandwidth estimates.
3. Host launch/sampling overhead and eager/graph behavior are uncalibrated.
4. V4 grouped-cache identity is approximated by scalar capacity/prefix reuse.
   Automatic transfer assumes the connector carries prepared draft KV; exact
   payload and initialization behavior need confirmation on the pinned runtime.
5. AgentX acceptance and fixed-verification identity are not yet qualified.
   The observed vLLM 3.77 convention is kept separate from cost-model error.
6. Missing GPU capacity prevents the controlled P/D measurement and paired
   replay. Missing evidence must not be reported as successful validation.

Phase 2 FPM design, collection planning and accuracy tests are deferred.

## Playground diagnostic reproduction

`diagnose_playground.py` retains separate SSE-gap and counter-derived round
observations, plus their source hashes. It queries the canonical op model at
the approximate start/end KV lengths of each wave, reporting predicted draft
and verify components. These components were not measured independently.
The difference percentages are diagnostic; eligible accuracy points remain zero.

`playground-diagnostics.json` contains 18 batch/depth points. At batch 1, the
predicted round ranges are 6.40–6.45 ms (K=3), 6.95–6.99 ms (K=5), and
7.43–7.58 ms (K=7), versus SSE-gap medians 9.55/10.17/10.64 ms. The gaps are
about -28.8% to -33.1%. Unmodeled host overhead, attention proxies and runtime
identity differences remain possible causes; this evidence cannot isolate them.

```sh
python/aisimulate/.venv/bin/python \
  benchmarks/evidence/dspark-fixed-20261014/diagnose_playground.py \
  --playground /path/to/AISimPlayground/playground/dsv4flash-b200-specdec-validation \
  --output benchmarks/evidence/dspark-fixed-20261014/playground-diagnostics.json
```
