<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AgentX replay with an explicit MTP assumption

Offline AgentX replay can use MTP on vLLM and SGLang engines, in aggregated or
prefill/decode-separated deployments. The engine and Dynamo stacks use the same
native speculative scheduler and canonical performance model. These runs test
simulation behavior and modeled iteration cost. They retain `functional_only`
qualification; they are not GPU measurements or AgentX benchmark results.

## Configure the method and acceptance

```yaml
engine:
  model: nvidia/GLM-5.2-NVFP4
  hardware: b300_sxm
  backend: sglang
  backend_version: 0.5.14
  estimation_mode: op_level
  fallback_policy: deny
  speculation:
    kind: mtp
    num_speculative_tokens: 3
    expected_accepted_tokens: 1.99
    seed: 42
```

`num_speculative_tokens` is the maximum proposed draft count, currently 1–5.
`expected_accepted_tokens` excludes the base token: the example expects 2.99
accepted tokens per decode forward including the base token, with four positions
verified. The mean is an explicit workload assumption, not a model prediction.
The sampler realizes a fractional mean using the two neighboring integer counts;
1.99 accepted drafts becomes one accepted draft with probability 0.01 and two
with probability 0.99. It does not recover an unobserved acceptance distribution.
Use the same seed and complete execution configuration to reproduce a run.
Sampling streams are worker-local. Changing routing or worker counts can change
which draws a request receives; a shared seed does not remove that A/B noise.

The op-level model prices widened target verification and approximate extra MTP
layers. Acceptance advances output progress in the scheduler; it does not divide
the cost of each forward. MTP can increase iteration latency while reducing the
number of iterations. A throughput improvement is not guaranteed at every batch
size or prompt/output length.

The existing `nextn` and `nextn_accepted` inputs remain available. Use one form
per engine: an explicit `speculation` block cannot be combined with active
legacy `nextn`. The explicit method is retained in saved configurations and cost
provenance. Acceptance and the sampling seed are replay inputs, outside the cost
model configuration.

All three names refer to accepted **draft** tokens: public
`expected_accepted_tokens`, legacy `nextn_accepted`, and report
`expected_accepted_draft_tokens`. Only reported mean AL includes the base token.
Zero legacy depth and `None` disable SD. Newly enabled Agentic replay requires
an explicit accepted-draft expectation or nonempty conditional rates even at
the native boundary; older non-Agentic NextN defaults remain compatible.
Agentic MTP rejects fixed/polynomial timing because those inputs do not supply
the required draft and target-verification cost model.

Adapters share `aisimulate.speculation.normalize_speculation_engine_args`.
Explicit methods require an AIC cost identity before lowering; the engine
runner resolves flat AIC inputs first. Ambiguous public speculation combined
with a directly supplied Dynamo `ais_perf_config` is rejected. Installing an
older package with the same `0.13.0` version does not supply this API: the Dynamo
SD adapter checks feature availability and gives a paired-source installation
error. SD-off paths do not depend on the new helper, but still require the
adapter's underlying replay API contract.

## Reference scenarios

The following original AISimulate examples use a small checked-in Weka trace,
B300 (`b300_sxm`), SGLang 0.5.14 and TP8 per worker:

| Scenario | Configuration | Draft count | Expected accepted drafts |
| --- | --- | --- | --- |
| GLM-5.2 NVFP4, aggregated | [GLM aggregated](../tests/e2e/configs/unified_cli/predict/engine/13-trace-weka-glm52-mtp-agg.yaml) | 3 | 1.99 |
| GLM-5.2 NVFP4, P/D | [GLM P/D](../tests/e2e/configs/unified_cli/predict/engine/14-trace-weka-glm52-mtp-disagg.yaml) | 3 | 1.99 |
| DSV4-Pro, aggregated | [DSV4 aggregated](../tests/e2e/configs/unified_cli/predict/engine/15-trace-weka-dsv4-mtp-agg.yaml) | 3 | 1.50 |
| DSV4-Pro, P/D | [DSV4 P/D](../tests/e2e/configs/unified_cli/predict/engine/16-trace-weka-dsv4-mtp-disagg.yaml) | 3 | 1.50 |

The GLM assumption references the K3, thinking-on AL2.99 entry in
[InferenceX's GLM-5.2 MTP acceptance reference](https://github.com/SemiAnalysisAI/InferenceX/blob/400dfe463877e50a1c4bb9705d99919b773c33bb/inferencex-e2e/infx/golden_al_distribution/glm5.2_mtp.yaml),
pinned at `400dfe463877e50a1c4bb9705d99919b773c33bb`. That source identifies
SPEED-Bench coding data, temperature 1.0, top-p 0.95, output length 4096, and
**GLM-5.2 FP8 on B300 with vLLM MTP**. Reusing that mean for this NVFP4/SGLang
simulation is an assumption, not an NVFP4 acceptance measurement. No upstream
deployment YAML is copied into these examples.

The DSV4 scenario explicitly selects target-shaped MTP as a hypothetical cost
model. It preserves the DSV4 target, attention and MoE identities; it does not
claim to reproduce its real DSpark draft model. Its 1.50 accepted-draft mean is
illustrative. Other target models may select the same method if their model
implementation supports that cost graph. Unsupported combinations still fail;
in particular, an explicit MTP override cannot relabel Kimi-K3's existing
DSpark-specific `nextn` graph.

Each example fixes 8192 rank-local KV blocks of 64 tokens. That is a scenario
capacity, not a measured GPU-memory fit. The current automatic replay capacity
estimator omits speculative state. For a deployment comparison, replace these
values with the available KV capacity for that deployment, including its draft
and graph reservations. P/D roles each receive their own capacity; do not divide
one aggregate block count between roles or silently copy an AR capacity into an
SD capacity qualification. The example's 400 GB/s KV link is also an assumption.

## Run and inspect

From the repository root with this checkout installed:

```bash
aisimulate predict \
  --config tests/e2e/configs/unified_cli/predict/engine/13-trace-weka-glm52-mtp-agg.yaml \
  --output-dir /tmp/agentx-mtp-glm-agg
```

Select any other row above for the P/D or DSV4 case. With a compatible Dynamo
installation, add `--stack dynamo` to run through the Dynamo adapter. The same
MTP fields apply; stack selection does not change the acceptance assumption.
For a vLLM scenario, set `backend: vllm` and a compatible literal version such as
`0.24.0`, and recheck performance-data coverage for the actual target and system.

For the longer corpus prepared by the [AgentX quickstart](agentx-quickstart.md),
replace `traffic.source.paths` with
`[/tmp/agentx-quickstart/plays-0000-0001.jsonl]` and set the source's block size to
64. The bundled tiny fixture uses hash blocks of four tokens. That trace hash
block size is independent of the engine KV block size. The longer corpus also
requires the quickstart's host-memory planning and an appropriate execution
budget.

Inspect the replay's effective method, configured acceptance, completed requests,
actual output tokens, decode forwards and `speculative_acceptance.mean_accept_length`.
`speculative_acceptance.mean_accept_length` includes the base token, excludes prefill, and is sampled
before output-limit truncation. A tiny trace need not realize the exact expected
mean; final bursts may emit fewer tokens than were accepted. Each request must
complete once, and its next turn or tool action must start only after completion.

The supported initial combination is HBM-only ordinary KV caching with op-level
timing. Host offload, grouped KV/FPM cache, incompatible `prefix_match_unit`, and
unimplemented draft methods remain rejected. Changing the speculative method
does not relax the target's cache or topology constraints. Disabling speculation
should retain the ordinary replay behavior.

## Measurement and qualification contract

`speculative_acceptance.decode_forwards` counts completed request decode forwards;
`accepted_tokens_including_base` counts their sampled acceptance before final
output clipping. Their ratio is `mean_accept_length` (null with no decode
forwards). Prefill and warmup do not enter this population. In a continuous
profile, include passes completed from the measurement barrier through the
response-grace deadline, including equality; exclude later cancellation drain.
The population can include earlier decode work from a request that is ultimately
canceled. It is separate from delivered-token throughput of successful requests.

Reports also retain the configured target, requested and resolved method, draft
depth, accepted-draft assumption, seed, cost approximation and capacity source.
For DSV4 P/D, `kv_transfer_approximation: scalar_bytes_per_token` explicitly means
the existing scalar transfer model. This qualification does not add precise
compressed/grouped-cache transfer modeling or validate its hardware accuracy.

These SD changes are based directly on each repository's main branch. Engine
replay supports its existing duration profiles, snapshots and warmup. Dynamo
replay uses its existing ordinary round-robin and KV-router paths in aggregated
and P/D deployments. The Dynamo adapter does not advertise profile, snapshot,
warmup, session-affinity or sibling-group-affinity support from the separate
routing integration.

The wider routing matrix in the
[historical stacked qualification](https://github.com/ai-dynamo/aisimulate/blob/69b18641d796557aaec52c5977dca5e958ccd8c6/docs/agentx-mtp-qualification.json)
was tested with unmerged routing code. It is evidence for that source pair only;
these standalone SD PRs do not import those routing changes or claim that the
same combinations are available on main. Requalify those combinations when
the routing integration is available.

Run installed-pair qualification with the two complete real Weka plays from
the [AgentX quickstart](agentx-quickstart.md):

```bash
/path/to/paired/venv/bin/python scripts/qualify_agentx_mtp.py \
  --cli /path/to/paired/venv/bin/aisimulate \
  --trace /tmp/agentx-quickstart/plays-0000-0001.jsonl \
  --scope all --duration 3600 --jobs 2 --output /tmp/agentx-mtp-qualification
```

The scopes have different stopping conditions:

| Scope | Scenarios | Execution boundary |
| --- | --- | --- |
| `--scope engine` | GLM vLLM/SGLang × agg/P-D (4); DSV4 SGLang × agg/P-D × SD off/on (4) | Each runs 3600 seconds of admission after snapshot preparation and warmup |
| `--scope dynamo` | GLM vLLM/SGLang × agg/P-D × round-robin/KV (8); DSV4 SGLang × agg/P-D × round-robin/KV × SD off/on (8) | Each replays both authored plays to completion, without a duration profile |
| `--scope all` | Both sets (24) | Preserve each scope's boundary; `--duration` applies only to Engine cases |

Every role has two TP8 replicas and fixed KV capacity. SD uses seed 42;
Engine snapshot selection independently uses seed 42. Engine-only qualification
requires only the AISimulate wheel. The Dynamo scope requires all three paired
wheels. The output directory retains the exact qualifier script, its hash,
installed wheel source/hash metadata, configurations and per-case receipts.
Checks remain enabled under `python -O`. CLI failures, missing reports and
failed checks retain an error receipt and make the matrix exit unsuccessfully.

## Standalone support and qualification

The [qualification summary](agentx-mtp-qualification.json) records the tested
main-based source pair, exact artifacts and separate execution scopes. The
real Weka two-play input has SHA-256
`e3a34f0617457a004694be52885d58748b998b6d3c22cf344ff6572a78757d5a`.
Engine checks validate the measurement/grace window and settled requests.
Dynamo checks require every authored play to complete and settle. Separate
boundary tests cover cancellation and acceptance exclusion during drain.

| Target and backend | Interface configurable | Standalone simulation coverage | Hardware accuracy |
| --- | --- | --- | --- |
| GLM-5.2 NVFP4 / vLLM 0.24.0 | MTP, depth 1–5 | Agg/P-D; Engine profiles and Dynamo round-robin/KV | Not evaluated |
| GLM-5.2 NVFP4 / SGLang 0.5.14 | MTP, depth 1–5 | Agg/P-D; Engine profiles and Dynamo round-robin/KV | Not evaluated |
| DSV4-Pro / SGLang 0.5.14 | Explicit hypothetical MTP | Same topologies/routes with SD off/on controls | Not evaluated |
| Dynamo profiles and session/sibling affinity | Separate routing integration | Not qualified by this main-based pair | Not evaluated |
| Ngram | Existing non-AgentX interface | AgentX rejected | Not evaluated |
| Other SD methods | No new public variant | Unsupported combinations rejected | Not evaluated |

The per-forward cost controls compare equivalent legacy NextN and explicit MTP
for both GLM backends and DSV4, and retain positive draft/verification cost
independently of acceptance. The compact JSON records every case, source
revision, script hash and wheel hash. A fresh qualifier run writes full logs,
configurations and receipts to the chosen output directory.

For this matrix, build **all three wheels**: AISimulate, `ai-dynamo-runtime`
with `ais-forward-pass`, and the `ai-dynamo` component package. With the
repositories checked out at the source revisions recorded in the summary and
Dynamo's native build prerequisites available, use absolute paths:

```bash
AIS_MTP_SOURCE=/path/to/aisimulate
DYNAMO_MTP_SOURCE=/path/to/dynamo
MTP_ENV=/tmp/agentx-mtp/venv
MTP_WHEELS=/tmp/agentx-mtp/wheels
uv venv --python 3.12 "$MTP_ENV"
uv pip install --python "$MTP_ENV/bin/python" maturin patchelf
mkdir -p "$MTP_WHEELS"
(cd "$AIS_MTP_SOURCE/python/aisimulate" && \
  "$MTP_ENV/bin/maturin" build --release \
  --interpreter "$MTP_ENV/bin/python" --out "$MTP_WHEELS")
(cd "$DYNAMO_MTP_SOURCE/lib/bindings/python" && \
  "$MTP_ENV/bin/maturin" build --locked --features ais-forward-pass \
  --interpreter "$MTP_ENV/bin/python" --out "$MTP_WHEELS")
(cd "$DYNAMO_MTP_SOURCE" && uv build --wheel --out-dir "$MTP_WHEELS")
uv pip install --python "$MTP_ENV/bin/python" "$MTP_WHEELS"/*.whl
```

Run the qualifier from the AISimulate checkout using this environment's Python
and CLI. The backend names select simulation models; no serving-backend extras
or GPU execution are needed for these cases.

These hashes identify historical source content, not a guarantee that an
unmerged branch commit remains fetchable forever. Both PRs target main. First
merge AISimulate, update every Dynamo source pin to that merged revision, then
rebuild and qualify the installed pair before merging Dynamo.
Keep the archived qualifier and wheel hashes with the corresponding receipt;
do not replace a historical script hash with the hash of a later script.

Rust consumers should also read the [source compatibility notes](core-api.md#choosing-a-forward-pass-api):
the new enum variant and report field require downstream source updates even
though legacy Python inputs retain their behavior.
