# AgentX GLM concurrency replay

CPU replay and original telemetry plots for the GLM-5.3/B300/SGLang concurrency curve, beginning with [InferenceX point 446063](https://inferencex.semianalysis.com/inference/agentic/446063), concurrency one. Each run reproduces the benchmark's preparation and 3,600-second profiling window with the complete source corpus loaded. It does **not** exhaust all 98,827 requests in the corpus.

## Frozen configuration

| Field | Value |
|---|---|
| AISim engine and native build | `6061bfbf11fe7e9651900a96d7f22f227ec2172e` |
| InferenceX source | `c093d2052d2e4f6aeaa8162e9e81219f31556fe4` |
| Actual model | `nvidia/GLM-5.3-NVFP4` |
| Model revision | `e3b8f9f2fad29727dfa95e766c5cac18ea1414bf` |
| Serving system | 8 B300 GPUs, aggregated SGLang 0.5.20 |
| Parallelism | TP8, MoE TP8, EP1, attention DP1 |
| Scheduler | 8,192 prefill tokens; 2 running requests |
| KV | FP8; 64-token blocks; 59,124 logical blocks; prefix caching |
| Context limit | 1,048,576 tokens |
| Speculation | NEXTN 3; expected accepted drafts 1.99 (acceptance length 2.99) |
| Replay | 1 agentic lane; live trajectory warmup; 3,600-second profile |
| Idle caps | 10 seconds globally; 300 seconds per tree |
| Grace | 30 seconds for responses; 10-second cancellation-drain ceiling |
| Telemetry | Native all-phase request/lifecycle records; 1,000-ms samples |
| Timing model | `op_level`, `SILICON`, fallback `deny` |

The logical KV capacity is 3,783,936 tokens per TP rank. The dashboard's 30,271,488-token sum counts replicated TP pools and must not be used as the logical capacity. Native model defaults preserve BF16 attention/shared experts, FP8 initial dense FFNs, and NVFP4 routed experts. No global GEMM precision override is applied.

The seeded initial selection is imported explicitly because AISim and AgentX use different seed samplers:

- Source root: `002001296e8a8c38ad9d7cc436d691afc602`.
- Source play: `weka:fffc8b0f1578cbcf:play:002001296e8a8c38ad9d7cc436d691afc602`.
- Cut: `107334653.55752248` ms; seed 42; corpus cursor 1.
- Keep input basename `traces.jsonl` and the original play order: both participate in native source identities.
- The complete corpus resolves nested timestamps to `absolute`; this is checked in each prediction.

## Inputs and provenance

- [Pinned serving recipe](https://github.com/SemiAnalysisAI/InferenceX/blob/c093d2052d2e4f6aeaa8162e9e81219f31556fe4/inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/glm5.3/sglang/b300-fp4-mtp/agentic.yaml).
- [Upstream benchmark run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39551262792/attempts/1), artifact `12364364163`.
- [Source dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/tree/23f152f6f0f9399a85901b89a6458def0ef16729), revision `23f152f6f0f9399a85901b89a6458def0ef16729`.
- Source trace: 1,847,151,435 bytes, 393 plays, 98,827 model requests. SHA256 `29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`.
- AgentX 1.0.12. Exact image digest, upstream files, download URLs, dataset card, and hashes are preserved in the external output bundle's `source/` directory.

The helper scripts here are self-authored. They do not import or copy InferenceX UI code. Downloaded artifacts and the modified trace remain outside the repository, with their source metadata and dataset license declaration.

Two replay variants preserve all source prompt lengths, block hashes, timestamps, play order, and dependencies:

1. **Original trace:** unchanged source output lengths.
2. **Observed output lengths:** replace only the output limits of exactly joined, successful profiling requests with their measured output lengths. Of 324 successful rows, 134 outputs are shorter than source and 190 are unchanged. Two profiling errors have no measured output length and remain unchanged. Unobserved requests also retain their source lengths. This is workload matching, not timing calibration.

The modified corpus SHA256 is `807ac1a7e4e587606519b1b7910e52a09572ac9cfcdcec5089d58428bb7f2484`. Its manifest lists every change. Reference requests are joined by source root and outer/inner indices; normalized conversation turn indices are not reliable source indices.

## Recorded results

These are single CPU runs on 2026-10-29 UTC, without latency calibration. The observed-output variant is the closer workload comparison; neither reproduces exact HTTP prompts.

| Metric | Reference | Observed output lengths | Original trace |
|---|---:|---:|---:|
| Complete CLI wall time | — | 110.59 s | 111.07 s |
| Native simulation timer | — | 1.31 s | 1.57 s |
| Mean profiling E2E latency | 3.8073 s | 5.4223 s | 9.0418 s |
| Mean E2E point APE (N=1) | — | 42.42% | 137.49% |
| Output throughput | 88.5379 tok/s | 75.4568 tok/s | 124.3124 tok/s |
| Output throughput point APE (N=1) | — | 14.77% | 40.41% |
| Source-matched request E2E MAPE | — | 36.78% (289 pairs) | 186.71% (241 pairs) |
| Equal-output subset E2E MAPE | — | 36.78% (289 pairs) | 27.62% (149 pairs) |
| Successful latency observations | 324 | 289 | 241 |

The observed-output run matches 289/324 reference successes (89.20%); the remaining 35 are beyond simulated progress in the fourth play. All 289 have matching output lengths, but only four also have identical input lengths. Original-trace source-matched MAPE includes output-length differences and is not a timing-only comparison; its equal-output subset covers fewer and different requests than the observed-output run.

Both native runs also retain one completed zero-output source request with null latency, corresponding to a known reference error. It is excluded from latency MAPE. Preparation completes at 19.165856 simulated seconds. Original-trace profiling drains for an additional 19.446014 simulated seconds after admission closes; the observed-output run has no active requests at the cutoff. Neither run cancels requests.

Native artifacts contain 253/301 terminal request rows, 3,642/3,623 samples, and 763/907 lifecycle events for original/observed variants. Output counters reconcile exactly: original 449,942 profiling tokens; observed 271,593; 11 preparation tokens each. Native phase boundaries and absolute timestamp semantics are checked. The initial local attempt was rejected before replay by resource supervision; its 0.31-second duration is preserved separately and is not a replay runtime.

## Remaining GLM concurrency points

The same pinned GLM-5.3/B300/SGLang curve also contains the following points. These replays use **recorded output lengths only**, with no additional original-trace control runs. The prior c1 recorded-output result is reused in the curve summary.

| Point | Concurrency | Scheduler max sequences | Reference successes | Changed output limits |
|---|---:|---:|---:|---:|
| [446068](https://inferencex.semianalysis.com/inference/agentic/446068) | 4 | 8 | 708 | 218 |
| [446069](https://inferencex.semianalysis.com/inference/agentic/446069) | 8 | 256 | 1,426 | 400 |
| [446060](https://inferencex.semianalysis.com/inference/agentic/446060) | 10 | 256 | 1,607 | 441 |
| [446061](https://inferencex.semianalysis.com/inference/agentic/446061) | 13 | 256 | 2,191 | 468 |
| [446062](https://inferencex.semianalysis.com/inference/agentic/446062) | 14 | 256 | 2,454 | 571 |

Per-point server exports confirm the same 59,124-block logical KV pool, runtime image, topology, precision, and speculation settings as c1. The first N corpus plays initialize the N lanes, using exact imported cuts; the shared corpus cursor starts at N. Lane 6 has multiple live branches at its cut, so preparation can have multiple primers per lane. Native lineage and phase records preserve these branches. Each point retains its own measured profile lengths, failures, snapshot metadata, raw profile hash, and derived-corpus hash. No source request identities repeat within a reference point.

`prepare_inputs.py --point-id ID` validates the reference counts, successful output-token sum, and raw profile hash against `source/source-manifest.json`. `score_replay.py --point-id ID` labels the point without changing matching or scoring rules. Both options default to 446063 for existing commands. Failed or unavailable reference requests retain their source output limit; requests beyond the measured cohort also retain their source lengths. Strict request MAPE includes only source-matched successes with equal delivered output lengths.

The external curve bundle uses `point-ID/{source,observed-input,observed-replay,observed-report}` plus `point-ID/observed-remote.yaml` and `point-ID/observed-replay-execution.json`. The complete corpus remains 393 plays / 98,827 requests for every point. Keep each corpus basename `traces.jsonl`; inputs can share a hardlink to the unchanged original corpus before derivation. Point configs retain the exact scheduler differences above and set `nested_timestamp_basis: absolute`.

Use the existing preparation, runner, scoring, and rendering commands with the point directory and `--point-id` supplied to preparation/scoring. Run points sequentially on the CPU host; wall times are single observations, not a repeated host-performance study. They include the same complete CLI scope as c1 and exclude source derivation, transfers, and plotting.

After scoring the points, build the linked dashboard index and curve plots:

```bash
python/aisimulate/.venv/bin/python \
  benchmarks/evidence/agentx-446063-replay/summarize_curve.py \
  --root /path/to/agentx-glm-concurrency \
  --baseline-root /path/to/agentx-glm-concurrency/point-446063
```

`selected-points.json` freezes the six point IDs. `index.html` links each standalone interactive report; `summary.csv`, `summary.json`, and `concurrency-curve.png/.svg` compare latency, throughput, strict request MAPE, pairing coverage, failures/cancellations, and complete/native wall time. Aggregate output separates the new five points from all six: pooled request MAPE weights each valid request equally; macro point errors weight points equally. Missing or failed runs remain explicit and are never assigned zero error. Bundling the reused c1 artifacts under `point-446063` keeps links portable without rerunning c1.

Interactive point IDs, concurrency labels, preparation counts, reference links, and download filenames come from the selected run. Reports with one recorded-output variant work without a control. Browser checks accept an optional expected-results JSON after the output directory, while retaining fixed c1 regression expectations when applicable.

All five additional native replays completed on 2026-10-29 UTC:

| Concurrency | Strict request E2E MAPE | Matched / reference successes | Mean E2E point APE | Output throughput point APE | Complete CLI wall | Native timer |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 37.62% | 601 / 708 | 44.64% | 12.69% | 112.74 s | 1.62 s |
| 8 | 49.78% | 1,352 / 1,426 | 47.42% | 5.17% | 115.86 s | 2.11 s |
| 10 | 59.45% | 1,545 / 1,607 | 57.89% | 4.42% | 139.25 s | 2.32 s |
| 13 | 85.37% | 2,137 / 2,191 | 71.94% | 3.48% | 112.75 s | 2.53 s |
| 14 | 83.15% | 2,336 / 2,454 | 43.84% | 4.76% | 113.19 s | 2.49 s |

The new five points have pooled request E2E MAPE **70.06%** over 7,971 pairs, covering 95.05% of 8,386 reference successes. Giving each point equal weight yields mean request MAPE 63.08%, mean-E2E aggregate MAPE 53.14%, and output-throughput MAPE **6.10%**. Complete CLI time totals **593.79 seconds**. Including reused c1 gives 68.89% pooled request MAPE over 8,260 pairs and 7.55% output-throughput MAPE across six points.

All scored pairs have equal output lengths. Most input lengths still differ because the imported source uses hash-block prompt lengths while the server measures tokenized HTTP prompts. These results therefore do not isolate timing error. Native profiles retain 604/1,358/1,557/2,181/2,352 completed requests, including zero-output or unmatched requests; none failed or were canceled. Reference errors remain in source records and are excluded from latency scoring. Profile throughput uses the successful cohort's observed start/end span, consistent with the benchmark, rather than assuming every denominator is exactly 3,600 seconds.

All six runs pass artifact-count, phase-boundary, full-window, config-hash, and per-phase delivered-token reconciliation checks. Independent raw-reference/native checks reproduce the c4 and c14 metrics. Preparation includes 4/7/9/12/15 primers plus 40/80/100/130/140 warmups for the five new points. C10, c13, and c14 drain active requests after the profile admission cutoff. Telemetry preserves those final partial intervals. The bundle's `validation.json` records the checks and counts.

## Running and reporting

Use the telemetry checkout's development environment. Native replay uses CPU; no GPU allocation is needed. The recorded runs used Linux, Python 3.12.3, a Ryzen 9 7950X, and 125 GiB host RAM. The supervisor retained a 96 GB ceiling; this is a safety limit, not observed consumption.

The external bundle contains `raw-remote.yaml` and `observed-remote.yaml`, exact execution records, logs, source manifests, and native results. Adjust the source paths and resource budget for the host before rerunning.

```bash
uv sync --project python/aisimulate --extra dev

python/aisimulate/.venv/bin/python \
  benchmarks/evidence/agentx-446063-replay/prepare_inputs.py \
  --source-dir /path/to/downloaded-source-bundle \
  --output-dir /path/to/results

python/aisimulate/.venv/bin/python \
  benchmarks/evidence/agentx-446063-replay/run_replay.py \
  --repo /path/to/aisimulate \
  --config /path/to/results/observed-remote.yaml \
  --output-dir /path/to/results/observed-replay \
  --variant observed_profile_output_lengths
```

The runner executes the public CLI:

```bash
aisimulate predict --config observed-remote.yaml \
  --capture-telemetry --capture-per-request \
  --telemetry-sample-interval-ms 1000 --output-dir observed-replay
```

Its adjacent `*-execution.json` records the full process wall time, command, config hash, host, commit, dirty status, exit status, and UTC boundaries. Wall time includes startup, resource preflight, full-corpus import, preparation, profiling/grace, and telemetry serialization. It excludes downloads/transfers, builds, scoring, and plotting. `prediction.json.wall_time_ms` is the narrower native simulation timer; report both separately. Reuse a fresh output name to preserve earlier evidence.

After scoring, render the standalone interactive HTML and PNG/SVG figures:

```bash
python/aisimulate/.venv/bin/python \
  benchmarks/evidence/agentx-446063-replay/score_replay.py \
  --run-dir /path/to/results/observed-replay \
  --reference /path/to/results/source/reference-requests.json \
  --namespace-map /path/to/results/source/source_namespace_map.json \
  --execution /path/to/results/observed-replay-execution.json \
  --output-dir /path/to/results/observed-report \
  --variant observed_profile_output_lengths

python/aisimulate/.venv/bin/python \
  benchmarks/evidence/agentx-446063-replay/render_report.py \
  --run-dir /path/to/results/observed-replay \
  --reference /path/to/results/source/reference-requests.json \
  --measurement /path/to/results/observed-report/measurement.json \
  --output-dir /path/to/results/observed-report
```

Repeat with the original trace config and separate output/report directories for the control. Rendering requires the development environment's Matplotlib, NumPy, and Plotly (recorded version 6.6.0, with Plotly.js 3.4.0).

Keep `observed-replay`, `raw-replay`, `observed-report`, and `raw-report` under the same results directory. Once both runs are scored, rendering either report discovers the sibling variant and embeds both in `report.html`. Rendering reads saved results; it does not rerun the simulation or change the scored metrics. The existing PNG/SVG figures remain alongside the interactive report.

Open `report.html` directly in a browser. Plotly, chart data, styles, and the license are embedded, so viewing and downloading report data require no network or local server. Reference links open external sites only when followed. The self-authored interface follows AgentX interaction patterns without copying its UI code or assets.

- Select an available replay variant, sort by latency or throughput, and use the Profiling/Warmup tabs. Warmup includes primers and ten live requests per lane.
- Hover for values, drag or scroll to zoom, double-click to reset, expand a chart, or export its PNG. Controls switch Distribution/In-flight average, P75/P90, TTFT/E2E, queue/completed requests, and input/decode throughput.
- Open Request timeline to group by conversation or worker, filter request IDs, pan or zoom, and click a request for timing, lengths, status, and source lineage.
- Open Aggregates across replays for MAPE, point APE, pairing coverage, and wall time. Evidence provides local JSON downloads of the report's request records, normalized chart samples, and accuracy/wall-time measurements; original native JSONL artifacts remain in each replay directory.
- Toggle the light/dark theme or open How to read for metric semantics. Unavailable metrics stay explicitly unavailable.

The browser checks use Playwright and Chromium (or an existing Chrome executable):

```bash
NODE_PATH=/path/to/node_modules \
AISIM_BROWSER_EXECUTABLE=/path/to/chrome \
node benchmarks/evidence/agentx-446063-replay/browser_checks.cjs \
  /path/to/results/observed-report/report.html /path/to/browser-checks
```

Checks cover both embedded variants, unchanged metrics, phase counts, chart controls,
request details, real timeline zoom/reset, JSON downloads, keyboard navigation,
mobile layout, and offline operation. Only Chromium/Chrome is browser-qualified.
The native data adapter also has an executable self-check:
`python/aisimulate/.venv/bin/python benchmarks/evidence/agentx-446063-replay/interactive_report.py`.

## Accuracy definitions

`APE = 100 × abs(simulated − measured) / measured`, for positive measured values. Request MAPE is the arithmetic mean of individual request APEs. A single point's mean-latency or throughput comparison is explicitly **N=1 APE**, not a request MAPE. There is no numerical acceptance threshold or timing-accuracy qualification in this experiment.

The raw benchmark has 326 profiling rows: 324 successes and 2 no-content errors. Successful reference rows have:

| Metric | Reference |
|---|---:|
| Mean request E2E latency | 3,807.279663 ms |
| Successful output tokens | 318,405 |
| Successful input tokens | 104,712,455 |
| First successful start to last successful end | 3,596.255252139 s |
| Output throughput | 88.537931 tokens/s |
| Input throughput | 29,117.080868 tokens/s |
| Total throughput | 29,205.618800 tokens/s |

These denominators match the dashboard. AIPerf's raw aggregate uses approximately 3,600 seconds and reports a slightly different throughput; do not mix the two definitions. Sampled throughput plots instead use native delivered output/executed prefill counters divided by each interval's exact duration. Executed prefill excludes cache hits and is not input-sequence-length throughput.

The scorer retains request pairs and exclusions, reports exact-source and output-length-matched subsets, flags input differences and ambiguous repeated instances, and records actual cohort sizes. Different simulated completion times change which requests fit inside the profile window; independent point aggregates therefore have different cohorts. Errors and cancellations are retained, not silently converted into successful measurements.

## Reading the plots and limits

- Request timelines separate primer/warmup and profiling. Static figures use the absolute virtual clock. Interactive charts use seconds since the selected phase began; request details and exports retain absolute timestamps, and the profiling origin aligns existing barrier-relative request output.
- Dedicated preparation request and cache figures keep the brief warmup interval readable alongside the hour-long profile.
- Request plots show input/output lengths, TTFT, E2E, TPOT, and token rate. TPOT is absent for fewer than two delivered tokens.
- Sample plots show output/prefill rates, arrivals/completions, queues, per-rank KV occupancy, prefix lookup, first-admission attribution, and speculative acceptance. Unavailable counters remain absent. This configuration has no host cache.
- Reference and simulation CDFs are descriptive independent cohorts. They do not imply per-request matching.
- Source prompt lengths differ from measured HTTP prompts by −64 to +427 tokens, averaging +125.06. Logical block identities do not reproduce exact tokenizer/chat-template payloads or client overhead.
- Native warmup completes one primer and ten one-token requests. The raw reference has seven valid warmup metric rows and four no-content errors, although all eleven wire requests returned. Native replay does not reproduce those HTTP failures or the two profiling no-content errors.
- Expected acceptance length 2.99 is the upstream recipe's provisional GLM-5.2-derived assumption, not fitted to these scores. SGLang 0.5.20 performance data can backward-fill missing operation families from 0.5.14; selecting 0.5.20 does not establish complete version-exact timing coverage.
- Server logs, power measurements, GPU execution, chart-export compatibility, and a new timing model are outside this experiment.

The report's `measurement.json` contains title, benchmark URL, wall time, simulated duration, methodology, limitations, and `comparisons` entries with `metric`, `unit`, `reference`, `simulated`, `mape_percent`, `count`, and `scope`. Large native telemetry arrays remain in their JSONL artifacts.

Validation includes the scorer's deterministic `--self-test`, independent recomputation of both runs' request and aggregate errors, native summary/counter reconciliation, Ruff checks, and visual inspection of the rendered figures. The actual supervised CLI runs validate both output-length variants, full corpus loading, native telemetry, and output serialization. No GPU or timing-accuracy acceptance test was performed.
