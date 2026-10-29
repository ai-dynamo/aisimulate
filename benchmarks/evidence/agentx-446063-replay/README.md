# AgentX point 446063 replay

CPU replay and original telemetry plots for [InferenceX point 446063](https://inferencex.semianalysis.com/inference/agentic/446063), concurrency one. This reproduces the benchmark's preparation and 3,600-second profiling window with the complete source corpus loaded. It does **not** exhaust all 98,827 requests in the corpus.

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

After scoring, render the standalone HTML and PNG/SVG figures:

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

Repeat with the original trace config and separate output/report directories for the control. Rendering requires the development environment's Matplotlib and NumPy.

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

- Request timelines separate primer/warmup and profiling. Dots and bars use the absolute virtual clock; the phase boundary aligns existing barrier-relative request output.
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
