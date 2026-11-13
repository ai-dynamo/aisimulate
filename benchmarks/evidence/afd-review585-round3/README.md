# AFD round-three evidence

This package preserves local modeling checks and five-pair host measurements after integration with main `472b9a650ab4b585c6b641520eb51987ba7ffda4`. Source HEAD is `257b58d867a49e50384a72382ec73c989786118a`, with different uncommitted changes in each snapshot. HEAD alone does not identify a build. No remote CI or physical-serving accuracy result is claimed.

## Files and source identity

- `sources.tar.gz`: exact original manifests, source overrides against pinned main and original measurement/checking scripts. Unchanged files come from main; each distinct override is stored once across the three snapshots. It also preserves the baseline integration patch.
- `reproduce.py`: restores the original archives, manifests, scripts and report paths into a new directory, checking their SHA256 identities.
- `replay-performance.json.gz`, `legacy-performance.json.gz`: original paired reports, compressed without changing their bytes.
- `forward-gate/`: original raw forward rounds and comparator result.
- `modeling-deltas.json.gz`, `layout-deltas.json.gz`, `memory.json`: independent modeling checks.
- `validation.json`: commands, results, exclusions, all three build identities, follow-up memory deltas and package checksums. Raw local test logs, drafts, worktree state and intermediate candidates are excluded.

| Source | Original source archive SHA256 | Loaded release binary SHA256 |
|---|---|---|
| Baseline | `d2970209f0f09a077a6775cd5bdac7c2a9a59fa13661b76dcf4ae08fed9b2a65` | `7ca8e37934292f81e16a661c0de4aee22159b5039bf24357025b7e903742a0fb` |
| Measured candidate | `1ab326013e8cc4dc649e452a1179cb57e8f1e0a1f7a3c66b618bfd618987c3a3` | `c1b97eec61a78ac749fde8a27e1c101d2926a2ef5fec7e96dd01cfa43a4f8b55` |
| Follow-up correction | `a32df48f080b5870ea07c07bad8b5f845ab48612f34b91e4cd2385f6510c7cdf` | `8a78d21a152ae8c863c693a6a2effc4d379ecdb7fa47a08ce14d486fbe9d56f7` |

All arms use data tree `912b4f8edaf5a8d53e36c3edd52bf6b5eb7f724c`. Original manifests retain per-file hashes, inventory hashes, source commit/tree and dirty-state markers. The baseline includes the recorded integration patch. The measured candidate and later prefill correction remain distinct: old measurements are not relabeled as results for the later correction. The package changes storage only; original source archives are reconstructed byte-for-byte with their original digests. The Git repository must contain immutable main `472b9a650ab4b585c6b641520eb51987ba7ffda4`; the package does not duplicate files already present in that commit.

## Modeling and validation

The original candidate fixes memory detail availability, ordinary-companion GPU identity, native AFD block-size validation, placeholder-model mocks and SDK search validation. Replay uses actual batch/context/MTP/effective microbatch depth; the frozen-pricing field and saved identity are rejected. Prefill fit and KV admission share bounded pool-wide token accounting with a full configured context/KV reservation.

`modeling-deltas.json.gz` separates historical → integrated default → explicit dynamic baseline → candidate. All six E2E cases conserve request/input/output totals. Five change serving metrics through frozen removal; the prefill-only case changes memory diagnostics only. None has a main-integration serving delta.

`layout-deltas.json.gz` checks fourteen layouts against independently executed explicit-dynamic baseline runs. Twelve change; `deepseek_v32_tp4_trace` and `qwen_tp2_pd_prefill` remain unchanged. Main integration already changes `mimo_tp2`, `mimo_tp4_trace` and `qwen_tp2_pd_decode`; frozen removal explains the other nine changed layouts. Final candidate values match the dynamic baseline. Historical generation identities and attribution remain intact.

`memory.json` records ISL=4096, configured batch=64, three microbatches, 2A+4F and context=4352. The baseline incorrectly rejects F prefill at 172.59 GiB against 141 GiB; the original corrected candidate reports 33.706 GiB A and 6.782 GiB F prefill, retaining 25.5 GiB A KV reservation. It completes one request with 4096 input and 128 output tokens. The earlier context=4224 investigation clips output and is excluded from equivalent-work evidence.

| Original measured-snapshot checks | Result |
|---|---|
| Rust library / public Rust API | 3172 passed, 2 ignored / 25 passed |
| Python AFD SDK and public contracts | 450 passed |
| Root callers, search and CLI | 2342 passed, 5 skipped |
| AFD/general parity | 425 passed |
| Ownership policy | 107 passed |

Ruff, Cargo formatting, whitespace, legal-file consistency and strict ownership passed locally. Ignored Rust cases require a manual semantic campaign and an R17 dataset. Five skipped Dynamo packaging checks belong to integration CI. Counts and full commands are recorded by snapshot in `validation.json`.

### Subsequent prefill correction

The follow-up snapshot removes `max_num_batched_tokens` from the SGLang prefill memory cap because that scheduler does not enforce this limit. Regression tests cover a 20000-token chunk with a smaller/default batched-token limit. Its validation and four explained memory-only Golden deltas are consolidated under `follow_up_prefill_batched_token_fix` in `validation.json`; serving metrics and untouched cases remain equal. The follow-up has targeted Rust/Python checks and 389 general plus 36 AFD parity passes across the recorded runs. It does not have a repeated full Rust campaign or a new paired performance campaign. Removing one scalar minimum at preparation adds no per-pass allocation/state; this is a source-level cost assessment, not a measured speed claim.

## Paired host measurements

Matched release extensions, Python 3.13.16 and the same dependencies/data ran on macOS 15.8 arm64. CPU affinity is unavailable; OMP/OpenBLAS/MKL/Rayon threads were one. Arms alternate over five pairs; builds/tests did not run concurrently with timings. Both sources, archives, data and loaded binaries were verified before and after each campaign. Baseline AFD explicitly selects dynamic pricing; the measured candidate has no frozen option. Intended default-prediction changes are assessed separately above.

The forward gate covers 64 available cases, five pairs, ten warmups and 100 repetitions. Predictions match and `blocking=false`. Of 128 cold/warm comparisons, 91 are stable (57 improvements, 34 unchanged), 37 are unstable (36 cold, one warm), with no consensus regression or invalid point. Noisy points remain inconclusive.

All thirteen replay/CLI workloads preserve complete request/input/output totals and have equal modeled reports between arms. Raw reports retain every configuration, sample, identity and phase timing. Each request has 1024 input and 128 output tokens. Saturated, underloaded, ragged 3+3+2, depth-16, restoration, ordinary/AFD Gemma4 and Hopper precision controls are included.

| Workload | Requests | Warm runner ms, B / C | Paired ratio | Full CLI ms, B / C | Paired ratio | CLI pair range | Worker peak MiB, B / C |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary_fixed | 16 | 0.755 / 0.750 | 0.990 | 647.65 / 645.34 | 0.996 | 0.985–1.006 | 134.72 / 134.12 |
| ordinary_op_level | 16 | 14.141 / 13.992 | 0.990 | 2083.83 / 2028.19 | 0.973 | 0.971–1.002 | 194.73 / 191.44 |
| ordinary_op_level | 1024 | 49.355 / 49.141 | 0.996 | 2090.30 / 2086.24 | 0.998 | 0.974–1.025 | 202.03 / 203.41 |
| afd_four_stage_low | 16 | 45.084 / 44.693 | 0.991 | 2331.91 / 2267.46 | 0.972 | 0.950–1.017 | 214.11 / 210.83 |
| afd_four_stage_busy | 288 | 37.904 / 37.469 | 0.989 | 2321.92 / 2258.50 | 0.977 | 0.943–1.006 | 233.05 / 231.41 |
| afd_two_stage | 128 | 34.558 / 34.299 | 0.995 | 2272.71 / 2278.51 | 1.003 | 0.974–1.023 | 217.12 / 222.20 |
| afd_ragged | 64 | 26.200 / 25.605 | 0.977 | 2317.86 / 2252.64 | 0.992 | 0.963–0.997 | 218.89 / 218.14 |
| afd_depth16 | 256 | 40.632 / 40.400 | 0.995 | 2299.38 / 2260.34 | 0.999 | 0.973–1.003 | 232.42 / 233.59 |
| afd_restore | 32 | 29.375 / 29.308 | 0.998 | 2943.74 / 2889.07 | 0.985 | 0.978–0.997 | 267.56 / 272.89 |
| ordinary_gemma4 | 16 | 56.128 / 48.325 | 0.862 | 1173.08 / 1112.09 | 0.948 | 0.941–0.955 | 149.75 / 150.69 |
| afd_gemma4 | 16 | 65.045 / 59.450 | 0.909 | 1402.46 / 1403.83 | 1.001 | 0.999–1.006 | 180.88 / 179.97 |
| ordinary_nvfp4_wo_hopper | 16 | 55.987 / 49.050 | 0.873 | 1108.99 / 1049.36 | 0.946 | 0.920–0.994 | 150.73 / 148.94 |
| afd_nvfp4_hopper | 16 | 66.332 / 59.723 | 0.898 | 1282.60 / 1285.42 | 1.002 | 0.997–1.005 | 174.67 / 175.45 |

Displayed B/C timings are independent arm medians; ratios are medians of paired candidate/baseline ratios. Warm runner medians range 0.862–0.998, full CLI medians 0.946–1.003; individual-pair variation remains visible. No universal host-speed improvement is claimed. The separate direct nvfp4-on-Hopper legacy control preserves its normalized numerical report over five pairs (median ratio 0.967, range 0.888–1.069); it is a static estimate rather than request replay.

Timing boundaries: `compile_ms` is prediction-to-ReplaySpec preparation; `empty_native_ms` constructs providers/memory/capacity with no traffic. First runner follows empty setup; warm timings are five complete calls after the first and three warmups, rebuilding providers each time. Native wall time includes replay setup/execution and collector finalization, excluding preceding model construction and subsequent evidence/power aggregation. Full CLI includes startup, supervision, preparation and serialization. RSS is whole-worker high-water memory, not retained cache or CLI-child peak. Release compiler durations are not a paired build-speed benchmark.

## Restore and reproduce

```sh
python/aisimulate/.venv/bin/python benchmarks/evidence/afd-review585-round3/reproduce.py --output .local-tmp/afd-round3-restored
```

Use `--repo /path/to/repository` when restoring from another location. Restoration reads the pinned Git objects without modifying Git state. The output contains the original `benchmarks/evidence/afd-review585-round3/` layout, including all three original source archives, manifests and exact measured harnesses. The utility verifies package checksums, source inventories, archive reconstruction digests and uncompressed report bytes; it refuses to overwrite an existing directory. Historical Golden reasons mentioning uncompressed JSON or original archive filenames refer to these restored paths. Original measured script hashes are preserved even though scripts are no longer loose files in this package.

For a rerun, use separate complete checkouts at the recorded HEAD/data revision. Copy the restored evidence layout into them and unpack each arm's original source archive over its respective checkout. Build matching release extensions using the recorded command, then verify/generate manifests for those builds before invoking the restored harnesses:

```sh
python benchmarks/evidence/afd-review585-round3/measure_forward.py --baseline /path/to/verified-baseline --candidate-revision snapshot-1ab326013e8cc4dc649e452a1179cb57e8f1e0a1f7a3c66b618bfd618987c3a3 --output /path/to/new-forward-results
python benchmarks/evidence/afd-review585-round3/measure.py --baseline /path/to/verified-baseline --candidate-revision snapshot-1ab326013e8cc4dc649e452a1179cb57e8f1e0a1f7a3c66b618bfd618987c3a3 --output /path/to/new-replay.json
python benchmarks/evidence/afd-review585-round3/measure_legacy.py --baseline /path/to/verified-baseline --output /path/to/new-legacy.json
```

These commands describe the original measured candidate, not the later prefill correction. Rebuilt binaries in another environment require their own identities and new evidence; never relabel the historical reports. Remote CI, ignored data/manual campaigns, cross-repository integration and physical AFD accuracy remain outside this package's qualification.
