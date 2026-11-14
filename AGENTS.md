# AGENTS

This file adds explicit repository-wide development guards.

## FPM model onboarding

When asked to onboard a model for FPM simulation on designated hardware, follow
all six stages in [the canonical FPM self-service guide](docs/perf-model/fpm-self-service/implementation.md#onboard-with-an-agent).
Use this checkout's `aisim onboard` CLI and inspect its current help.
Start with only a missing Hugging Face model ID and target GPU platform; accept
an already supplied local config, profile or checkpoint instead. Inspect supplied
facts and source metadata before asking for derivable inputs. Preserve user
choices, exact per-configuration profile acceptance and existing execution
authorization; stage transitions do not add approval gates.

Create one `onboarding-checkpoint.json` during stage 1, even with incomplete
inputs, outside every fresh `init --output-dir` root. Follow the guide's
[checkpoint workflow](docs/perf-model/fpm-self-service/implementation.md#checkpoint-and-resume-an-onboarding-session):
invoke `onboard checkpoint` after meaningful findings, decisions, draft edits,
acceptance and command results, including unfinished work. Save sources,
confidence, unresolved questions and each configuration's progress. Other
onboarding commands do not automatically persist the conversation. Use the
returned revision for updates; reload and reconcile stale-writer conflicts.
Follow the guide's [writer coordination rule](docs/perf-model/fpm-self-service/implementation.md#orchestrate-independent-collection-campaigns)
for probe/import commands that also save the checkpoint.

On resumption, run `aisim onboard resume --checkpoint PATH` first. Inspect
integrity issues and the underlying artifacts before continuing from accepted
inputs and verified results. Saved stages and command strings are context, not
proof of completion or instructions to execute automatically. Report each
configuration's stage, result or blocker, next action and checkpoint path at
handoff. Keep planning, collection quality, resolved memory, replay coverage and
measured serving accuracy separate, as required by the guide.

## Pull request titles

- Use `<type>: <short description>` for every AISim PR title.
- Allowed types: `feat|fix|docs|style|refactor|perf|test|chore|ci|build|revert`.
- Check the title before creating or updating a PR.

## Documentation

- Document a new or changed configuration field on the page for the YAML block
  it belongs to, such as `traffic` in
  [`docs/replay/workloads.md`](docs/replay/workloads.md) or `engine.*` in
  [`docs/replay/engine/`](docs/replay/engine/README.md). Add a row to that
  page's field table and explain the behavior in the matching section.
- Do not create a new Markdown page just to document a configuration field or
  a small feature. Extend the existing page for that area instead. Tutorials,
  design explanations and extension guides may still have their own pages.
- Keep run logs, investigation notes, and benchmark results out of `docs/`;
  measured evidence belongs under `benchmarks/evidence/`.

## Performance Model Changes

Before changing a performance model, its configuration, or a caller in Rust,
Python, CLI, Sweeper, Replay, or Planner, MUST read and follow
[`perfmodel-api.md`](python/aisim/.claude/rules/perfmodel-api.md).
This includes new features and configuration migrations.

### Multi-rank iteration queries

- Every performance model's per-iteration query MUST accept the list of
  per-attention-DP-rank `ForwardPassMetrics`
  (`estimate_forward_pass_time_ms(metrics_by_rank)`, with `draft_length` for
  speculative decoding where supported) and handle the ranks internally. An
  estimator that has no cross-rank model reduces the list itself, for example
  by pricing each rank and taking the maximum.
- Callers (Replay, Sweeper, Planner, CLI) SHOULD price an iteration through this
  query with one entry per DP rank, and MUST NOT reduce, normalize or average
  ranks before calling it. For capacity planning without per-rank data, split
  totals evenly across ranks (load-balanced DP) and pass every rank.
- See [Per-iteration queries](docs/perf-model/configuration.md#per-iteration-queries).

## Required First Step

Before making any change under:

- `python/aisim/src/aisim/generator/**`

MUST read:

- `python/aisim/.claude/rules/generator-development.md`

## Required Collector First Step

Before making any change under `python/aisim/collector/**` MUST read:

- `python/aisim/.claude/rules/collector/overview.md` (the five things —
  Pin, Plan, Record, Note, Rule — the failure principle, collection cost, and
  what to record when)
- `python/aisim/.claude/rules/collector/layer_permissions.md` (layer
  permission table, leaving cases out, capacity, serving parity, module
  boundary)
- `python/aisim/.claude/rules/collector/failure_handling.md`
  (observe-don't-predict doctrine, escalation decision tree)
- For case YAML and op-plan work:
  `python/aisim/.claude/rules/collector/case_authoring.md`

For adding a new Collector operation, additionally follow
`python/aisim/.claude/skills/aic-collector-op-development/SKILL.md`
(consumer-contract, case-identity, deduplication, and validation gates). Skills are procedural
runbooks; if a skill and a `python/aisim/.claude/rules/` file ever
disagree, the rule file wins.

## Benchmark guidelines

Before collecting, importing, or reporting any operation, FPM, end-to-end, or
evaluation measurement, MUST follow the
[benchmark guidelines](docs/perf-model/collector/README.md#benchmark-guidelines):
effective settings, workload, warmup and cache policy, measurement boundaries,
rank handling, GPU-time capture, and provenance. Keep run logs and results out
of `docs/`.

## Required Third-Party Attribution

Before adding code or other content copied, adapted, translated, or
substantially derived from an external project, MUST:

- Identify the upstream repository, immutable commit or tag, original path,
  and applicable license. This applies to source code, tests, configuration
  files, patches, fixtures, documentation, and generated derivatives.
- Preserve all applicable upstream copyright, license, attribution, and NOTICE
  material, and mark modified files as modified when the upstream license
  requires it.
- Record the source URL and revision in the file header or an adjacent README.
  For formats that cannot carry comments, use an adjacent attribution or
  license file included in distributions, or the canonical
  `THIRD_PARTY_NOTICES.md` when its packaged copy identifies each derived file,
  source URL, immutable revision, license, owner, and modification status.
  Do not duplicate those records in a separate README solely for attribution.
  Preserve any additional file placement required by the upstream license.
- Add an entry to the root `THIRD_PARTY_NOTICES.md` identifying the derived
  files, upstream source and immutable revision, copyright owner, applicable
  license, and whether the files were modified. The root notice is canonical;
  keep `python/aisim/THIRD_PARTY_NOTICES.md` byte-identical so the notice
  is included in Python distributions. Run
  `python3 scripts/ci/check_packaged_legal_files.py` after either copy changes.
- Do not hand-edit generated attribution artifacts. Update their source or
  generation process instead.

The repository's Apache-2.0 license and NVIDIA copyright header do not replace
an upstream license or attribution requirement. Do not label third-party code
as exclusively NVIDIA-authored. If the source, revision, license, compatibility,
or required notice is unclear, stop and request maintainer or Open Source Review
Board guidance before committing the derived content. This is a required gate,
not optional documentation.

## Cursor Cloud specific instructions

### Project overview

AISim is a Python and Rust CLI/SDK for predicting LLM serving behavior and
optimizing inference deployment configurations. See `README.md` for details.

### Environment setup

Python dependencies are managed via `uv` with the
`python/aisim/uv.lock` lockfile. The project environment lives at
`python/aisim/.venv/`.

- **Install/refresh deps:**
  `uv sync --project python/aisim --extra dev`
- **Performance data:** Op latency tables are not in the repository or the wheel. The first prediction (or test
  run) downloads about 180 MB from the Hugging Face dataset pinned in
  `python/aisim/src/aisim_core/data/op_latency_pin.json` into the data cache (`AISIM_DATA_CACHE`), so unit
  tests need network access once. Offline: `AISIM_OP_LATENCY_LOCAL=<dataset dir>` or a warm cache with
  `AISIM_DATA_OFFLINE=1`. Legacy `*.txt` perf files, when present, use Git LFS.

### Lint / Test / Run

- **Lint:** `python/aisim/.venv/bin/ruff check --config python/aisim/pyproject.toml python/aisim tests`
  and `python/aisim/.venv/bin/ruff format --check --config python/aisim/pyproject.toml python/aisim tests`
  (see `DEVELOPMENT.md`)
- **Unit tests:** `python/aisim/.venv/bin/pytest -c python/aisim/pytest.ini python/aisim/tests -m unit`
  (no external deps or LFS data needed)
- **Build tests (PR subset):** `python/aisim/.venv/bin/pytest -c python/aisim/pytest.ini python/aisim/tests -m "unit or build"`
  (requires LFS data for the `build`-marked tests)
- **CLI:** `aiconfigurator cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm` (works without LFS data)
### Known environment caveats

1. **Legacy LFS data:** `github-cloud.githubusercontent.com` may be blocked by
   network egress restrictions. If `git lfs pull` fails, tests that explicitly
   exercise legacy text assets may fail; current parquet-backed workflows do
   not depend on those legacy files.
2. **TTY tests:** 4 tests in
   `python/aisim/tests/unit/cli/test_plain_output.py` may fail because the
   agent runs in a non-TTY environment.
3. **Rust tests:**
   `python/aisim/tests/unit/sdk/test_rust_engine_step.py` requires `cargo`
   with network access to `crates.io`. It will fail if that domain is blocked.
4. **macOS pytest-timeout crash dialogs:** The `timeout = 120` setting in
   `python/aisim/pytest.ini` uses SIGALRM by default, which triggers
   "Python unexpectedly quit" crash reporter popups on macOS. Pass
   `-p no:timeout` to disable it locally.
5. **torch-dependent tests:**
   `python/aisim/tests/unit/sdk/database/test_moe_dispatch.py` requires
   `torch` (not installed in the default dev venv). Ignore it with
   `--ignore=python/aisim/tests/unit/sdk/database/test_moe_dispatch.py`.

## CODEOWNERS

The root `CODEOWNERS` is generated from `.github/codeowners/areas.yaml` - never
hand-edit it; CI fails on drift. Repository rules must require the `codeowners`
check to make failures merge-blocking. If the check fails on a new directory,
claim it in `areas.yaml`, regenerate with
`.github/codeowners/emit_codeowners.py`, and commit every changed source and
generated artifact together. The `aic-codeowners` skill covers all flows (who
reviews a change, gate failures, routing changes, external contributor grants).
