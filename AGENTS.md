# AGENTS

This file adds explicit repository-wide development guards.

## FPM model onboarding

When asked to onboard a model for FPM simulation on designated hardware, follow
[Onboard with an agent](docs/fpm-self-service.md#onboard-with-an-agent) in the FPM
self-service guide. Use the checkout's `aisimulate onboard` CLI and current help.
Follow its six stages: inspect the model and target; choose the worker and
review runtime/collection limits; derive, review and save the profile; plan
collection; collect and verify data; validate replay and run ordinary
prediction/recommendation. Start by asking only for a missing
Hugging Face model ID (`organization/model-name`) and target GPU platform. Accept
an already supplied local config, profile or checkpoint path instead of requiring
a Hub ID; do not ask for both an ID and a config upfront. For a Hub ID, retrieve
the config using available Hub access as described in stage 1. Reuse supplied
facts and inspect the config/profile before asking for derivable metadata; defer
other questions to their stage and help the user choose the worker topology.
At stage 2, inspect `aisimulate onboard init --model-config PATH --suggest-parallel`
with the actual checkpoint/revision, runtime, GPU, interconnect and collection
options. This read-only JSON preview reports model/hardware-aware TP choices and
MoE TP/DEP/TEP alternatives, exact flags, resource sources and missing inputs.
Ask for unresolved shared precision/layout facts rather than guessing. Only a
complete declared/estimated byte budget can establish an `estimated_fit` default;
the shortlist is not a performance ranking or runtime qualification. Present the
default and alternatives, choose one exact topology per plan, and preserve any
explicit topology choice. If no default exists, explain the missing inputs and
select a candidate before collecting its rank-local bounds. Byte overrides are
specific to the chosen tuple; use explicit topology flags with those bounds.
Derive the minimum collection GPUs from attention TP times attention DP (TP4
requires four GPUs). Do not ask for total available GPUs, cluster node allocation
or replica budgets during onboarding. Preserve target hardware, runtime and
interconnect characteristics; verify actual collection resources before execution.
Generated predict/recommend configs validate one worker. Deployment replicas and
optimization GPU budgets belong to ordinary predict/recommend configurations.

Review context, scheduler and prefill capture limits independently of validation
traffic. Fresh config/profile defaults use the smaller of the declared context
and 256,000 tokens, profile scheduler bounds or 8,192 tokens/256 sequences, and a
2,048-token prefill CUDA graph capture limit. Explain and allow edits to these
initial policies; they establish neither capacity nor timing coverage. Do not
require fixed input/output lengths, concurrency, TTFT or TPOT during intake.
Those flags customize optional synthetic examples only. Maximum sequences does
not reserve maximum context for every sequence or request every prefill batch.
Follow the [shared collection policy](docs/fpm-self-service.md#how-the-collection-grid-is-determined):
AISimulate sets runtime limits, prefill capture sizes and some sample caps;
Dynamo combines them with the deployed image's sampling defaults and runtime
feasibility checks to generate the exact grid. Prefill capture overrides change
the engine configuration and must match the serving target. A complete generated
grid does not establish AgentX/direct-FPM query coverage; do not invent a separate
AgentX collection grid.

After verifying the formal data pair, use `aisimulate onboard validate-fpm`
with a local Weka trace and a separate validation output directory. Follow the
guide's pinned AgentX reference and current cold aggregated, one-lane, HBM-only,
non-speculative scope. Preserve target model projection and strict direct FPM
with denied fallback. Read `validation.json` together with native query coverage
and request completion evidence. Coverage counts native lookup resolutions,
including interpolation and failures; cached timing reuse is not another query.
Missing timing stops replay and preserves partial evidence, which cannot certify
the remainder of the trace. Report coverage and silicon accuracy separately.
Reuse verified v2 collection plans for validation-only changes; legacy v1 plans
need a new directory as described in the guide.

Do not reject a checkpoint solely because it is multimodal. Read its unambiguous
`text_config`, or its flat text-decoder fields, and explain that FPM models only
the text decoder. Multimodal encoders, projectors, preprocessing and other
non-text components and their resource costs are excluded; full multimodal
deployment memory and latency are not modeled. Preserve this scope in the
reviewed profile's provenance and require explicit bounds for unknown decoder
resources. Keep validation of incompatible decoder/cache semantics intact.

Report the current stage, its result or blocker, and the next action. Resume from
validated artifacts and accepted decisions instead of repeating the intake.
Stage transitions are not additional approval gates. Preserve explicit review
and acceptance of the exact profile, user overrides/provenance, and existing
execution authorization as described in the guide's terminal and headless flows.
The config/profile route requires neither an op-level model class nor
per-operation silicon data. Report planning, collection, simulation and measured
accuracy separately.

## Performance Model Changes

Before changing a performance model, its configuration, or a caller in Rust,
Python, CLI, Sweeper, Replay, or Planner, MUST read and follow
[`perfmodel-api.md`](python/aisimulate/.claude/rules/perfmodel-api.md).
This includes new features and configuration migrations.

## Required First Step

Before making any change under:

- `python/aisimulate/src/aisimulate/generator/**`

MUST read:

- `python/aisimulate/.claude/rules/generator-development.md`

## Required Collector First Step

Before making any change under `python/aisimulate/collector/**` MUST read:

- `python/aisimulate/.claude/rules/collector/layer_permissions.md` (layer
  permission table, module boundary, dispatch-vs-skip rule)
- `python/aisimulate/.claude/rules/collector/failure_handling.md`
  (observe-don't-predict doctrine, escalation decision tree)
- For case YAML work:
  `python/aisimulate/.claude/rules/collector/case_authoring.md`

For adding a new Collector operation, additionally follow
`python/aisimulate/.claude/skills/aic-collector-op-development/SKILL.md`
(consumer-contract, case-identity, deduplication, and validation gates). Skills are procedural
runbooks; if a skill and a `python/aisimulate/.claude/rules/` file ever
disagree, the rule file wins.

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
  license file that is included in distributions.
- Add an entry to the root `THIRD_PARTY_NOTICES.md` identifying the derived
  files, upstream source and immutable revision, copyright owner, applicable
  license, and whether the files were modified. The root notice is canonical;
  keep `python/aisimulate/THIRD_PARTY_NOTICES.md` byte-identical so the notice
  is included in Python distributions. Run
  `python3 scripts/check_packaged_legal_files.py` after either copy changes.
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

AISimulate is a Python and Rust CLI/SDK for predicting LLM serving behavior and
optimizing inference deployment configurations. See `README.md` for details.

### Environment setup

Python dependencies are managed via `uv` with the
`python/aisimulate/uv.lock` lockfile. The project environment lives at
`python/aisimulate/.venv/`.

- **Install/refresh deps:**
  `uv sync --project python/aisimulate --extra dev`
- **Performance data:** Current op profiles are parquet files under
  `python/aisimulate/src/aisimulate_core/systems/data/<system>/<family>/<backend>/<version>/`
  and are checked in directly. Legacy `*.txt` perf files, when present, use Git
  LFS; run `git lfs pull` only when working with those legacy assets.

### Lint / Test / Run

- **Lint:** `python/aisimulate/.venv/bin/ruff check --config python/aisimulate/pyproject.toml python/aisimulate tests`
  and `python/aisimulate/.venv/bin/ruff format --check --config python/aisimulate/pyproject.toml python/aisimulate tests`
  (see `DEVELOPMENT.md`)
- **Unit tests:** `python/aisimulate/.venv/bin/pytest -c python/aisimulate/pytest.ini python/aisimulate/tests -m unit`
  (no external deps or LFS data needed)
- **Build tests (PR subset):** `python/aisimulate/.venv/bin/pytest -c python/aisimulate/pytest.ini python/aisimulate/tests -m "unit or build"`
  (requires LFS data for the `build`-marked tests)
- **CLI:** `aiconfigurator cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm` (works without LFS data)
### Known environment caveats

1. **Legacy LFS data:** `github-cloud.githubusercontent.com` may be blocked by
   network egress restrictions. If `git lfs pull` fails, tests that explicitly
   exercise legacy text assets may fail; current parquet-backed workflows do
   not depend on those legacy files.
2. **TTY tests:** 4 tests in
   `python/aisimulate/tests/unit/cli/test_plain_output.py` may fail because the
   agent runs in a non-TTY environment.
3. **Rust tests:**
   `python/aisimulate/tests/unit/sdk/test_rust_engine_step.py` requires `cargo`
   with network access to `crates.io`. It will fail if that domain is blocked.
4. **macOS pytest-timeout crash dialogs:** The `timeout = 120` setting in
   `python/aisimulate/pytest.ini` uses SIGALRM by default, which triggers
   "Python unexpectedly quit" crash reporter popups on macOS. Pass
   `-p no:timeout` to disable it locally.
5. **torch-dependent tests:**
   `python/aisimulate/tests/unit/sdk/database/test_moe_dispatch.py` requires
   `torch` (not installed in the default dev venv). Ignore it with
   `--ignore=python/aisimulate/tests/unit/sdk/database/test_moe_dispatch.py`.

## CODEOWNERS

The root `CODEOWNERS` is generated from `.github/codeowners/areas.yaml` - never
hand-edit it; CI fails on drift. Repository rules must require the `codeowners`
check to make failures merge-blocking. If the check fails on a new directory,
claim it in `areas.yaml`, regenerate with
`.github/codeowners/emit_codeowners.py`, and commit every changed source and
generated artifact together. The `aic-codeowners` skill covers all flows (who
reviews a change, gate failures, routing changes, external contributor grants).
