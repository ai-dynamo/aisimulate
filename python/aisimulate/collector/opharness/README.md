# opharness — workflows and instruments for op-path support & upgrades

The recurring jobs here are workflows over one question: **given a
generator-rendered config, what does the framework actually deploy — and does
our collected data agree?** A model is measured whole (evidence only exists in
full-model context); verdicts land on ops (the collector's ledger).

## Layout

```
opharness/
  targets.yaml        INPUT: what to probe & how — roster, one pinned version
                      per backend, per-model generate args (each with a fact
                      citation), dummy overrides, signed owner exclusions
  configs/            INPUT data: probe roster, curated per-model signals
  components/         INSTRUMENTS: deterministic, single-question, reusable
                      across workflows — scripts exist precisely where an AI
                      would otherwise drift (free-hand benchmarking, label
                      renames, code-reading instead of executing)
  workflows/          ORCHESTRATION: thin, human+AI-readable step tables that
                      only ever call components for anything that must be
                      reproducible
  results/            OUTPUT: results/<sm>/<framework>-<version>.yaml (verdict
                      + deployed identity per checkpoint; versions coexist) and
                      findings.yaml (root-caused conclusions, deduplicated,
                      with evidence and version pins)
```

## Components

| Component | Question it answers |
|---|---|
| `dummies.py` | build depth-cut, width-true dummy checkpoints (quant/dispatch behave like the real model, load in minutes) |
| `probe_driver.py` | the G1 driver: golden `cli generate` render -> per-GPU probe queues -> curated records -> results matrices — records and matrix cells carry `golden_facts` (did the render apply the generator's model facts, or did the pipeline swallow a facts-resolution failure), and a checkpoint under `dummy_overrides.family` is probed only on that adapter's dummy (a dir built by another adapter is skipped as `dummy not built`) (`--plan/--emit-queues/--records/--matrix/--check-coverage`, `--only` to scope) |
| `probes/` + `inject/` | in-container identity probes per framework; `inject/sitecustomize.py` is the multi-rank (tp/ep) leg — the filename is the mechanism |
| `kernel_taxonomy_<sm>.yaml` | per-SM kernel-name -> canonical-backend vocabulary (both sides of a verdict translate through the SAME file; SMs never share one) |
| `path_diff.py` | collector op path vs serving, same profiler, same vocabulary (stub) |
| `decompose.py` | observed execution -> op families (taxonomy roles x backend labels) + residue (kernels no family names); results/<sm>/decompose/ |
| `e2e_align.py` | SDK prediction vs one live measurement of the golden deployment (explicit measurement file); results/<sm>/e2e/ — the campaign needs a GPU matching an SDK system entry |
| `evidence_bundle.py` | pack a campaign's evidence (raw + fingerprints, records, captures, full reports, kernel-level decompositions) into one content-addressed tar.gz and index it in results/evidence_index.yaml |
| `build_images.sh` | rebuild probe images + the generator venv (this checkout) from targets.yaml pins |
| `op_smoke.py` | one case of one registered op through the executor's (get_func, run_func) contract — "can this collector build and time a case on the new pin" |
| `executor_smoke.py` | the same op through collect.py's REAL path (`--model-cases-full` plan, checkpoint, `--resume`, finalize) -> results/<sm>/executor_smoke/; `--shards` checks a pipeline shard plan against the case plan on a CPU |
| `lane_evidence.py` + `lane_evidence.yaml` | collector version/SM lane guards graded against the identity records: which checkpoints are the evidence for a lane, does the guard agree (open+confirmed / closed+contradicted) |

Evidence layering (owner decision 2026-09-26): the repo carries CONCLUSIONS — matrices, verdict summaries (identities, sha/fingerprint of both inputs, per-role counts, the deciding names when red), decompose summaries (role -> backend -> kernel count, residue), findings. EVIDENCE — raw probes and their `.fp` sidecars, records.jsonl, collector captures, full path_diff reports and kernel-level decompositions (`archive/evidence/` in the workspace) — stays out of git and is packed per campaign by `components/evidence_bundle.py`; `results/evidence_index.yaml` names each bundle by campaign id, sha256 and location, and every committed conclusion carries the ids/fingerprints that resolve into it.

Run-time layout: components run from this checkout (queues invoke `components/probes/` inside the container; the checkout is visible under the workspace mount or mounted read-only at `/harness`); the workspace (`AIS_PROBE_WORKSPACE`) holds only data — configs/, dummy_models/, archive/ (plan, run_sh, raw + `.fp` fingerprint sidecars, records), facts/, jitcache/, venv_ais/. Every plan run carries an execution fingerprint (engine invocation, dummy config, probe code, image, kv); a raw counts as evidence only when its sidecar or recorded argv matches, and the matrix marks the rest `stale evidence` / `unverified`.

## Workflows

- `workflows/upgrade_op.md` — framework version / collector change for one op family
- `workflows/onboard_model.md` — new checkpoint: identity, decomposition, coverage, admission
- `workflows/new_op_collector.md` — a family no collector measures yet

More workflows are expected; they reuse components rather than growing new
ad-hoc scripts.

## How workflows are driven (and kept from drifting)

A workflow run is a loop between an agent and `components/workflow_check.py`:

```
while not workflow_check(<workflow>, params).all_done:
    do the FIRST todo step (script steps call components; ai steps produce
    declared artifacts; owner steps stop for a signed decision)
    re-run workflow_check      # artifacts decide progress, never the agent
```

Each `workflows/<name>.md` has a sibling `<name>.yaml` manifest whose per-step
`done_when` predicates consult artifacts only (results matrices, findings,
targets declarations, workspace evidence). Steps depending on a
not-yet-implemented component report `blocked` — an honest roadmap, distinct
from actionable `todo`. Every check appends an observation to
`results/campaigns/<slug>.jsonl`, so how a campaign actually progressed is a
complete, append-only record; the ledger is audit-only and never read back as
state, which makes runs resumable across interrupted sessions and agents.

## Boundary rules

1. **Instruments measure, AI judges, owners choose.** Deterministic +
   repeatable -> component. Interpreting evidence (root cause, gap vs bug) ->
   AI, output lands as findings or declarations, never as pipeline branches.
   Trade-offs (exclusions, equivalences, granularity) -> signed owner entries.
2. **No claim from reading code.** "The framework does X" requires a
   component-produced record; minimal repros that don't reproduce the full
   execution path don't count either (both failure modes are documented in
   findings history).
3. **Self-contained on purpose.** This directory duplicates rather than
   references other collector machinery (vocabularies, catalogs) so new
   workflow design is not constrained by existing implementations; dedupe is a
   later, deliberate step.
4. **Facts carry provenance.** Every result file pins its framework version;
   every finding states evidence paths and dates; golden commands stamp the
   generator commit — reproduce by checking out that commit.

## Execution home

Probing runs on a GPU workspace (`AIC_PROBE_WORKSPACE`) holding dummy_models/,
archive/ (raw evidence + records.jsonl), and fetched configs; this directory
holds the instruments and the durable inputs/outputs. See probe_driver's
docstring for the invocation set.

## Per-SM runs (sm90 / sm100 / sm103 / sm120)

Every verdict-plane artifact is keyed by SM and nothing is shared across SMs:
`components/kernel_taxonomy_<sm>.yaml` (vocabulary — a new SM starts with its
own file, seeded from the closest SM and then labelled from THAT box's raws;
probe_driver refuses to run without it), `results/<sm>/<fw>-<ver>.yaml`
(matrix), `results/pathdiff/<sm>/<fw>-<ver>/` (gate verdicts),
`results/retests/<sm>/` (customization retests).

Which SM a command works on:

1. `targets.yaml` `platform.sm` is the source of truth — it names the box this
   checkout is pinned to (`h20_sm90` here). `probe_driver.current_sm()`,
   `workflow_check` (`--param sm=` default) and the `captures/verdicts_*.sh`
   scripts all read it.
2. `export AIS_SM=<sm>` OVERRIDES it, and is only for grading another SM's
   evidence from this box (e.g. re-running `verdicts_vllm_0300.sh` over an
   imported sm120 workspace). Never leave it exported by accident: until
   2026-09-30 the default was a hard-coded sm90, and an sm120 box that forgot
   the export labelled its records with the sm90 vocabulary silently.
3. On a new box: set `platform` in targets.yaml (name, sm, SDK `system` used
   for the golden render), add `kernel_taxonomy_<sm>.yaml`, then run the
   workflow exactly as on sm90; `workflow_check upgrade_op --param fw=... --param
   version=...` shows honest todos for the new SM until its own evidence exists.

SM103 (B300) is treated as equivalent to SM100 (owner decision 2026-10-05): it is the same capability MAJOR and every framework selects
kernels by that family (sglang `major == 10`, trtllm/vllm `in (100, 103)`), so a
collector that tests `sm == 100` silently measures the wrong lane on B300
(sglang encoder, fixed 2026-10-05). Branch on the family the framework's own
selector uses. `kernel_taxonomy_sm103.yaml` is seeded from the sm100 file and
still needs labelling from B300 raws; `results/sm103/` does not exist yet.

Gate declarations are SM-aware: a `run ...` line in `captures/verdicts_*.sh`
declares the gate for every SM; prefix `FLOOR_SM=<sm> FLOOR_NOTE="<framework
fact>"` when the gate has no serving instance on that SM by framework fact
(sm120: fp8-KV dense MLA, TRITON_MLA smem). On that SM the script writes a
`platform-floor` verdict file and `path_aligned` does not wait for it;
`gates_declared` still counts it as coverage. Explained deviations
(`OUT=$OUT_EXPLAINED run ...`) are not gates on any SM.

Run captures inside the framework image with the checkout mounted and
`PYTHONPATH=<checkout>/python/aisimulate`; every declared gate has its script
under `components/captures/` (kda_gen sets `AIS_KDA_DECODE_PATHS=fused` itself).
The workspace may live outside the checkout (`AIS_PROBE_WORKSPACE`); the golden
render runs with the workspace as cwd for that reason.

Run ids include `targets.platform.name` (since 2026-09-30, so two boxes never
produce the same id for one case). A workspace probed before that is migrated
once with `probe_driver.py --migrate-run-ids --apply` (dry run without
`--apply`), then `--records` / `--matrix` as usual. Customizations in
`targets.yaml` take an optional `sms: [sm120]`; gated repos without a token fall
back to the SDK's bundled config with a PROVENANCE note (tokenizer via
`tokenizer_from`).

## Structural policy (owner decisions, 2026-09-20)

1. **Mechanism freeze.** Components are capped at the current set. New
   capability lands as a field or predicate on an existing component
   (config_delta is the precedent), never as a new file, unless something is
   deleted in the same change. The data plane (taxonomy rules, findings,
   verdicts) grows freely — it is output, not mechanism.
2. **Versions are not forked — SM is.** Framework-version history lives in
   git: collectors upgrade IN PLACE when the manifest pin moves (old code is
   `git checkout <tag>` away; old DATA is permanent under its version key).
   Family pins below the target version upgrade with it; a pin on an odd
   version (preview/dev build) gets one validation run or a code/PR-inclusion
   check against the target release before folding in. SM, by contrast, IS
   forked — on both planes: per-SM collector files (falls due at the next pin
   move, together with folding the 029 lanes back into base) and per-SM
   verdict-plane data (kernel_taxonomy_<sm>.yaml, results/pathdiff/<sm>/,
   results/retests/<sm>/). Rationale: version copies are serial (one alive at
   a time — git's case); SM copies are parallel, maintained by different
   sessions on different machines — physical separation replaces the
   cross-arch audit discipline and staleness machinery a shared file would
   demand. Arch-NEUTRAL fixes found on one SM are propagated to sibling files
   via an explicit cross-SM work order (the B300->H20 block-table fix is the
   template), never assumed.
3. **Workspace provisioning** (documents the manual B300 setup): a workspace
   root holds targets.yaml, dummy_models/ (gen_dummy_models.py), configs/
   (fetch step), archive/ (run_sh + raw + records.jsonl), facts/, jitcache/,
   aic/ (predecessor-toolchain checkout for golden renders) + venv_aic
   (python3.12, aic-core wheel via maturin; see build_images.sh), and the
   framework images by digest from framework_manifest.yaml. Point
   AIS_PROBE_WORKSPACE at the root; probes run in-container with the
   workspace mounted at /work and MPS bypassed
   (CUDA_MPS_PIPE_DIRECTORY=/nonexistent-no-mps).
