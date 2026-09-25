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
| `probe_driver.py` | the G1 driver: golden `cli generate` render -> per-GPU probe queues -> curated records -> results matrices (`--plan/--emit-queues/--records/--matrix/--check-coverage`, `--only` to scope) |
| `probes/` + `inject/` | in-container identity probes per framework; `inject/sitecustomize.py` is the multi-rank (tp/ep) leg — the filename is the mechanism |
| `kernel_taxonomy_<sm>.yaml` | per-SM kernel-name -> canonical-backend vocabulary (both sides of a verdict translate through the SAME file; SMs never share one) |
| `path_diff.py` | collector op path vs serving, same profiler, same vocabulary (stub) |
| `decompose.py` | observed execution -> op families (taxonomy roles x backend labels) + residue (kernels no family names); results/<sm>/decompose/ |
| `e2e_align.py` | SDK prediction vs one live measurement of the golden deployment (explicit measurement file); results/<sm>/e2e/ — the campaign needs a GPU matching an SDK system entry |
| `build_images.sh` | rebuild probe images + generator venv from targets.yaml pins |

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
