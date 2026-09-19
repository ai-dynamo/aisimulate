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
| `kernel_taxonomy.yaml` | the single kernel-name -> canonical-backend vocabulary every instrument translates through |
| `decompose.py` | model -> op families + residue (stub; contract in module docstring) |
| `path_diff.py` | collector op path vs serving, same profiler, same vocabulary (stub) |
| `e2e_align.py` | AIC prediction vs live measurement on the golden deployment (stub) |
| `build_images.sh` | rebuild probe images + generator venv from targets.yaml pins |

## Workflows

- `workflows/upgrade_op.md` — framework version / collector change for one op family
- `workflows/onboard_model.md` — new checkpoint: identity, decomposition, coverage, admission
- `workflows/new_op_collector.md` — a family no collector measures yet

More workflows are expected; they reuse components rather than growing new
ad-hoc scripts.

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
