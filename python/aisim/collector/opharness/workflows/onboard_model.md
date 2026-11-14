# Workflow: onboard a model (extend coverage)

Entry: a new checkpoint (HF repo) on the pinned (framework, version, SM).
Exit: the model boots under a generator-rendered command (or its failure is a
root-caused terminal fact), its kernels are covered by collector ops, and its
cells appear in `results/`.

| # | Step | Who | Instrument / artifact |
|---|------|-----|------------------------|
| 1 | Fetch inputs | script | `components/fetch_inputs.py REPO...`: config/hf_quant into `configs/`, every non-weight file (tokenizer, processor, custom code) into `configs/aux_files/<org>_<name>/`; gated repo or missing config = OWNER DECISION (recorded in `targets.yaml` roster.excluded, signed) — never a silent skip. Then add the repo to `targets.yaml` roster.extra_repos |
| 2 | Build dummies | script | `components/dummies.py` (depth-cut, width-true; per-repo exceptions only as `dummy_overrides` declarations) |
| 3 | Identity probe | script | a `probe_driver.py` campaign per system in scope (`--system --only <repo> --emit-shards`, the collection pipeline, `--import-raws --records`): golden `cli generate` render -> probe -> records. `--records` prints each run as `ok` or with the error that left it without serving evidence |
| 4 | Rescue or root-cause | AI | for fails: A/B designed by AI but EXECUTED through the probe; workable extra generate args -> `targets.yaml cli_extra_args` (with fact citation); a terminal wall -> a `collector/skips.py` entry when the backend cannot serve the model, citing the run |
| 5 | Coverage gap -> dev | AI | missing collector modules / case declarations (this hands off to the new_op_collector workflow when a family has no collector at all) |
| 6 | Collect + run diff | script | existing collector for the model's cases; `run_diff.py failures <result dirs>` for the new rows' failures, and `run_diff.py tables` when the model's rows land in tables an earlier release shipped |
| 7 | Path alignment | script | a `components/gates.yaml` row per op the model newly exercises (`repo` = this checkpoint, `select` naming its planned case), then `gates.py plan` -> the collection pipeline -> `gates.py grade` per SM: collector path vs the step-3 serving records |

Identity ≠ compute precision: a quantized checkpoint that binds an NVFP4
identity but executes a W4A16 dequant kernel family is recorded exactly so
(the `moe quant->kernel` arrow in results); admission never equates
"checkpoint loads" with "precision measured". Kernels no taxonomy rule labels are the
record's `unclassified_kernels`; covering them is the coverage-gap step. End-to-end accuracy
is the `aic-silicon-align` skill and the e2e accuracy CI, not a harness step.

`components/gates.py grade --sm <sm> --repo <repo> --results <job dirs>` exits 0 when the path gates that name the repo are aligned.
