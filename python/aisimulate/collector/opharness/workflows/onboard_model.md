# Workflow: onboard a model (decompose into ops, extend coverage)

Entry: a new checkpoint (HF repo) on the pinned (framework, version, SM).
Exit: the model boots under a generator-rendered command (or its failure is a
root-caused terminal fact), its op decomposition is covered, and its cells
appear in `results/`.

| # | Step | Who | Instrument / artifact |
|---|------|-----|------------------------|
| 1 | Fetch inputs | script | `components/fetch_inputs.py REPO...`: config/hf_quant into `configs/`, every non-weight file (tokenizer, processor, custom code) into `configs/aux_files/<org>_<name>/`; gated repo or missing config = OWNER DECISION (recorded in `targets.yaml` roster.excluded, signed) — never a silent skip. Then add the repo to `configs/repos.txt` and `targets.yaml` roster.extra_repos |
| 2 | Build dummies | script | `components/dummies.py` (depth-cut, width-true; per-repo exceptions only as `dummy_overrides` declarations) |
| 3 | Identity probe | script | `probe_driver.py` per backend: golden `cli generate` render -> probe -> records. Verdict: pass / pass+custom / fail |
| 4 | Rescue or root-cause | AI | for fails: A/B designed by AI but EXECUTED through the probe; workable extra generate args -> `targets.yaml cli_extra_args` (with fact citation); terminal walls -> findings |
| 5 | Decompose | script | decompose component (not built yet; step stays blocked until it exists): observed execution -> op families + residue |
| 6 | Granularity decision | owner | non-empty residue = the model has execution no family covers; deciding "new family / new table / absorb" is a human call, recorded before any collection |
| 7 | Coverage gap -> dev | AI | missing collector modules / case declarations (this hands off to the new_op_collector workflow when a family has no collector at all) |
| 8 | Collect + sanity | script | existing collector for the model's cases; row-level sanity |
| 9 | Path alignment | script | `components/path_diff.py`: collector path vs step-3 serving records |
| 10 | E2E admission | script | e2e_align component (not built yet): prediction vs live measurement on the golden deployment |

Identity ≠ compute precision: a quantized checkpoint that binds an NVFP4
identity but executes a W4A16 dequant kernel family is recorded exactly so
(the `moe quant->kernel` arrow in results); admission never equates
"checkpoint loads" with "precision measured".

Progress is derived, not self-reported: `components/workflow_check.py onboard_model --param ...` evaluates the sibling `onboard_model.yaml` manifest against artifacts and names the first actionable step.
