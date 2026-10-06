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
| 5 | Decompose | script | `components/decompose.py`: observed execution -> op families (taxonomy roles) + residue |
| 6 | Granularity decision | owner | non-empty residue = the model has execution no family covers; deciding "new family / new table / absorb" is a human call, recorded before any collection |
| 7a | SDK manifest | script | `components/e2e_align.py --sdk-manifest REPO`: the SDK graph's measured identities (component x layer x structure key) per backend -> `results/<sm>/manifest/`; requires `op_family` on the checkpoint (targets.yaml roster.checkpoint_overrides), e.g. `dsv411` for DeepSeek-V4.1 |
| 7b | Module identity | script | every manifest component has an observed serving role in the step-5 decomposition (`module_identity_aligned`); a missing role is a taxonomy rule or a granularity call, never a collector fix |
| 7c | Coverage gap -> dev | AI | missing collector modules / case declarations (this hands off to the new_op_collector workflow when a family has no collector at all) |
| 8 | Collect + admit | script | the family's producers on the declared grid (`collector/cases/base_ops/<family>*.yaml`), admitted by the family contract; the publisher records the admission (`publish.py --admission-record` -> `results/<sm>/admission/`) |
| 9 | Path alignment | script | `components/path_diff.py`: collector path vs step-3 serving records (`components/captures/<family>_*.py`, one cell per component x phase) |
| 10 | Publication | script | the published table loads with strict provenance and answers a SILICON prediction on `platform.perf_system` (`published_loadable`) |
| 11 | E2E admission | script | `components/e2e_align.py`: prediction vs live measurement on the golden deployment |

Identity ≠ compute precision: a quantized checkpoint that binds an NVFP4
identity but executes a W4A16 dequant kernel family is recorded exactly so
(the `moe quant->kernel` arrow in results); admission never equates
"checkpoint loads" with "precision measured".

Progress is derived, not self-reported: `components/workflow_check.py onboard_model --param ...` evaluates the sibling `onboard_model.yaml` manifest against artifacts and names the first actionable step.
