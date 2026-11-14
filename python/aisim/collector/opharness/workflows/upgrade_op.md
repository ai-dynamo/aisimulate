# Workflow: upgrade an op (framework version or collector change)

Entry: one op family, a new pin per framework. Exit: the family's tables are
recollected where its kernels changed (reuse declared for the rest), the
collector's cells are shown to run what serving runs on the new pin, and every
difference from the previous release's tables is explained.

Collection itself follows `docs/perf-model/collector/upgrade.md` and the
collector overview's "Every pin bump" row; this table adds the three checks of
the README. One op family moves at a time, across the three frameworks.

| # | Step | Who | How |
|---|------|-----|-----|
| 1 | Bump the pin | owner | `collector/framework_manifest.yaml`: the default, or a family pin for the ops in scope; the collector modules' `__compat__` follow |
| 2 | Source check | AI | per framework, diff the two releases for the family's selector, kernels, fusions, graph mode, autotuning and metadata on **every SM with shipped data**, not only the SMs the gates run on. Re-read in the same pass every `collector/skips.py` entry and deviation note of the family (rule 7; `collect.py --plan-only` lists each entry with the cases it leaves out): keep it with its source, or delete it. Fix what the check finds, with a file-and-symbol citation, before any GPU job |
| 3 | Path gates | script | per SM generation: `probe_driver.py --system <system> --campaign <name> --gates --op <op> --emit-shards --url-base <share>` -> `gates.py plan --sm <sm> --op <op> --serving <campaign dir> --out <dir>` -> the collection pipeline, one job per (backend, release) -> `probe_driver.py --import-raws <job dir> --records` -> `gates.py grade --sm <sm> --op <op> --results <job dirs>`. Done when it exits 0. Before the first job, `gates.py paths --sm <sm> --op <op>` (no GPU) lists every path the family's plans collect and the gate on it: a path step 2 found changed and no gate names gets a `gates.yaml` row, or `--derive` on the three commands when its cases name a roster checkpoint |
| 4 | Collect | script | the family's full plan on every system. After the first window, `run_diff.py failures <result dirs>` on what has landed: a key bucket that fails as a whole is fixed or left out in `collector/skips.py` before the rest of the run pays for it (the 2026-11 run met its SM89 and SM120 failures only here: the gates ran on SM90 and SM100) |
| 5 | Run diff | script | `run_diff.py tables <packaged tables> --base <shipped data> --detail` and `run_diff.py failures <result dirs>`. A table whose compute-bound cases slowed while its launch-bound cases held is recollected on another node before anything else is concluded from it |
| 6 | Triage | AI | every lead of step 5 and every new failure group against the previous release ends as one of: a fix; a `collector/skips.py` entry with its reason (a model the backend cannot serve, a block that cannot run); a deviation note at the code (a path the table measures differently); an unavailable key reported in the PR; or a sentence in the commit naming the collector change that explains it |
| 7 | Integrate | script | the tables with their `collection_meta.yaml`, the data checks, and the unit and golden tests against the data |

## Across SMs

An upgrade validated on one SM is validated only for the fields and code
paths that SM reads (the 0.29 MLA block-table width passed every SM90 smoke and
crashed on B300).

1. Prefer values asked from the framework (`get_supported_kernel_block_sizes`,
   spec fields) over literals; a platform-dependent literal is a review finding.
2. When the framework branches by SM, mirror its branch with a citation.
3. Every other SM with shipped data gets a smoke and its own path verdicts, or
   an `unverified_sms` mark in the registry; never leave a sibling SM silently
   green.
4. A new SM adds taxonomy rules scoped to it with `sms:`.
