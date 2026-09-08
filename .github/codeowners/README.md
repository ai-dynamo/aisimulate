# CODEOWNERS as Code

This directory follows the CODEOWNERS management design used by the Dynamo and
AIConfigurator repositories. The root `CODEOWNERS` file is generated from one
declarative source. **Do not hand-edit `CODEOWNERS`**: change `areas.yaml`,
regenerate, and commit the source and generated artifacts together.

## Who Reviews My Change?

GitHub auto-requests the team that owns the files changed by a pull request. To
check the routing before pushing:

```bash
# all owners for the files changed by this branch
python .github/codeowners/who_owns.py --codeowners CODEOWNERS --changed --base main

# owners for specific paths
python .github/codeowners/who_owns.py --codeowners CODEOWNERS \
  crates/core/src/engine/runtime.rs src/aisimulate/sweeper/search.py
```

A line with more than one owner is co-ownership. Under GitHub's code-owner
review rule, any one listed owner can satisfy the gate; co-ownership improves
review visibility without requiring one approval from every listed team.

## Files

| File | Purpose |
|------|---------|
| `areas.yaml` | Single source of truth for subsystem paths and GitHub teams. **Edit this.** |
| `external_contributors.yaml` | Area-scoped external code owners and source for `CONTRIBUTORS.md`. **Edit this when applicable.** |
| `codeowners_match.py` | Canonical matcher and policy resolver shared by every tool. |
| `build_codeowners.py` | Validates 100% explicit ownership coverage of the tracked tree. |
| `emit_codeowners.py` | Generates root `CODEOWNERS` and `CONTRIBUTORS.md` from policy inputs. |
| `who_owns.py` | Reports the owners for paths or a branch diff. |
| `test_codeowners.py` | Generic matcher, resolver, emitter, and validation tests. |
| `test_aisimulate_policy.py` | AISimulate-specific routing contract. |

## Change Ownership

From the repository root:

```bash
python -m pip install pyyaml pytest
python .github/codeowners/build_codeowners.py \
  --areas .github/codeowners/areas.yaml --repo . --strict
python .github/codeowners/emit_codeowners.py \
  --areas .github/codeowners/areas.yaml \
  --out CODEOWNERS \
  --external .github/codeowners/external_contributors.yaml \
  --contributors-out CONTRIBUTORS.md
python -m pytest -c /dev/null .github/codeowners/test_*.py -q \
  -p no:cacheprovider
```

## How It Stays Correct

The `codeowners` workflow runs on every pull request and main-branch push. It
rejects shadow CODEOWNERS files, runs the shared and repository-specific tests,
validates explicit path coverage, and fails when generated artifacts drift from
their declarative sources.

The workflow makes policy problems visible. Repository rules must separately
require the `codeowners` status check and code-owner review for those results to
be merge-blocking.

## Migrated AIConfigurator Snapshot

The full AIC source import preserves `python/aisimulate/.github/codeowners/` as
inactive provenance, but its generated `CODEOWNERS` file is not retained.
GitHub only discovers CODEOWNERS at the repository root, `.github/`, or `docs/`,
so the generated root file is the only active repository policy. The root
policy routes changes to the preserved ownership tooling to AISimulate Infra
plus maintainers to keep the active and historical policies from being
confused. The DevOps team is not listed in the active policy and is never a
required code-owner approver.
