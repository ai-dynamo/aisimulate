# CODEOWNERS as Code

The root `CODEOWNERS` file is generated from `areas.yaml`. Change the policy
source, regenerate the file, and commit both changes together.

The policy uses the repository-specific Forward Pass Engine, Sweeper, Replay,
Mocker, Infrastructure, Maintainers, and DevOps teams. GitHub treats multiple
owners on one line as “any one approves”: co-ownership adds review visibility
and fallback coverage, not one mandatory approval from every listed team.

## Change the policy

```bash
python -m pip install pyyaml pytest
python .github/codeowners/codeowners.py \
  --policy .github/codeowners/areas.yaml \
  --out CODEOWNERS \
  --repo . \
  --write \
  --validate
python -m pytest -c /dev/null .github/codeowners/test_*.py -q
```

Validation rejects shadow files, fails when a tracked path has only the root
fallback owner, and checks that `CODEOWNERS` matches the YAML source exactly.
Repository rules must separately require code-owner review and the CODEOWNERS
check before review requests and validation failures become merge gates.
