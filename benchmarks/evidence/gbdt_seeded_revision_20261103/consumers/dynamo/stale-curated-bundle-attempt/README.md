# Initial Dynamo rerun with stale example bytes

This initial attempt used the rebuilt seeded-only native runtime but the previous
curated example bundle, which still declared `bootstrap_recipe`. Four bundle
cases therefore stopped at schema validation. This is a staged-fixture mismatch,
not an unchanged consumer adapter failure. The raw attempt is retained here; the
final fresh run uses the regenerated bundle with `training_recipe`.

The ordinary regression and legacy checkpoint paths passed (11 cases). Two
attention-DP capacity assertions reproduced their merged-baseline failures.
