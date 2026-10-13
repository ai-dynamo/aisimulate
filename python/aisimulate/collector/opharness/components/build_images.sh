#!/bin/bash
# Rebuild probe images + the generator venv. ALL version pins come from
# targets.yaml (single source; bump versions there, never here). Re-run
# whenever docker prune eats images on this shared box.
#
# Layout (review 2026-09-25): the harness runs FROM THIS CHECKOUT — probes,
# driver, taxonomy — and keeps only data in the workspace ($AIS_PROBE_WORKSPACE:
# configs/, dummy_models/, archive/, facts/, jitcache/, the generator venv).
set -euxo pipefail
HARNESS="$(cd "$(dirname "$0")/.." && pwd)"                 # collector/opharness
CHECKOUT="$(cd "$HARNESS/../.." && pwd)"                    # python/aisimulate (the package)
WS="${AIS_PROBE_WORKSPACE:-$PWD}"
mkdir -p "$WS/jitcache" "$WS/archive" "$WS/configs" "$WS/dummy_models"

readarray -t PINS < <(python3 - "$HARNESS/targets.yaml" <<'PY'
import sys, yaml
t = yaml.safe_load(open(sys.argv[1]))
for be, cfg in t['backends'].items():
    for ver, img in cfg['images'].items():
        print(f"{be}|{ver}|{img}")
PY
)
for pin in "${PINS[@]}"; do
  IFS='|' read -r be ver img <<< "$pin"
  docker pull "$img"
done

# --- generator CLI venv (golden pipeline) ------------------------------------
# The golden loop invokes THIS repository's `aiconfigurator cli generate`
# (components/probe_driver.py GEN_CLI, override AIS_GENERATOR_CLI). The native
# runtime is built from the checkout with maturin (needs ~/.cargo/bin).
export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
command -v uv >/dev/null || python3 -m pip install -q uv
cd "$WS"
uv venv venv_ais --python 3.12
VIRTUAL_ENV="$WS/venv_ais" uv pip install -e "$CHECKOUT"
"$WS/venv_ais/bin/aiconfigurator" --help >/dev/null
echo "generator venv: $WS/venv_ais (checkout $(git -C "$CHECKOUT" rev-parse --short HEAD))"
