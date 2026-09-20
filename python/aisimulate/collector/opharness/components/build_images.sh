#!/bin/bash
# Rebuild probe images + the generator-CLI venv. ALL version pins come from
# targets.yaml (single source; bump versions there, never here). Re-run
# whenever docker prune eats images on this shared box.
set -euxo pipefail
cd "$(dirname "$0")/.."

readarray -t PINS < <(python3 - <<'PY'
import yaml
t = yaml.safe_load(open('targets.yaml'))
for be, cfg in t['backends'].items():
    for ver, img in cfg['images'].items():
        print(f"{be}|{ver}|{img}")
PY
)

# the checkout itself declares which compiled core it needs — read, don't pin
CORE_WHEEL=$(python3 -c "
import re
print(re.search(r'aiconfigurator-core==([\w.]+)', open('aic/pyproject.toml').read()).group(1))")
for pin in "${PINS[@]}"; do
  IFS='|' read -r be ver img <<< "$pin"
  case "$be" in
    vllm)     docker pull "$img" ;;
    *)        docker pull "$img" ;;
  esac
done

# (vllm-probe:<ver>-fix retired with the 0.29.0 pin: official images since
#  0.27.1 ship tilelang without the broken libcudart stub — findings
#  vllm_024_image_tilelang_stub records the full history.)

# --- generator CLI venv (golden pipeline) ------------------------------------
# The golden loop invokes the REAL generator `cli generate` command
# (currently the predecessor aiconfigurator toolchain's venv checkout).
# The compiled core is built FROM THE CHECKOUT (the PyPI wheel lags upstream
# ABI; the crate's abi3 floor is py3.11 -> use python3.12).
PY312=${PY312:-/root/.local/bin/python3.12}
export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
"$PY312" -m venv venv_aic
./venv_aic/bin/pip install -q maturin jinja2 packaging numpy pandas plotext plotly prettytable pydantic pyarrow pyyaml tqdm matplotlib
./venv_aic/bin/maturin build --release -m aic/aic-core/rust/aiconfigurator-core/Cargo.toml -o /tmp/aic_corewheel
./venv_aic/bin/pip install -q /tmp/aic_corewheel/aiconfigurator_core-*.whl
./venv_aic/bin/pip install -q -e ./aic --no-deps
ln -sf "$(pwd)/$(ls venv_aic/lib/python3.*/site-packages/aiconfigurator_core/_aiconfigurator_core*.so | head -1)" \
       aic/aic-core/src/aiconfigurator_core/_aiconfigurator_core.abi3.so
