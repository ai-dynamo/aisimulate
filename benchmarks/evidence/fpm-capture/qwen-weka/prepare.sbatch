#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=qwen-fpm-prepare
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
# Modified from benchmarks/evidence/accuracy/qwen-weka/prepare.sbatch at 024d22d68:
# also installs the Dynamo scheduler wheels into a separate target directory.
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
storage_root=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/simonec
image="$storage_root/images/vllm-0310-cu130-a4a4c043.sqsh"
cd "$storage_root"
test -f "$CAMPAIGN_ROOT/aisim/python/aisim/src/aisim_core/fpm_record.py"
git -C "$CAMPAIGN_ROOT/aisim" rev-parse HEAD > "$CAMPAIGN_ROOT/aisim-commit.txt"
test -z "$(git -C "$CAMPAIGN_ROOT/aisim" status --porcelain)"
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_DATA_PATH="$CAMPAIGN_ROOT/preparation/enroot"
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
mkdir -p "$ENROOT_DATA_PATH" "$CAMPAIGN_ROOT/preparation"
srun --ntasks=1 --container-image="$image" --container-mounts="$storage_root:/workspace" \
    bash -c 'set -euo pipefail
      root="${1/#\/lustre\/fsw\/portfolios\/coreai\/projects\/coreai_comparch_inferencex\/users\/simonec/\/workspace}"
      export HF_HOME="$root/preparation/hf" PIP_CACHE_DIR="$root/preparation/pip"
      python3 -m venv "$root/client"
      "$root/client/bin/pip" install "agentx-harness==1.0.12"
      "$root/client/bin/python" "$root/code/client.py" --prepare "$root/preparation"
      "$root/client/bin/pip" freeze > "$root/preparation/client-packages.txt"
      sha256sum /workspace/images/vllm-0310-cu130-a4a4c043.sqsh > "$root/preparation/image.sha256"
      test "$(cut -d " " -f 1 "$root/preparation/image.sha256")" = 35b6881618f1ab48ef054126d659e18aebf2a009dd918b726f4220b32a274eae
      "$root/client/bin/agentx" profile --help > "$root/preparation/profile-help.txt"
      # Dynamo scheduler: no dependencies, outside the image site-packages.
      python3 -m pip freeze > "$root/preparation/server-packages-before.txt"
      python3 -m pip install --no-deps --target "$root/preparation/dynamo-site" \
        "ai-dynamo==1.6.1" "ai-dynamo-runtime==1.6.1"
      python3 -m pip freeze > "$root/preparation/server-packages-after.txt"
      cmp "$root/preparation/server-packages-before.txt" "$root/preparation/server-packages-after.txt"
      ls "$root/preparation/dynamo-site" > "$root/preparation/dynamo-site.txt"
      PYTHONPATH="$root/preparation/dynamo-site" python3 -c "
import vllm, dynamo.vllm.instrumented_scheduler as s
assert vllm.__version__.startswith(\"0.31.0\"), vllm.__version__
assert vllm.__file__.startswith(\"/usr\"), vllm.__file__
print(vllm.__version__, s.__file__)" > "$root/preparation/dynamo-check.txt"
    ' bash "$CAMPAIGN_ROOT"
