<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Verify the Dynamo Planner installation

This example runs the public AISimulate CLI with Planner enabled. It uses
synthetic traffic, local model metadata, fixed timing, and fixed KV capacity;
no model weights, GPU, or Kubernetes cluster are needed. Its timings are
synthetic and do not measure model performance or qualify scaling accuracy.

## Install the matching dependencies

Use Linux with Python 3.12, matching the release validation environment. From
the AISimulate repository root, place the compatible AISimulate, `ai-dynamo`,
and `ai-dynamo-runtime` wheel artifacts in `./wheels/` (one of each):

```bash
python3 -m venv .venv-planner
source .venv-planner/bin/activate
python3 -m pip install \
  ./wheels/aisimulate-*.whl \
  ./wheels/ai_dynamo-*.whl \
  ./wheels/ai_dynamo_runtime-*.whl
```

For an RC, select the exact release artifacts rather than relying on a
package version shared by multiple builds. RC wheels may not be available
from the public package index.

The basic [With Dynamo installation](../../../../README.md#with-dynamo)
does not install all Planner dependencies. Install the complete Planner
requirements from the **same Dynamo tag or commit as those wheels**:

```bash
# Dynamo 1.5.0 RC9 example; replace with the revision of your installed build.
DYNAMO_REF=ffd7c1a90eb403c0d43911690c5c9b8457acd826
python3 -m pip install -r \
  "https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REF}/container/deps/requirements.planner.txt"
python3 -m pip check
```

Use the full requirements file, which includes `scikit-learn` and other
Planner dependencies. The supported prebuilt alternative is the matching
`dynamo-planner` image, which already includes these prerequisites. Use the
image from the same Dynamo release you intend to validate.

## Run a Planner-enabled prediction

From the AISimulate repository root, using the environment above:

```bash
cd docs/cli/examples/dynamo-planner
python3 -m aisimulate predict \
  --stack dynamo \
  --config prediction.yaml \
  --output-dir ./planner-output \
  --capture-per-request \
  --format json
```

Keep this working directory: `prediction.yaml` resolves its `./model` path
relative to it. The included `model/config.json` is synthetic metadata based
on this repository's unified CLI test fixture, with no model weights.

A successful run exits zero, completes all 12 requests, and writes
`planner-output/prediction.json` plus `planner-output/requests.jsonl`.
The configuration explicitly enables load-based Planner scaling, with a
five-second adjustment interval and a two-GPU simulated budget. The
`--stack dynamo` option is required to resolve its top-level `planner` section.
For another run, choose a new output directory or pass `--overwrite` to
replace the known output files.

If the command fails while loading `dynamo.planner` with
`ModuleNotFoundError: No module named 'sklearn'`, the active Python environment
is missing Planner prerequisites. Install the matching requirements there,
run `python3 -m pip check`, and repeat this prediction. A successful dependency
check, basic prediction without Planner, or `predict --help` alone does not
verify that Planner loads.
