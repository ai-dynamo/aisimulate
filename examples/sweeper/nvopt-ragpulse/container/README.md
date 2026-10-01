<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate / Dynamo CPU environment

Image: `nvcr.io/nvidian/dynamo-dev/aisimulate:hzhou-0930-02`.
Use the immutable pull reference in `push-result.json` when launching experiments.

Linux amd64, Ubuntu 24.04/Python 3.12, Dynamo `c7241c2f153efba10b57c38c2144b70d82194a4d`, AISimulate `0.13.0.dev202609300000000061`, Google Vizier 0.1.21, JAX/JAXlib 0.4.38. Exact wheel SHA256 hashes and source identities are in `image-provenance/`. The native Dynamo wheel requires GLIBC 2.39 and uses the `ais-forward-pass,replay-bench` release features. This image simulates GPUs; no GPU engine or model weights are installed.

The default command runs only a small offline smoke. It checks DeepSeek-V3 config/tokenizer, all 11 predictor presets, static/KV Router/Planner replay (24 requests), Planner history bootstrap and 7 ticks, and a Vizier GP-bandit suggest/complete. 2 CPUs/4 GB RAM and UID 1000 with networking disabled passed. This does not qualify full campaigns or search quality. The separate design check makes zero predictor training or simulation calls.

```bash
docker run --rm --network none --cpus 2 --memory 4g \
  --user 1000:1000 \
  nvcr.io/nvidian/dynamo-dev/aisimulate:hzhou-0930-02
```

Mount experiment code/configs to `/workspace` and traces to `/data`, then override the command with the desired Python entrypoint. The image defaults to offline HF access and contains 7.9 MB of DeepSeek-V3 metadata/tokenizer/license files at the recorded HF revision. Config paths must use container locations.

This source bundle contains only text. Wheel files and HF cache files must be retained separately. Stage an exact Docker context from those artifacts; the staging helper copies only names listed in the manifests and rejects checksum mismatches, existing output directories, and missing artifacts. It never copies credentials or an entire HF cache.

```bash
python stage_context.py \
  --wheelhouse /path/to/retained/wheelhouse \
  --hf-hub-cache /path/to/retained/hf-cache/hub \
  --output /path/to/new/build-context
docker build --platform linux/amd64 \
  -t nvcr.io/nvidian/dynamo-dev/aisimulate:hzhou-0930-02 \
  /path/to/new/build-context
```

Python installation uses 144 exact hashed wheels with `--no-index --require-hashes`. The base image is digest-pinned. OS package versions are recorded inside `/opt/provenance/dpkg-packages.txt`; a future rebuild can receive updated Ubuntu packages, so bit-identical OS layers are not claimed. The pure Python ai-dynamo wheel was built from a copy of the pinned source; the native Dynamo/AIS wheels are reused unchanged. Compared with the borrowed host environment, aiohttp 3.14.3 satisfies the pinned Dynamo source dependency and kubernetes_asyncio 32.0.0 follows its Planner requirements. Numerical pins are unchanged.


The external supplemental probe in `supplemental/` ran against the immutable
pushed image with a read-only source mount. It fed four completed synthetic
2D observations to `VizierGPUCBPEBandit` before requesting another suggestion,
which exercised actual GP fitting and acquisition optimization beyond the seed
path. Reduced budgets (20 ARD iterations, 100 acquisition evaluations) keep this
a functional check; it makes no search-quality claim and calls Dynamo zero times.
Its result and source hash are saved separately because this helper is not baked
into the image.


The final image is `hzhou-0930-02`, which explicitly sets
`JAX_ENABLE_X64=true` to match the frozen experiment's `scripts/run_broad.py`.
The earlier `hzhou-0930-01` image remains an immutable historical artifact and
is not the final experiment environment. The base, OS dependencies, Python
wheels, and HF metadata layers are reused unchanged. Historical environment
smoke results identify the image on which they ran. The supplemental GP-UCB-PE
probe runs against the new immutable image, asserts x64 is enabled and the
actual default floating dtype is float64, then exercises GP fitting and
acquisition after four completed synthetic observations. No full replay was
rerun for this environment-variable-only numerical correction.
