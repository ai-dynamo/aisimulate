# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Same multi-architecture nightly index used by the B200 Slurm experiment.
ARG BASE_IMAGE=nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly@sha256:cc84ea52fc8e66fb61a54692a30b2dcfa4d9b2349aa39d565039657cba8c26bb
FROM ${BASE_IMAGE}

USER root
ARG TARGETARCH
ARG AIPERF_COMMIT=754356e9a39acc6cc6afb242d123bb57c3fb6f75
LABEL ai.dynamo.experiment="agentx-gb200-hicache" \
      ai.dynamo.aiperf.repository="https://github.com/SemiAnalysisAI/aiperf" \
      ai.dynamo.aiperf.revision="${AIPERF_COMMIT}"

# Keep client dependencies out of the engine environment. Do not put this venv
# first on PATH: python3 must continue to resolve to the original Dynamo runtime.
RUN test "$TARGETARCH" = arm64 \
    && test "$(uname -m)" = aarch64 \
    && python3 -m venv /opt/agentx-aiperf \
    && /opt/agentx-aiperf/bin/pip install --no-cache-dir \
       "aiperf @ https://github.com/SemiAnalysisAI/aiperf/archive/${AIPERF_COMMIT}.tar.gz" \
    && /opt/agentx-aiperf/bin/pip check \
    && /opt/agentx-aiperf/bin/pip freeze > /opt/agentx-aiperf/freeze.txt

USER dynamo
RUN /opt/agentx-aiperf/bin/aiperf profile --help > /tmp/agentx-aiperf-help.txt \
    && grep -q warmup-requests-per-lane /tmp/agentx-aiperf-help.txt \
    && python3 -c 'import importlib.metadata as m; print({p: m.version(p) for p in ["ai-dynamo", "sglang", "flashinfer-python"]})'

# Inherit the base NVIDIA entrypoint. Use /opt/agentx-aiperf/bin/aiperf for load,
# and the base python3 for dynamo.frontend / dynamo.sglang.
