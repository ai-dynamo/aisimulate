# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# syntax=docker/dockerfile:1
FROM ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3

LABEL org.opencontainers.image.title="Dynamo AISimulate offline experiment environment" \
      org.opencontainers.image.revision="c7241c2f153efba10b57c38c2144b70d82194a4d" \
      org.opencontainers.image.description="CPU-only DeepSeek-V3/H200 simulation with frozen AISimulate, native Dynamo, Router, Planner and Vizier"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv ca-certificates libstdc++6 libgcc-s1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    JAX_PLATFORMS=cpu \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    DYN_LOG=warn \
    TMPDIR=/tmp
COPY requirements.lock /opt/provenance/requirements.lock
RUN --mount=type=bind,source=wheelhouse,target=/wheels,ro \
    python -m pip install --no-cache-dir --no-index --find-links=/wheels \
        --require-hashes -r /opt/provenance/requirements.lock \
    && python -m pip check
COPY hf-cache/ /opt/huggingface/
COPY image-provenance/ /opt/provenance/
COPY smoke/ /opt/smoke/
RUN mkdir -p /workspace /data \
    && chmod 1777 /workspace /data \
    && chmod -R a+rX /opt/huggingface /opt/provenance /opt/smoke \
    && python -m pip freeze > /opt/provenance/pip-freeze.txt \
    && dpkg-query -W > /opt/provenance/dpkg-packages.txt
# Match the frozen experiment's explicit Vizier numerical precision setting.
ENV JAX_ENABLE_X64=true
WORKDIR /workspace
CMD ["python", "/opt/smoke/environment_smoke.py"]
