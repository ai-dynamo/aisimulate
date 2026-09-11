# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build both linux/amd64 and linux/arm64. Supply an immutable digest of the
# existing runner image; retain its entrypoint and runtime configuration.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
RUN apt-get -o Acquire::Retries=3 update \
    && apt-get -o Acquire::Retries=3 install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
USER runner
