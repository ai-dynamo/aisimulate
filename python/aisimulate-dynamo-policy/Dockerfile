# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build from the repository root. The source checkout determines both package
# versions; build_dynamo_policy.py checks Rust/Python pins and the Dynamo lock.
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git pkg-config libssl-dev cmake \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.96.1
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CARGO_BUILD_JOBS=2
RUN python -m pip install 'maturin>=1.12,<2'
WORKDIR /source
COPY . .
ARG AISIMULATE_SOURCE_REVISION
RUN python scripts/build_dynamo_policy.py --output-dir /wheels \
    --source-revision "${AISIMULATE_SOURCE_REVISION}"

FROM python:3.12-slim
COPY --from=build /wheels /wheels
RUN python -m pip install /wheels/*.whl && python -m pip check \
    && python -c 'from aisimulate_dynamo_policy.runner import DynamoPolicyRunnerFactory; DynamoPolicyRunnerFactory()'
WORKDIR /work
ENTRYPOINT ["aisimulate"]
