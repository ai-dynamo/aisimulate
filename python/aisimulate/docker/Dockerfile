
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG WHEEL_BUILD_BASE=quay.io/pypa/manylinux_2_28_x86_64

FROM nvcr.io/nvidia/base/ubuntu:jammy-20250619 AS base

WORKDIR /workspace

ENV VIRTUAL_ENV=/opt/aiconfigurator/venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN mkdir /opt/aiconfigurator && \
    uv venv --python 3.13 ${VIRTUAL_ENV} && \
    chmod a+rx /root && \
    chmod -R a+rX /root/.local
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV MPLCONFIGDIR=/tmp/matplotlib

# One `aisimulate` wheel owns the application, compatibility namespaces,
# SDK/data payload, Replay runtime, and unified Rust extension. It is built in
# the PyPA manylinux_2_28 image so the native PyO3 `.so` links glibc 2.28 and
# maturin's auditwheel stamps the portable
# `manylinux_2_28_<arch>` tag. We add `uv` and a rustup toolchain so cargo can
# fetch crates over TLS.
#
# CI selects the architecture-matched PyPA image with WHEEL_BUILD_BASE. The
# x86_64 default preserves existing local and build-test invocations.
FROM ${WHEEL_BUILD_BASE} AS build
WORKDIR /workspace
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo \
    PATH=/opt/rust/cargo/bin:${PATH}
# Download to disk (not `curl | sh`, which hides a failed download behind sh's
# exit status); pin the toolchain for reproducible release wheels.
RUN curl --proto '=https' --tlsv1.2 -sSf --retry 5 --retry-connrefused --retry-delay 2 \
        https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain 1.96.0 \
    && rm /tmp/rustup-init.sh

# Pin to the image's CPython 3.13 so uv/maturin never fetch a standalone one.
ENV UV_PYTHON=/opt/python/cp313-cp313/bin/python3
COPY Cargo.toml Cargo.lock /workspace/
COPY crates/ /workspace/crates/
COPY python/aisimulate/src/ /workspace/python/aisimulate/src/
COPY python/aisimulate/LICENSE python/aisimulate/THIRD_PARTY_NOTICES.md python/aisimulate/pyproject.toml python/aisimulate/README.md /workspace/python/aisimulate/
COPY python/aisimulate/tools/verify_release_wheels.py /workspace/python/aisimulate/tools/verify_release_wheels.py
COPY python/aisimulate/tools/verify_installed_package_layers.py /workspace/python/aisimulate/tools/verify_installed_package_layers.py
COPY python/aisimulate/tools/build_legacy_monolith_fixture.py /workspace/python/aisimulate/tools/build_legacy_monolith_fixture.py
# Fail fast if Git LFS content wasn't materialized before the build — otherwise
# the wheel ships pointer stubs in place of perf data (see DYN-2725).
RUN set -e; \
    bad=$(grep -rlE "^version https://git-lfs" /workspace/python/aisimulate/src/aiconfigurator_core/systems 2>/dev/null || true); \
    if [ -n "$bad" ]; then \
        echo "ERROR: Unmaterialized Git LFS pointer files found:" >&2; \
        echo "$bad" >&2; \
        echo "Run 'git lfs install && git lfs pull' in the source tree before building." >&2; \
        exit 1; \
    fi
# `uv build` runs maturin's PEP517 backend, which defaults to `--compatibility
# off`; MATURIN_PEP517_ARGS overrides it to stamp the manylinux_2_28 tag (maturin
# appends it last, so it wins). `python` isn't on PATH here -> verify via UV_PYTHON.
RUN cd /workspace/python/aisimulate && \
    MATURIN_PEP517_ARGS="--compatibility manylinux_2_28" \
        uv build --wheel --out-dir /workspace/dist && \
    "${UV_PYTHON}" tools/verify_release_wheels.py /workspace/dist

# Exercise the one-wheel boundary outside the checkout, including all
# compatibility imports, bundled data, and the native estimator.
RUN uv venv --python 3.13 /tmp/aisimulate-wheel-test && \
    AISIMULATE_WHL=$(ls /workspace/dist/aisimulate-*.whl) && \
    uv pip install --python /tmp/aisimulate-wheel-test/bin/python "${AISIMULATE_WHL}" && \
    uv pip check --python /tmp/aisimulate-wheel-test/bin/python && \
    cd /tmp && /tmp/aisimulate-wheel-test/bin/python \
        /workspace/python/aisimulate/tools/verify_installed_package_layers.py --exercise-engine

# Exercise the supported clean migration from the historical monolithic 0.9
# wheel. Remove the old owner before installing `aisimulate` so pip cannot later
# delete a compatibility file now owned by the unified distribution.
RUN "${UV_PYTHON}" /workspace/python/aisimulate/tools/build_legacy_monolith_fixture.py /tmp/legacy-dist && \
    LEGACY_WHL=$(ls /tmp/legacy-dist/aiconfigurator-0.9.0-*.whl) && \
    AISIMULATE_WHL=$(ls /workspace/dist/aisimulate-*.whl) && \
    "${UV_PYTHON}" -m venv /tmp/aic-pip-migration && \
    /tmp/aic-pip-migration/bin/pip install "${LEGACY_WHL}" && \
    /tmp/aic-pip-migration/bin/pip uninstall -y aiconfigurator aiconfigurator-core && \
    /tmp/aic-pip-migration/bin/pip install "${AISIMULATE_WHL}" && \
    cd /tmp && /tmp/aic-pip-migration/bin/python \
        /workspace/python/aisimulate/tools/verify_installed_package_layers.py --exercise-engine

# Export the verified unified wheel for CI staging jobs.
FROM scratch AS wheel-out
COPY --from=build /workspace/dist/ /

FROM base AS runtime
COPY --from=build /workspace/dist /wheelhouse
RUN uv pip install /wheelhouse/aisimulate-*.whl && \
    rm -rf /wheelhouse

FROM runtime AS test

# The perf-database tool tests shell out to git (changed_ops revision diffing,
# migrate_* execute paths use git mv/rm/add). Test stage only — the shipped
# runtime image stays git-free.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Runtime already installed the unified wheel. Add its development extras
# before copying the test sources.
COPY --from=build /workspace/dist /wheelhouse
RUN AISIMULATE_WHL=$(ls /wheelhouse/aisimulate-*.whl) && \
    uv pip install "${AISIMULATE_WHL}[dev]" && \
    rm -rf /wheelhouse

COPY python/aisimulate/pytest.ini /workspace/
COPY python/aisimulate/tests/ /workspace/tests/
COPY python/aisimulate/src/ /workspace/src/
COPY python/aisimulate/tools/ /workspace/tools/
COPY python/aisimulate/collector/ /workspace/collector/
COPY python/aisimulate/.agents/ /workspace/.agents/
