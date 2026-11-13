# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep learned-model dependency imports independent of collector Torch stubs."""

import sys
from contextlib import contextmanager

import pytest


@contextmanager
def _without_stub_torch():
    # Collector modules need their import-time stub for fork-worker tests.
    # SciPy's array API expects torch.Tensor to be a class, so hide only an
    # incompatible stub and restore it after import or the individual test.
    with pytest.MonkeyPatch.context() as patch:
        torch = sys.modules.get("torch")
        if torch is not None and not isinstance(getattr(torch, "Tensor", None), type):
            patch.delitem(sys.modules, "torch")
        yield


def import_sklearn():
    """Preserve dependency skips while protecting collection-time imports."""
    with _without_stub_torch():
        return pytest.importorskip("sklearn")


@pytest.fixture(autouse=True)
def isolate_training_dependencies():
    """Protect SciPy's runtime array checks as well as its initial import."""
    with _without_stub_torch():
        yield
