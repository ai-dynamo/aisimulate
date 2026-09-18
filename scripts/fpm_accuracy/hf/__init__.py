# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Hugging Face source-evidence access for AISim FPM Gym."""

from fpm_accuracy.hf.dataset import DEFAULT_REPO_ID, HfDataset
from fpm_accuracy.hf.models import (
    CaseStatus,
    ConfigurationSnapshot,
    FpmArtifact,
    MeasurementCase,
    MeasurementFile,
    MeasurementIssue,
    MeasurementObservation,
    MeasurementReference,
    MeasurementState,
    OrderingKind,
)
from fpm_accuracy.hf.overrides import HfCaseOverride, HfOverrides, load_overrides
from fpm_accuracy.hf.protocols import (
    SUPPORTED_EVIDENCE_FORMAT_IDS,
    SUPPORTED_PROTOCOL_IDS,
    ProtocolAdapter,
    adapter_for,
)

__all__ = [
    "DEFAULT_REPO_ID",
    "SUPPORTED_EVIDENCE_FORMAT_IDS",
    "SUPPORTED_PROTOCOL_IDS",
    "CaseStatus",
    "ConfigurationSnapshot",
    "FpmArtifact",
    "HfCaseOverride",
    "HfDataset",
    "HfOverrides",
    "MeasurementCase",
    "MeasurementFile",
    "MeasurementIssue",
    "MeasurementObservation",
    "MeasurementReference",
    "MeasurementState",
    "OrderingKind",
    "ProtocolAdapter",
    "adapter_for",
    "load_overrides",
]
