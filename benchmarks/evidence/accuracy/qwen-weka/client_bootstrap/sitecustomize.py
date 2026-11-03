# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin and observe the official loader in every AgentX client subprocess."""

import json
import os
from dataclasses import asdict
from pathlib import Path

from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader
from aiperf.dataset.loader.semianalysis_cc_traces_weka import SemiAnalysisCCTracesWekaLoader
from aiperf.timing.trajectory_source import TrajectorySource

BaseHFDatasetLoader.hf_revision = "23f152f6f0f9399a85901b89a6458def0ef16729"
original_validate = SemiAnalysisCCTracesWekaLoader._validate_rows


def recorded_loader(self, dataset):
    selected = original_validate(self, dataset)
    root = Path(os.environ["AGENTX_COLLECTION_RECORD"])
    root.mkdir(parents=True, exist_ok=True)
    identities = list(selected)
    if len(identities) != 220:
        raise ValueError("AgentX whole-play filter did not select 220 plays")
    (root / "resolved-dataset.json").write_text(self.run.cfg.model_dump_json(indent=2))
    (root / "eligible-plays.json").write_text(json.dumps(identities, indent=2))
    return selected


SemiAnalysisCCTracesWekaLoader._validate_rows = recorded_loader
original_trajectory = TrajectorySource._build_timestamped_trajectory


def recorded_trajectory(self, root_id, lane_index=None):
    result = original_trajectory(self, root_id, lane_index)
    if result is not None and result.snapshot is not None:
        root = Path(os.environ["AGENTX_COLLECTION_RECORD"])
        root.mkdir(parents=True, exist_ok=True)
        row = {
            "root_id": root_id,
            "lane_index": lane_index,
            "snapshot": asdict(result.snapshot),
            "start_turn_index": result.start_turn_index,
        }
        descriptor = os.open(root / "trajectory-cuts.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, (json.dumps(row, default=str) + "\n").encode())
        finally:
            os.close(descriptor)
        metadata = root / "dataset-metadata.json"
        if not metadata.exists():
            metadata.write_text(self.dataset_metadata.model_dump_json(indent=2))
    return result


TrajectorySource._build_timestamped_trajectory = recorded_trajectory
