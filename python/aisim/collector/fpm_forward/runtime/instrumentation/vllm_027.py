# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.27.0 adapter; source mapping and limitations are in vllm-0.27.0.md."""

from dynamo.vllm.instrumented_scheduler import InstrumentedScheduler
from vllm.v1.worker.gpu_worker import Worker

from .hooks import SchedulerObservation, WorkerObservation


class ObservedWorker(WorkerObservation, Worker):
    observation_version = "0.27.0"


class ObservedInstrumentedScheduler(SchedulerObservation, InstrumentedScheduler):
    observation_version = "0.27.0"
