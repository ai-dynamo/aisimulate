// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{HostBlockKey, TransferId};
use uuid::Uuid;

/// One framework-owned logical position for a successfully prepared host block.
///
/// This mapping is emitted only through the opt-in artifact observer. The
/// framework-neutral host tier never consumes or stores request-local indices.
#[derive(Clone, Copy)]
pub(crate) struct HostStoreBlockMapping {
    pub(crate) block: HostBlockKey,
    pub(crate) logical_block_index: usize,
}

/// Host-offload transition emitted only when a caller installs an observer.
pub(crate) struct HostOffloadObservation {
    pub(crate) request_id: Uuid,
    pub(crate) event: HostOffloadObservationData,
}

pub(crate) enum HostOffloadObservationData {
    StorePrepared {
        at_ms: f64,
        transfer_id: TransferId,
        blocks: Vec<HostBlockKey>,
    },
    /// Framework-owned identity seam for the corresponding prepared store.
    StoreBlockMappings {
        at_ms: f64,
        transfer_id: TransferId,
        mappings: Vec<HostStoreBlockMapping>,
    },
    StoreSubmitted {
        at_ms: f64,
        completes_at_ms: f64,
        transfer_id: TransferId,
        blocks: Vec<HostBlockKey>,
    },
    StoreCompleted {
        at_ms: f64,
        transfer_id: TransferId,
        blocks: Vec<HostBlockKey>,
    },
    LoadQueued {
        at_ms: f64,
        completes_at_ms: f64,
        transfer_id: TransferId,
        blocks: Vec<HostBlockKey>,
    },
    LoadCompleted {
        at_ms: f64,
        transfer_id: TransferId,
        blocks: Vec<HostBlockKey>,
    },
    LoadCancelled {
        at_ms: f64,
        transfer_id: TransferId,
        blocks: Vec<HostBlockKey>,
    },
    Evicted {
        at_ms: f64,
        block: HostBlockKey,
    },
    CapacityRetry {
        at_ms: f64,
        blocks: Vec<HostBlockKey>,
        structurally_unfittable: bool,
    },
}

/// Optional synchronous destination for parity observations.
pub(crate) trait HostOffloadObserver: Send + Sync {
    fn record(&self, observation: HostOffloadObservation);
}
