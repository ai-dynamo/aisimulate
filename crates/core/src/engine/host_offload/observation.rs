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

/// Borrowed host-offload transition emitted only when a caller installs an
/// observer. The observer owns any retention or serialization policy.
pub(crate) struct HostOffloadObservation<'a> {
    pub(crate) request_id: Uuid,
    pub(crate) event: HostOffloadObservationData<'a>,
}

pub(crate) enum HostOffloadObservationData<'a> {
    StorePrepared {
        at_ms: f64,
        transfer_id: TransferId,
        blocks: &'a [HostBlockKey],
    },
    /// Framework-owned identity seam for the corresponding prepared store.
    StoreBlockMappings {
        at_ms: f64,
        transfer_id: TransferId,
        mappings: &'a [HostStoreBlockMapping],
    },
    StoreSubmitted {
        at_ms: f64,
        completes_at_ms: f64,
        transfer_id: TransferId,
        blocks: &'a [HostBlockKey],
    },
    StoreCompleted {
        at_ms: f64,
        transfer_id: TransferId,
        blocks: &'a [HostBlockKey],
    },
    LoadQueued {
        at_ms: f64,
        completes_at_ms: f64,
        transfer_id: TransferId,
        blocks: &'a [HostBlockKey],
    },
    LoadCompleted {
        at_ms: f64,
        transfer_id: TransferId,
        blocks: &'a [HostBlockKey],
    },
    LoadCancelled {
        at_ms: f64,
        transfer_id: TransferId,
        blocks: &'a [HostBlockKey],
    },
    Evicted {
        at_ms: f64,
        block: HostBlockKey,
    },
    CapacityRetry {
        at_ms: f64,
        blocks: &'a [HostBlockKey],
        structurally_unfittable: bool,
    },
}

/// Optional synchronous destination for parity observations.
pub(crate) trait HostOffloadObserver: Send + Sync {
    fn record(&self, observation: HostOffloadObservation<'_>);
}
