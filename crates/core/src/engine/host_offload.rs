// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Framework-neutral host-residency and virtual-transfer mechanics.
//!
//! Framework adapters decide when to look up, store, load, or touch blocks.
//! This module owns only G2 capacity, LRU ordering, pool ownership and
//! deterministic D2H/H2D service. Transfers are synchronous state transitions
//! driven by virtual time; no runtime, thread, or distributed-storage
//! dependency is involved.

mod observation;
mod registry;
mod tier;

pub(crate) use observation::{
    HostOffloadObservation, HostOffloadObservationData, HostOffloadObserver, HostStoreBlockMapping,
};
pub use registry::SharedG2Pool;
pub(crate) use registry::{G2Binding, G2Registry, HostClient};
#[cfg(test)]
pub(crate) use tier::tests::{C, private};
pub(crate) use tier::{
    CompletedTransfer, HostBlockKey, HostBlockMeta, HostLink, HostTier, HostTierConfig,
    LoadOutcome, Lookup, StoreOutcome, TransferId,
};
