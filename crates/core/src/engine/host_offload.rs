// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Framework-neutral host-residency and virtual-transfer mechanics.
//!
//! Framework adapters decide when to look up, store, load, or touch blocks.
//! This module owns only G2 capacity, LRU ordering, and deterministic D2H/H2D
//! deadlines. Transfers are synchronous state transitions driven by virtual
//! time; no runtime, thread, or distributed-storage dependency is involved.

mod observation;
mod tier;

pub(crate) use observation::{
    HostOffloadObservation, HostOffloadObservationData, HostOffloadObserver, HostStoreBlockMapping,
};
pub(crate) use tier::{
    CompletedTransfer, HostBlockKey, HostTier, HostTierConfig, LoadOutcome, Lookup, StoreOutcome,
    TransferId,
};
