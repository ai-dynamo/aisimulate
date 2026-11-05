// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use anyhow::Result;
use uuid::Uuid;

use crate::replay::protocol::DirectRequest;

/// A finite request source evaluated only when replay has a free concurrency slot.
///
/// The factory receives the original zero-based request index, independent of
/// completion order. It must not retain previously generated requests. Replay
/// preserves the factory's request metadata and replaces the arrival timestamp
/// when admitting the request, just as it does for an eager concurrency queue.
pub struct GeneratedRequests {
    request_count: usize,
    next_index: usize,
    factory: Box<dyn FnMut(usize) -> Result<DirectRequest> + Send>,
    canonical_ids: bool,
}

impl fmt::Debug for GeneratedRequests {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeneratedRequests")
            .field("request_count", &self.request_count)
            .field("next_index", &self.next_index)
            .finish_non_exhaustive()
    }
}

impl GeneratedRequests {
    pub fn new(
        request_count: usize,
        factory: impl FnMut(usize) -> Result<DirectRequest> + Send + 'static,
    ) -> Self {
        Self {
            request_count,
            next_index: 0,
            factory: Box::new(factory),
            canonical_ids: false,
        }
    }

    pub(crate) fn remaining(&self) -> usize {
        self.request_count - self.next_index
    }

    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    pub(crate) fn set_canonical_ids(&mut self) {
        self.canonical_ids = true;
    }

    pub(crate) fn pop_front(&mut self) -> Result<Option<DirectRequest>> {
        if self.remaining() == 0 {
            return Ok(None);
        }
        let mut request = (self.factory)(self.next_index)?;
        if self.canonical_ids {
            request.uuid = Some(Uuid::from_u128(self.next_index as u128 + 1));
        }
        self.next_index += 1;
        Ok(Some(request))
    }
}
