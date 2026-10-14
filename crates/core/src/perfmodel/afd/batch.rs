// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Balanced replica then microbatch partition, represented by at most three runs.
use crate::AicError;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct BatchShape {
    /// Contiguous replica groups, larger replicas first. Zero groups are inert.
    pub groups: [(u32, u32); 2],
}
impl BatchShape {
    pub fn full(batch: u32, replicas: u32) -> Self {
        Self {
            groups: [
                (
                    batch / replicas + u32::from(batch % replicas != 0),
                    batch % replicas,
                ),
                (batch / replicas, replicas - batch % replicas),
            ],
        }
    }
    pub fn max(self) -> u32 {
        self.groups
            .into_iter()
            .filter(|&(_, n)| n > 0)
            .map(|(b, _)| b)
            .max()
            .unwrap_or(0)
    }
    pub fn total(self) -> u32 {
        self.groups.into_iter().map(|(b, n)| b * n).sum()
    }
    pub fn scaled(self, width: u32) -> Result<Self, AicError> {
        let mut result = self;
        for (b, _) in &mut result.groups {
            *b = b
                .checked_mul(width)
                .ok_or_else(|| AicError::InvalidEngineConfig("AFD token count overflow".into()))?;
        }
        // The pool total is a query argument, not only each replica's count.
        self.total()
            .checked_mul(width)
            .ok_or_else(|| AicError::InvalidEngineConfig("AFD pool token count overflow".into()))?;
        Ok(result)
    }
}

pub(super) struct Microbatches {
    pub runs: [(BatchShape, u32); 3],
    pub len: usize,
    pub count: u32,
}
impl Microbatches {
    pub fn new(batch: u32, replicas: u32, configured: u32) -> Self {
        debug_assert!(batch > 0 && replicas > 0 && configured > 0);
        let full = BatchShape::full(batch, replicas);
        let m = configured.min(full.max());
        let lo = batch / replicas;
        let hi = lo + u32::from(batch % replicas != 0);
        let mut edges = [0, lo % m, hi % m, m];
        edges.sort_unstable();
        let mut result = Self {
            runs: Default::default(),
            len: 0,
            count: m,
        };
        for pair in edges.windows(2) {
            if pair[0] == pair[1] {
                continue;
            }
            let shape = BatchShape {
                groups: full
                    .groups
                    .map(|(b, n)| (b / m + u32::from(pair[0] < b % m), n)),
            };
            result.runs[result.len] = (shape, pair[1] - pair[0]);
            result.len += 1;
        }
        result
    }
    pub fn run(&self, micro: usize) -> usize {
        let mut end = 0;
        for i in 0..self.len {
            end += self.runs[i].1 as usize;
            if micro < end {
                return i;
            }
        }
        unreachable!("microbatch index")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partition_conserves_requests_and_never_invents_idle_replicas() {
        for r in [1, 2, 8, 64] {
            for b in 1..130 {
                for m in [1, 2, 3, 4, 7, 16] {
                    let p = Microbatches::new(b, r, m);
                    assert!(p.len <= 3);
                    assert_eq!(
                        p.runs[..p.len]
                            .iter()
                            .map(|(s, n)| s.total() * n)
                            .sum::<u32>(),
                        b
                    );
                    assert_eq!(p.runs[..p.len].iter().map(|(_, n)| n).sum::<u32>(), p.count);
                }
            }
        }
        let p = Microbatches::new(1, 8, 3);
        assert_eq!(p.runs[0].0.total(), 1);
        let p = Microbatches::new(8, 1, 3);
        assert_eq!(
            p.runs[..p.len]
                .iter()
                .map(|(s, n)| (s.total(), *n))
                .collect::<Vec<_>>(),
            vec![(3, 2), (2, 1)]
        );
    }
}
