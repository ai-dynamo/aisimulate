// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private exact k-d tree used by repeated performance-model lookups.

/// Read-only point storage for a [`KdTree`].
///
/// The tree stores only sample indices. Callers keep their existing coordinate
/// layout and provide each point as a slice during build and search.
pub(crate) trait PointSet {
    fn len(&self) -> usize;
    fn point(&self, sample: usize) -> &[f64];
}

impl PointSet for [Vec<f64>] {
    fn len(&self) -> usize {
        <[Vec<f64>]>::len(self)
    }

    fn point(&self, sample: usize) -> &[f64] {
        &self[sample]
    }
}

/// Bounded nearest-neighbour state supplied by each lookup.
///
/// Implementations choose their exact ordering rule. The squared cutoff is
/// used only for exact far-branch pruning.
pub(crate) trait NeighborCollector {
    fn consider(&mut self, sample: usize, distance_squared: f64);
    fn cutoff_distance_squared(&self) -> Option<f64>;
}

#[derive(Debug, Clone)]
struct KdNode {
    sample: usize,
    axis: usize,
    left: Option<usize>,
    right: Option<usize>,
}

/// Immutable exact nearest-neighbour index over finite, equal-sized points.
#[derive(Debug, Clone)]
pub(crate) struct KdTree {
    nodes: Vec<KdNode>,
    root: usize,
    dims: usize,
    samples: usize,
}

impl KdTree {
    /// Build an index for at least three finite, equal-sized points.
    ///
    /// Smaller or malformed point sets return `None` so callers can keep their
    /// existing linear lookup.
    pub(crate) fn build<P: PointSet + ?Sized>(points: &P) -> Option<Self> {
        if points.len() < 3 {
            return None;
        }
        let dims = points.point(0).len();
        if dims == 0
            || (0..points.len()).any(|sample| {
                let point = points.point(sample);
                point.len() != dims || point.iter().any(|value| !value.is_finite())
            })
        {
            return None;
        }

        fn build_nodes<P: PointSet + ?Sized>(
            points: &P,
            samples: &mut [usize],
            depth: usize,
            dims: usize,
            nodes: &mut Vec<KdNode>,
        ) -> Option<usize> {
            if samples.is_empty() {
                return None;
            }
            let axis = depth % dims;
            let middle = samples.len() / 2;
            samples.select_nth_unstable_by(middle, |&left, &right| {
                points.point(left)[axis]
                    .total_cmp(&points.point(right)[axis])
                    .then_with(|| left.cmp(&right))
            });
            let (left_samples, middle_and_right) = samples.split_at_mut(middle);
            let (sample, right_samples) = middle_and_right
                .split_first_mut()
                .expect("non-empty k-d tree partition");

            let node = nodes.len();
            nodes.push(KdNode {
                sample: *sample,
                axis,
                left: None,
                right: None,
            });
            let left = build_nodes(points, left_samples, depth + 1, dims, nodes);
            let right = build_nodes(points, right_samples, depth + 1, dims, nodes);
            nodes[node].left = left;
            nodes[node].right = right;
            Some(node)
        }

        let samples_count = points.len();
        let mut sample_indices = (0..samples_count).collect::<Vec<_>>();
        let mut nodes = Vec::with_capacity(samples_count);
        let root = build_nodes(points, &mut sample_indices, 0, dims, &mut nodes)
            .expect("non-empty point set has a k-d tree root");
        Some(Self {
            nodes,
            root,
            dims,
            samples: samples_count,
        })
    }

    /// Return whether this tree can search the given query without changing
    /// the caller's former fallback behavior.
    pub(crate) fn can_query(&self, query: &[f64]) -> bool {
        query.len() == self.dims && query.iter().all(|value| value.is_finite())
    }

    /// Search the tree and pass exact candidates to the caller's bounded
    /// collector.
    pub(crate) fn search<P, C>(&self, points: &P, query: &[f64], nearest: &mut C)
    where
        P: PointSet + ?Sized,
        C: NeighborCollector + ?Sized,
    {
        debug_assert_eq!(points.len(), self.samples);
        debug_assert!(self.can_query(query));

        fn visit<P, C>(tree: &KdTree, points: &P, query: &[f64], node_index: usize, nearest: &mut C)
        where
            P: PointSet + ?Sized,
            C: NeighborCollector + ?Sized,
        {
            let node = &tree.nodes[node_index];
            let point = points.point(node.sample);
            let mut distance_squared = 0.0;
            for axis in 0..tree.dims {
                let delta = point[axis] - query[axis];
                distance_squared += delta * delta;
            }
            nearest.consider(node.sample, distance_squared);

            let delta = query[node.axis] - point[node.axis];
            let (near, far) = if delta.is_sign_negative() {
                (node.left, node.right)
            } else {
                (node.right, node.left)
            };
            if let Some(near) = near {
                visit(tree, points, query, near, nearest);
            }
            // Equality must visit the far branch. It can contain an equally
            // distant sample with an earlier original index. Keep this test in
            // squared space so sqrt rounding cannot prune a valid candidate.
            if nearest
                .cutoff_distance_squared()
                .is_none_or(|cutoff| delta * delta <= cutoff)
            {
                if let Some(far) = far {
                    visit(tree, points, query, far, nearest);
                }
            }
        }

        visit(self, points, query, self.root, nearest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Candidate {
        sample: usize,
        distance_squared: f64,
    }

    impl Candidate {
        fn cmp(self, other: Self) -> Ordering {
            self.distance_squared
                .total_cmp(&other.distance_squared)
                .then_with(|| self.sample.cmp(&other.sample))
        }
    }

    struct TopK {
        limit: usize,
        candidates: Vec<Candidate>,
    }

    impl TopK {
        fn new(limit: usize) -> Self {
            Self {
                limit,
                candidates: Vec::with_capacity(limit),
            }
        }
    }

    impl NeighborCollector for TopK {
        fn consider(&mut self, sample: usize, distance_squared: f64) {
            if self.limit == 0 {
                return;
            }
            let candidate = Candidate {
                sample,
                distance_squared,
            };
            if self.candidates.len() == self.limit {
                if !candidate
                    .cmp(*self.candidates.last().expect("full top-k set"))
                    .is_lt()
                {
                    return;
                }
                self.candidates.pop();
            }
            let position = self
                .candidates
                .partition_point(|current| !current.cmp(candidate).is_gt());
            self.candidates.insert(position, candidate);
        }

        fn cutoff_distance_squared(&self) -> Option<f64> {
            (self.limit > 0 && self.candidates.len() == self.limit).then(|| {
                self.candidates
                    .last()
                    .expect("full top-k set")
                    .distance_squared
            })
        }
    }

    fn linear(points: &[Vec<f64>], query: &[f64], limit: usize) -> Vec<Candidate> {
        let mut candidates = (0..points.len())
            .map(|sample| {
                let mut distance_squared = 0.0;
                for axis in 0..query.len() {
                    let delta = points[sample][axis] - query[axis];
                    distance_squared += delta * delta;
                }
                Candidate {
                    sample,
                    distance_squared,
                }
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| left.cmp(*right));
        candidates.truncate(limit);
        candidates
    }

    #[test]
    fn exact_top_k_matches_linear_in_two_to_four_dimensions() {
        for dims in 2..=4 {
            let points = (0..257)
                .map(|sample| {
                    (0..dims)
                        .map(|axis| ((sample * (axis + 3) * 17 + axis * 29) % 251) as f64 / 7.0)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let tree = KdTree::build(points.as_slice()).expect("regular points are indexed");
            for query_number in 0..19 {
                let query = (0..dims)
                    .map(|axis| ((query_number * (axis + 5) * 13 + 3) % 239) as f64 / 11.0)
                    .collect::<Vec<_>>();
                for limit in [1, 2, 12, 300] {
                    let mut indexed = TopK::new(limit);
                    tree.search(points.as_slice(), &query, &mut indexed);
                    assert_eq!(indexed.candidates, linear(&points, &query, limit));
                }
            }
        }
    }

    #[test]
    fn exact_top_k_preserves_ties_underflow_and_large_limits() {
        let tiny = f64::MIN_POSITIVE;
        let points = vec![vec![0.0], vec![tiny], vec![3.0 * tiny]];
        let tree = KdTree::build(points.as_slice()).unwrap();
        let query = [2.0 * tiny];
        let mut indexed = TopK::new(12);
        tree.search(points.as_slice(), &query, &mut indexed);
        assert_eq!(indexed.candidates, linear(&points, &query, 12));
        assert_eq!(
            indexed
                .candidates
                .iter()
                .map(|candidate| candidate.sample)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn malformed_points_and_non_finite_queries_use_the_linear_fallback() {
        assert!(KdTree::build([vec![0.0], vec![1.0]].as_slice()).is_none());
        assert!(KdTree::build([vec![0.0], vec![1.0, 2.0], vec![3.0]].as_slice()).is_none());
        assert!(KdTree::build([vec![0.0], vec![f64::NAN], vec![3.0]].as_slice()).is_none());

        let points = [vec![0.0], vec![1.0], vec![2.0]];
        let tree = KdTree::build(points.as_slice()).unwrap();
        assert!(!tree.can_query(&[]));
        assert!(!tree.can_query(&[f64::INFINITY]));
        assert!(!tree.can_query(&[f64::NAN]));
    }
}
