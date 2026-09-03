// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared grid types and scalar helpers for the per-family loaders.
//!
//! Normal table queries use [`super::perf_interp`] (the v2 resolver engine,
//! mirroring `sdk/perf_interp/engine.py`). Specialized calibration paths can
//! use the small clamped-axis helper below when they must preserve a Python
//! scalar interpolation contract outside that engine.

use std::collections::BTreeMap;

/// 3-level nested grid used by loaders while assembling tables:
/// `axis0 -> axis1 -> axis2 -> value`.
pub type Grid3<T> = BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, T>>>;

/// Linear interpolation over one sorted axis with nearest-value extrapolation.
/// Two bounded range searches find an exact point or the nearest resolvable
/// bracketing points. A resolver can return `None` to omit an empty nested
/// branch; the axis returns `None` only when no branch resolves.
pub(super) fn interpolate_clamped<T>(
    points: &BTreeMap<u32, T>,
    coordinate: u32,
    resolve: impl Fn(&T) -> Option<f64>,
) -> Option<f64> {
    let left = points
        .range(..=coordinate)
        .rev()
        .find_map(|(&point, value)| resolve(value).map(|value| (point, value)));
    if let Some((point, value)) = left
        && point == coordinate
    {
        return Some(value);
    }
    let right = points
        .range(coordinate..)
        .find_map(|(&point, value)| resolve(value).map(|value| (point, value)));
    match (left, right) {
        (None, None) => None,
        (Some((_, value)), None) | (None, Some((_, value))) => Some(value),
        (Some((left_coordinate, left)), Some((right_coordinate, right))) => {
            let weight = (f64::from(coordinate) - f64::from(left_coordinate))
                / (f64::from(right_coordinate) - f64::from(left_coordinate));
            Some(left * (1.0 - weight) + right * weight)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamped_interpolation_handles_empty_exact_boundaries_and_interior() {
        let empty = BTreeMap::<u32, f64>::new();
        assert_eq!(interpolate_clamped(&empty, 10, |value| Some(*value)), None);

        let points = BTreeMap::from([(10, 1.0), (20, 3.0)]);
        for (coordinate, expected) in [(10, 1.0), (0, 1.0), (30, 3.0), (15, 2.0)] {
            assert_eq!(
                interpolate_clamped(&points, coordinate, |value| Some(*value)),
                Some(expected)
            );
        }
    }

    #[test]
    fn clamped_interpolation_skips_unresolved_branches() {
        let empty_exact = BTreeMap::from([(10, Some(1.0)), (20, None), (30, Some(5.0))]);
        assert_eq!(
            interpolate_clamped(&empty_exact, 20, |value| *value),
            Some(3.0)
        );

        let empty_bracket = BTreeMap::from([(10, None), (20, Some(3.0))]);
        assert_eq!(
            interpolate_clamped(&empty_bracket, 15, |value| *value),
            Some(3.0)
        );

        let all_empty = BTreeMap::from([(10, None), (20, None)]);
        assert_eq!(interpolate_clamped(&all_empty, 15, |value| *value), None);
    }
}
