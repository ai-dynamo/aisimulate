// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serde adapters that let in-memory estimator state derive its own encoding
//! without changing the runtime layout.
//!
//! - [`arr`] and [`arr2`] encode const-generic arrays as fixed-length tuples;
//!   serde only implements arrays with literal lengths.
//! - [`sorted_map`] encodes an ordered map as a sequence of pairs and rejects
//!   unsorted or duplicate keys on decode, so each state has one encoding and
//!   a corrupted map cannot silently collapse entries.

use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;

use serde::de::{Error as _, SeqAccess, Visitor};
use serde::ser::SerializeTuple;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub(crate) mod arr {
    use super::*;

    pub(crate) fn serialize<S: Serializer, T: Serialize, const N: usize>(
        value: &[T; N],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(N)?;
        for item in value {
            tuple.serialize_element(item)?;
        }
        tuple.end()
    }

    struct ArrayVisitor<T, const N: usize>(PhantomData<T>);

    impl<'de, T: Deserialize<'de>, const N: usize> Visitor<'de> for ArrayVisitor<T, N> {
        type Value = [T; N];

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "an array of length {N}")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut items = Vec::with_capacity(N);
            for index in 0..N {
                items.push(
                    seq.next_element()?
                        .ok_or_else(|| A::Error::invalid_length(index, &self))?,
                );
            }
            items
                .try_into()
                .map_err(|_| A::Error::custom("array length mismatch"))
        }
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>, T: Deserialize<'de>, const N: usize>(
        deserializer: D,
    ) -> Result<[T; N], D::Error> {
        deserializer.deserialize_tuple(N, ArrayVisitor::<T, N>(PhantomData))
    }
}

pub(crate) mod arr2 {
    use super::*;

    struct Row<'a, T, const M: usize>(&'a [T; M]);

    impl<T: Serialize, const M: usize> Serialize for Row<'_, T, M> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            arr::serialize(self.0, serializer)
        }
    }

    struct OwnedRow<T, const M: usize>([T; M]);

    impl<'de, T: Deserialize<'de>, const M: usize> Deserialize<'de> for OwnedRow<T, M> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            arr::deserialize(deserializer).map(OwnedRow)
        }
    }

    pub(crate) fn serialize<S: Serializer, T: Serialize, const M: usize, const N: usize>(
        value: &[[T; M]; N],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(N)?;
        for row in value {
            tuple.serialize_element(&Row(row))?;
        }
        tuple.end()
    }

    struct MatrixVisitor<T, const M: usize, const N: usize>(PhantomData<T>);

    impl<'de, T: Deserialize<'de>, const M: usize, const N: usize> Visitor<'de>
        for MatrixVisitor<T, M, N>
    {
        type Value = [[T; M]; N];

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "a {N}x{M} matrix")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut rows = Vec::with_capacity(N);
            for index in 0..N {
                let row: OwnedRow<T, M> = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::invalid_length(index, &self))?;
                rows.push(row.0);
            }
            rows.try_into()
                .map_err(|_| A::Error::custom("matrix length mismatch"))
        }
    }

    pub(crate) fn deserialize<
        'de,
        D: Deserializer<'de>,
        T: Deserialize<'de>,
        const M: usize,
        const N: usize,
    >(
        deserializer: D,
    ) -> Result<[[T; M]; N], D::Error> {
        deserializer.deserialize_tuple(N, MatrixVisitor::<T, M, N>(PhantomData))
    }
}

pub(crate) mod sorted_map {
    use super::*;

    pub(crate) fn serialize<S: Serializer, K: Serialize, V: Serialize>(
        value: &BTreeMap<K, V>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(value.iter())
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>, K, V>(
        deserializer: D,
    ) -> Result<BTreeMap<K, V>, D::Error>
    where
        K: Deserialize<'de> + Ord,
        V: Deserialize<'de>,
    {
        let pairs = Vec::<(K, V)>::deserialize(deserializer)?;
        let mut map = BTreeMap::new();
        for (key, value) in pairs {
            if map.last_key_value().is_some_and(|(last, _)| *last >= key) {
                return Err(D::Error::custom(
                    "map keys must be strictly increasing (unsorted or duplicate key)",
                ));
            }
            map.insert(key, value);
        }
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Generic<const D: usize> {
        #[serde(with = "arr")]
        row: [f64; D],
        #[serde(with = "arr2")]
        matrix: [[f64; D]; D],
        #[serde(with = "sorted_map")]
        map: BTreeMap<Vec<usize>, f64>,
    }

    fn options() -> impl bincode::Options {
        use bincode::Options;
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .reject_trailing_bytes()
    }

    #[test]
    fn const_generic_arrays_and_maps_round_trip_bit_exactly() {
        use bincode::Options;
        let value = Generic::<3> {
            row: [f64::INFINITY, -0.0, f64::from_bits(1)],
            matrix: [
                [1.0, 2.0, 3.0],
                [f64::NEG_INFINITY, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            map: BTreeMap::from([(vec![0, 1], 0.5), (vec![1, 0], 2.0)]),
        };
        let bytes = options().serialize(&value).unwrap();
        let decoded: Generic<3> = options().deserialize(&bytes).unwrap();
        assert_eq!(decoded.row.map(f64::to_bits), value.row.map(f64::to_bits));
        assert_eq!(
            decoded,
            Generic {
                row: decoded.row,
                ..value
            }
        );
    }

    #[test]
    fn unsorted_or_duplicate_map_keys_are_rejected() {
        use bincode::Options;
        #[derive(Serialize)]
        struct Raw {
            row: [f64; 1],
            matrix: [[f64; 1]; 1],
            map: Vec<(Vec<usize>, f64)>,
        }
        for map in [
            vec![(vec![1], 1.0), (vec![0], 2.0)],
            vec![(vec![0], 1.0), (vec![0], 2.0)],
        ] {
            let bytes = options()
                .serialize(&Raw {
                    row: [0.0],
                    matrix: [[0.0]],
                    map,
                })
                .unwrap();
            let error = options().deserialize::<Generic<1>>(&bytes).unwrap_err();
            assert!(error.to_string().contains("strictly increasing"), "{error}");
        }
    }
}
