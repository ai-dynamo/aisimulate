// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The one result memo of forward-pass pricing.
//!
//! Every pricing result cache is a [`Memo`], entered only through
//! [`Memo::get_or_price`], so all of them keep one contract:
//!
//! - An entry is a pure function of its key. The owner keys on everything
//!   the pricing reads, and owns the memo for as long as that holds.
//! - A miss stores the worst empirical provenance tier its pricing noted on
//!   this thread, and every hit notes that tier again, so a hit reports what
//!   its miss did. Tiers a nested miss notes also count toward the enclosing
//!   one.
//! - Errors are never stored, nor values the memo's `storable` check rejects.
//! - Debug builds can re-price every hit and require the same result.
//!
//! Callers that must observe every lookup, such as FPM query coverage, price
//! without their memo.

use std::cell::Cell;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Mutex, PoisonError};

use quick_cache::sync::{Cache, DefaultLifecycle};
use quick_cache::{DefaultHashBuilder, Equivalent, OptionsBuilder, UnitWeighter};
use rustc_hash::FxHashMap;

use crate::operators::util_empirical::ProvenanceTier;
use crate::perf_database::PerfDatabase;

thread_local! {
    /// The worst tier noted on this thread since the innermost open miss
    /// began.
    static NOTED: Cell<u8> = const { Cell::new(ProvenanceTier::Silicon as u8) };
}

/// Counts `tier` toward the miss open on this thread, if any.
/// [`PerfDatabase::note_provenance`] calls it for every tier it notes.
#[inline]
pub(crate) fn note_on_thread(tier: ProvenanceTier) {
    NOTED.with(|noted| {
        if tier as u8 > noted.get() {
            noted.set(tier as u8);
        }
    });
}

/// `price`'s result and the worst tier noted on this thread while it ran.
/// Those tiers also count toward the enclosing miss, whether `price`
/// succeeds or fails.
fn attribute<T>(price: impl FnOnce() -> T) -> (T, ProvenanceTier) {
    let enclosing = NOTED.replace(ProvenanceTier::Silicon as u8);
    let value = price();
    // No unwind guard: a panic also unwinds every enclosing miss, which
    // then stores nothing.
    let noted = NOTED.get();
    NOTED.set(noted.max(enclosing));
    (value, ProvenanceTier::from_rank(noted))
}

/// Where a hit notes the tier its miss stored.
pub(crate) trait NoteProvenance {
    fn note_provenance(&self, tier: ProvenanceTier);
}

impl NoteProvenance for PerfDatabase {
    fn note_provenance(&self, tier: ProvenanceTier) {
        PerfDatabase::note_provenance(self, tier);
    }
}

/// For memos below the operators, whose pricing notes no tier: a hit counts
/// its tier only toward the miss open on this thread.
pub(crate) struct OnThread;

impl NoteProvenance for OnThread {
    fn note_provenance(&self, tier: ProvenanceTier) {
        note_on_thread(tier);
    }
}

enum Store<K, V> {
    /// One map, emptied when full: the cheapest hit, for small values looked
    /// up very often.
    Flat {
        entries: Mutex<FxHashMap<K, (V, ProvenanceTier)>>,
        capacity: usize,
    },
    /// quick_cache's sharded store, which evicts cold entries and keeps hot
    /// ones.
    Clock(Cache<K, (V, ProvenanceTier)>),
}

/// A bounded memo of pricing results under the module contract.
pub(crate) struct Memo<K, V> {
    store: Store<K, V>,
    storable: fn(&V) -> bool,
    verify_hits: bool,
}

impl<K, V> Debug for Memo<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Memo").finish_non_exhaustive()
    }
}

impl<K, V> Memo<K, V>
where
    K: Eq + Hash,
    V: Clone + Debug,
{
    fn with_store(store: Store<K, V>) -> Self {
        Self {
            store,
            storable: |_| true,
            verify_hits: false,
        }
    }

    /// A single map of at most `capacity` entries, emptied when full.
    pub(crate) fn flat(capacity: usize) -> Self {
        Self::with_store(Store::Flat {
            entries: Mutex::default(),
            capacity,
        })
    }

    /// A quick_cache store of at most `capacity` entries.
    pub(crate) fn clock(capacity: usize) -> Self {
        Self::with_store(Store::Clock(Cache::new(capacity)))
    }

    /// A quick_cache store of at most `capacity` entries in `shards` shards.
    pub(crate) fn clock_sharded(capacity: usize, shards: usize) -> Self {
        let options = OptionsBuilder::new()
            .estimated_items_capacity(capacity)
            .weight_capacity(capacity as u64)
            .shards(shards)
            .build()
            .expect("valid memo options");
        Self::with_store(Store::Clock(Cache::with_options(
            options,
            UnitWeighter,
            DefaultHashBuilder::default(),
            DefaultLifecycle::default(),
        )))
    }

    /// Stores only the values `storable` accepts; the others are returned
    /// and priced again next time.
    pub(crate) fn storing_only(mut self, storable: fn(&V) -> bool) -> Self {
        self.storable = storable;
        self
    }

    /// Debug builds re-price every hit and panic unless it prices the same.
    pub(crate) fn verifying_hits(mut self) -> Self {
        self.verify_hits = true;
        self
    }

    /// The value at `key`: the stored one, or `price`'s. A hit notes the
    /// stored tier to `notes`.
    pub(crate) fn get_or_price<E: Debug>(
        &self,
        notes: &impl NoteProvenance,
        key: K,
        price: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E> {
        let stored = match &self.store {
            Store::Flat { entries, .. } => entries
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(&key)
                .cloned(),
            Store::Clock(cache) => cache.get(&key),
        };
        match stored {
            Some(entry) => Ok(self.hit(notes, entry, price)),
            None => self.miss(key, price),
        }
    }

    /// [`Self::get_or_price`] looked up by a borrowed form of the key,
    /// which `owned` converts only to store a miss.
    pub(crate) fn get_or_price_by<Q, E: Debug>(
        &self,
        notes: &impl NoteProvenance,
        key: &Q,
        owned: impl FnOnce(&Q) -> K,
        price: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        match &self.store {
            Store::Clock(cache) => match cache.get(key) {
                Some(entry) => Ok(self.hit(notes, entry, price)),
                None => self.miss(owned(key), price),
            },
            Store::Flat { .. } => self.get_or_price(notes, owned(key), price),
        }
    }

    fn hit<E: Debug>(
        &self,
        notes: &impl NoteProvenance,
        (value, tier): (V, ProvenanceTier),
        price: impl FnOnce() -> Result<V, E>,
    ) -> V {
        #[cfg(debug_assertions)]
        if self.verify_hits {
            let repriced = price();
            assert!(
                matches!(&repriced, Ok(repriced) if format!("{repriced:?}") == format!("{value:?}")),
                "a memo hit prices as {repriced:?}, not its stored {value:?}"
            );
        }
        #[cfg(not(debug_assertions))]
        let _ = price;
        if tier != ProvenanceTier::Silicon {
            notes.note_provenance(tier);
        }
        value
    }

    fn miss<E>(&self, key: K, price: impl FnOnce() -> Result<V, E>) -> Result<V, E> {
        let (priced, tier) = attribute(price);
        let value = priced?;
        if (self.storable)(&value) {
            match &self.store {
                Store::Flat { entries, capacity } => {
                    let mut entries = entries.lock().unwrap_or_else(PoisonError::into_inner);
                    if entries.len() >= *capacity {
                        entries.clear();
                    }
                    entries.insert(key, (value.clone(), tier));
                }
                Store::Clock(cache) => cache.insert(key, (value.clone(), tier)),
            }
        }
        Ok(value)
    }
}

#[cfg(test)]
impl<K, V> Memo<K, V>
where
    K: Eq + Hash,
    V: Clone + Debug,
{
    pub(crate) fn clear(&self) {
        match &self.store {
            Store::Flat { entries, .. } => entries
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clear(),
            Store::Clock(cache) => cache.clear(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match &self.store {
            Store::Flat { entries, .. } => {
                entries.lock().unwrap_or_else(PoisonError::into_inner).len()
            }
            Store::Clock(cache) => cache.len(),
        }
    }

    /// The stored value and tier at `key`, without noting it.
    pub(crate) fn peek(&self, key: &K) -> Option<(V, ProvenanceTier)> {
        match &self.store {
            Store::Flat { entries, .. } => entries
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(key)
                .cloned(),
            Store::Clock(cache) => cache.peek(key),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell as Calls;

    /// Records what a hit notes.
    #[derive(Default)]
    struct Recorder(Calls<Option<ProvenanceTier>>);

    impl NoteProvenance for Recorder {
        fn note_provenance(&self, tier: ProvenanceTier) {
            self.0.set(Some(tier));
            note_on_thread(tier);
        }
    }

    fn noting(tier: ProvenanceTier, value: f64) -> impl FnOnce() -> Result<f64, String> {
        move || {
            note_on_thread(tier);
            Ok(value)
        }
    }

    #[test]
    fn hits_return_the_stored_value_and_note_its_tier() {
        for memo in [Memo::flat(8), Memo::clock(8)] {
            let notes = Recorder::default();
            let miss = memo.get_or_price(&notes, 1u64, noting(ProvenanceTier::XShape, 2.0));
            assert_eq!((miss, notes.0.get()), (Ok(2.0), None));
            assert_eq!(memo.peek(&1), Some((2.0, ProvenanceTier::XShape)));
            let hit = memo.get_or_price(&notes, 1, || Ok::<_, String>(2.0));
            assert_eq!(
                (hit, notes.0.get()),
                (Ok(2.0), Some(ProvenanceTier::XShape))
            );
        }
    }

    #[test]
    fn silicon_hits_note_nothing() {
        let memo = Memo::clock(8);
        let notes = Recorder::default();
        memo.get_or_price(&notes, 1u64, || Ok::<_, String>(1.0))
            .unwrap();
        memo.get_or_price(&notes, 1u64, || Ok::<_, String>(1.0))
            .unwrap();
        assert_eq!(notes.0.get(), None);
    }

    #[test]
    fn errors_and_rejected_values_are_never_stored() {
        let memo = Memo::clock(8).storing_only(|value: &f64| value.is_finite());
        let notes = OnThread;
        assert!(
            memo.get_or_price(&notes, 1u64, || Err("no".to_owned()))
                .is_err()
        );
        assert_eq!(
            memo.get_or_price(&notes, 2u64, || Ok::<_, String>(f64::NAN))
                .map(f64::is_nan),
            Ok(true)
        );
        assert_eq!(memo.len(), 0);
        assert_eq!(
            memo.get_or_price(&notes, 1u64, || Ok::<_, String>(3.0)),
            Ok(3.0)
        );
        assert_eq!(memo.len(), 1);
    }

    #[test]
    fn a_full_flat_memo_starts_over() {
        let memo = Memo::flat(2);
        for key in 0u64..3 {
            memo.get_or_price(&OnThread, key, || Ok::<_, String>(key as f64))
                .unwrap();
        }
        assert_eq!(memo.len(), 1);
        assert_eq!(memo.peek(&2), Some((2.0, ProvenanceTier::Silicon)));
    }

    #[test]
    fn nested_and_failed_misses_count_toward_the_enclosing_miss() {
        let outer = Memo::clock(8);
        let inner = Memo::clock(8);
        let failing: Memo<u64, f64> = Memo::clock(8);
        outer
            .get_or_price(&OnThread, 1u64, || {
                note_on_thread(ProvenanceTier::Empirical);
                inner.get_or_price(&OnThread, 1u64, noting(ProvenanceTier::XShape, 1.0))?;
                let _ = failing.get_or_price(&OnThread, 1u64, || {
                    note_on_thread(ProvenanceTier::XQuant);
                    Err::<f64, String>("no".to_owned())
                });
                Ok::<_, String>(2.0)
            })
            .unwrap();
        assert_eq!(inner.peek(&1).unwrap().1, ProvenanceTier::XShape);
        assert_eq!(outer.peek(&1).unwrap().1, ProvenanceTier::XQuant);
        // A hit on the inner memo inside a new outer miss counts too.
        outer
            .get_or_price(&OnThread, 2u64, || {
                inner.get_or_price(&OnThread, 1u64, || Ok::<_, String>(1.0))
            })
            .unwrap();
        assert_eq!(outer.peek(&2).unwrap().1, ProvenanceTier::XShape);
    }

    #[test]
    fn another_threads_tiers_never_reach_a_miss() {
        let memo = std::sync::Arc::new(Memo::clock(8));
        let (started, release) = (std::sync::Barrier::new(2), std::sync::Barrier::new(2));
        std::thread::scope(|scope| {
            scope.spawn(|| {
                started.wait();
                note_on_thread(ProvenanceTier::XOp);
                release.wait();
            });
            memo.get_or_price(&OnThread, 1u64, || {
                started.wait();
                release.wait();
                Ok::<_, String>(1.0)
            })
            .unwrap();
        });
        assert_eq!(memo.peek(&1).unwrap().1, ProvenanceTier::Silicon);
    }

    #[test]
    fn borrowed_lookups_convert_the_key_only_to_store_a_miss() {
        let memo: Memo<String, f64> = Memo::clock(8);
        let converted = Calls::new(0);
        let owned = |key: &str| {
            converted.set(converted.get() + 1);
            key.to_owned()
        };
        for _ in 0..2 {
            let value = memo.get_or_price_by(&OnThread, "a", owned, || Ok::<_, String>(1.0));
            assert_eq!(value, Ok(1.0));
        }
        assert_eq!(converted.get(), 1);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "a memo hit prices as")]
    fn verified_hits_reject_a_different_price() {
        let memo = Memo::flat(8).verifying_hits();
        memo.get_or_price(&OnThread, 1u64, || Ok::<_, String>(1.0))
            .unwrap();
        let _ = memo.get_or_price(&OnThread, 1u64, || Ok::<_, String>(-0.0 + 2.0));
    }
}
