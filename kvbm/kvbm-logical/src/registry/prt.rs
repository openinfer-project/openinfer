// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Position-keyed two-level map backing the registry: an outer concurrent
//! map keyed by block position, an inner plain `HashMap` keyed by sequence
//! hash.
//!
//! This replaces `dynamo_tokens::PositionalRadixTree` for the registry's
//! use. That implementation's inner level is itself a `DashMap`, so every
//! first registration at a new block position allocates a full sharded map
//! — shard count scales with core count (4×cores rounded up to a power of
//! two; 1024 shards on a 192-core node) — which measured 21µs of a 25µs
//! per-block registration on a 48-core box, ~35× the cost of the actual
//! slot transition. The sharding buys nothing here: every inner-map access
//! goes through the outer entry's `RefMut`, which already holds the outer
//! shard's write lock exclusively. A plain `HashMap` (allocation-free
//! `Default`) drops registration to ~2µs/block.
//!
//! The outer entry guard doubles as the per-position critical section the
//! handle-drop race protection relies on — see the comment in
//! `handle.rs::Drop`.

use std::collections::HashMap;
use std::hash::Hash;

use dashmap::DashMap;
use dashmap::mapref::one::RefMut;
use dynamo_tokens::PositionalHash;

pub(crate) struct PositionalRadixTree<V, K: PositionalHash + Hash + Eq> {
    map: DashMap<u64, HashMap<K, V>>,
}

impl<V, K: PositionalHash + Hash + Eq> PositionalRadixTree<V, K> {
    pub(crate) fn new() -> Self {
        Self {
            map: DashMap::new(),
        }
    }

    /// Entry for the key's position level, created empty on first touch.
    /// The returned guard holds the outer shard's write lock, serializing
    /// all access to this position's inner map.
    pub(crate) fn prefix(&self, key: &K) -> RefMut<'_, u64, HashMap<K, V>> {
        self.map.entry(key.position()).or_default()
    }

    /// Total number of registered entries across all positions.
    pub(crate) fn len(&self) -> usize {
        self.map.iter().map(|level| level.len()).sum()
    }
}
