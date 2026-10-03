// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `dsv411_module_perf.parquet`: DeepSeek-V4.1 `dsv411` family measurements.
//!
//! Explicit numeric key columns (no serialized geometry): `component, role,
//! compress_ratio, phase, tp_size, batch_size, query, kv_len` plus the
//! component's exact-match structural columns. Each (component, structure,
//! phase, tp_size, batch_size) owns a 2-axis grid over (query, kv_len) resolved
//! by the shared `perf_interp` engine (bracket/blend inside, util-hold outside).
//! Latency is in ms. A missing file is a coverage gap; a malformed present file
//! is fatal in every mode. Primary source only — no sibling/cross-backend donors.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

use super::SourceResolver;
use super::parquet_loader::{PerfReader, PerfRow};
use super::perf_interp::{LeafValue, Node, OpInterpConfig, PreparedGrid};
use crate::common::error::AicError;
use crate::operators::dsv411::{
    COMPONENT_ATTENTION_CORE, COMPONENT_ENGRAM, COMPONENT_INDEXER, COMPONENT_MHC,
    COMPONENT_SHARED_LINEAR, Coordinates, structure_key,
};

pub const BASENAME: &str = "dsv411_module_perf.parquet";
pub const REGIME_CONTEXT: &str = "eager_drained";
pub const REGIME_GENERATION: &str = "cuda_graph";
pub const REGIME_GENERATION_EXCEPTION: &str = "eager_exception";

pub struct Dsv411Table {
    path: Option<PathBuf>,
    grids: OnceLock<Result<Grids, String>>,
}

#[derive(Default)]
struct Grids {
    grids: BTreeMap<Key, PreparedGrid>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    component: String,
    structure: String,
    is_context: bool,
    tp_size: u32,
    batch_size: u32,
}

#[derive(Debug, PartialEq, Eq)]
struct Identity {
    source_sha256: String,
    config_sha256: String,
    runtime_digest: String,
}

fn invalid(message: impl Into<String>) -> AicError {
    AicError::InvalidPerfData(message.into())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Column handles for the structural (exact-match) fields; the loader rebuilds
/// each component's `structure()` string from them in the operator's field order.
struct Columns {
    component: usize,
    role: usize,
    compress_ratio: usize,
    phase: usize,
    tp_size: usize,
    batch_size: usize,
    query: usize,
    kv_len: usize,
    latency: usize,
    sample_count: usize,
    kernel_source: usize,
    measurement_scope: usize,
    measurement_regime: usize,
    kv_seed_regime: usize,
    source_sha256: usize,
    config_sha256: usize,
    runtime_digest: usize,
    used_cuda_graph: usize,
    num_heads: Option<usize>,
    head_dim: Option<usize>,
    q_lora_rank: Option<usize>,
    o_lora_rank: Option<usize>,
    o_groups: Option<usize>,
    window_size: Option<usize>,
    index_topk: Option<usize>,
    index_n_heads: Option<usize>,
    index_head_dim: Option<usize>,
    is_candidate_source: Option<usize>,
    candidate_limit: Option<usize>,
    quant_mode: Option<usize>,
    num_embeddings: Option<usize>,
    hash_columns: Option<usize>,
    hc_mult: Option<usize>,
    sinkhorn_iters: Option<usize>,
    sharding: Option<usize>,
    hidden_size: Option<usize>,
    n: Option<usize>,
    k: Option<usize>,
}

impl Columns {
    fn open(reader: &PerfReader) -> Result<Self, AicError> {
        Ok(Self {
            component: reader.col("component")?,
            role: reader.col("role")?,
            compress_ratio: reader.col("compress_ratio")?,
            phase: reader.col("phase")?,
            tp_size: reader.col("tp_size")?,
            batch_size: reader.col("batch_size")?,
            query: reader.col("query")?,
            kv_len: reader.col("kv_len")?,
            latency: reader.col("latency")?,
            sample_count: reader.col("sample_count")?,
            kernel_source: reader.col("kernel_source")?,
            measurement_scope: reader.col("measurement_scope")?,
            measurement_regime: reader.col("measurement_regime")?,
            kv_seed_regime: reader.col("kv_seed_regime")?,
            source_sha256: reader.col("source_sha256")?,
            config_sha256: reader.col("config_sha256")?,
            runtime_digest: reader.col("runtime_digest")?,
            used_cuda_graph: reader.col("used_cuda_graph")?,
            num_heads: reader.col_optional("num_heads"),
            head_dim: reader.col_optional("head_dim"),
            q_lora_rank: reader.col_optional("q_lora_rank"),
            o_lora_rank: reader.col_optional("o_lora_rank"),
            o_groups: reader.col_optional("o_groups"),
            window_size: reader.col_optional("window_size"),
            index_topk: reader.col_optional("index_topk"),
            index_n_heads: reader.col_optional("index_n_heads"),
            index_head_dim: reader.col_optional("index_head_dim"),
            is_candidate_source: reader.col_optional("is_candidate_source"),
            candidate_limit: reader.col_optional("candidate_limit"),
            quant_mode: reader.col_optional("quant_mode"),
            num_embeddings: reader.col_optional("num_embeddings"),
            hash_columns: reader.col_optional("hash_columns"),
            hc_mult: reader.col_optional("hc_mult"),
            sinkhorn_iters: reader.col_optional("sinkhorn_iters"),
            sharding: reader.col_optional("sharding"),
            hidden_size: reader.col_optional("hidden_size"),
            n: reader.col_optional("n"),
            k: reader.col_optional("k"),
        })
    }
}

fn need_u32(row: &PerfRow, col: Option<usize>, name: &str) -> Result<u32, AicError> {
    row.u32_optional(col)?
        .ok_or_else(|| invalid(format!("dsv411 row requires column {name}")))
}

fn need_str<'a>(row: &'a PerfRow, col: Option<usize>, name: &str) -> Result<&'a str, AicError> {
    row.str_optional(col)?
        .ok_or_else(|| invalid(format!("dsv411 row requires column {name}")))
}

/// Rebuild the operator's canonical structure string from the row's columns.
fn row_structure(row: &PerfRow, c: &Columns, component: &str) -> Result<String, AicError> {
    let s = |v: u32| v.to_string();
    Ok(match component {
        COMPONENT_ATTENTION_CORE => structure_key(&[
            ("role", row.str(c.role)?.to_string()),
            ("compress_ratio", s(row.u32(c.compress_ratio)?)),
            ("num_heads", s(need_u32(row, c.num_heads, "num_heads")?)),
            ("head_dim", s(need_u32(row, c.head_dim, "head_dim")?)),
            ("q_lora_rank", s(need_u32(row, c.q_lora_rank, "q_lora_rank")?)),
            ("o_lora_rank", s(need_u32(row, c.o_lora_rank, "o_lora_rank")?)),
            ("o_groups", s(need_u32(row, c.o_groups, "o_groups")?)),
            ("window_size", s(need_u32(row, c.window_size, "window_size")?)),
            ("index_topk", s(need_u32(row, c.index_topk, "index_topk")?)),
            ("quant_mode", need_str(row, c.quant_mode, "quant_mode")?.to_string()),
        ]),
        COMPONENT_INDEXER => structure_key(&[
            ("compress_ratio", s(row.u32(c.compress_ratio)?)),
            ("index_n_heads", s(need_u32(row, c.index_n_heads, "index_n_heads")?)),
            ("index_head_dim", s(need_u32(row, c.index_head_dim, "index_head_dim")?)),
            ("index_topk", s(need_u32(row, c.index_topk, "index_topk")?)),
            ("is_candidate_source", (need_u32(row, c.is_candidate_source, "is_candidate_source")? != 0).to_string()),
            ("candidate_limit", s(need_u32(row, c.candidate_limit, "candidate_limit")?)),
            ("q_lora_rank", s(need_u32(row, c.q_lora_rank, "q_lora_rank")?)),
            ("quant_mode", need_str(row, c.quant_mode, "quant_mode")?.to_string()),
        ]),
        COMPONENT_ENGRAM => structure_key(&[
            ("num_embeddings", row.u64(c.num_embeddings.ok_or_else(|| invalid("dsv411 engram rows require num_embeddings"))?)?.to_string()),
            ("head_dim", s(need_u32(row, c.head_dim, "head_dim")?)),
            ("hash_columns", s(need_u32(row, c.hash_columns, "hash_columns")?)),
            ("hc_mult", s(need_u32(row, c.hc_mult, "hc_mult")?)),
            ("sharding", need_str(row, c.sharding, "sharding")?.to_string()),
            ("quant_mode", need_str(row, c.quant_mode, "quant_mode")?.to_string()),
        ]),
        COMPONENT_MHC => structure_key(&[
            ("hidden_size", s(need_u32(row, c.hidden_size, "hidden_size")?)),
            ("hc_mult", s(need_u32(row, c.hc_mult, "hc_mult")?)),
            ("sinkhorn_iters", s(need_u32(row, c.sinkhorn_iters, "sinkhorn_iters")?)),
        ]),
        COMPONENT_SHARED_LINEAR => structure_key(&[
            ("n", s(need_u32(row, c.n, "n")?)),
            ("k", s(need_u32(row, c.k, "k")?)),
            ("quant_mode", need_str(row, c.quant_mode, "quant_mode")?.to_string()),
        ]),
        other => return Err(invalid(format!("unknown dsv411 component {other:?}"))),
    })
}

impl Dsv411Table {
    pub fn new(data_root: PathBuf) -> Self {
        Self { path: Some(data_root.join(BASENAME)), grids: OnceLock::new() }
    }

    /// Primary source only, belonging to the requested system/backend/version
    /// (same admission rule as the `dsv41` table).
    pub fn with_sources(data_root: &Path, resolver: &SourceResolver) -> Result<Self, AicError> {
        let primary = resolver
            .prioritized_sources_for(BASENAME, data_root)?
            .into_iter()
            .find(|source| source.channel == "primary");
        if primary.as_ref().is_some_and(|source| source.source.kernel_sources().is_some()) {
            return Err(invalid("dsv411 primary kernel_sources filters are not supported"));
        }
        let path = primary.map(|source| source.source.0);
        if let Some(path) = &path {
            let system_root = data_root
                .parent()
                .and_then(Path::parent)
                .ok_or_else(|| invalid("dsv411 data root must include system/backend/version"))?;
            let belongs = |source: &Path, root: &Path| {
                let Ok(relative) = source.strip_prefix(root) else { return false };
                let parts: Vec<_> = relative.components().collect();
                matches!(parts.len(), 3 | 4)
                    && parts.iter().all(|part| matches!(part, Component::Normal(_)))
                    && source.file_name().is_some_and(|name| name == BASENAME)
                    && source.parent().and_then(Path::file_name) == data_root.file_name()
                    && source.parent().and_then(Path::parent).and_then(Path::file_name)
                        == data_root.parent().and_then(Path::file_name)
            };
            if !belongs(path, system_root) {
                return Err(invalid("dsv411 primary source must belong to the requested system, backend and version"));
            }
            if path.try_exists().map_err(|e| invalid(e.to_string()))? {
                let resolved_source = path.canonicalize().map_err(|e| invalid(e.to_string()))?;
                let resolved_root = system_root.canonicalize().map_err(|e| invalid(e.to_string()))?;
                if !belongs(&resolved_source, &resolved_root) {
                    return Err(invalid("dsv411 primary source resolves outside the requested system, backend and version"));
                }
            }
        }
        Ok(Self { path, grids: OnceLock::new() })
    }

    /// Exact (component, structure, phase, tp, batch) bucket; (query, kv_len)
    /// interpolated. `sol(query, kv_len)` supplies the analytic reference.
    pub fn query(
        &self,
        component: &str,
        structure: &str,
        is_context: bool,
        tp_size: u32,
        coords: Coordinates,
        sol: &dyn Fn(f64, f64) -> Result<f64, AicError>,
    ) -> Result<Option<LeafValue>, AicError> {
        let grids = self.grids.get_or_init(|| match &self.path {
            Some(path) => load(path).map_err(|e| format!("{}: {e}", path.display())),
            None => Ok(Grids::default()),
        });
        let grids = grids.as_ref().map_err(|e| invalid(e.clone()))?;
        let key = Key {
            component: component.into(),
            structure: structure.into(),
            is_context,
            tp_size,
            batch_size: coords.batch,
        };
        let Some(grid) = grids.grids.get(&key) else { return Ok(None) };
        let failure = RefCell::new(None);
        let sol_fn = |c: &[f64]| match sol(c[0], c[1]) {
            Ok(v) => v,
            Err(err) => {
                *failure.borrow_mut() = Some(err);
                f64::NAN
            }
        };
        let cfg = if is_context {
            OpInterpConfig::grid_sqrt_axis(&["query", "kv_len"], 0, &sol_fn)
        } else {
            OpInterpConfig::grid(&["query", "kv_len"], &sol_fn)
        };
        let result = grid.query_value(&cfg, &[f64::from(coords.query), f64::from(coords.kv_len)]);
        if let Some(err) = failure.into_inner() {
            return Err(err);
        }
        let leaf = result?;
        if !leaf.latency.is_finite() || leaf.latency <= 0.0 {
            return Err(invalid("dsv411 interpolation produced an invalid latency"));
        }
        Ok(Some(leaf))
    }
}

fn load(path: &Path) -> Result<Grids, AicError> {
    if !path.try_exists().map_err(|e| invalid(e.to_string()))? {
        return Ok(Grids::default());
    }
    let reader = PerfReader::open(path)?;
    let c = Columns::open(&reader)?;
    let mut identity: Option<Identity> = None;
    let mut nodes: BTreeMap<Key, Node> = BTreeMap::new();
    let mut seen: BTreeMap<Key, Vec<(u32, u32)>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let component = row.str(c.component)?;
        let structure = row_structure(&row, &c, component)?;
        let phase = row.str(c.phase)?;
        let is_context = match phase {
            "context" => true,
            "generation" => false,
            other => return Err(invalid(format!("unknown dsv411 phase {other:?}"))),
        };
        let (tp, batch, query, kv_len) = (row.u32(c.tp_size)?, row.u32(c.batch_size)?, row.u32(c.query)?, row.u32(c.kv_len)?);
        let latency = row.f64(c.latency)?;
        if tp == 0 || batch == 0 || query == 0 || !latency.is_finite() || latency <= 0.0 || row.u32(c.sample_count)? == 0 {
            return Err(invalid("dsv411 sample requires positive tp, batch, query, latency and sample_count"));
        }
        if row.str(c.kernel_source)?.trim().is_empty() || row.str(c.measurement_scope)? != "local_compute" {
            return Err(invalid("dsv411 samples require a kernel_source and local_compute scope"));
        }
        let regime = row.str(c.measurement_regime)?;
        let graph = row.bool_strict(c.used_cuda_graph)?;
        let regime_ok = if is_context {
            regime == REGIME_CONTEXT && !graph
        } else {
            (regime == REGIME_GENERATION && graph) || (regime == REGIME_GENERATION_EXCEPTION && !graph)
        };
        if !regime_ok {
            return Err(invalid(format!(
                "dsv411 {phase} row has measurement_regime {regime:?} with used_cuda_graph={graph}"
            )));
        }
        let seed = row.str(c.kv_seed_regime)?;
        let attention_like = matches!(component, COMPONENT_ATTENTION_CORE | COMPONENT_INDEXER);
        if attention_like {
            if (!is_context || kv_len > 0) && seed != "real_kv" {
                return Err(invalid("dsv411 decode/cached-prefill rows require real_kv seeding"));
            }
            if !is_context && query != 1 {
                return Err(invalid("dsv411 generation rows measure one query token"));
            }
        } else if batch != 1 || kv_len != 0 || seed != "n/a" {
            return Err(invalid("dsv411 token-only components require batch=1, kv_len=0, kv_seed_regime=n/a"));
        }
        let current = Identity {
            source_sha256: row.str_owned(c.source_sha256)?,
            config_sha256: row.str_owned(c.config_sha256)?,
            runtime_digest: row.str_owned(c.runtime_digest)?,
        };
        if !valid_sha256(&current.source_sha256)
            || !valid_sha256(&current.config_sha256)
            || !current.runtime_digest.strip_prefix("sha256:").is_some_and(valid_sha256)
        {
            return Err(invalid("dsv411 provenance requires complete SHA256 identities"));
        }
        match &identity {
            Some(expected) if expected != &current => {
                return Err(invalid("dsv411 table mixes runtime/source/config identities"));
            }
            None => identity = Some(current),
            _ => {}
        }
        let key = Key { component: component.into(), structure, is_context, tp_size: tp, batch_size: batch };
        let coords = seen.entry(key.clone()).or_default();
        if coords.contains(&(query, kv_len)) {
            return Err(invalid("duplicate dsv411 physical key and (query, kv_len) coordinate"));
        }
        coords.push((query, kv_len));
        nodes
            .entry(key)
            .or_insert_with(Node::branch)
            .insert_value(&[query, kv_len], LeafValue::with_power(latency, 0.0));
    }
    if nodes.is_empty() {
        return Err(invalid("present dsv411 module table is empty"));
    }
    Ok(Grids { grids: nodes.into_iter().map(|(key, node)| (key, PreparedGrid::new(node))).collect() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};

    fn col_name(col: &Col) -> &str {
        match col {
            Col::Str(name, _) | Col::I64(name, _) | Col::F64(name, _) | Col::Bool(name, _) => name,
        }
    }

    const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn linear_rows(n: usize) -> Vec<Col> {
        vec![
            Col::Str("component", vec![COMPONENT_SHARED_LINEAR; n]),
            Col::Str("role", vec![""; n]),
            Col::I64("compress_ratio", vec![0; n]),
            Col::Str("phase", vec!["context"; n]),
            Col::I64("tp_size", vec![2; n]),
            Col::I64("batch_size", vec![1; n]),
            Col::I64("kv_len", vec![0; n]),
            Col::Str("kernel_source", vec!["observed.kernel"; n]),
            Col::Str("measurement_scope", vec!["local_compute"; n]),
            Col::Str("measurement_regime", vec![REGIME_CONTEXT; n]),
            Col::Str("kv_seed_regime", vec!["n/a"; n]),
            Col::Str("source_sha256", vec![SHA; n]),
            Col::Str("config_sha256", vec![SHA; n]),
            Col::Str("runtime_digest", vec![DIGEST; n]),
            Col::Bool("used_cuda_graph", vec![false; n]),
            Col::I64("sample_count", vec![5; n]),
            Col::I64("n", vec![64; n]),
            Col::I64("k", vec![32; n]),
            Col::Str("quant_mode", vec!["fp8_block"; n]),
        ]
    }

    fn fixture() -> Vec<Col> {
        let mut cols = linear_rows(2);
        cols.push(Col::I64("query", vec![10, 20]));
        cols.push(Col::F64("latency", vec![1.0, 3.0]));
        cols
    }

    fn load(columns: &[Col]) -> (tempfile::TempDir, Dsv411Table) {
        let root = tempfile::tempdir().unwrap();
        write_parquet(&root.path().join(BASENAME), columns);
        let table = Dsv411Table::new(root.path().to_owned());
        (root, table)
    }

    const STRUCTURE: &str = "n=64|k=32|quant_mode=fp8_block";

    fn lookup(table: &Dsv411Table, query: u32) -> Result<Option<LeafValue>, AicError> {
        table.query(COMPONENT_SHARED_LINEAR, STRUCTURE, true, 2, Coordinates::tokens(query), &|q, _| Ok(q * q))
    }

    #[test]
    fn exact_hits_and_interpolation_along_query() {
        let (_root, table) = load(&fixture());
        assert_eq!(lookup(&table, 10).unwrap().unwrap().latency, 1.0);
        assert_eq!(lookup(&table, 20).unwrap().unwrap().latency, 3.0);
        let mid = lookup(&table, 15).unwrap().unwrap().latency;
        assert!(mid > 1.0 && mid < 3.0, "{mid}");
    }

    #[test]
    fn unknown_structure_phase_or_batch_is_a_coverage_gap() {
        let (_root, table) = load(&fixture());
        assert!(table.query(COMPONENT_SHARED_LINEAR, "n=65|k=32|quant_mode=fp8_block", true, 2, Coordinates::tokens(10), &|q, _| Ok(q)).unwrap().is_none());
        assert!(table.query(COMPONENT_SHARED_LINEAR, STRUCTURE, false, 2, Coordinates::tokens(10), &|q, _| Ok(q)).unwrap().is_none());
        assert!(table.query(COMPONENT_SHARED_LINEAR, STRUCTURE, true, 4, Coordinates::tokens(10), &|q, _| Ok(q)).unwrap().is_none());
    }

    #[test]
    fn missing_file_is_a_gap_but_malformed_rows_are_fatal() {
        let root = tempfile::tempdir().unwrap();
        let table = Dsv411Table::new(root.path().to_owned());
        assert!(lookup(&table, 10).unwrap().is_none());
        for (name, cols) in [
            ("regime", {
                let mut c = fixture();
                c.retain(|col| col_name(col) != "measurement_regime");
                c.push(Col::Str("measurement_regime", vec![REGIME_GENERATION; 2]));
                c
            }),
            ("duplicate", {
                let mut c = linear_rows(2);
                c.push(Col::I64("query", vec![10, 10]));
                c.push(Col::F64("latency", vec![1.0, 2.0]));
                c
            }),
            ("sha", {
                let mut c = fixture();
                c.retain(|col| col_name(col) != "source_sha256");
                c.push(Col::Str("source_sha256", vec!["short"; 2]));
                c
            }),
        ] {
            let (_root, table) = load(&cols);
            assert!(lookup(&table, 10).is_err(), "{name} must be rejected");
        }
    }

    #[test]
    fn generation_rows_require_cuda_graph_unless_declared_exception() {
        let cols = || {
            let mut cols = linear_rows(2);
            cols.retain(|col| !matches!(col_name(col), "phase" | "measurement_regime" | "used_cuda_graph" | "kv_seed_regime"));
            cols.push(Col::I64("query", vec![10, 20]));
            cols.push(Col::F64("latency", vec![1.0, 3.0]));
            cols.push(Col::Str("phase", vec!["generation"; 2]));
            cols.push(Col::Str("kv_seed_regime", vec!["n/a"; 2]));
            cols
        };
        let mut eager = cols();
        eager.push(Col::Str("measurement_regime", vec![REGIME_GENERATION; 2]));
        eager.push(Col::Bool("used_cuda_graph", vec![false; 2]));
        let (_r, t) = load(&eager);
        assert!(t.query(COMPONENT_SHARED_LINEAR, STRUCTURE, false, 2, Coordinates::tokens(10), &|q, _| Ok(q)).is_err());
        let mut exception = cols();
        exception.push(Col::Str("measurement_regime", vec![REGIME_GENERATION_EXCEPTION; 2]));
        exception.push(Col::Bool("used_cuda_graph", vec![false; 2]));
        let (_r, t) = load(&exception);
        assert_eq!(t.query(COMPONENT_SHARED_LINEAR, STRUCTURE, false, 2, Coordinates::tokens(10), &|q, _| Ok(q)).unwrap().unwrap().latency, 1.0);
    }
}
