// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin parquet reader shared by every `perf_database/*.rs` table loader.
//!
//! Design notes:
//! - Each loader resolves column names to integer positions ONCE per file
//!   load via `PerfReader::col`, then iterates rows with positional accessors.
//!   That avoids the per-row string-keyed lookup the `Row::get_*` API
//!   forces when callers go through `(name, field)` pairs.
//! - Rows are decoded column by column, [`BATCH_ROWS`] at a time, with
//!   parquet's typed column readers; a `PerfRow` is a shared handle on its
//!   batch plus a row index. The parquet record (`Row`) API this replaced
//!   allocated every column name and string cell of every row, which made it
//!   most of the cost of a table load. Accessors keep that API's error
//!   contract: a null cell or a column of another type is
//!   `ParquetError::General`.
//! - Accessors borrow strings out of the decoded batch (no copy); loaders
//!   that need to store a string in a BTreeMap key call
//!   `PerfRow::str_owned`.
//! - Integer columns in the AIC perf tables are stored as INT64 in parquet
//!   but always fit in u32 in practice. `PerfRow::u32` narrows and errors
//!   on overflow so a perf-DB collection bug surfaces loudly instead of
//!   wrapping silently.
//! - All errors carry the source file path so the failure message points at
//!   the exact perf-DB file (mirrors the path-context error contract used
//!   by the perf_database layer).

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use parquet::basic::{ConvertedType, Type as PhysicalType};
use parquet::column::reader::{ColumnReader, ColumnReaderImpl};
use parquet::data_type::{BoolType, ByteArrayType, DataType, DoubleType, Int64Type};
use parquet::errors::ParquetError;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::schema::types::ColumnDescriptor;

use crate::common::error::AicError;

const LFS_POINTER_PREFIX: &[u8] = b"version https://git-lfs";

/// Rows decoded per column at a time. Small, so a caller that stops at the
/// first matching row (the readiness probes) decodes little of the file.
const BATCH_ROWS: usize = 1024;

/// Opens a parquet file and pre-caches column names so each loader resolves
/// `col("name") -> position` once before the row loop.
pub struct PerfReader {
    file_reader: SerializedFileReader<File>,
    path: PathBuf,
    column_names: Vec<String>,
}

impl PerfReader {
    pub fn open(path: &Path) -> Result<Self, AicError> {
        let mut file = File::open(path).map_err(|source| AicError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        // Surface unresolved git-lfs pointers with a clear message before
        // the parquet reader emits its less-specific "Invalid Parquet"
        // error. Parquet starts with the magic bytes "PAR1"; an unresolved
        // LFS pointer starts with "version https://git-lfs".
        let mut head = [0u8; LFS_POINTER_PREFIX.len()];
        let read = file.read(&mut head).map_err(|source| AicError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if read >= LFS_POINTER_PREFIX.len() && &head == LFS_POINTER_PREFIX {
            return Err(AicError::PerfDatabase(format!(
                "perf file is an unresolved git-lfs pointer: {}; run `git lfs pull`",
                path.display()
            )));
        }
        // Reopen since we consumed the head bytes (the parquet reader needs
        // to start from offset 0 to read the footer offset).
        let file = File::open(path).map_err(|source| AicError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let file_reader = SerializedFileReader::new(file).map_err(|source| AicError::Parquet {
            path: path.to_path_buf(),
            source,
        })?;
        let column_names = file_reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .map(|c| c.name().to_string())
            .collect();
        Ok(Self {
            file_reader,
            path: path.to_path_buf(),
            column_names,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the position of `name` in the parquet schema. Errors when the
    /// column is absent, with the surviving column names in the message so
    /// schema drift is visible without opening the file.
    pub fn col(&self, name: &str) -> Result<usize, AicError> {
        self.column_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| {
                AicError::PerfDatabase(format!(
                    "parquet column {name:?} missing at {} (have: [{}])",
                    self.path.display(),
                    self.column_names.join(", ")
                ))
            })
    }

    /// Same as `col` but returns `None` instead of erroring — for columns
    /// that are present in some perf-DB versions and absent in others.
    pub fn col_optional(&self, name: &str) -> Option<usize> {
        self.column_names.iter().position(|n| n == name)
    }

    /// Iterate over every row, in file order. A decode error (corrupt page,
    /// invalid UTF-8 in a string column) is yielded as an `Err` item and ends
    /// the iteration.
    pub fn rows(&self) -> Result<impl Iterator<Item = Result<PerfRow, AicError>> + '_, AicError> {
        Ok(PerfRows {
            reader: self,
            next_row_group: 0,
            remaining: 0,
            cursors: Vec::new(),
            batch: None,
            next_row: 0,
            done: false,
        })
    }
}

/// Row iterator behind [`PerfReader::rows`]: decodes the current row group
/// one batch at a time and yields a handle per row.
struct PerfRows<'a> {
    reader: &'a PerfReader,
    next_row_group: usize,
    /// Rows of the current row group not decoded yet.
    remaining: usize,
    /// One reader per column of the current row group.
    cursors: Vec<ColumnCursor>,
    batch: Option<Rc<RowBatch>>,
    next_row: usize,
    done: bool,
}

impl Iterator for PerfRows<'_> {
    type Item = Result<PerfRow, AicError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(batch) = &self.batch
                && self.next_row < batch.len
            {
                let row = PerfRow {
                    batch: Rc::clone(batch),
                    row: self.next_row,
                };
                self.next_row += 1;
                return Some(Ok(row));
            }
            if self.done {
                return None;
            }
            match self.read_batch() {
                Ok(true) => {}
                Ok(false) => self.done = true,
                Err(source) => {
                    self.done = true;
                    return Some(Err(AicError::Parquet {
                        path: self.reader.path.clone(),
                        source,
                    }));
                }
            }
        }
    }
}

impl PerfRows<'_> {
    /// Decodes the next batch, opening the next row group when the current
    /// one is used up. `Ok(false)` once every row group is consumed.
    fn read_batch(&mut self) -> Result<bool, ParquetError> {
        let file = &self.reader.file_reader;
        while self.remaining == 0 {
            if self.next_row_group == file.num_row_groups() {
                return Ok(false);
            }
            let row_group = file.get_row_group(self.next_row_group)?;
            let num_rows = row_group.metadata().num_rows();
            self.remaining = usize::try_from(num_rows).map_err(|_| {
                ParquetError::General(format!("row group has a negative row count {num_rows}"))
            })?;
            self.cursors = (0..row_group.num_columns())
                .map(|i| {
                    let descr = row_group.metadata().column(i).column_descr();
                    Ok(ColumnCursor::new(row_group.get_column_reader(i)?, descr))
                })
                .collect::<Result<_, ParquetError>>()?;
            self.next_row_group += 1;
        }
        let len = self.remaining.min(BATCH_ROWS);
        let columns = self
            .cursors
            .iter_mut()
            .map(|cursor| cursor.read(len))
            .collect::<Result<_, _>>()?;
        self.remaining -= len;
        self.batch = Some(Rc::new(RowBatch {
            path: self.reader.path.clone(),
            len,
            columns,
        }));
        self.next_row = 0;
        Ok(true)
    }
}

/// Typed reader for one column of the current row group.
struct ColumnCursor {
    kind: CursorKind,
    /// Definition level of a present cell; lower levels are nulls.
    max_def: i16,
    def_levels: Vec<i16>,
}

enum CursorKind {
    Bool(ColumnReaderImpl<BoolType>),
    Long(ColumnReaderImpl<Int64Type>),
    Double(ColumnReaderImpl<DoubleType>),
    Str(ColumnReaderImpl<ByteArrayType>),
    /// A type no perf table uses (repeated, INT32, FLOAT, binary, ...). It
    /// is never decoded, and every typed access to it fails as a type
    /// mismatch, as it did through the record API.
    Other(PhysicalType),
}

impl ColumnCursor {
    /// Picks the decoding the record API's `Field` conversion would give the
    /// column: INT64 is `Long` only without a converted type (or `INT_64`),
    /// BYTE_ARRAY is `Str` only for UTF8, ENUM and JSON.
    fn new(reader: ColumnReader, descr: &ColumnDescriptor) -> Self {
        let kind = match (reader, descr.converted_type()) {
            _ if descr.max_rep_level() > 0 => CursorKind::Other(descr.physical_type()),
            (ColumnReader::BoolColumnReader(r), _) => CursorKind::Bool(r),
            (ColumnReader::Int64ColumnReader(r), ConvertedType::NONE | ConvertedType::INT_64) => {
                CursorKind::Long(r)
            }
            (ColumnReader::DoubleColumnReader(r), _) => CursorKind::Double(r),
            (
                ColumnReader::ByteArrayColumnReader(r),
                ConvertedType::UTF8 | ConvertedType::ENUM | ConvertedType::JSON,
            ) => CursorKind::Str(r),
            _ => CursorKind::Other(descr.physical_type()),
        };
        Self {
            kind,
            max_def: descr.max_def_level(),
            def_levels: Vec::new(),
        }
    }

    fn read(&mut self, len: usize) -> Result<ColumnBatch, ParquetError> {
        let (max_def, def_levels) = (self.max_def, &mut self.def_levels);
        Ok(match &mut self.kind {
            CursorKind::Bool(r) => {
                let (values, nulls) = read_typed(r, len, max_def, def_levels)?;
                ColumnBatch {
                    values: Values::Bool(spread(values, &nulls)),
                    nulls,
                }
            }
            CursorKind::Long(r) => {
                let (values, nulls) = read_typed(r, len, max_def, def_levels)?;
                ColumnBatch {
                    values: Values::Long(spread(values, &nulls)),
                    nulls,
                }
            }
            CursorKind::Double(r) => {
                let (values, nulls) = read_typed(r, len, max_def, def_levels)?;
                ColumnBatch {
                    values: Values::Double(spread(values, &nulls)),
                    nulls,
                }
            }
            CursorKind::Str(r) => {
                let (values, nulls) = read_typed(r, len, max_def, def_levels)?;
                let mut text = String::new();
                let mut ends = Vec::with_capacity(len);
                let mut present = values.iter();
                for row in 0..len {
                    if !nulls.get(row).copied().unwrap_or(false)
                        && let Some(value) = present.next()
                    {
                        let s = std::str::from_utf8(value.data()).map_err(|e| {
                            ParquetError::General(format!(
                                "Error reading BYTE_ARRAY as String. Bytes: {:?} Error: {e:?}",
                                value.data()
                            ))
                        })?;
                        text.push_str(s);
                    }
                    ends.push(text.len());
                }
                ColumnBatch {
                    values: Values::Str { text, ends },
                    nulls,
                }
            }
            CursorKind::Other(physical) => ColumnBatch {
                values: Values::Other(*physical),
                nulls: Vec::new(),
            },
        })
    }
}

/// Reads `len` rows of one column: the present values, packed, and a null
/// flag per row (empty when the batch has no nulls).
fn read_typed<T: DataType>(
    reader: &mut ColumnReaderImpl<T>,
    len: usize,
    max_def: i16,
    def_levels: &mut Vec<i16>,
) -> Result<(Vec<T::T>, Vec<bool>), ParquetError> {
    let mut values = Vec::with_capacity(len);
    def_levels.clear();
    let (rows, present, _) = reader.read_records(len, Some(def_levels), None, &mut values)?;
    if rows != len {
        return Err(ParquetError::General(format!(
            "column ended after {rows} of {len} rows"
        )));
    }
    let nulls = if present == len {
        Vec::new()
    } else {
        def_levels.iter().map(|&level| level < max_def).collect()
    };
    Ok((values, nulls))
}

/// Places packed present values at their rows; null rows hold a default.
fn spread<V: Default>(packed: Vec<V>, nulls: &[bool]) -> Vec<V> {
    if nulls.is_empty() {
        return packed;
    }
    let mut packed = packed.into_iter();
    nulls
        .iter()
        .map(|&null| {
            if null {
                V::default()
            } else {
                packed.next().unwrap_or_default()
            }
        })
        .collect()
}

/// One decoded batch: row-aligned values for every column.
struct RowBatch {
    path: PathBuf,
    len: usize,
    columns: Vec<ColumnBatch>,
}

struct ColumnBatch {
    values: Values,
    /// Null flag per row; empty when the batch has no nulls.
    nulls: Vec<bool>,
}

enum Values {
    Bool(Vec<bool>),
    Long(Vec<i64>),
    Double(Vec<f64>),
    /// Row `i` is `text[ends[i - 1]..ends[i]]` (from 0 for the first row).
    Str {
        text: String,
        ends: Vec<usize>,
    },
    Other(PhysicalType),
}

impl Values {
    /// The record API's `Field` type name, for type-mismatch errors.
    fn type_name(&self) -> String {
        match self {
            Values::Bool(_) => "Bool".to_string(),
            Values::Long(_) => "Long".to_string(),
            Values::Double(_) => "Double".to_string(),
            Values::Str { .. } => "Str".to_string(),
            Values::Other(physical) => physical.to_string(),
        }
    }
}

/// One row of a decoded batch with positional, typed accessors. Cloning the
/// batch handle is the only work per row.
pub struct PerfRow {
    batch: Rc<RowBatch>,
    row: usize,
}

impl PerfRow {
    fn path(&self) -> PathBuf {
        self.batch.path.clone()
    }

    /// Typed cell access with the record API's error contract: a null cell
    /// or a column of another type is `General`, a column past the last
    /// `IndexOutOfBound`.
    fn cell<'a, T>(
        &'a self,
        col: usize,
        want: &str,
        get: impl FnOnce(&'a Values, usize) -> Option<T>,
    ) -> Result<T, ParquetError> {
        let columns = &self.batch.columns;
        let column = columns
            .get(col)
            .ok_or(ParquetError::IndexOutOfBound(col, columns.len()))?;
        if column.nulls.get(self.row).copied().unwrap_or(false) {
            return Err(ParquetError::General(format!(
                "Cannot access Null as {want}"
            )));
        }
        get(&column.values, self.row).ok_or_else(|| {
            ParquetError::General(format!(
                "Cannot access {} as {want}",
                column.values.type_name()
            ))
        })
    }

    fn get_bool(&self, col: usize) -> Result<bool, ParquetError> {
        self.cell(col, "Bool", |values, row| match values {
            Values::Bool(v) => Some(v[row]),
            _ => None,
        })
    }

    fn get_long(&self, col: usize) -> Result<i64, ParquetError> {
        self.cell(col, "Long", |values, row| match values {
            Values::Long(v) => Some(v[row]),
            _ => None,
        })
    }

    fn get_double(&self, col: usize) -> Result<f64, ParquetError> {
        self.cell(col, "Double", |values, row| match values {
            Values::Double(v) => Some(v[row]),
            _ => None,
        })
    }

    fn get_string(&self, col: usize) -> Result<&str, ParquetError> {
        self.cell(col, "Str", |values, row| match values {
            Values::Str { text, ends } => {
                let start = if row == 0 { 0 } else { ends[row - 1] };
                Some(&text[start..ends[row]])
            }
            _ => None,
        })
    }

    /// Borrowed string view into the row. Use this when the value is parsed
    /// or compared but not stored.
    pub fn str(&self, col: usize) -> Result<&str, AicError> {
        self.get_string(col).map_err(|source| AicError::Parquet {
            path: self.path(),
            source,
        })
    }

    /// Cloned String for BTreeMap keys.
    pub fn str_owned(&self, col: usize) -> Result<String, AicError> {
        self.str(col).map(|s| s.to_string())
    }

    /// Narrow INT64 to u32; the AIC perf tables always fit. Errors if not.
    pub fn u32(&self, col: usize) -> Result<u32, AicError> {
        let v = self.get_long(col).map_err(|source| AicError::Parquet {
            path: self.path(),
            source,
        })?;
        u32::try_from(v).map_err(|_| {
            AicError::PerfDatabase(format!(
                "parquet column[{col}] = {v} does not fit in u32 at {}",
                self.batch.path.display()
            ))
        })
    }

    /// Optional INT64 → u32. Returns None when the cell is missing/null.
    pub fn u32_optional(&self, col: Option<usize>) -> Result<Option<u32>, AicError> {
        let Some(idx) = col else {
            return Ok(None);
        };
        // `get_long` errors when the field is null; treat NullFound as absence.
        match self.get_long(idx) {
            Ok(v) => u32::try_from(v).map(Some).map_err(|_| {
                AicError::PerfDatabase(format!(
                    "parquet column[{idx}] = {v} does not fit in u32 at {}",
                    self.batch.path.display()
                ))
            }),
            Err(parquet::errors::ParquetError::General(_)) => Ok(None),
            Err(source) => Err(AicError::Parquet {
                path: self.path(),
                source,
            }),
        }
    }

    /// Optional INT64 kept SIGNED. Returns None when the cell is missing/null
    /// (or stored under a non-integer physical type). The table-view folds use
    /// this to mirror Python's `int(row[...])`, which never narrowed: a
    /// negative key cell loaded as a negative key instead of erroring.
    pub fn i64_optional(&self, col: Option<usize>) -> Result<Option<i64>, AicError> {
        let Some(idx) = col else {
            return Ok(None);
        };
        match self.get_long(idx) {
            Ok(v) => Ok(Some(v)),
            Err(parquet::errors::ParquetError::General(_)) => Ok(None),
            Err(source) => Err(AicError::Parquet {
                path: self.path(),
                source,
            }),
        }
    }

    /// INT64 → u64. Used by columns that legitimately need the full range
    /// (e.g. NCCL `message_size`).
    pub fn u64(&self, col: usize) -> Result<u64, AicError> {
        let v = self.get_long(col).map_err(|source| AicError::Parquet {
            path: self.path(),
            source,
        })?;
        u64::try_from(v).map_err(|_| {
            AicError::PerfDatabase(format!(
                "parquet column[{col}] = {v} is negative; expected u64 at {}",
                self.batch.path.display()
            ))
        })
    }

    pub fn f64(&self, col: usize) -> Result<f64, AicError> {
        self.get_double(col).map_err(|source| AicError::Parquet {
            path: self.path(),
            source,
        })
    }

    /// BOOLEAN column, with fallbacks mirroring Python's `_to_bool` string
    /// coercion (`str(value).strip().lower() in {"1","true","yes","y"}`) for
    /// files that store flags as INT64 or strings.
    pub fn bool(&self, col: usize) -> Result<bool, AicError> {
        if let Ok(v) = self.get_bool(col) {
            return Ok(v);
        }
        if let Ok(v) = self.get_long(col) {
            return Ok(v == 1);
        }
        let s = self.str(col)?;
        Ok(matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "y"
        ))
    }

    /// BOOLEAN column with NO type coercion: identity flags (FPM schema v6
    /// `enable_*`) must be real booleans — Python's loader rejects anything
    /// else, so coercing a stringly-typed flag here would silently diverge
    /// from it. A type mismatch is a loud data bug.
    pub fn bool_strict(&self, col: usize) -> Result<bool, AicError> {
        self.get_bool(col).map_err(|source| AicError::Parquet {
            path: self.path(),
            source,
        })
    }

    /// Optional double. Returns None when the column lookup is None OR the
    /// cell is null. Used by dispatch tables whose split-latency columns
    /// can legitimately be absent for one mode of the schema (DeepEP normal
    /// vs LL).
    pub fn f64_optional(&self, col: Option<usize>) -> Result<Option<f64>, AicError> {
        let Some(idx) = col else {
            return Ok(None);
        };
        match self.get_double(idx) {
            Ok(v) => Ok(Some(v)),
            Err(parquet::errors::ParquetError::General(_)) => Ok(None),
            Err(source) => Err(AicError::Parquet {
                path: self.path(),
                source,
            }),
        }
    }

    /// Borrowed string accessor for columns that may be absent or null.
    /// Returns None if the column lookup is None OR the cell is null.
    pub fn str_optional(&self, col: Option<usize>) -> Result<Option<&str>, AicError> {
        let Some(idx) = col else {
            return Ok(None);
        };
        match self.get_string(idx) {
            Ok(s) => Ok(Some(s)),
            Err(parquet::errors::ParquetError::General(_)) => Ok(None),
            Err(source) => Err(AicError::Parquet {
                path: self.path(),
                source,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parquet::data_type::{ByteArray, Int32Type};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::record::RowAccessor;
    use parquet::schema::parser::parse_message_type;

    use super::*;

    const ROWS_PER_GROUP: usize = BATCH_ROWS + 476;

    /// Writes two row groups of OPTIONAL columns (nulls on different rows
    /// per column) plus one REQUIRED string and one INT32 column, a type the
    /// columnar reader does not decode.
    fn write_fixture(path: &Path) {
        let schema = "message fixture { OPTIONAL BYTE_ARRAY s (UTF8); REQUIRED BYTE_ARRAY r (UTF8); \
             OPTIONAL INT64 i; OPTIONAL DOUBLE d; OPTIONAL BOOLEAN b; OPTIONAL INT32 x; }";
        let schema = Arc::new(parse_message_type(schema).unwrap());
        let props = Arc::new(WriterProperties::builder().build());
        let mut writer =
            SerializedFileWriter::new(File::create(path).unwrap(), schema, props).unwrap();
        for group in 0..2 {
            let rows: Vec<usize> = (group * ROWS_PER_GROUP..(group + 1) * ROWS_PER_GROUP).collect();
            let defs = |every: usize| -> Vec<i16> {
                rows.iter().map(|r| i16::from(r % every != 0)).collect()
            };
            let present = |every: usize| rows.iter().copied().filter(move |r| r % every != 0);
            let mut rg = writer.next_row_group().unwrap();
            let strings: Vec<ByteArray> = present(7)
                .map(|r| ByteArray::from(format!("µs-{}", r % 5).as_str()))
                .collect();
            let required: Vec<ByteArray> = rows
                .iter()
                .map(|r| ByteArray::from(format!("k{r}").as_str()))
                .collect();
            let longs: Vec<i64> = present(5).map(|r| r as i64 - 1000).collect();
            let doubles: Vec<f64> = present(3).map(|r| r as f64 * 0.1).collect();
            let bools: Vec<bool> = present(11).map(|r| r % 2 == 0).collect();
            let ints: Vec<i32> = present(13).map(|r| r as i32).collect();
            let mut col = rg.next_column().unwrap().unwrap();
            col.typed::<ByteArrayType>()
                .write_batch(&strings, Some(&defs(7)), None)
                .unwrap();
            col.close().unwrap();
            let mut col = rg.next_column().unwrap().unwrap();
            col.typed::<ByteArrayType>()
                .write_batch(&required, None, None)
                .unwrap();
            col.close().unwrap();
            let mut col = rg.next_column().unwrap().unwrap();
            col.typed::<Int64Type>()
                .write_batch(&longs, Some(&defs(5)), None)
                .unwrap();
            col.close().unwrap();
            let mut col = rg.next_column().unwrap().unwrap();
            col.typed::<DoubleType>()
                .write_batch(&doubles, Some(&defs(3)), None)
                .unwrap();
            col.close().unwrap();
            let mut col = rg.next_column().unwrap().unwrap();
            col.typed::<BoolType>()
                .write_batch(&bools, Some(&defs(11)), None)
                .unwrap();
            col.close().unwrap();
            let mut col = rg.next_column().unwrap().unwrap();
            col.typed::<Int32Type>()
                .write_batch(&ints, Some(&defs(13)), None)
                .unwrap();
            col.close().unwrap();
            rg.close().unwrap();
        }
        writer.close().unwrap();
    }

    /// `Ok` values compare exactly; errors compare by kind (`General` for a
    /// null or a type mismatch, `IndexOutOfBound` past the last column).
    fn outcome<T: std::fmt::Debug>(result: Result<T, ParquetError>) -> String {
        match result {
            Ok(v) => format!("{v:?}"),
            Err(ParquetError::General(_)) => "General".to_string(),
            Err(other) => format!("{other:?}"),
        }
    }

    #[test]
    fn columnar_rows_match_the_record_api_cell_for_cell() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("fixture.parquet");
        write_fixture(&path);
        let reader = PerfReader::open(&path).unwrap();
        let oracle: Vec<parquet::record::Row> = reader
            .file_reader
            .get_row_iter(None)
            .unwrap()
            .map(Result::unwrap)
            .collect();
        let rows: Vec<PerfRow> = reader.rows().unwrap().map(Result::unwrap).collect();
        assert_eq!(rows.len(), 2 * ROWS_PER_GROUP);
        assert_eq!(rows.len(), oracle.len());
        for (index, (row, expected)) in rows.iter().zip(&oracle).enumerate() {
            for col in 0..=reader.column_names.len() {
                let at = format!("row {index} column {col}");
                assert_eq!(
                    outcome(row.get_string(col)),
                    outcome(expected.get_string(col).map(String::as_str)),
                    "{at}"
                );
                assert_eq!(
                    outcome(row.get_long(col)),
                    outcome(expected.get_long(col)),
                    "{at}"
                );
                assert_eq!(
                    outcome(row.get_double(col).map(f64::to_bits)),
                    outcome(expected.get_double(col).map(f64::to_bits)),
                    "{at}"
                );
                assert_eq!(
                    outcome(row.get_bool(col)),
                    outcome(expected.get_bool(col)),
                    "{at}"
                );
            }
        }
        // A null cell keeps the record API's message, which loaders surface.
        let null = rows[0].get_long(2).unwrap_err().to_string();
        assert_eq!(null, oracle[0].get_long(2).unwrap_err().to_string());
    }
}
