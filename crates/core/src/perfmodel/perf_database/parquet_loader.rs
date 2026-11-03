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

    /// Row count from the file footer; decodes no data. A negative count in
    /// a corrupt footer reads as 0.
    pub fn num_rows(&self) -> u64 {
        let rows = self.file_reader.metadata().file_metadata().num_rows();
        u64::try_from(rows).unwrap_or(0)
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
            CursorKind::Bool(r) => read_spread(r, len, max_def, def_levels, Values::Bool)?,
            CursorKind::Long(r) => read_spread(r, len, max_def, def_levels, Values::Long)?,
            CursorKind::Double(r) => read_spread(r, len, max_def, def_levels, Values::Double)?,
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

/// Reads `len` rows of a fixed-width column as row-aligned values.
fn read_spread<T: DataType>(
    reader: &mut ColumnReaderImpl<T>,
    len: usize,
    max_def: i16,
    def_levels: &mut Vec<i16>,
    wrap: fn(Vec<T::T>) -> Values,
) -> Result<ColumnBatch, ParquetError> {
    let (values, nulls) = read_typed(reader, len, max_def, def_levels)?;
    Ok(ColumnBatch {
        values: wrap(spread(values, &nulls)),
        nulls,
    })
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

/// One cell of a decoded row, as the record API's `Field` would hold it.
#[derive(Debug, PartialEq)]
enum Cell<'a> {
    Null,
    Bool(bool),
    Long(i64),
    Double(f64),
    Str(&'a str),
    /// Any cell, null or not, of a column the reader does not decode. Type
    /// errors name its physical type (`INT32`) where the record API named
    /// the `Field` (`Int`, or `Null` for a null cell).
    Other(PhysicalType),
}

/// One row of a decoded batch with positional, typed accessors. Cloning the
/// batch handle is the only work per row.
pub struct PerfRow {
    batch: Rc<RowBatch>,
    row: usize,
}

impl PerfRow {
    /// The cell at `col`; a column past the last is `IndexOutOfBound`.
    fn cell(&self, col: usize) -> Result<Cell<'_>, AicError> {
        let (columns, row) = (&self.batch.columns, self.row);
        let column = columns
            .get(col)
            .ok_or_else(|| self.error(ParquetError::IndexOutOfBound(col, columns.len())))?;
        if column.nulls.get(row).copied().unwrap_or(false) {
            return Ok(Cell::Null);
        }
        Ok(match &column.values {
            Values::Bool(v) => Cell::Bool(v[row]),
            Values::Long(v) => Cell::Long(v[row]),
            Values::Double(v) => Cell::Double(v[row]),
            Values::Str { text, ends } => {
                let start = if row == 0 { 0 } else { ends[row - 1] };
                Cell::Str(&text[start..ends[row]])
            }
            Values::Other(physical) => Cell::Other(*physical),
        })
    }

    /// `cell` for a column that may be absent; `None` when it is.
    fn cell_optional(&self, col: Option<usize>) -> Result<Option<Cell<'_>>, AicError> {
        col.map(|col| self.cell(col)).transpose()
    }

    /// The record API's error for reading `cell` as `want`: `General`, also
    /// for a null cell.
    fn mismatch(&self, cell: Cell<'_>, want: &str) -> AicError {
        let have = match cell {
            Cell::Null => "Null",
            Cell::Bool(_) => "Bool",
            Cell::Long(_) => "Long",
            Cell::Double(_) => "Double",
            Cell::Str(_) => "Str",
            Cell::Other(physical) => &physical.to_string(),
        };
        self.error(ParquetError::General(format!(
            "Cannot access {have} as {want}"
        )))
    }

    fn error(&self, source: ParquetError) -> AicError {
        AicError::Parquet {
            path: self.batch.path.clone(),
            source,
        }
    }

    fn long(&self, col: usize) -> Result<i64, AicError> {
        match self.cell(col)? {
            Cell::Long(v) => Ok(v),
            cell => Err(self.mismatch(cell, "Long")),
        }
    }

    fn fit_u32(&self, col: usize, v: i64) -> Result<u32, AicError> {
        u32::try_from(v).map_err(|_| {
            AicError::PerfDatabase(format!(
                "parquet column[{col}] = {v} does not fit in u32 at {}",
                self.batch.path.display()
            ))
        })
    }

    /// Borrowed string view into the row. Use this when the value is parsed
    /// or compared but not stored.
    pub fn str(&self, col: usize) -> Result<&str, AicError> {
        match self.cell(col)? {
            Cell::Str(s) => Ok(s),
            cell => Err(self.mismatch(cell, "Str")),
        }
    }

    /// Cloned String for BTreeMap keys.
    pub fn str_owned(&self, col: usize) -> Result<String, AicError> {
        self.str(col).map(|s| s.to_string())
    }

    /// Narrow INT64 to u32; the AIC perf tables always fit. Errors if not.
    pub fn u32(&self, col: usize) -> Result<u32, AicError> {
        self.fit_u32(col, self.long(col)?)
    }

    /// Optional INT64 → u32. Returns None when the cell is missing/null.
    pub fn u32_optional(&self, col: Option<usize>) -> Result<Option<u32>, AicError> {
        let Some(idx) = col else {
            return Ok(None);
        };
        match self.cell(idx)? {
            Cell::Long(v) => self.fit_u32(idx, v).map(Some),
            _ => Ok(None),
        }
    }

    /// Optional INT64 kept SIGNED. Returns None when the cell is missing/null
    /// (or stored under a non-integer physical type). The table-view folds use
    /// this to mirror Python's `int(row[...])`, which never narrowed: a
    /// negative key cell loaded as a negative key instead of erroring.
    pub fn i64_optional(&self, col: Option<usize>) -> Result<Option<i64>, AicError> {
        Ok(match self.cell_optional(col)? {
            Some(Cell::Long(v)) => Some(v),
            _ => None,
        })
    }

    /// INT64 → u64. Used by columns that legitimately need the full range
    /// (e.g. NCCL `message_size`).
    pub fn u64(&self, col: usize) -> Result<u64, AicError> {
        let v = self.long(col)?;
        u64::try_from(v).map_err(|_| {
            AicError::PerfDatabase(format!(
                "parquet column[{col}] = {v} is negative; expected u64 at {}",
                self.batch.path.display()
            ))
        })
    }

    pub fn f64(&self, col: usize) -> Result<f64, AicError> {
        match self.cell(col)? {
            Cell::Double(v) => Ok(v),
            cell => Err(self.mismatch(cell, "Double")),
        }
    }

    /// BOOLEAN column, with fallbacks mirroring Python's `_to_bool` string
    /// coercion (`str(value).strip().lower() in {"1","true","yes","y"}`) for
    /// files that store flags as INT64 or strings.
    pub fn bool(&self, col: usize) -> Result<bool, AicError> {
        match self.cell(col)? {
            Cell::Bool(v) => Ok(v),
            Cell::Long(v) => Ok(v == 1),
            Cell::Str(s) => Ok(matches!(
                s.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "y"
            )),
            cell => Err(self.mismatch(cell, "Str")),
        }
    }

    /// BOOLEAN column with NO type coercion: identity flags (FPM schema v6
    /// `enable_*`) must be real booleans — Python's loader rejects anything
    /// else, so coercing a stringly-typed flag here would silently diverge
    /// from it. A type mismatch is a loud data bug.
    pub fn bool_strict(&self, col: usize) -> Result<bool, AicError> {
        match self.cell(col)? {
            Cell::Bool(v) => Ok(v),
            cell => Err(self.mismatch(cell, "Bool")),
        }
    }

    /// Optional double. Returns None when the column lookup is None OR the
    /// cell is null. Used by dispatch tables whose split-latency columns
    /// can legitimately be absent for one mode of the schema (DeepEP normal
    /// vs LL).
    pub fn f64_optional(&self, col: Option<usize>) -> Result<Option<f64>, AicError> {
        Ok(match self.cell_optional(col)? {
            Some(Cell::Double(v)) => Some(v),
            _ => None,
        })
    }

    /// Borrowed string accessor for columns that may be absent or null.
    /// Returns None if the column lookup is None OR the cell is null.
    pub fn str_optional(&self, col: Option<usize>) -> Result<Option<&str>, AicError> {
        Ok(match self.cell_optional(col)? {
            Some(Cell::Str(s)) => Some(s),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parquet::data_type::{ByteArray, Int32Type};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::{SerializedFileWriter, SerializedRowGroupWriter};
    use parquet::record::{Field, Row, RowAccessor};
    use parquet::schema::parser::parse_message_type;

    use super::*;

    const ROWS_PER_GROUP: usize = BATCH_ROWS + 476;

    /// Writes `groups` row groups of `schema`; `fill` writes each group's
    /// columns in schema order.
    fn write_file(
        path: &Path,
        schema: &str,
        props: WriterProperties,
        groups: usize,
        mut fill: impl FnMut(usize, &mut SerializedRowGroupWriter<'_, File>),
    ) {
        let schema = Arc::new(parse_message_type(schema).unwrap());
        let file = File::create(path).unwrap();
        let mut writer = SerializedFileWriter::new(file, schema, Arc::new(props)).unwrap();
        for group in 0..groups {
            let mut rg = writer.next_row_group().unwrap();
            fill(group, &mut rg);
            rg.close().unwrap();
        }
        writer.close().unwrap();
    }

    fn column<T: DataType>(
        rg: &mut SerializedRowGroupWriter<'_, File>,
        values: &[T::T],
        defs: Option<&[i16]>,
        reps: Option<&[i16]>,
    ) {
        let mut col = rg.next_column().unwrap().unwrap();
        col.typed::<T>().write_batch(values, defs, reps).unwrap();
        col.close().unwrap();
    }

    /// Two row groups of OPTIONAL columns (nulls on different rows per
    /// column), a REQUIRED string, the annotations the reader decodes
    /// (INT_64, ENUM) and types it leaves undecoded (INT32, TIMESTAMP_MICROS,
    /// raw BYTE_ARRAY, REPEATED). 64-row pages and a small dictionary limit
    /// make every batch cross pages and a dictionary-to-plain fallback.
    fn write_fixture(path: &Path) {
        let schema = "message fixture { OPTIONAL BYTE_ARRAY s (UTF8); REQUIRED BYTE_ARRAY r (UTF8); \
             OPTIONAL INT64 i; OPTIONAL DOUBLE d; OPTIONAL BOOLEAN b; OPTIONAL INT32 x; \
             OPTIONAL INT64 a (INT_64); OPTIONAL INT64 ts (TIMESTAMP_MICROS); \
             OPTIONAL BYTE_ARRAY raw; OPTIONAL BYTE_ARRAY e (ENUM); REPEATED INT64 rep; }";
        let props = WriterProperties::builder()
            .set_write_batch_size(64)
            .set_data_page_row_count_limit(64)
            .set_dictionary_page_size_limit(512)
            .build();
        write_file(path, schema, props, 2, |group, rg| {
            let rows: Vec<usize> = (group * ROWS_PER_GROUP..(group + 1) * ROWS_PER_GROUP).collect();
            let defs = |every: usize| -> Vec<i16> {
                rows.iter().map(|r| i16::from(r % every != 0)).collect()
            };
            let present = |every: usize| rows.iter().copied().filter(move |r| r % every != 0);
            let bytes = |s: String| ByteArray::from(s.into_bytes());
            let strings: Vec<_> = present(7).map(|r| bytes(format!("µs-{}", r % 5))).collect();
            let required: Vec<_> = rows.iter().map(|r| bytes(format!("k{r}"))).collect();
            let raw: Vec<_> = present(23).map(|r| bytes(format!("b{r}"))).collect();
            let enums: Vec<_> = present(29).map(|r| bytes(format!("e{}", r % 3))).collect();
            let longs: Vec<i64> = present(5).map(|r| r as i64 - 1000).collect();
            let doubles: Vec<f64> = present(3).map(|r| r as f64 * 0.1).collect();
            let bools: Vec<bool> = present(11).map(|r| r % 2 == 0).collect();
            let ints: Vec<i32> = present(13).map(|r| r as i32).collect();
            let annotated: Vec<i64> = present(17).map(|r| r as i64).collect();
            let micros: Vec<i64> = present(19).map(|r| r as i64 * 1000).collect();
            let repeated: Vec<i64> = rows.iter().map(|&r| r as i64).collect();
            column::<ByteArrayType>(rg, &strings, Some(&defs(7)), None);
            column::<ByteArrayType>(rg, &required, None, None);
            column::<Int64Type>(rg, &longs, Some(&defs(5)), None);
            column::<DoubleType>(rg, &doubles, Some(&defs(3)), None);
            column::<BoolType>(rg, &bools, Some(&defs(11)), None);
            column::<Int32Type>(rg, &ints, Some(&defs(13)), None);
            column::<Int64Type>(rg, &annotated, Some(&defs(17)), None);
            column::<Int64Type>(rg, &micros, Some(&defs(19)), None);
            column::<ByteArrayType>(rg, &raw, Some(&defs(23)), None);
            column::<ByteArrayType>(rg, &enums, Some(&defs(29)), None);
            // One element per row: present (def 1), each starting a record (rep 0).
            let ones = vec![1; rows.len()];
            column::<Int64Type>(rg, &repeated, Some(&ones), Some(&vec![0; rows.len()]));
        });
    }

    #[test]
    fn columnar_rows_match_the_record_api_cell_for_cell() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("fixture.parquet");
        write_fixture(&path);
        let reader = PerfReader::open(&path).unwrap();
        let oracle: Vec<Row> = reader
            .file_reader
            .get_row_iter(None)
            .unwrap()
            .map(Result::unwrap)
            .collect();
        let rows: Vec<PerfRow> = reader.rows().unwrap().map(Result::unwrap).collect();
        assert_eq!(rows.len(), 2 * ROWS_PER_GROUP);
        assert_eq!(rows.len(), oracle.len());
        assert_eq!(reader.num_rows(), oracle.len() as u64);
        for (index, (row, expected)) in rows.iter().zip(&oracle).enumerate() {
            assert_eq!(expected.len(), reader.column_names.len());
            for (col, (_, field)) in expected.get_column_iter().enumerate() {
                let want = match field {
                    Field::Null => Some(Cell::Null),
                    Field::Bool(v) => Some(Cell::Bool(*v)),
                    Field::Long(v) => Some(Cell::Long(*v)),
                    Field::Double(v) => Some(Cell::Double(*v)),
                    Field::Str(s) => Some(Cell::Str(s)),
                    _ => None,
                };
                match (row.cell(col).unwrap(), want) {
                    // An undecoded column reads every cell as `Other`, nulls too.
                    (Cell::Other(_), None | Some(Cell::Null)) => {}
                    (cell, want) => assert_eq!(Some(cell), want, "row {index} column {col}"),
                }
            }
        }
        let past = reader.column_names.len();
        assert!(matches!(
            rows[0].cell(past),
            Err(AicError::Parquet {
                source: ParquetError::IndexOutOfBound(col, len),
                ..
            }) if col == past && len == past
        ));
        // Nulls and type mismatches keep the record API's messages, which
        // loaders surface: `i` is null in row 0 and a Long in row 1.
        for index in [0, 1] {
            let Err(AicError::Parquet { source, .. }) = rows[index].f64(2) else {
                panic!("row {index} column 2 read as Double");
            };
            let expected = oracle[index].get_double(2).unwrap_err();
            assert_eq!(source.to_string(), expected.to_string());
        }
    }

    #[test]
    fn invalid_utf8_ends_rows_with_one_path_qualified_error() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad.parquet");
        let schema = "message bad { REQUIRED BYTE_ARRAY s (UTF8); }";
        let props = WriterProperties::builder().build();
        write_file(&path, schema, props, 1, |_, rg| {
            let values = [ByteArray::from(b"ok".to_vec()), ByteArray::from(vec![0xff])];
            column::<ByteArrayType>(rg, &values, None, None);
        });
        let reader = PerfReader::open(&path).unwrap();
        let mut rows = reader.rows().unwrap();
        assert!(matches!(
            rows.next(),
            Some(Err(AicError::Parquet {
                path: at,
                source: ParquetError::General(_),
            })) if at == path
        ));
        assert!(rows.next().is_none());
    }
}
