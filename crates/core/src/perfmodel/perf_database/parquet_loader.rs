// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin parquet reader shared by every `perf_database/*.rs` table loader.
//!
//! Design notes:
//! - Each loader resolves column names to integer positions ONCE per file
//!   load via `PerfReader::col`, then iterates rows with positional accessors.
//! - Parquet's Arrow reader decodes [`BATCH_ROWS`] rows at a time; a
//!   `PerfRow` is a shared handle on its batch plus a row index.
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
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use parquet::basic::{ConvertedType, Type as PhysicalType};
use parquet::errors::ParquetError;
use parquet::schema::types::{SchemaDescriptor, Type};

use crate::common::error::AicError;

const LFS_POINTER_PREFIX: &[u8] = b"version https://git-lfs";

/// Rows decoded per batch. Small, so a caller that stops at the first
/// matching row (the readiness probes) decodes little of the file.
const BATCH_ROWS: usize = 1024;

/// Opens a parquet file and pre-caches column names so each loader resolves
/// `col("name") -> position` once before the row loop.
pub struct PerfReader {
    file: File,
    metadata: ArrowReaderMetadata,
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
        // Column types come from the parquet schema alone: the Arrow schema
        // pyarrow embeds could make string columns dictionary or large-string
        // arrays, which the accessors do not read.
        let parquet_error = |source| parquet_error(path, source);
        let options = ArrowReaderOptions::new().with_skip_arrow_metadata(true);
        let mut metadata =
            ArrowReaderMetadata::load(&file, options.clone()).map_err(parquet_error)?;
        let retyped = retyped_schema(metadata.parquet_schema()).map_err(parquet_error)?;
        if let Some(schema) = retyped {
            // Decode the footer again against the retyped columns.
            let options = options.with_parquet_schema(Arc::new(schema));
            metadata = ArrowReaderMetadata::load(&file, options).map_err(parquet_error)?;
        }
        let column_names = metadata
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        Ok(Self {
            file,
            metadata,
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

    /// Rows that `rows` will decode: the sum of the row groups' counts in the
    /// footer, which is what the Arrow reader iterates (not the file-level
    /// count). Decodes no data. A negative count is a corrupt footer.
    pub fn num_rows(&self) -> Result<u64, AicError> {
        let groups = self.metadata.metadata().row_groups();
        groups
            .iter()
            .try_fold(0u64, |total, group| {
                total.checked_add(u64::try_from(group.num_rows()).ok()?)
            })
            .ok_or_else(|| {
                AicError::PerfDatabase(format!(
                    "parquet footer has an invalid row-group row count at {}",
                    self.path.display()
                ))
            })
    }

    /// Iterate over every row, in file order. A decode error (corrupt page,
    /// invalid UTF-8 in a string column) is yielded as one `Err` item in
    /// place of its whole batch, the rows before the bad cell included, and
    /// ends the iteration.
    pub fn rows(&self) -> Result<impl Iterator<Item = Result<PerfRow, AicError>> + '_, AicError> {
        let parquet_error = |source| parquet_error(&self.path, source);
        let file = self.file.try_clone().map_err(|source| AicError::Io {
            path: self.path.clone(),
            source,
        })?;
        let batches =
            ParquetRecordBatchReaderBuilder::new_with_metadata(file, self.metadata.clone())
                .with_batch_size(BATCH_ROWS)
                .build()
                .map_err(parquet_error)?;
        let (mut batches, mut batch, mut row) = (Some(batches), None::<Rc<RowBatch>>, 0);
        Ok(std::iter::from_fn(move || {
            loop {
                if let Some(batch) = &batch
                    && row < batch.columns.num_rows()
                {
                    row += 1;
                    let batch = Rc::clone(batch);
                    return Some(Ok(PerfRow {
                        batch,
                        row: row - 1,
                    }));
                }
                match batches.as_mut()?.next()? {
                    Ok(columns) => {
                        batch = Some(Rc::new(RowBatch {
                            path: self.path.clone(),
                            columns,
                        }));
                        row = 0;
                    }
                    Err(source) => {
                        // The reader's state after an error is unspecified;
                        // stop here.
                        batches = None;
                        return Some(Err(parquet_error(source.into())));
                    }
                }
            }
        }))
    }
}

fn parquet_error(path: &Path, source: ParquetError) -> AicError {
    AicError::Parquet {
        path: path.to_path_buf(),
        source,
    }
}

/// `schema` with two kinds of top-level column retyped, or `None` when the
/// file has neither:
/// - ENUM and JSON `BYTE_ARRAY` columns become `UTF8`, so their cells read
///   as strings and are UTF-8 checked. The decoder checks only columns
///   annotated `UTF8`, and types ENUM as binary.
/// - `INT64` columns with a logical type but no converted type (nanosecond
///   timestamp or time) become plain `INT64`, so their cells read as
///   integers.
fn retyped_schema(schema: &SchemaDescriptor) -> Result<Option<SchemaDescriptor>, ParquetError> {
    let root = schema.root_schema();
    let mut retyped = false;
    let fields = root.get_fields().iter().map(|field| {
        let Type::PrimitiveType {
            basic_info: info,
            physical_type,
            ..
        } = field.as_ref()
        else {
            return Ok(Arc::clone(field));
        };
        let converted = match (physical_type, info.converted_type()) {
            (PhysicalType::BYTE_ARRAY, ConvertedType::ENUM | ConvertedType::JSON) => {
                ConvertedType::UTF8
            }
            (PhysicalType::INT64, ConvertedType::NONE) if info.logical_type_ref().is_some() => {
                ConvertedType::NONE
            }
            _ => return Ok(Arc::clone(field)),
        };
        retyped = true;
        Type::primitive_type_builder(info.name(), *physical_type)
            .with_repetition(info.repetition())
            .with_converted_type(converted)
            .build()
            .map(Arc::new)
    });
    let fields = fields.collect::<Result<Vec<_>, ParquetError>>()?;
    if !retyped {
        return Ok(None);
    }
    let root = Type::group_type_builder(root.name())
        .with_fields(fields)
        .build()?;
    Ok(Some(SchemaDescriptor::new(Arc::new(root))))
}

/// One decoded batch and the file it came from.
struct RowBatch {
    path: PathBuf,
    columns: RecordBatch,
}

/// One typed cell of a decoded row.
#[derive(Debug, PartialEq)]
enum Cell<'a> {
    Null,
    Bool(bool),
    Long(i64),
    Double(f64),
    Str(&'a str),
    /// A present cell of a type no accessor reads (INT32, UINT_64,
    /// timestamps, unannotated binary, lists).
    Other(&'a DataType),
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
        let (columns, row) = (self.batch.columns.columns(), self.row);
        let array = columns
            .get(col)
            .ok_or_else(|| self.error(ParquetError::IndexOutOfBound(col, columns.len())))?;
        if array.is_null(row) {
            return Ok(Cell::Null);
        }
        Ok(match array.data_type() {
            DataType::Boolean => Cell::Bool(array.as_boolean().value(row)),
            DataType::Int64 => Cell::Long(array.as_primitive::<Int64Type>().value(row)),
            DataType::Float64 => Cell::Double(array.as_primitive::<Float64Type>().value(row)),
            DataType::Utf8 => Cell::Str(array.as_string::<i32>().value(row)),
            other => Cell::Other(other),
        })
    }

    /// `cell` for a column that may be absent; `None` when it is.
    fn cell_optional(&self, col: Option<usize>) -> Result<Option<Cell<'_>>, AicError> {
        col.map(|col| self.cell(col)).transpose()
    }

    /// The error for reading a null cell, or a cell of another type, as `want`.
    fn mismatch(&self, col: usize, cell: Cell<'_>, want: &str) -> AicError {
        self.error(ParquetError::General(format!(
            "parquet column[{col}] is {cell:?}, expected {want}"
        )))
    }

    fn error(&self, source: ParquetError) -> AicError {
        parquet_error(&self.batch.path, source)
    }

    fn long(&self, col: usize) -> Result<i64, AicError> {
        match self.cell(col)? {
            Cell::Long(v) => Ok(v),
            cell => Err(self.mismatch(col, cell, "Long")),
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
            cell => Err(self.mismatch(col, cell, "Str")),
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
            cell => Err(self.mismatch(col, cell, "Double")),
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
            cell => Err(self.mismatch(col, cell, "Bool")),
        }
    }

    /// BOOLEAN column with NO type coercion: identity flags (FPM schema v6
    /// `enable_*`) must be real booleans — Python's loader rejects anything
    /// else, so coercing a stringly-typed flag here would silently diverge
    /// from it. A type mismatch is a loud data bug.
    pub fn bool_strict(&self, col: usize) -> Result<bool, AicError> {
        match self.cell(col)? {
            Cell::Bool(v) => Ok(v),
            cell => Err(self.mismatch(col, cell, "Bool")),
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
    use parquet::data_type::{
        BoolType, ByteArray, ByteArrayType, DataType, DoubleType, Int32Type, Int64Type,
    };
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::file::writer::{SerializedFileWriter, SerializedRowGroupWriter};
    use parquet::record::{Field, Row};
    use parquet::schema::parser::parse_message_type;

    use super::*;

    /// Enough rows to cross a batch boundary.
    const ROWS: usize = BATCH_ROWS + 476;

    /// Writes `groups` identical row groups of `schema`; `fill` writes one
    /// group's columns in schema order.
    fn write_file(
        path: &Path,
        schema: &str,
        groups: usize,
        fill: impl Fn(&mut SerializedRowGroupWriter<'_, File>),
    ) {
        let schema = Arc::new(parse_message_type(schema).unwrap());
        let file = File::create(path).unwrap();
        let mut writer = SerializedFileWriter::new(file, schema, Default::default()).unwrap();
        for _ in 0..groups {
            let mut rg = writer.next_row_group().unwrap();
            fill(&mut rg);
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

    /// An OPTIONAL column of `ROWS` rows that is null on every `every`-th row.
    fn optional<T: DataType>(
        rg: &mut SerializedRowGroupWriter<'_, File>,
        every: usize,
        value: impl Fn(usize) -> T::T,
    ) {
        let present = |r: &usize| !r.is_multiple_of(every);
        let defs: Vec<i16> = (0..ROWS).map(|r| i16::from(present(&r))).collect();
        let values: Vec<T::T> = (0..ROWS).filter(present).map(value).collect();
        column::<T>(rg, &values, Some(&defs), None);
    }

    /// Two row groups of `ROWS` rows: OPTIONAL columns (nulls on different
    /// rows per column), a REQUIRED string, annotations the accessors read
    /// (INT_64, nanosecond timestamp, JSON, ENUM) and types they do not
    /// (INT32, TIMESTAMP_MICROS, raw BYTE_ARRAY, REPEATED).
    fn write_fixture(path: &Path) {
        let schema = "message fixture { OPTIONAL BYTE_ARRAY s (UTF8); REQUIRED BYTE_ARRAY r (UTF8); \
             OPTIONAL INT64 i; OPTIONAL DOUBLE d; OPTIONAL BOOLEAN b; OPTIONAL INT32 x; \
             OPTIONAL INT64 a (INT_64); OPTIONAL INT64 ts (TIMESTAMP_MICROS); \
             OPTIONAL INT64 ns (TIMESTAMP(NANOS,true)); \
             OPTIONAL BYTE_ARRAY raw; OPTIONAL BYTE_ARRAY j (JSON); \
             OPTIONAL BYTE_ARRAY e (ENUM); REPEATED INT64 rep; }";
        write_file(path, schema, 2, |rg| {
            let bytes = |s: String| ByteArray::from(s.into_bytes());
            let required: Vec<_> = (0..ROWS).map(|r| bytes(format!("k{r}"))).collect();
            let repeated: Vec<i64> = (0..ROWS).map(|r| r as i64).collect();
            optional::<ByteArrayType>(rg, 7, |r| bytes(format!("µs-{}", r % 5)));
            column::<ByteArrayType>(rg, &required, None, None);
            optional::<Int64Type>(rg, 5, |r| r as i64 - 1000);
            optional::<DoubleType>(rg, 3, |r| r as f64 * 0.1);
            optional::<BoolType>(rg, 11, |r| r % 2 == 0);
            optional::<Int32Type>(rg, 13, |r| r as i32);
            optional::<Int64Type>(rg, 17, |r| r as i64);
            optional::<Int64Type>(rg, 19, |r| r as i64 * 1000);
            optional::<Int64Type>(rg, 37, |r| r as i64 * 1_000_000);
            optional::<ByteArrayType>(rg, 23, |r| bytes(format!("b{r}")));
            optional::<ByteArrayType>(rg, 29, |r| bytes(format!("{{\"r\":{r}}}")));
            optional::<ByteArrayType>(rg, 31, |r| bytes(format!("E{}", r % 3)));
            // One element per row: present (def 1), each starting a record (rep 0).
            column::<Int64Type>(rg, &repeated, Some(&[1; ROWS]), Some(&[0; ROWS]));
        });
    }

    #[test]
    fn columnar_rows_match_the_record_api_cell_for_cell() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("fixture.parquet");
        write_fixture(&path);
        let reader = PerfReader::open(&path).unwrap();
        let oracle: Vec<Row> = SerializedFileReader::new(File::open(&path).unwrap())
            .unwrap()
            .get_row_iter(None)
            .unwrap()
            .map(Result::unwrap)
            .collect();
        let rows: Vec<PerfRow> = reader.rows().unwrap().map(Result::unwrap).collect();
        assert_eq!(rows.len(), 2 * ROWS);
        assert_eq!(rows.len(), oracle.len());
        assert_eq!(reader.num_rows().unwrap(), oracle.len() as u64);
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
                    (Cell::Other(_), None) => {}
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
        // `i` (column 2) is null in row 0 and a Long in row 1.
        for index in [0, 1] {
            assert!(matches!(
                rows[index].f64(2),
                Err(AicError::Parquet {
                    source: ParquetError::General(_),
                    ..
                })
            ));
        }
    }

    /// The record API read an ENUM cell as a string, so an ENUM
    /// `kernel_source` matched a source filter; read as absent, the filter
    /// would drop the row.
    #[test]
    fn enum_kernel_source_passes_the_source_filter() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("enum.parquet");
        let schema =
            "message t { REQUIRED BYTE_ARRAY kernel_source (ENUM); REQUIRED DOUBLE latency; }";
        write_file(&path, schema, 1, |rg| {
            column::<ByteArrayType>(rg, &[ByteArray::from("measured_kernel")], None, None);
            column::<DoubleType>(rg, &[1.5], None, None);
        });
        let reader = PerfReader::open(&path).unwrap();
        let ks_col = reader.col_optional("kernel_source");
        let row = reader.rows().unwrap().next().unwrap().unwrap();
        assert_eq!(row.str_optional(ks_col).unwrap(), Some("measured_kernel"));
        let allow = ["measured_kernel".to_string()];
        assert!(crate::perf_database::kernel_source_ok(Some(&allow), ks_col, &row).unwrap());
    }

    /// Every annotation read as a string: the decoder itself checks only
    /// `UTF8` columns and would hand out the other two unchecked.
    #[test]
    fn invalid_utf8_ends_rows_with_one_path_qualified_error() {
        let tmp = tempfile::tempdir().unwrap();
        for annotation in ["UTF8", "ENUM", "JSON"] {
            let path = tmp.path().join(format!("{annotation}.parquet"));
            let schema = format!("message bad {{ REQUIRED BYTE_ARRAY s ({annotation}); }}");
            write_file(&path, &schema, 1, |rg| {
                let values = [ByteArray::from(b"ok".to_vec()), ByteArray::from(vec![0xff])];
                column::<ByteArrayType>(rg, &values, None, None);
            });
            let reader = PerfReader::open(&path).unwrap();
            let mut rows = reader.rows().unwrap();
            assert!(
                matches!(
                    rows.next(),
                    Some(Err(AicError::Parquet { path: at, .. })) if at == path
                ),
                "{annotation}"
            );
            assert!(rows.next().is_none(), "{annotation}");
        }
    }
}
