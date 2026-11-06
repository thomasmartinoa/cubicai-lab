//! GGUF header parsing: metadata and tensor directory. Works on a full file (mmap) or on just the
//! first bytes of a remote file, in which case it asks for more with [`ParseError::Truncated`].

use crate::types::GgmlType;
use std::collections::HashMap;
use std::fmt;

/// Arrays longer than this keep only their first items (tokenizer vocabularies run to 256k strings).
const KEEP_STRINGS: usize = 16;
const KEEP_NUMBERS: usize = 8192;

#[derive(Clone, Debug)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Array),
}

#[derive(Clone, Debug)]
pub struct Array {
    pub item_type: u32,
    pub len: u64,
    /// The first items (all of them unless the array is long).
    pub items: Vec<Value>,
}

impl Value {
    pub fn as_f64(&self) -> Option<f64> {
        Some(match *self {
            Value::U8(v) => v as f64,
            Value::I8(v) => v as f64,
            Value::U16(v) => v as f64,
            Value::I16(v) => v as f64,
            Value::U32(v) => v as f64,
            Value::I32(v) => v as f64,
            Value::F32(v) => v as f64,
            Value::U64(v) => v as f64,
            Value::I64(v) => v as f64,
            Value::F64(v) => v,
            Value::Bool(v) => v as u8 as f64,
            _ => return None,
        })
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_f64().filter(|v| *v >= 0.0).map(|v| v as u64)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => self.as_f64().map(|v| v != 0.0),
        }
    }

    pub fn as_array(&self) -> Option<&Array> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Short text for the metadata table.
    pub fn display(&self) -> String {
        match self {
            Value::Str(s) => {
                if s.chars().count() > 160 {
                    format!("{}…", s.chars().take(160).collect::<String>())
                } else {
                    s.clone()
                }
            }
            Value::Bool(b) => b.to_string(),
            Value::F32(v) => format!("{v}"),
            Value::F64(v) => format!("{v}"),
            Value::Array(a) => {
                let shown: Vec<String> = a.items.iter().take(8).map(|v| v.display()).collect();
                let more = if a.len as usize > shown.len() { ", …" } else { "" };
                format!("[{}{}] ({} items)", shown.join(", "), more, a.len)
            }
            other => other.as_f64().map(|v| format!("{v}")).unwrap_or_default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: dims[0] is the row length (the input side of a weight matrix).
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset inside the file's data section.
    pub offset: u64,
    /// Which part of a split model holds it (0 for single files).
    pub file: usize,
    pub n_elements: u64,
    pub n_bytes: u64,
}

impl TensorInfo {
    pub fn row_len(&self) -> u64 {
        self.dims.first().copied().unwrap_or(1)
    }

    pub fn n_rows(&self) -> u64 {
        self.dims.iter().skip(1).product::<u64>().max(1)
    }

    pub fn shape_text(&self) -> String {
        // Shown the way people write matrices: outputs × inputs (reverse of ggml order).
        self.dims.iter().rev().map(|d| d.to_string()).collect::<Vec<_>>().join(" × ")
    }
}

#[derive(Clone, Debug)]
pub struct Gguf {
    pub version: u32,
    pub metadata: Vec<(String, Value)>,
    index: HashMap<String, usize>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    /// Where the tensor data starts in the file.
    pub data_offset: u64,
    /// Data start of each part of a split model (index = `TensorInfo::file`).
    pub part_offsets: Vec<u64>,
}

#[derive(Debug, Clone)]
pub enum ParseError {
    /// The bytes ended before the header did; `have` is how many we had.
    Truncated { have: usize },
    Invalid(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::Truncated { have } => write!(f, "file header longer than the {have} bytes read"),
            ParseError::Invalid(m) => write!(f, "not a valid GGUF file: {m}"),
        }
    }
}

impl std::error::Error for ParseError {}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ParseError> {
        let end = self.pos.checked_add(n).ok_or_else(|| ParseError::Invalid("length overflow".into()))?;
        if end > self.b.len() {
            return Err(ParseError::Truncated { have: self.b.len() });
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn arr<const N: usize>(&mut self) -> Result<[u8; N], ParseError> {
        Ok(self.take(N)?.try_into().unwrap())
    }

    fn u32(&mut self) -> Result<u32, ParseError> {
        Ok(u32::from_le_bytes(self.arr()?))
    }

    fn u64(&mut self) -> Result<u64, ParseError> {
        Ok(u64::from_le_bytes(self.arr()?))
    }

    fn string(&mut self) -> Result<String, ParseError> {
        let len = self.u64()?;
        if len > (1 << 32) {
            return Err(ParseError::Invalid(format!("string of {len} bytes")));
        }
        let bytes = self.take(len as usize)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn skip_string(&mut self) -> Result<(), ParseError> {
        let len = self.u64()?;
        if len > (1 << 32) {
            return Err(ParseError::Invalid(format!("string of {len} bytes")));
        }
        self.take(len as usize).map(|_| ())
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<Value, ParseError> {
        Ok(match ty {
            0 => Value::U8(self.arr::<1>()?[0]),
            1 => Value::I8(self.arr::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.arr()?)),
            3 => Value::I16(i16::from_le_bytes(self.arr()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.arr()?)),
            6 => Value::F32(f32::from_le_bytes(self.arr()?)),
            7 => Value::Bool(self.arr::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                if depth > 4 {
                    return Err(ParseError::Invalid("arrays nested too deep".into()));
                }
                let item_type = self.u32()?;
                let len = self.u64()?;
                let keep = if item_type == 8 || item_type == 9 { KEEP_STRINGS } else { KEEP_NUMBERS };
                let mut items = Vec::with_capacity((len as usize).min(keep));
                if let Some(size) = fixed_size(item_type).filter(|_| len as usize > keep) {
                    for _ in 0..keep {
                        items.push(self.value(item_type, depth + 1)?);
                    }
                    let rest = (len - keep as u64)
                        .checked_mul(size)
                        .ok_or_else(|| ParseError::Invalid("array too long".into()))?;
                    self.take(rest as usize)?;
                } else {
                    for i in 0..len {
                        if (i as usize) < keep {
                            items.push(self.value(item_type, depth + 1)?);
                        } else if item_type == 8 {
                            self.skip_string()?;
                        } else {
                            self.value(item_type, depth + 1)?;
                        }
                    }
                }
                Value::Array(Array { item_type, len, items })
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.arr()?)),
            12 => Value::F64(f64::from_le_bytes(self.arr()?)),
            other => return Err(ParseError::Invalid(format!("unknown value type {other}"))),
        })
    }
}

fn fixed_size(ty: u32) -> Option<u64> {
    Some(match ty {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}

impl Gguf {
    /// Parses the header. `file_len` (when known) lets sizes of unknown tensor types be worked out
    /// from where the next tensor starts.
    pub fn parse(bytes: &[u8], file_len: Option<u64>) -> Result<Gguf, ParseError> {
        let mut c = Cursor { b: bytes, pos: 0 };
        let magic = c.arr::<4>()?;
        if &magic != b"GGUF" {
            return Err(ParseError::Invalid("missing GGUF magic".into()));
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return Err(ParseError::Invalid(format!("version {version} (big-endian or unsupported)")));
        }
        let n_tensors = c.u64()?;
        let n_kv = c.u64()?;
        if n_tensors > 1_000_000 || n_kv > 1_000_000 {
            return Err(ParseError::Invalid("implausible counts".into()));
        }
        let mut metadata = Vec::with_capacity(n_kv as usize);
        let mut index = HashMap::new();
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            let value = c.value(ty, 0)?;
            index.insert(key.clone(), metadata.len());
            metadata.push((key, value));
        }
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()?;
            if n_dims > 8 {
                return Err(ParseError::Invalid(format!("{name}: {n_dims} dimensions")));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(c.u64()?);
            }
            let ty = GgmlType::from_id(c.u32()?);
            let offset = c.u64()?;
            let n_elements = dims.iter().product::<u64>().max(1);
            let n_bytes = ty.bytes_for(n_elements).unwrap_or(0);
            tensors.push(TensorInfo { name, dims, ty, offset, file: 0, n_elements, n_bytes });
        }
        let alignment = index
            .get("general.alignment")
            .and_then(|&i| metadata[i].1.as_u64())
            .filter(|a| *a > 0)
            .unwrap_or(32);
        let data_offset = (c.pos as u64).div_ceil(alignment) * alignment;
        let mut g = Gguf { version, metadata, index, tensors, alignment, data_offset, part_offsets: vec![data_offset] };
        g.fill_unknown_sizes(file_len);
        Ok(g)
    }

    fn fill_unknown_sizes(&mut self, file_len: Option<u64>) {
        if self.tensors.iter().all(|t| t.n_bytes > 0) {
            return;
        }
        let mut order: Vec<usize> = (0..self.tensors.len()).collect();
        order.sort_by_key(|&i| self.tensors[i].offset);
        for w in 0..order.len() {
            let i = order[w];
            if self.tensors[i].n_bytes > 0 {
                continue;
            }
            let end = match order.get(w + 1) {
                Some(&j) => Some(self.tensors[j].offset),
                None => file_len.map(|l| l.saturating_sub(self.data_offset)),
            };
            if let Some(end) = end {
                self.tensors[i].n_bytes = end.saturating_sub(self.tensors[i].offset);
            }
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.index.get(key).map(|&i| &self.metadata[i].1)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(|v| match v {
            // Some per-layer values are arrays; the largest is the useful single number.
            Value::Array(a) => a.items.iter().filter_map(Value::as_u64).max(),
            other => other.as_u64(),
        })
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(Value::as_f64)
    }

    pub fn arch(&self) -> &str {
        self.get_str("general.architecture").unwrap_or("unknown")
    }

    /// `{arch}.{key}`, the per-architecture hyperparameters.
    pub fn arch_u64(&self, key: &str) -> Option<u64> {
        self.get_u64(&format!("{}.{key}", self.arch()))
    }

    pub fn arch_f64(&self, key: &str) -> Option<f64> {
        self.get_f64(&format!("{}.{key}", self.arch()))
    }

    /// A per-layer value (`{arch}.{key}` as an array or a single number) for layer `layer`.
    pub fn arch_per_layer(&self, key: &str, layer: usize) -> Option<u64> {
        match self.get(&format!("{}.{key}", self.arch()))? {
            Value::Array(a) => a.items.get(layer).and_then(Value::as_u64),
            v => v.as_u64(),
        }
    }

    pub fn split_count(&self) -> u64 {
        self.get_u64("split.count").unwrap_or(1).max(1)
    }

    /// Where a tensor's bytes start in its file.
    pub fn tensor_start(&self, t: &TensorInfo) -> u64 {
        self.part_offsets.get(t.file).copied().unwrap_or(self.data_offset) + t.offset
    }

    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.n_bytes).sum()
    }

    /// Adds the tensors of another part of a split model (`file` is its index).
    pub fn merge_split(&mut self, other: Gguf, file: usize) {
        if self.part_offsets.len() <= file {
            self.part_offsets.resize(file + 1, 0);
        }
        self.part_offsets[file] = other.data_offset;
        for mut t in other.tensors {
            t.file = file;
            self.tensors.push(t);
        }
    }
}
