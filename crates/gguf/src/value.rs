//! GGUF metadata value types.

use std::fmt;

use crate::error::{GgufError, Result};

/// Tag byte for a metadata value type. Matches the GGUF spec wire values.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl ValueType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            other => return Err(GgufError::UnknownValueType(other)),
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::Bool => "bool",
            Self::String => "string",
            Self::Array => "array",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F64 => "f64",
        }
    }
}

/// Owned, parsed GGUF metadata value.
///
/// Arrays are intentionally typed (e.g. `Array(StringArray(..))` rather than `Vec<Value>`)
/// — GGUF disallows nested arrays and this lets callers pattern-match without box-allocating.
#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Array),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug, Clone)]
pub enum Array {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl Array {
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn element_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::F32(_) => ValueType::F32,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F64(_) => ValueType::F64,
        }
    }
}

impl Value {
    pub fn type_str(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::Array(_) => "array",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F64(_) => "f64",
        }
    }
}

/// Helpers used by `GgufFile::get_*` accessors.
impl Value {
    /// Coerce any small integer Value to u64 (loses sign for negatives — caller's job to choose).
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Self::U8(v) => v as u64,
            Self::I8(v) if v >= 0 => v as u64,
            Self::U16(v) => v as u64,
            Self::I16(v) if v >= 0 => v as u64,
            Self::U32(v) => v as u64,
            Self::I32(v) if v >= 0 => v as u64,
            Self::U64(v) => v,
            Self::I64(v) if v >= 0 => v as u64,
            Self::Bool(b) => b as u64,
            _ => return None,
        })
    }

    pub fn as_i64(&self) -> Option<i64> {
        Some(match *self {
            Self::U8(v) => v as i64,
            Self::I8(v) => v as i64,
            Self::U16(v) => v as i64,
            Self::I16(v) => v as i64,
            Self::U32(v) => v as i64,
            Self::I32(v) => v as i64,
            Self::U64(v) if v <= i64::MAX as u64 => v as i64,
            Self::I64(v) => v,
            _ => return None,
        })
    }

    pub fn as_f32(&self) -> Option<f32> {
        Some(match *self {
            Self::F32(v) => v,
            Self::F64(v) => v as f32,
            _ => return None,
        })
    }

    pub fn as_f64(&self) -> Option<f64> {
        Some(match *self {
            Self::F32(v) => v as f64,
            Self::F64(v) => v,
            _ => return None,
        })
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(b) = *self { Some(b) } else { None }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(s) = self { Some(s.as_str()) } else { None }
    }

    pub fn as_array(&self) -> Option<&Array> {
        if let Self::Array(a) = self { Some(a) } else { None }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U8(v) => write!(f, "{v}"),
            Self::I8(v) => write!(f, "{v}"),
            Self::U16(v) => write!(f, "{v}"),
            Self::I16(v) => write!(f, "{v}"),
            Self::U32(v) => write!(f, "{v}"),
            Self::I32(v) => write!(f, "{v}"),
            Self::F32(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::String(s) => {
                if s.len() > 80 {
                    write!(f, "{:?}…({} bytes)", &s[..80], s.len())
                } else {
                    write!(f, "{s:?}")
                }
            }
            Self::Array(a) => write!(f, "<{} of len {}>", a.element_type().as_str(), a.len()),
            Self::U64(v) => write!(f, "{v}"),
            Self::I64(v) => write!(f, "{v}"),
            Self::F64(v) => write!(f, "{v}"),
        }
    }
}
