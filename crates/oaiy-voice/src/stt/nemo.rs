//! A NeMo `.nemo` checkpoint, read without running any of it.
//!
//! A `.nemo` is an uncompressed tar holding `model_config.yaml`,
//! `model_weights.ckpt` (a zip-format `torch.save` of the state dict) and the
//! tokenizer files under hash-prefixed names. The tar and the zip inside it
//! are both stored, not compressed, so a tensor is read straight from its
//! place in the file.
//!
//! The pickle is interpreted, not executed: only plain data (dicts, lists,
//! tuples, numbers, strings), `OrderedDict`, tensor storages and
//! `torch._utils._rebuild_tensor_v2` are understood, and any other global is
//! an error. This is what `torch.load(weights_only=True)` allows. (The same
//! reader as `oaiy-media`'s `sound/pth.rs`, with 64-bit integer storages for
//! BatchNorm's `num_batches_tracked` and a zip that starts inside the tar.)

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::{bad, Result};

/// Element types a checkpoint's storages hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageType {
    F32,
    F16,
    BF16,
    I64,
    I32,
    U8,
    Bool,
}

impl StorageType {
    pub fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I64 => 8,
            Self::U8 | Self::Bool => 1,
        }
    }
}

/// A value from the pickle.
#[derive(Clone, Debug)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    /// A dict or `OrderedDict`, in order.
    Dict(Vec<(Value, Value)>),
    /// A tensor: its storage's zip entry and type, element offset, shape and stride.
    Tensor { storage: String, dtype: StorageType, offset: usize, shape: Vec<usize>, stride: Vec<usize> },
    /// A storage (a persistent id), before a tensor is built on it.
    Storage { key: String, dtype: StorageType },
    /// A global the pickle may call: only the allowed ones get this far.
    Global(String),
    Mark,
}

impl Value {
    fn ints(&self) -> Option<Vec<i64>> {
        match self {
            Value::List(v) | Value::Tuple(v) => v.iter().map(|x| if let Value::Int(i) = x { Some(*i) } else { None }).collect(),
            _ => None,
        }
    }
}

/// One stored member of the tar, or entry of the zip: where its bytes start
/// in the file and how many there are.
#[derive(Clone, Copy, Debug)]
struct Span {
    start: u64,
    len: u64,
}

/// A tensor's raw little-endian bytes and layout.
pub struct RawTensor {
    pub dtype: StorageType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

pub struct NemoArchive {
    path: PathBuf,
    file: File,
    members: Vec<(String, Span)>,
    /// The checkpoint's zip entries (absolute spans in the file).
    entries: HashMap<String, Span>,
    /// The zip's folder (entries are `<folder>/data.pkl`, `<folder>/data/<key>`).
    folder: String,
    /// The state dict: parameter name to tensor.
    tensors: HashMap<String, Value>,
    order: Vec<String>,
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}
fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}
fn u64_at(b: &[u8], i: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&b[i..i + 8]);
    u64::from_le_bytes(w)
}

fn read_at(file: &mut File, start: u64, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    file.seek(SeekFrom::Start(start))?;
    file.read_exact(&mut buf)?;
    Ok(buf)
}

/// The members of a POSIX (ustar or GNU) tar: name and data span. Long names
/// (GNU `L` records and pax `path=` headers) are followed.
fn tar_members(file: &mut File) -> Result<Vec<(String, Span)>> {
    let size = file.metadata()?.len();
    let mut out = Vec::new();
    let mut at = 0u64;
    let mut long_name: Option<String> = None;
    while at + 512 <= size {
        let h = read_at(file, at, 512)?;
        if h.iter().all(|&b| b == 0) {
            break;
        }
        let field = |a: usize, b: usize| -> String {
            let s = &h[a..b];
            let end = s.iter().position(|&c| c == 0).unwrap_or(s.len());
            String::from_utf8_lossy(&s[..end]).into_owned()
        };
        let octal = field(124, 136);
        let len = if h[124] & 0x80 != 0 {
            // GNU base-256 size.
            h[125..136].iter().fold(0u64, |acc, &b| (acc << 8) | b as u64)
        } else {
            u64::from_str_radix(octal.trim_matches(|c: char| c == ' ' || c == '\0'), 8).map_err(|_| bad(format!("tar: bad size {octal:?}")))?
        };
        let kind = h[156];
        let data = at + 512;
        let padded = len.div_ceil(512) * 512;
        if data + len > size {
            return Err(bad("tar: a member runs past the end of the file"));
        }
        match kind {
            b'L' => {
                let n = read_at(file, data, len as usize)?;
                long_name = Some(String::from_utf8_lossy(&n).trim_end_matches('\0').to_string());
            }
            b'x' => {
                let n = read_at(file, data, len as usize)?;
                for rec in String::from_utf8_lossy(&n).lines() {
                    if let Some(p) = rec.split_once(' ').and_then(|(_, kv)| kv.strip_prefix("path=")) {
                        long_name = Some(p.to_string());
                    }
                }
            }
            b'0' | 0 => {
                let prefix = field(345, 500);
                let name = long_name.take().unwrap_or_else(|| if prefix.is_empty() { field(0, 100) } else { format!("{prefix}/{}", field(0, 100)) });
                let name = name.trim_start_matches("./").to_string();
                out.push((name, Span { start: data, len }));
            }
            _ => long_name = None,
        }
        at = data + padded;
    }
    Ok(out)
}

impl NemoArchive {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let members = tar_members(&mut file).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let ckpt = members.iter().find(|(n, _)| n.ends_with("model_weights.ckpt")).map(|(_, s)| *s).ok_or_else(|| bad(format!("{}: no model_weights.ckpt (not a .nemo?)", path.display())))?;
        let entries = zip_entries(&mut file, ckpt)?;
        let pkl = entries.keys().find(|k| k.ends_with("/data.pkl")).cloned().ok_or_else(|| bad("the checkpoint has no data.pkl"))?;
        let folder = pkl.trim_end_matches("/data.pkl").to_string();
        let span = entries[&pkl];
        let bytes = read_at(&mut file, span.start, span.len as usize)?;
        let root = unpickle(&bytes)?;
        // A plain state dict, or Lightning's {"state_dict": {...}}.
        let dict = match &root {
            Value::Dict(items) => items.iter().find(|(k, _)| matches!(k, Value::Str(s) if s == "state_dict")).map(|(_, v)| v.clone()).unwrap_or(root.clone()),
            _ => return Err(bad("the checkpoint is not a dict")),
        };
        let Value::Dict(items) = dict else { return Err(bad("the state dict is not a dict")) };
        let mut tensors = HashMap::new();
        let mut order = Vec::new();
        for (k, v) in items {
            if let (Value::Str(k), t @ Value::Tensor { .. }) = (k, v) {
                order.push(k.clone());
                tensors.insert(k, t);
            }
        }
        Ok(Self { path: path.to_path_buf(), file, members, entries, folder, tensors, order })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The state dict's names, in file order.
    pub fn names(&self) -> &[String] {
        &self.order
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// A tensor's shape, without reading it.
    pub fn shape(&self, name: &str) -> Option<Vec<usize>> {
        match self.tensors.get(name) {
            Some(Value::Tensor { shape, .. }) => Some(shape.clone()),
            _ => None,
        }
    }

    /// A tar member (by a suffix of its name, since the tokenizer files carry
    /// a hash prefix), read whole.
    pub fn member(&mut self, suffix: &str) -> Result<Option<Vec<u8>>> {
        let Some(span) = self.members.iter().find(|(n, _)| n.ends_with(suffix)).map(|(_, s)| *s) else { return Ok(None) };
        if span.len > 256 << 20 {
            return Err(bad(format!("{suffix}: {} bytes is too large a member to read whole", span.len)));
        }
        Ok(Some(read_at(&mut self.file, span.start, span.len as usize)?))
    }

    /// One tensor's bytes and shape.
    pub fn tensor(&mut self, name: &str) -> Result<RawTensor> {
        let Some(Value::Tensor { storage, dtype, offset, shape, stride }) = self.tensors.get(name) else { return Err(bad(format!("tensor {name} not in {}", self.path.display()))) };
        // Only contiguous row-major tensors (what a state dict holds).
        let mut expect = 1;
        for (d, s) in shape.iter().zip(stride).rev() {
            if *d > 1 && *s != expect {
                return Err(bad(format!("{name} is not contiguous")));
            }
            expect *= d;
        }
        let numel: usize = shape.iter().product();
        let size = dtype.size();
        let key = format!("{}/data/{storage}", self.folder);
        let span = *self.entries.get(&key).ok_or_else(|| bad(format!("{name}: no storage {key}")))?;
        let (from, len) = (offset * size, numel * size);
        if (from + len) as u64 > span.len {
            return Err(bad(format!("{name} runs past its storage")));
        }
        let bytes = read_at(&mut self.file, span.start + from as u64, len)?;
        Ok(RawTensor { dtype: *dtype, shape: shape.clone(), bytes })
    }
}

/// The entries of the stored zip at `zip` (a span of the file), with their
/// data spans made absolute.
fn zip_entries(file: &mut File, zip: Span) -> Result<HashMap<String, Span>> {
    let size = zip.len;
    // The end of central directory record (and the zip64 one, past 4 GB).
    let tail_len = size.min(65_536 + 22);
    let tail = read_at(file, zip.start + size - tail_len, tail_len as usize)?;
    let eocd = (0..tail.len().saturating_sub(21)).rev().find(|&i| u32_at(&tail, i) == 0x0605_4b50).ok_or_else(|| bad("model_weights.ckpt is not a zip (a legacy torch.save file?)"))?;
    let mut count = u16_at(&tail, eocd + 10) as u64;
    let mut cd_size = u32_at(&tail, eocd + 12) as u64;
    let mut cd_start = u32_at(&tail, eocd + 16) as u64;
    if cd_start == 0xffff_ffff || count == 0xffff || cd_size == 0xffff_ffff {
        let loc = eocd.checked_sub(20).filter(|&i| u32_at(&tail, i) == 0x0706_4b50).ok_or_else(|| bad("zip64 locator missing"))?;
        let at = u64_at(&tail, loc + 8);
        let rec = read_at(file, zip.start + at, 56)?;
        count = u64_at(&rec, 32);
        cd_size = u64_at(&rec, 40);
        cd_start = u64_at(&rec, 48);
    }
    if cd_start + cd_size > size || cd_size > 64 << 20 {
        return Err(bad("bad zip central directory"));
    }
    let cd = read_at(file, zip.start + cd_start, cd_size as usize)?;
    let mut entries = HashMap::new();
    let mut i = 0;
    for _ in 0..count {
        if i + 46 > cd.len() || u32_at(&cd, i) != 0x0201_4b50 {
            return Err(bad("bad zip central directory"));
        }
        let method = u16_at(&cd, i + 10);
        let mut len = u32_at(&cd, i + 24) as u64;
        let name_len = u16_at(&cd, i + 28) as usize;
        let extra_len = u16_at(&cd, i + 30) as usize;
        let comment_len = u16_at(&cd, i + 32) as usize;
        let mut local = u32_at(&cd, i + 42) as u64;
        if i + 46 + name_len + extra_len > cd.len() {
            return Err(bad("bad zip central directory"));
        }
        let name = String::from_utf8_lossy(&cd[i + 46..i + 46 + name_len]).into_owned();
        // Zip64 sizes and offset, in the extra field.
        let mut e = i + 46 + name_len;
        let end = e + extra_len;
        while e + 4 <= end {
            let (id, n) = (u16_at(&cd, e), u16_at(&cd, e + 2) as usize);
            if id == 1 {
                let mut f = e + 4;
                if u32_at(&cd, i + 24) == 0xffff_ffff {
                    len = u64_at(&cd, f);
                    f += 8;
                }
                if u32_at(&cd, i + 20) == 0xffff_ffff {
                    f += 8;
                }
                if u32_at(&cd, i + 42) == 0xffff_ffff {
                    local = u64_at(&cd, f);
                }
            }
            e += 4 + n;
        }
        if method != 0 {
            return Err(bad(format!("{name} is compressed")));
        }
        // The data starts after the local header, whose extra field may differ.
        let lh = read_at(file, zip.start + local, 30)?;
        let start = local + 30 + u16_at(&lh, 26) as u64 + u16_at(&lh, 28) as u64;
        if start + len > size {
            return Err(bad(format!("{name} runs past the end of the zip")));
        }
        entries.insert(name, Span { start: zip.start + start, len });
        i = end + comment_len;
    }
    Ok(entries)
}

fn storage_type(name: &str) -> Result<StorageType> {
    Ok(match name {
        "torch.FloatStorage" => StorageType::F32,
        "torch.HalfStorage" => StorageType::F16,
        "torch.BFloat16Storage" => StorageType::BF16,
        "torch.LongStorage" => StorageType::I64,
        "torch.IntStorage" => StorageType::I32,
        "torch.ByteStorage" => StorageType::U8,
        "torch.BoolStorage" => StorageType::Bool,
        other => return Err(bad(format!("unsupported storage {other}"))),
    })
}

/// The only globals a checkpoint may name.
fn global(module: &str, name: &str) -> Result<Value> {
    let full = format!("{module}.{name}");
    match full.as_str() {
        "collections.OrderedDict" | "torch._utils._rebuild_tensor_v2" => Ok(Value::Global(full)),
        s if s.starts_with("torch.") && s.ends_with("Storage") => {
            storage_type(s)?;
            Ok(Value::Global(full))
        }
        _ => Err(bad(format!("refusing {full}: only tensors and plain data are read"))),
    }
}

/// Interpret a pickle (protocols 2 to 5) with only data-building operations.
pub fn unpickle(b: &[u8]) -> Result<Value> {
    let mut stack: Vec<Value> = Vec::new();
    let mut memo: HashMap<u32, Value> = HashMap::new();
    let mut i = 0;
    let take = |i: &mut usize, n: usize| -> Result<&[u8]> {
        let s = b.get(*i..*i + n).ok_or_else(|| bad("truncated pickle"))?;
        *i += n;
        Ok(s)
    };
    let pop = |stack: &mut Vec<Value>| stack.pop().ok_or_else(|| bad("empty pickle stack"));
    let pop_mark = |stack: &mut Vec<Value>| -> Result<Vec<Value>> {
        let at = stack.iter().rposition(|v| matches!(v, Value::Mark)).ok_or_else(|| bad("no pickle mark"))?;
        let items = stack.split_off(at + 1);
        stack.pop();
        Ok(items)
    };
    let u32_le = |s: &[u8]| u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
    loop {
        let op = *b.get(i).ok_or_else(|| bad("truncated pickle"))?;
        i += 1;
        match op {
            0x80 => {
                take(&mut i, 1)?;
            } // PROTO
            0x95 => {
                take(&mut i, 8)?;
            } // FRAME
            b'.' => return pop(&mut stack),
            b'(' => stack.push(Value::Mark),
            b'}' => stack.push(Value::Dict(Vec::new())),
            b']' => stack.push(Value::List(Vec::new())),
            b')' => stack.push(Value::Tuple(Vec::new())),
            b'N' => stack.push(Value::None),
            0x88 => stack.push(Value::Bool(true)),
            0x89 => stack.push(Value::Bool(false)),
            b'K' => {
                let v = take(&mut i, 1)?[0];
                stack.push(Value::Int(v as i64));
            }
            b'M' => {
                let v = take(&mut i, 2)?;
                stack.push(Value::Int(u16::from_le_bytes([v[0], v[1]]) as i64));
            }
            b'J' => {
                let v = take(&mut i, 4)?;
                stack.push(Value::Int(u32_le(v) as i32 as i64));
            }
            0x8a => {
                let n = take(&mut i, 1)?[0] as usize;
                let v = take(&mut i, n)?;
                if n > 8 {
                    return Err(bad("an integer wider than 64 bits"));
                }
                let mut w = [if v.last().is_some_and(|x| x & 0x80 != 0) { 0xff } else { 0 }; 8];
                w[..n].copy_from_slice(v);
                stack.push(Value::Int(i64::from_le_bytes(w)));
            }
            b'G' => {
                let v = take(&mut i, 8)?;
                let mut w = [0u8; 8];
                w.copy_from_slice(v);
                stack.push(Value::Float(f64::from_be_bytes(w)));
            }
            b'X' => {
                let n = u32_le(take(&mut i, 4)?) as usize;
                stack.push(Value::Str(String::from_utf8_lossy(take(&mut i, n)?).into_owned()));
            }
            0x8c => {
                let n = take(&mut i, 1)?[0] as usize;
                stack.push(Value::Str(String::from_utf8_lossy(take(&mut i, n)?).into_owned()));
            }
            b'q' => {
                let k = take(&mut i, 1)?[0] as u32;
                memo.insert(k, stack.last().cloned().ok_or_else(|| bad("empty pickle stack"))?);
            }
            b'r' => {
                let k = u32_le(take(&mut i, 4)?);
                memo.insert(k, stack.last().cloned().ok_or_else(|| bad("empty pickle stack"))?);
            }
            0x94 => {
                let k = memo.len() as u32;
                memo.insert(k, stack.last().cloned().ok_or_else(|| bad("empty pickle stack"))?);
            }
            b'h' => {
                let k = take(&mut i, 1)?[0] as u32;
                stack.push(memo.get(&k).cloned().ok_or_else(|| bad("bad pickle memo"))?);
            }
            b'j' => {
                let k = u32_le(take(&mut i, 4)?);
                stack.push(memo.get(&k).cloned().ok_or_else(|| bad("bad pickle memo"))?);
            }
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(Value::Tuple(items));
            }
            0x85 => {
                let a = pop(&mut stack)?;
                stack.push(Value::Tuple(vec![a]));
            }
            0x86 => {
                let b2 = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Value::Tuple(vec![a, b2]));
            }
            0x87 => {
                let c = pop(&mut stack)?;
                let b2 = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Value::Tuple(vec![a, b2, c]));
            }
            b'a' => {
                let v = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Value::List(l)) => l.push(v),
                    _ => return Err(bad("APPEND to a non-list")),
                }
            }
            b'e' => {
                let items = pop_mark(&mut stack)?;
                match stack.last_mut() {
                    Some(Value::List(l)) => l.extend(items),
                    _ => return Err(bad("APPENDS to a non-list")),
                }
            }
            b's' => {
                let v = pop(&mut stack)?;
                let k = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Value::Dict(d)) => d.push((k, v)),
                    _ => return Err(bad("SETITEM on a non-dict")),
                }
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                let Some(Value::Dict(d)) = stack.last_mut() else { return Err(bad("SETITEMS on a non-dict")) };
                for pair in items.chunks(2) {
                    if let [k, v] = pair {
                        d.push((k.clone(), v.clone()));
                    }
                }
            }
            b'c' => {
                let line = |i: &mut usize| -> Result<String> {
                    let end = b[*i..].iter().position(|&c| c == b'\n').ok_or_else(|| bad("truncated pickle global"))?;
                    let s = String::from_utf8_lossy(&b[*i..*i + end]).into_owned();
                    *i += end + 1;
                    Ok(s)
                };
                let module = line(&mut i)?;
                let name = line(&mut i)?;
                stack.push(global(&module, &name)?);
            }
            0x93 => {
                let name = pop(&mut stack)?;
                let module = pop(&mut stack)?;
                let (Value::Str(m), Value::Str(n)) = (module, name) else { return Err(bad("bad STACK_GLOBAL")) };
                stack.push(global(&m, &n)?);
            }
            b'Q' => {
                // A persistent id: ('storage', storage type, key, location, numel).
                let pid = pop(&mut stack)?;
                let Value::Tuple(t) = pid else { return Err(bad("bad persistent id")) };
                match t.as_slice() {
                    [Value::Str(kind), Value::Global(storage), Value::Str(key), ..] if kind == "storage" => {
                        stack.push(Value::Storage { key: key.clone(), dtype: storage_type(storage)? });
                    }
                    _ => return Err(bad("unsupported persistent id")),
                }
            }
            b'R' => {
                let args = pop(&mut stack)?;
                let f = pop(&mut stack)?;
                let Value::Global(f) = f else { return Err(bad("REDUCE on a non-global")) };
                let Value::Tuple(args) = args else { return Err(bad("REDUCE without a tuple")) };
                stack.push(match f.as_str() {
                    "collections.OrderedDict" => Value::Dict(Vec::new()),
                    "torch._utils._rebuild_tensor_v2" => match args.as_slice() {
                        [Value::Storage { key, dtype }, Value::Int(offset), shape, stride, ..] => Value::Tensor {
                            storage: key.clone(),
                            dtype: *dtype,
                            offset: usize::try_from(*offset).map_err(|_| bad("negative tensor offset"))?,
                            shape: shape.ints().ok_or_else(|| bad("bad tensor shape"))?.into_iter().map(|d| d as usize).collect(),
                            stride: stride.ints().ok_or_else(|| bad("bad tensor stride"))?.into_iter().map(|d| d as usize).collect(),
                        },
                        _ => return Err(bad("bad _rebuild_tensor_v2 arguments")),
                    },
                    other => return Err(bad(format!("{other} is not callable here"))),
                });
            }
            // BUILD: the state set on an OrderedDict or a tensor (none that matters here).
            b'b' => {
                pop(&mut stack)?;
            }
            other => return Err(bad(format!("unsupported pickle opcode 0x{other:02x}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tar with one member, as Python's tarfile writes it (ustar).
    fn tar_with(name: &str, data: &[u8]) -> Vec<u8> {
        let mut h = vec![0u8; 512];
        h[..name.len()].copy_from_slice(name.as_bytes());
        let size = format!("{:011o}\0", data.len());
        h[124..136].copy_from_slice(size.as_bytes());
        h[156] = b'0';
        h[257..263].copy_from_slice(b"ustar\0");
        let mut out = h;
        out.extend_from_slice(data);
        out.resize(out.len().div_ceil(512) * 512, 0);
        out.extend_from_slice(&[0u8; 1024]);
        out
    }

    #[test]
    fn tar_members_are_found_with_their_spans() {
        let dir = std::env::temp_dir().join(format!("oaiy-voice-tar-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.tar");
        std::fs::write(&path, tar_with("./abc_tokenizer.model", b"hello")).unwrap();
        let mut f = File::open(&path).unwrap();
        let m = tar_members(&mut f).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].0, "abc_tokenizer.model");
        assert_eq!((m[0].1.start, m[0].1.len), (512, 5));
        drop(f);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn pickle_reads_a_state_dict_and_refuses_code() {
        // {"w": _rebuild_tensor_v2(storage("0", Float), 0, (2,), (1,))} by hand.
        let mut p = vec![0x80, 2, b'}', b'q', 0];
        p.extend_from_slice(b"X\x01\x00\x00\x00w");
        p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
        p.extend_from_slice(b"((X\x07\x00\x00\x00storagectorch\nLongStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x02tQ");
        p.extend_from_slice(b"K\x00K\x02\x85K\x01\x85\x89tR");
        p.extend_from_slice(b"s.");
        let v = unpickle(&p).unwrap();
        let Value::Dict(items) = v else { panic!("not a dict") };
        assert!(matches!(&items[0].1, Value::Tensor { dtype: StorageType::I64, shape, .. } if shape == &vec![2]));
        let evil = b"\x80\x02cos\nsystem\n.";
        assert!(unpickle(evil).is_err());
    }
}
