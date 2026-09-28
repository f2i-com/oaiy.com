//! A PyTorch `.pth` checkpoint (a zip of a pickle and its tensors' storages) read
//! without running it. The pickle is interpreted, not executed: only plain data
//! (dicts, lists, tuples, numbers, strings), `OrderedDict`, tensor storages and
//! `torch._utils._rebuild_tensor_v2` are understood, and any other global is an
//! error. This is what `torch.load(weights_only=True)` allows, and it is enough
//! for a state dict and its constructor arguments.
use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

fn bad(s: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(format!("pth: {}", s.into()))
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
    /// A tensor: its storage's zip entry and dtype, element offset, shape and stride.
    Tensor { storage: String, dtype: DType, offset: usize, shape: Vec<usize>, stride: Vec<usize> },
    /// A storage (a persistent id), before a tensor is built on it.
    Storage { key: String, dtype: DType },
    /// A global the pickle may call: only the three allowed ones get this far.
    Global(String),
    Mark,
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Dict(items) => items.iter().find(|(k, _)| matches!(k, Value::Str(s) if s == key)).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn ints(&self) -> Option<Vec<i64>> {
        match self {
            Value::List(v) | Value::Tuple(v) => v.iter().map(Value::as_i64).collect(),
            _ => None,
        }
    }
}

/// One stored (uncompressed) zip entry: where its bytes start and how many.
struct Entry {
    start: u64,
    len: u64,
}

pub struct Pth {
    file: File,
    entries: HashMap<String, Entry>,
    /// The archive's folder (its entries are `<folder>/data.pkl`, `<folder>/data/<key>`).
    folder: String,
    pub root: Value,
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}
fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}
fn u64_at(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

impl Pth {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let size = file.metadata()?.len();
        // The end of central directory record (and the zip64 one, past 4 GB).
        let tail_len = size.min(65_536 + 22);
        file.seek(SeekFrom::Start(size - tail_len))?;
        let mut tail = vec![0u8; tail_len as usize];
        file.read_exact(&mut tail)?;
        let eocd = (0..tail.len().saturating_sub(21)).rev().find(|&i| u32_at(&tail, i) == 0x0605_4b50).ok_or_else(|| bad(format!("{}: not a zip (a legacy torch.save file?)", path.display())))?;
        let mut count = u16_at(&tail, eocd + 10) as u64;
        let mut cd_size = u32_at(&tail, eocd + 12) as u64;
        let mut cd_start = u32_at(&tail, eocd + 16) as u64;
        if cd_start == 0xffff_ffff || count == 0xffff {
            let loc = eocd.checked_sub(20).filter(|&i| u32_at(&tail, i) == 0x0706_4b50).ok_or_else(|| bad("zip64 locator missing"))?;
            let at = u64_at(&tail, loc + 8);
            let mut rec = [0u8; 56];
            file.seek(SeekFrom::Start(at))?;
            file.read_exact(&mut rec)?;
            count = u64_at(&rec, 32);
            cd_size = u64_at(&rec, 40);
            cd_start = u64_at(&rec, 48);
        }
        let mut cd = vec![0u8; cd_size as usize];
        file.seek(SeekFrom::Start(cd_start))?;
        file.read_exact(&mut cd)?;
        let mut entries = HashMap::new();
        let mut i = 0;
        for _ in 0..count {
            if u32_at(&cd, i) != 0x0201_4b50 {
                return Err(bad("bad central directory"));
            }
            let method = u16_at(&cd, i + 10);
            let mut len = u32_at(&cd, i + 24) as u64;
            let name_len = u16_at(&cd, i + 28) as usize;
            let extra_len = u16_at(&cd, i + 30) as usize;
            let comment_len = u16_at(&cd, i + 32) as usize;
            let mut local = u32_at(&cd, i + 42) as u64;
            let name = String::from_utf8_lossy(&cd[i + 46..i + 46 + name_len]).into_owned();
            // Zip64 sizes and offset, in the extra field.
            let mut e = i + 46 + name_len;
            let end = e + extra_len;
            while e + 4 <= end {
                let (id, n) = (u16_at(&cd, e), u16_at(&cd, e + 2) as usize);
                if id == 1 {
                    let mut f = e + 4;
                    if u32_at(&cd, i + 24) == 0xffff_ffff { len = u64_at(&cd, f); f += 8; }
                    if u32_at(&cd, i + 20) == 0xffff_ffff { f += 8; }
                    if u32_at(&cd, i + 42) == 0xffff_ffff { local = u64_at(&cd, f); }
                }
                e += 4 + n;
            }
            if method != 0 {
                return Err(bad(format!("{name} is compressed")));
            }
            // The data starts after the local header, whose extra field may differ.
            let mut lh = [0u8; 30];
            file.seek(SeekFrom::Start(local))?;
            file.read_exact(&mut lh)?;
            let start = local + 30 + u16_at(&lh, 26) as u64 + u16_at(&lh, 28) as u64;
            entries.insert(name, Entry { start, len });
            i = end + comment_len;
        }
        let pkl = entries.keys().find(|k| k.ends_with("/data.pkl")).cloned().ok_or_else(|| bad("no data.pkl"))?;
        let folder = pkl.trim_end_matches("/data.pkl").to_string();
        let mut out = Self { file, entries, folder, root: Value::None };
        let bytes = out.read_entry(&pkl)?;
        out.root = unpickle(&bytes)?;
        Ok(out)
    }

    fn read_entry(&mut self, name: &str) -> Result<Vec<u8>> {
        let e = self.entries.get(name).ok_or_else(|| bad(format!("no entry {name}")))?;
        let mut buf = vec![0u8; e.len as usize];
        self.file.seek(SeekFrom::Start(e.start))?;
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// A tensor value's data, as F32 on `dev`.
    pub fn tensor(&mut self, value: &Value, dev: &Device) -> Result<Tensor> {
        let Value::Tensor { storage, dtype, offset, shape, stride } = value else { return Err(bad("not a tensor")) };
        // Only contiguous row-major tensors (what a state dict holds).
        let mut expect = 1;
        for (d, s) in shape.iter().zip(stride).rev() {
            if *d > 1 && *s != expect {
                return Err(bad("a non-contiguous tensor"));
            }
            expect *= d;
        }
        let numel: usize = shape.iter().product();
        let size = dtype.size_in_bytes();
        let bytes = self.read_entry(&format!("{}/data/{storage}", self.folder))?;
        let data = bytes.get(offset * size..(offset + numel) * size).ok_or_else(|| bad("a tensor past its storage"))?;
        let values: Vec<f32> = match dtype {
            DType::F32 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            DType::F16 => data.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            DType::BF16 => data.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            _ => return Err(bad(format!("unsupported dtype {dtype:?}"))),
        };
        Tensor::from_vec(values, shape.clone(), dev)
    }
}

fn storage_dtype(name: &str) -> Result<DType> {
    Ok(match name {
        "torch.FloatStorage" => DType::F32,
        "torch.HalfStorage" => DType::F16,
        "torch.BFloat16Storage" => DType::BF16,
        other => return Err(bad(format!("unsupported storage {other}"))),
    })
}

/// Interpret a pickle (protocols 2 to 4) with only data-building operations.
fn unpickle(b: &[u8]) -> Result<Value> {
    let mut stack: Vec<Value> = Vec::new();
    let mut memo: HashMap<u32, Value> = HashMap::new();
    let mut i = 0;
    let take = |i: &mut usize, n: usize| -> Result<&[u8]> {
        let s = b.get(*i..*i + n).ok_or_else(|| bad("truncated pickle"))?;
        *i += n;
        Ok(s)
    };
    let pop = |stack: &mut Vec<Value>| stack.pop().ok_or_else(|| bad("empty stack"));
    let pop_mark = |stack: &mut Vec<Value>| -> Result<Vec<Value>> {
        let at = stack.iter().rposition(|v| matches!(v, Value::Mark)).ok_or_else(|| bad("no mark"))?;
        let items = stack.split_off(at + 1);
        stack.pop();
        Ok(items)
    };
    loop {
        let op = *b.get(i).ok_or_else(|| bad("truncated pickle"))?;
        i += 1;
        match op {
            0x80 => { take(&mut i, 1)?; } // PROTO
            0x95 => { take(&mut i, 8)?; } // FRAME
            b'.' => return pop(&mut stack),
            b'(' => stack.push(Value::Mark),
            b'}' => stack.push(Value::Dict(Vec::new())),
            b']' => stack.push(Value::List(Vec::new())),
            b')' => stack.push(Value::Tuple(Vec::new())),
            b'N' => stack.push(Value::None),
            0x88 => stack.push(Value::Bool(true)),
            0x89 => stack.push(Value::Bool(false)),
            b'K' => { let v = take(&mut i, 1)?[0]; stack.push(Value::Int(v as i64)); }
            b'M' => { let v = take(&mut i, 2)?; stack.push(Value::Int(u16::from_le_bytes([v[0], v[1]]) as i64)); }
            b'J' => { let v = take(&mut i, 4)?; stack.push(Value::Int(i32::from_le_bytes(v.try_into().unwrap()) as i64)); }
            0x8a => {
                let n = take(&mut i, 1)?[0] as usize;
                let v = take(&mut i, n)?;
                if n > 8 { return Err(bad("an integer wider than 64 bits")); }
                let mut w = [if v.last().is_some_and(|x| x & 0x80 != 0) { 0xff } else { 0 }; 8];
                w[..n].copy_from_slice(v);
                stack.push(Value::Int(i64::from_le_bytes(w)));
            }
            b'G' => { let v = take(&mut i, 8)?; stack.push(Value::Float(f64::from_be_bytes(v.try_into().unwrap()))); }
            b'X' => {
                let n = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()) as usize;
                stack.push(Value::Str(String::from_utf8_lossy(take(&mut i, n)?).into_owned()));
            }
            0x8c => {
                let n = take(&mut i, 1)?[0] as usize;
                stack.push(Value::Str(String::from_utf8_lossy(take(&mut i, n)?).into_owned()));
            }
            b'q' => { let k = take(&mut i, 1)?[0] as u32; memo.insert(k, stack.last().cloned().ok_or_else(|| bad("empty stack"))?); }
            b'r' => { let k = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()); memo.insert(k, stack.last().cloned().ok_or_else(|| bad("empty stack"))?); }
            0x94 => { let k = memo.len() as u32; memo.insert(k, stack.last().cloned().ok_or_else(|| bad("empty stack"))?); }
            b'h' => { let k = take(&mut i, 1)?[0] as u32; stack.push(memo.get(&k).cloned().ok_or_else(|| bad("bad memo"))?); }
            b'j' => { let k = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()); stack.push(memo.get(&k).cloned().ok_or_else(|| bad("bad memo"))?); }
            b't' => { let items = pop_mark(&mut stack)?; stack.push(Value::Tuple(items)); }
            0x85 => { let a = pop(&mut stack)?; stack.push(Value::Tuple(vec![a])); }
            0x86 => { let b2 = pop(&mut stack)?; let a = pop(&mut stack)?; stack.push(Value::Tuple(vec![a, b2])); }
            0x87 => { let c = pop(&mut stack)?; let b2 = pop(&mut stack)?; let a = pop(&mut stack)?; stack.push(Value::Tuple(vec![a, b2, c])); }
            b'a' => { let v = pop(&mut stack)?; match stack.last_mut() { Some(Value::List(l)) => l.push(v), _ => return Err(bad("APPEND to a non-list")) } }
            b'e' => { let items = pop_mark(&mut stack)?; match stack.last_mut() { Some(Value::List(l)) => l.extend(items), _ => return Err(bad("APPENDS to a non-list")) } }
            b's' => {
                let v = pop(&mut stack)?;
                let k = pop(&mut stack)?;
                match stack.last_mut() { Some(Value::Dict(d)) => d.push((k, v)), _ => return Err(bad("SETITEM on a non-dict")) }
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                let Some(Value::Dict(d)) = stack.last_mut() else { return Err(bad("SETITEMS on a non-dict")) };
                for pair in items.chunks(2) {
                    if let [k, v] = pair { d.push((k.clone(), v.clone())); }
                }
            }
            b'c' => {
                let line = |i: &mut usize| -> Result<String> {
                    let end = b[*i..].iter().position(|&c| c == b'\n').ok_or_else(|| bad("truncated global"))?;
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
                        stack.push(Value::Storage { key: key.clone(), dtype: storage_dtype(storage)? });
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
                            offset: *offset as usize,
                            shape: shape.ints().ok_or_else(|| bad("bad tensor shape"))?.into_iter().map(|d| d as usize).collect(),
                            stride: stride.ints().ok_or_else(|| bad("bad tensor stride"))?.into_iter().map(|d| d as usize).collect(),
                        },
                        _ => return Err(bad("bad _rebuild_tensor_v2 arguments")),
                    },
                    other => return Err(bad(format!("{other} is not callable here"))),
                });
            }
            // BUILD: the state set on an OrderedDict or a tensor (none that matters here).
            b'b' => { pop(&mut stack)?; }
            other => return Err(bad(format!("unsupported pickle opcode 0x{other:02x}"))),
        }
    }
}

/// The only globals a checkpoint may name.
fn global(module: &str, name: &str) -> Result<Value> {
    let full = format!("{module}.{name}");
    match full.as_str() {
        "collections.OrderedDict" | "torch._utils._rebuild_tensor_v2" | "torch.FloatStorage" | "torch.HalfStorage" | "torch.BFloat16Storage" => Ok(Value::Global(full)),
        _ => Err(bad(format!("refusing {full}: only tensors and plain data are read"))),
    }
}
