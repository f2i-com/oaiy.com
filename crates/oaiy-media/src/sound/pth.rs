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

impl Drop for Value {
    fn drop(&mut self) {
        let mut work = Vec::new();
        match self {
            Value::List(xs) | Value::Tuple(xs) => work.append(xs),
            Value::Dict(pairs) => {
                for (k, v) in std::mem::take(pairs) {
                    work.push(k);
                    work.push(v);
                }
            }
            _ => return,
        }
        while let Some(mut cur) = work.pop() {
            match &mut cur {
                Value::List(xs) | Value::Tuple(xs) => work.append(xs),
                Value::Dict(pairs) => {
                    for (k, v) in std::mem::take(pairs) {
                        work.push(k);
                        work.push(v);
                    }
                }
                _ => {}
            }
        }
    }
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

const MAX_CD_BYTES: u64 = 64 << 20;
const MAX_PICKLE_BYTES: u64 = 64 << 20;
const MAX_PICKLE_NODES: usize = 2_000_000;
const MAX_PICKLE_CLONE_BYTES: usize = 128 << 20;
const MAX_PICKLE_DEPTH: usize = 128;

pub struct Pth {
    file: File,
    file_size: u64,
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
        if cd_start == 0xffff_ffff || count == 0xffff || cd_size == 0xffff_ffff {
            let loc = eocd.checked_sub(20).filter(|&i| u32_at(&tail, i) == 0x0706_4b50).ok_or_else(|| bad("zip64 locator missing"))?;
            let at = u64_at(&tail, loc + 8);
            if at.checked_add(56).is_none_or(|end| end > size) {
                return Err(bad("zip64 end of central directory past end of file"));
            }
            let mut rec = [0u8; 56];
            file.seek(SeekFrom::Start(at))?;
            file.read_exact(&mut rec)?;
            if u32_at(&rec, 0) != 0x0606_4b50 {
                return Err(bad("bad zip64 end of central directory signature"));
            }
            count = u64_at(&rec, 32);
            cd_size = u64_at(&rec, 40);
            cd_start = u64_at(&rec, 48);
        }
        if cd_size > MAX_CD_BYTES || cd_start.checked_add(cd_size).is_none_or(|end| end > size) {
            return Err(bad("bad central directory bounds"));
        }
        let cd_len = usize::try_from(cd_size).map_err(|_| bad("central directory size exceeds usize"))?;
        let mut cd = vec![0u8; cd_len];
        file.seek(SeekFrom::Start(cd_start))?;
        file.read_exact(&mut cd)?;
        if count > (cd.len() / 46) as u64 {
            return Err(bad("bad central directory count"));
        }
        let mut entries = HashMap::new();
        let mut i: usize = 0;
        for _ in 0..count {
            if i.checked_add(46).is_none_or(|end| end > cd.len()) || u32_at(&cd, i) != 0x0201_4b50 {
                return Err(bad("bad central directory"));
            }
            let method = u16_at(&cd, i + 10);
            let mut len = u32_at(&cd, i + 24) as u64;
            let name_len = u16_at(&cd, i + 28) as usize;
            let extra_len = u16_at(&cd, i + 30) as usize;
            let comment_len = u16_at(&cd, i + 32) as usize;
            let mut local = u32_at(&cd, i + 42) as u64;
            let name_start = i + 46;
            let name_end = name_start.checked_add(name_len).ok_or_else(|| bad("bad central directory"))?;
            let extra_end = name_end.checked_add(extra_len).ok_or_else(|| bad("bad central directory"))?;
            let next_i = extra_end.checked_add(comment_len).ok_or_else(|| bad("bad central directory"))?;
            if next_i > cd.len() {
                return Err(bad("bad central directory"));
            }
            let name = String::from_utf8_lossy(&cd[name_start..name_end]).into_owned();
            // Zip64 sizes and offset, in the extra field.
            let mut e = name_end;
            let end = extra_end;
            while e + 4 <= end {
                let (id, n) = (u16_at(&cd, e), u16_at(&cd, e + 2) as usize);
                let field_end = (e + 4).checked_add(n).ok_or_else(|| bad("bad zip extra field"))?;
                if field_end > end {
                    return Err(bad("truncated zip extra field"));
                }
                if id == 1 {
                    let mut f = e + 4;
                    if u32_at(&cd, i + 24) == 0xffff_ffff {
                        if f + 8 > field_end {
                            return Err(bad("truncated zip64 uncompressed size"));
                        }
                        len = u64_at(&cd, f);
                        f += 8;
                    }
                    if u32_at(&cd, i + 20) == 0xffff_ffff {
                        if f + 8 > field_end {
                            return Err(bad("truncated zip64 compressed size"));
                        }
                        f += 8;
                    }
                    if u32_at(&cd, i + 42) == 0xffff_ffff {
                        if f + 8 > field_end {
                            return Err(bad("truncated zip64 local header offset"));
                        }
                        local = u64_at(&cd, f);
                    }
                }
                e = field_end;
            }
            if method != 0 {
                return Err(bad(format!("{name} is compressed")));
            }
            // The data starts after the local header, whose extra field may differ.
            if local.checked_add(30).is_none_or(|lh_end| lh_end > size) {
                return Err(bad(format!("{name}: local header past end of file")));
            }
            let mut lh = [0u8; 30];
            file.seek(SeekFrom::Start(local))?;
            file.read_exact(&mut lh)?;
            if u32_at(&lh, 0) != 0x0403_4b50 {
                return Err(bad(format!("{name}: bad local header signature")));
            }
            let start = local
                .checked_add(30)
                .and_then(|v| v.checked_add(u16_at(&lh, 26) as u64))
                .and_then(|v| v.checked_add(u16_at(&lh, 28) as u64))
                .ok_or_else(|| bad(format!("{name}: local header overflow")))?;
            if start.checked_add(len).is_none_or(|entry_end| entry_end > size) {
                return Err(bad(format!("{name} runs past end of file")));
            }
            entries.insert(name, Entry { start, len });
            i = next_i;
        }
        let pkl = entries.keys().find(|k| k.ends_with("/data.pkl")).cloned().ok_or_else(|| bad("no data.pkl"))?;
        let folder = pkl.trim_end_matches("/data.pkl").to_string();
        let mut out = Self { file, file_size: size, entries, folder, root: Value::None };
        let bytes = out.read_entry_capped(&pkl, MAX_PICKLE_BYTES)?;
        out.root = unpickle(&bytes)?;
        Ok(out)
    }

    fn read_entry_capped(&mut self, name: &str, max_len: u64) -> Result<Vec<u8>> {
        let e = self.entries.get(name).ok_or_else(|| bad(format!("no entry {name}")))?;
        if e.len > max_len {
            return Err(bad(format!("entry {name} exceeds limit ({} > {max_len})", e.len)));
        }
        if e.start.checked_add(e.len).is_none_or(|end| end > self.file_size) {
            return Err(bad(format!("entry {name} runs past end of file")));
        }
        let len = usize::try_from(e.len).map_err(|_| bad(format!("entry {name} length exceeds usize")))?;
        let mut buf = vec![0u8; len];
        self.file.seek(SeekFrom::Start(e.start))?;
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// A tensor value's data, as F32 on `dev`.
    pub fn tensor(&mut self, value: &Value, dev: &Device) -> Result<Tensor> {
        let Value::Tensor { storage, dtype, offset, shape, stride } = value else { return Err(bad("not a tensor")) };
        if shape.len() != stride.len() {
            return Err(bad("tensor shape/stride length mismatch"));
        }
        // Only contiguous row-major tensors (what a state dict holds).
        let mut expect: usize = 1;
        for (&d, &s) in shape.iter().zip(stride).rev() {
            if d > 1 && s != expect {
                return Err(bad("a non-contiguous tensor"));
            }
            expect = expect.checked_mul(d).ok_or_else(|| bad("tensor shape overflow"))?;
        }
        let nonzero = shape
            .iter()
            .filter(|&&d| d != 0)
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| bad("tensor shape overflow"))?;
        let numel = if shape.contains(&0) { 0 } else { nonzero };
        let size = dtype.size_in_bytes();
        let byte_off = offset.checked_mul(size).ok_or_else(|| bad("tensor offset overflow"))?;
        let byte_len = numel.checked_mul(size).ok_or_else(|| bad("tensor byte length overflow"))?;
        let byte_end = byte_off.checked_add(byte_len).ok_or_else(|| bad("tensor byte range overflow"))?;
        let key = format!("{}/data/{storage}", self.folder);
        let e = self.entries.get(&key).ok_or_else(|| bad(format!("no entry {key}")))?;
        if (byte_end as u64) > e.len {
            return Err(bad("a tensor past its storage"));
        }
        let read_start = e.start.checked_add(byte_off as u64).ok_or_else(|| bad("tensor file offset overflow"))?;
        if read_start.checked_add(byte_len as u64).is_none_or(|end| end > self.file_size) {
            return Err(bad("tensor runs past end of file"));
        }
        let mut data = vec![0u8; byte_len];
        self.file.seek(SeekFrom::Start(read_start))?;
        self.file.read_exact(&mut data)?;
        let values: Vec<f32> = match dtype {
            DType::F32 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            DType::F16 => data.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            DType::BF16 => data.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            _ => return Err(bad(format!("unsupported dtype {dtype:?}"))),
        };
        Tensor::from_vec(values, shape.clone(), dev)
    }
}

fn charge_node(budget: &mut usize) -> Result<()> {
    if *budget == 0 {
        return Err(bad("pickle exceeds node budget"));
    }
    *budget -= 1;
    Ok(())
}

fn charge_bytes(budget: &mut usize, n: usize) -> Result<()> {
    if n > *budget {
        return Err(bad("pickle exceeds byte budget"));
    }
    *budget -= n;
    Ok(())
}

fn clone_budgeted(v: &Value, nodes: &mut usize, bytes: &mut usize) -> Result<Value> {
    let mut work = vec![(v, 0usize)];
    while let Some((cur, depth)) = work.pop() {
        if depth > MAX_PICKLE_DEPTH {
            return Err(bad("pickle nesting too deep"));
        }
        charge_node(nodes)?;
        match cur {
            Value::Str(s) | Value::Global(s) => {
                charge_bytes(bytes, s.len())?;
            }
            Value::Storage { key, .. } => {
                charge_bytes(bytes, key.len())?;
            }
            Value::Tensor { storage, shape, stride, .. } => {
                let extra = shape
                    .len()
                    .saturating_add(stride.len())
                    .saturating_mul(std::mem::size_of::<usize>());
                charge_bytes(bytes, storage.len().saturating_add(extra))?;
            }
            Value::List(xs) | Value::Tuple(xs) => {
                charge_bytes(bytes, xs.len().saturating_mul(std::mem::size_of::<Value>()))?;
                for x in xs {
                    work.push((x, depth + 1));
                }
            }
            Value::Dict(pairs) => {
                charge_bytes(bytes, pairs.len().saturating_mul(std::mem::size_of::<(Value, Value)>()))?;
                for (k, val) in pairs {
                    work.push((k, depth + 1));
                    work.push((val, depth + 1));
                }
            }
            _ => {}
        }
    }
    Ok(v.clone())
}

fn storage_dtype(name: &str) -> Result<DType> {
    Ok(match name {
        "torch.FloatStorage" => DType::F32,
        "torch.HalfStorage" => DType::F16,
        "torch.BFloat16Storage" => DType::BF16,
        other => return Err(bad(format!("unsupported storage {other}"))),
    })
}

fn check_depth(depth: usize) -> Result<usize> {
    if depth > MAX_PICKLE_DEPTH {
        return Err(bad("pickle nesting too deep"));
    }
    Ok(depth)
}

/// Interpret a pickle (protocols 2 to 4) with only data-building operations.
fn unpickle(b: &[u8]) -> Result<Value> {
    let mut stack: Vec<(Value, usize)> = Vec::new();
    let mut memo: HashMap<u32, (Value, usize)> = HashMap::new();
    let mut budget = MAX_PICKLE_NODES;
    let mut byte_budget = MAX_PICKLE_CLONE_BYTES;
    let mut i: usize = 0;
    let take = |i: &mut usize, n: usize| -> Result<&[u8]> {
        let end = i.checked_add(n).ok_or_else(|| bad("truncated pickle"))?;
        let s = b.get(*i..end).ok_or_else(|| bad("truncated pickle"))?;
        *i = end;
        Ok(s)
    };
    let pop = |stack: &mut Vec<(Value, usize)>| stack.pop().ok_or_else(|| bad("empty stack"));
    let pop_mark = |stack: &mut Vec<(Value, usize)>| -> Result<Vec<(Value, usize)>> {
        let at = stack.iter().rposition(|(v, _)| matches!(v, Value::Mark)).ok_or_else(|| bad("no mark"))?;
        let items = stack.split_off(at + 1);
        stack.pop();
        Ok(items)
    };
    loop {
        let op = *b.get(i).ok_or_else(|| bad("truncated pickle"))?;
        i += 1;
        charge_node(&mut budget)?;
        match op {
            0x80 => { take(&mut i, 1)?; } // PROTO
            0x95 => { take(&mut i, 8)?; } // FRAME
            b'.' => return Ok(pop(&mut stack)?.0),
            b'(' => stack.push((Value::Mark, 0)),
            b'}' => stack.push((Value::Dict(Vec::new()), 0)),
            b']' => stack.push((Value::List(Vec::new()), 0)),
            b')' => stack.push((Value::Tuple(Vec::new()), 0)),
            b'N' => stack.push((Value::None, 0)),
            0x88 => stack.push((Value::Bool(true), 0)),
            0x89 => stack.push((Value::Bool(false), 0)),
            b'K' => { let v = take(&mut i, 1)?[0]; stack.push((Value::Int(v as i64), 0)); }
            b'M' => { let v = take(&mut i, 2)?; stack.push((Value::Int(u16::from_le_bytes([v[0], v[1]]) as i64), 0)); }
            b'J' => { let v = take(&mut i, 4)?; stack.push((Value::Int(i32::from_le_bytes(v.try_into().unwrap()) as i64), 0)); }
            0x8a => {
                let n = take(&mut i, 1)?[0] as usize;
                let v = take(&mut i, n)?;
                if n > 8 { return Err(bad("an integer wider than 64 bits")); }
                let mut w = [if v.last().is_some_and(|x| x & 0x80 != 0) { 0xff } else { 0 }; 8];
                w[..n].copy_from_slice(v);
                stack.push((Value::Int(i64::from_le_bytes(w)), 0));
            }
            b'G' => { let v = take(&mut i, 8)?; stack.push((Value::Float(f64::from_be_bytes(v.try_into().unwrap())), 0)); }
            b'X' => {
                let n = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()) as usize;
                let raw = take(&mut i, n)?;
                charge_bytes(&mut byte_budget, raw.len())?;
                let s = String::from_utf8_lossy(raw).into_owned();
                charge_bytes(&mut byte_budget, s.len().saturating_sub(raw.len()))?;
                stack.push((Value::Str(s), 0));
            }
            0x8c => {
                let n = take(&mut i, 1)?[0] as usize;
                let raw = take(&mut i, n)?;
                charge_bytes(&mut byte_budget, raw.len())?;
                let s = String::from_utf8_lossy(raw).into_owned();
                charge_bytes(&mut byte_budget, s.len().saturating_sub(raw.len()))?;
                stack.push((Value::Str(s), 0));
            }
            b'q' => {
                let k = take(&mut i, 1)?[0] as u32;
                let (top, depth) = stack.last().ok_or_else(|| bad("empty stack"))?;
                memo.insert(k, (clone_budgeted(top, &mut budget, &mut byte_budget)?, *depth));
            }
            b'r' => {
                let k = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap());
                let (top, depth) = stack.last().ok_or_else(|| bad("empty stack"))?;
                memo.insert(k, (clone_budgeted(top, &mut budget, &mut byte_budget)?, *depth));
            }
            0x94 => {
                let k = memo.len() as u32;
                let (top, depth) = stack.last().ok_or_else(|| bad("empty stack"))?;
                memo.insert(k, (clone_budgeted(top, &mut budget, &mut byte_budget)?, *depth));
            }
            b'h' => {
                let k = take(&mut i, 1)?[0] as u32;
                let (val, depth) = memo.get(&k).ok_or_else(|| bad("bad memo"))?;
                stack.push((clone_budgeted(val, &mut budget, &mut byte_budget)?, *depth));
            }
            b'j' => {
                let k = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap());
                let (val, depth) = memo.get(&k).ok_or_else(|| bad("bad memo"))?;
                stack.push((clone_budgeted(val, &mut budget, &mut byte_budget)?, *depth));
            }
            b't' => {
                let items = pop_mark(&mut stack)?;
                let depth = match items.iter().map(|(_, d)| *d).max() {
                    Some(m) => check_depth(m.saturating_add(1))?,
                    None => 0,
                };
                stack.push((Value::Tuple(items.into_iter().map(|(v, _)| v).collect()), depth));
            }
            0x85 => {
                let (a, da) = pop(&mut stack)?;
                let depth = check_depth(da.saturating_add(1))?;
                stack.push((Value::Tuple(vec![a]), depth));
            }
            0x86 => {
                let (b2, db) = pop(&mut stack)?;
                let (a, da) = pop(&mut stack)?;
                let depth = check_depth(da.max(db).saturating_add(1))?;
                stack.push((Value::Tuple(vec![a, b2]), depth));
            }
            0x87 => {
                let (c, dc) = pop(&mut stack)?;
                let (b2, db) = pop(&mut stack)?;
                let (a, da) = pop(&mut stack)?;
                let depth = check_depth(da.max(db).max(dc).saturating_add(1))?;
                stack.push((Value::Tuple(vec![a, b2, c]), depth));
            }
            b'a' => {
                let (v, dv) = pop(&mut stack)?;
                match stack.last_mut() {
                    Some((Value::List(l), dl)) => {
                        *dl = check_depth((*dl).max(dv.saturating_add(1)))?;
                        l.push(v);
                    }
                    _ => return Err(bad("APPEND to a non-list")),
                }
            }
            b'e' => {
                let items = pop_mark(&mut stack)?;
                let max_d = items.iter().map(|(_, d)| *d).max();
                match stack.last_mut() {
                    Some((Value::List(l), dl)) => {
                        if let Some(m) = max_d {
                            *dl = check_depth((*dl).max(m.saturating_add(1)))?;
                        }
                        l.extend(items.into_iter().map(|(v, _)| v));
                    }
                    _ => return Err(bad("APPENDS to a non-list")),
                }
            }
            b's' => {
                let (v, dv) = pop(&mut stack)?;
                let (k, dk) = pop(&mut stack)?;
                match stack.last_mut() {
                    Some((Value::Dict(d), dd)) => {
                        *dd = check_depth((*dd).max(dk.max(dv).saturating_add(1)))?;
                        d.push((k, v));
                    }
                    _ => return Err(bad("SETITEM on a non-dict")),
                }
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                if items.len() % 2 != 0 {
                    return Err(bad("SETITEMS with odd item count"));
                }
                let max_d = items.iter().map(|(_, d)| *d).max();
                let Some((Value::Dict(d), dd)) = stack.last_mut() else { return Err(bad("SETITEMS on a non-dict")) };
                if let Some(m) = max_d {
                    *dd = check_depth((*dd).max(m.saturating_add(1)))?;
                }
                let mut it = items.into_iter().map(|(v, _)| v);
                while let (Some(k), Some(v)) = (it.next(), it.next()) {
                    d.push((k, v));
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
                stack.push((global(&module, &name)?, 0));
            }
            0x93 => {
                let (name, _) = pop(&mut stack)?;
                let (module, _) = pop(&mut stack)?;
                let (Value::Str(m), Value::Str(n)) = (&module, &name) else { return Err(bad("bad STACK_GLOBAL")) };
                stack.push((global(m, n)?, 0));
            }
            b'Q' => {
                // A persistent id: ('storage', storage type, key, location, numel).
                let (pid, _) = pop(&mut stack)?;
                let Value::Tuple(t) = &pid else { return Err(bad("bad persistent id")) };
                match t.as_slice() {
                    [Value::Str(kind), Value::Global(storage), Value::Str(key), ..] if kind == "storage" => {
                        charge_bytes(&mut byte_budget, key.len())?;
                        stack.push((Value::Storage { key: key.clone(), dtype: storage_dtype(storage)? }, 0));
                    }
                    _ => return Err(bad("unsupported persistent id")),
                }
            }
            b'R' => {
                let (args, _) = pop(&mut stack)?;
                let (f, _) = pop(&mut stack)?;
                let Value::Global(f) = &f else { return Err(bad("REDUCE on a non-global")) };
                let Value::Tuple(args) = &args else { return Err(bad("REDUCE without a tuple")) };
                let out = match f.as_str() {
                    "collections.OrderedDict" => Value::Dict(Vec::new()),
                    "torch._utils._rebuild_tensor_v2" => match args.as_slice() {
                        [Value::Storage { key, dtype }, Value::Int(offset), shape, stride, ..] => {
                            let offset = usize::try_from(*offset).map_err(|_| bad("negative tensor offset"))?;
                            let shape: Vec<usize> = shape
                                .ints()
                                .ok_or_else(|| bad("bad tensor shape"))?
                                .into_iter()
                                .map(|d| usize::try_from(d).map_err(|_| bad("negative tensor dim")))
                                .collect::<Result<Vec<_>>>()?;
                            let stride: Vec<usize> = stride
                                .ints()
                                .ok_or_else(|| bad("bad tensor stride"))?
                                .into_iter()
                                .map(|d| usize::try_from(d).map_err(|_| bad("negative tensor stride")))
                                .collect::<Result<Vec<_>>>()?;
                            if shape.len() != stride.len() {
                                return Err(bad("tensor shape/stride length mismatch"));
                            }
                            charge_bytes(&mut byte_budget, key.len())?;
                            Value::Tensor {
                                storage: key.clone(),
                                dtype: *dtype,
                                offset,
                                shape,
                                stride,
                            }
                        }
                        _ => return Err(bad("bad _rebuild_tensor_v2 arguments")),
                    },
                    other => return Err(bad(format!("{other} is not callable here"))),
                };
                stack.push((out, 0));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pickle_rejects_clone_bomb_negative_dims_and_rank_mismatch() {
        // Exponential clone bomb via memo.
        let mut bomb = vec![0x80, 2, b'K', 1, b'q', 0];
        for k in 1u8..=25 {
            bomb.extend_from_slice(&[b'h', k - 1, b'h', k - 1, 0x86, b'q', k]);
        }
        bomb.push(b'.');
        assert!(unpickle(&bomb).is_err(), "clone bomb must fail");

        // Negative tensor offset must fail.
        let mut neg_off = vec![0x80, 2];
        neg_off.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
        neg_off.extend_from_slice(b"((X\x07\x00\x00\x00storagectorch\nFloatStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x01tQ");
        neg_off.extend_from_slice(b"J\xff\xff\xff\xffK\x01\x85K\x01\x85\x89tR.");
        assert!(unpickle(&neg_off).is_err(), "negative offset must fail");

        // Shape/stride rank mismatch must fail.
        let mut mismatch = vec![0x80, 2];
        mismatch.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
        mismatch.extend_from_slice(b"((X\x07\x00\x00\x00storagectorch\nFloatStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x02tQ");
        mismatch.extend_from_slice(b"K\x00K\x01K\x02\x86K\x01\x85\x89tR.");
        assert!(unpickle(&mismatch).is_err(), "shape/stride rank mismatch must fail");
    }

    #[test]
    fn pickle_rejects_deep_non_memo_nesting_without_stack_overflow() {
        // 1500 chained TUPLE1 (0x85) opcodes without any memo ops must fail cleanly via depth cap, never stack overflow.
        let mut deep_tuple = vec![0x80, 2, b'N'];
        deep_tuple.resize(deep_tuple.len() + 1500, 0x85);
        deep_tuple.push(b'.');
        let err = unpickle(&deep_tuple).expect_err("deep TUPLE1 chain must fail");
        assert!(err.to_string().contains("too deep"), "{err}");

        // Deep list nesting via APPEND (b'a') without memo ops must also fail cleanly.
        let mut deep_list = vec![0x80, 2];
        for _ in 0..=MAX_PICKLE_DEPTH + 10 {
            deep_list.push(b']');
        }
        for _ in 0..=MAX_PICKLE_DEPTH + 9 {
            deep_list.push(b'a');
        }
        deep_list.push(b'.');
        assert!(unpickle(&deep_list).is_err(), "deep list APPEND chain must fail");

        // Exact boundary MAX_PICKLE_DEPTH succeeds and drops iteratively.
        let mut at_limit = vec![0x80, 2, b'N'];
        at_limit.resize(at_limit.len() + MAX_PICKLE_DEPTH, 0x85);
        at_limit.push(b'.');
        assert!(unpickle(&at_limit).is_ok(), "nesting at MAX_PICKLE_DEPTH should succeed");
    }

    #[test]
    fn pickle_rejects_memo_string_clone_oom_bomb() {
        // Store a 1 MiB string in memo slot 0, then clone it 200 times via BINGET (h 0) into a tuple.
        // Total node count is tiny (~210 << MAX_PICKLE_NODES), but total cloned bytes (~201 MiB)
        // exceeds MAX_PICKLE_CLONE_BYTES (128 MiB) and must fail cleanly before OOMing.
        let payload_len: u32 = 1 << 20;
        let mut bomb = Vec::with_capacity(payload_len as usize + 512);
        bomb.extend_from_slice(&[0x80, 2, b'(', b'X']);
        bomb.extend_from_slice(&payload_len.to_le_bytes());
        bomb.resize(bomb.len() + payload_len as usize, b'a');
        bomb.extend_from_slice(&[b'q', 0]);
        for _ in 0..200 {
            bomb.extend_from_slice(&[b'h', 0]);
        }
        bomb.extend_from_slice(&[b't', b'.']);
        let err = unpickle(&bomb).expect_err("memo string clone bomb must fail");
        assert!(err.to_string().contains("byte budget"), "{err}");
    }

    #[test]
    fn zip_rejects_oversized_cd_and_truncated_zip64_extra() {
        let dir = std::env::temp_dir().join(format!("oaiy-media-pth-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Central directory size past file end.
        let mut bad_cd = vec![0u8; 22];
        bad_cd[0..4].copy_from_slice(&0x0605_4b50u32.to_le_bytes());
        bad_cd[10..12].copy_from_slice(&1u16.to_le_bytes());
        bad_cd[12..16].copy_from_slice(&(100u32 << 20).to_le_bytes());
        let p1 = dir.join("bad_cd.pth");
        std::fs::write(&p1, &bad_cd).unwrap();
        assert!(Pth::open(&p1).is_err());

        // Truncated Zip64 extra field (claims 0xffffffff size, extra len = 4 < 8).
        let mut zip = Vec::new();
        zip.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        zip.resize(30, 0);
        let cd_start = zip.len() as u32;
        let mut cd = vec![0u8; 46];
        cd[0..4].copy_from_slice(&0x0201_4b50u32.to_le_bytes());
        cd[24..28].copy_from_slice(&0xffff_ffffu32.to_le_bytes());
        cd[28..30].copy_from_slice(&1u16.to_le_bytes());
        cd[30..32].copy_from_slice(&8u16.to_le_bytes());
        cd.push(b'a');
        cd.extend_from_slice(&1u16.to_le_bytes());
        cd.extend_from_slice(&4u16.to_le_bytes());
        cd.extend_from_slice(&[0u8; 4]);
        let cd_size = cd.len() as u32;
        zip.extend_from_slice(&cd);
        let mut eocd = vec![0u8; 22];
        eocd[0..4].copy_from_slice(&0x0605_4b50u32.to_le_bytes());
        eocd[10..12].copy_from_slice(&1u16.to_le_bytes());
        eocd[12..16].copy_from_slice(&cd_size.to_le_bytes());
        eocd[16..20].copy_from_slice(&cd_start.to_le_bytes());
        zip.extend_from_slice(&eocd);
        let p2 = dir.join("bad_z64.pth");
        std::fs::write(&p2, &zip).unwrap();
        assert!(Pth::open(&p2).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
