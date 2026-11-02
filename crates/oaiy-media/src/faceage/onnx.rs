//! Just enough of ONNX's protobuf to read a small convolutional network as published: its nodes (operator, inputs,
//! outputs, attributes), its inputs and outputs, and its initialisers (float and 64-bit integer tensors). The file is
//! only read: the network is run by `yunet`, which takes the operators it knows and refuses any other.
use std::collections::HashMap;

/// The largest file read: a face detector is a fraction of this.
const MAX_FILE: usize = 64 << 20;

#[derive(Clone, Debug, PartialEq)]
pub enum Attr {
    Int(i64),
    Ints(Vec<i64>),
    Float(f32),
    Floats(Vec<f32>),
    Text(String),
    Other,
}

#[derive(Clone, Debug, Default)]
pub struct Node {
    pub op: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, Attr>,
}

impl Node {
    pub fn ints(&self, name: &str) -> Option<&[i64]> {
        match self.attrs.get(name) {
            Some(Attr::Ints(v)) => Some(v),
            _ => None,
        }
    }

    pub fn int(&self, name: &str) -> Option<i64> {
        match self.attrs.get(name) {
            Some(Attr::Int(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn text(&self, name: &str) -> Option<&str> {
        match self.attrs.get(name) {
            Some(Attr::Text(v)) => Some(v),
            _ => None,
        }
    }
}

/// An initialiser: its shape and its values, as floats or as 64-bit integers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Initializer {
    pub dims: Vec<usize>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub initializers: HashMap<String, Initializer>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

type R<T> = Result<T, String>;

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn done(&self) -> bool {
        self.at >= self.bytes.len()
    }

    fn varint(&mut self) -> R<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self.bytes.get(self.at).ok_or("the ONNX file ends inside a number")?;
            self.at += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("an ONNX number longer than ten bytes".into())
    }

    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.bytes.len()).ok_or("the ONNX file ends inside a field")?;
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn delimited(&mut self) -> R<&'a [u8]> {
        let n = usize::try_from(self.varint()?).map_err(|_| "an ONNX field too large")?;
        self.take(n)
    }

    /// The next field's number and wire type, or None at the end.
    fn key(&mut self) -> R<Option<(u64, u8)>> {
        if self.done() {
            return Ok(None);
        }
        let key = self.varint()?;
        Ok(Some((key >> 3, (key & 7) as u8)))
    }

    fn skip(&mut self, wire: u8) -> R<()> {
        match wire {
            0 => self.varint().map(|_| ()),
            1 => self.take(8).map(|_| ()),
            2 => self.delimited().map(|_| ()),
            5 => self.take(4).map(|_| ()),
            other => Err(format!("an ONNX field of wire type {other}, which this reader does not take")),
        }
    }

    fn text(&mut self) -> R<String> {
        String::from_utf8(self.delimited()?.to_vec()).map_err(|_| "an ONNX name that is not UTF-8".into())
    }

    /// Repeated integers, packed or one a field.
    fn ints(&mut self, wire: u8, out: &mut Vec<i64>) -> R<()> {
        if wire == 2 {
            let mut inner = Reader::new(self.delimited()?);
            while !inner.done() {
                out.push(inner.varint()? as i64);
            }
        } else {
            out.push(self.varint()? as i64);
        }
        Ok(())
    }

    /// Repeated floats, packed or one a field.
    fn floats(&mut self, wire: u8, out: &mut Vec<f32>) -> R<()> {
        let bytes = if wire == 2 { self.delimited()? } else { self.take(4)? };
        if bytes.len() % 4 != 0 {
            return Err("ONNX floats that are not whole".into());
        }
        out.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
        Ok(())
    }
}

fn attribute(bytes: &[u8]) -> R<(String, Attr)> {
    let mut r = Reader::new(bytes);
    let (mut name, mut attr) = (String::new(), Attr::Other);
    let (mut ints, mut floats) = (Vec::new(), Vec::new());
    while let Some((field, wire)) = r.key()? {
        match (field, wire) {
            (1, 2) => name = r.text()?,
            (2, 5) => attr = Attr::Float(f32::from_le_bytes(r.take(4)?.try_into().expect("four bytes"))),
            (3, 0) => attr = Attr::Int(r.varint()? as i64),
            (4, 2) => attr = Attr::Text(String::from_utf8_lossy(r.delimited()?).into_owned()),
            (7, _) => r.floats(wire, &mut floats)?,
            (8, _) => r.ints(wire, &mut ints)?,
            _ => r.skip(wire)?,
        }
    }
    if !ints.is_empty() {
        attr = Attr::Ints(ints);
    } else if !floats.is_empty() {
        attr = Attr::Floats(floats);
    }
    Ok((name, attr))
}

fn node(bytes: &[u8]) -> R<Node> {
    let mut r = Reader::new(bytes);
    let mut n = Node::default();
    while let Some((field, wire)) = r.key()? {
        match (field, wire) {
            (1, 2) => n.inputs.push(r.text()?),
            (2, 2) => n.outputs.push(r.text()?),
            (3, 2) => n.name = r.text()?,
            (4, 2) => n.op = r.text()?,
            (5, 2) => {
                let (name, attr) = attribute(r.delimited()?)?;
                n.attrs.insert(name, attr);
            }
            _ => r.skip(wire)?,
        }
    }
    Ok(n)
}

/// ONNX's data types this reader keeps.
const FLOAT: i64 = 1;
const INT64: i64 = 7;

fn tensor(bytes: &[u8]) -> R<(String, Initializer)> {
    let mut r = Reader::new(bytes);
    let (mut name, mut dims, mut kind, mut raw) = (String::new(), Vec::new(), 0i64, None);
    let mut t = Initializer::default();
    while let Some((field, wire)) = r.key()? {
        match (field, wire) {
            (1, _) => r.ints(wire, &mut dims)?,
            (2, 0) => kind = r.varint()? as i64,
            (4, _) => r.floats(wire, &mut t.floats)?,
            (7, _) => r.ints(wire, &mut t.ints)?,
            (8, 2) => name = r.text()?,
            (9, 2) => raw = Some(r.delimited()?),
            (13, _) | (14, _) => return Err(format!("{name}: an ONNX tensor kept outside the file, which this reader does not follow")),
            _ => r.skip(wire)?,
        }
    }
    t.dims = dims.iter().map(|&d| usize::try_from(d).map_err(|_| format!("{name}: a negative dimension"))).collect::<R<_>>()?;
    let count: usize = t.dims.iter().product();
    if let Some(raw) = raw {
        match kind {
            FLOAT if raw.len() == count * 4 => t.floats = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            INT64 if raw.len() == count * 8 => t.ints = raw.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().expect("eight bytes"))).collect(),
            FLOAT | INT64 => return Err(format!("{name}: its data is not its shape's size")),
            _ => {}
        }
    }
    let have = if kind == FLOAT { t.floats.len() } else if kind == INT64 { t.ints.len() } else { count };
    if have != count {
        return Err(format!("{name}: {have} values for a shape of {count}"));
    }
    Ok((name, t))
}

fn value_name(bytes: &[u8]) -> R<String> {
    let mut r = Reader::new(bytes);
    let mut name = String::new();
    while let Some((field, wire)) = r.key()? {
        if (field, wire) == (1, 2) {
            name = r.text()?;
        } else {
            r.skip(wire)?;
        }
    }
    Ok(name)
}

fn graph(bytes: &[u8]) -> R<Graph> {
    let mut r = Reader::new(bytes);
    let mut g = Graph::default();
    while let Some((field, wire)) = r.key()? {
        match (field, wire) {
            (1, 2) => g.nodes.push(node(r.delimited()?)?),
            (5, 2) => {
                let (name, t) = tensor(r.delimited()?)?;
                g.initializers.insert(name, t);
            }
            (11, 2) => g.inputs.push(value_name(r.delimited()?)?),
            (12, 2) => g.outputs.push(value_name(r.delimited()?)?),
            _ => r.skip(wire)?,
        }
    }
    Ok(g)
}

/// An ONNX model's graph.
pub fn read(bytes: &[u8]) -> R<Graph> {
    if bytes.len() > MAX_FILE {
        return Err("an ONNX file larger than a face detector".into());
    }
    let mut r = Reader::new(bytes);
    let mut found = None;
    while let Some((field, wire)) = r.key()? {
        if (field, wire) == (7, 2) {
            found = Some(graph(r.delimited()?)?);
        } else {
            r.skip(wire)?;
        }
    }
    let mut g = found.ok_or("an ONNX file with no graph")?;
    // A graph may list its initialisers among its inputs too (older exporters do).
    g.inputs.retain(|i| !g.initializers.contains_key(i));
    Ok(g)
}
