use std::io::{BufWriter, Write};
use std::path::Path;
use dsv41::moe::Route;
use nrob::{Result, json::Json};

pub struct RouteLog {
    out: BufWriter<std::fs::File>,
    request: u64,
    phase: String,
    token: Option<u32>,
    text: String,
}
impl RouteLog {
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self {out: BufWriter::new(std::fs::OpenOptions::new().write(true).create_new(true).open(path)?),
            request: 0, phase: "prompt".into(), token: None, text: String::new()})
    }
    pub fn context(&mut self, request:u64, phase:&str, token:Option<u32>, text:&str) {
        self.request=request; self.phase=phase.into(); self.token=token; self.text=text.into();
    }
    pub fn routes(&mut self, layer:usize, position:usize, routes:&[Route], ternary:bool) -> Result<()> {
        for (i,r) in routes.iter().enumerate() {
            writeln!(self.out, "{{\"request\":{},\"phase\":{},\"position\":{},\"input_token\":{},\"input_text\":{},\"layer\":{},\"precision\":\"{}\",\"experts\":{:?},\"router_weights\":{:?}}}",
                self.request,Json::str(&self.phase).to_json(),position+i,
                self.token.map(|n|n.to_string()).unwrap_or_else(||"null".into()),Json::str(&self.text).to_json(),layer,
                if ternary {"w2g128"} else {"mxfp4"},r.experts,r.weights)?;
        }
        Ok(())
    }
    pub fn flush(&mut self) -> Result<()> {self.out.flush()?;Ok(())}
}
