//! Conservative runtime compatibility inside an explicit DSML calls envelope.
//! The reference/golden parser remains strict. Never repair missing closing tags,
//! invent arguments, or rewrite raw string parameter contents.
use super::{format_err, split_tool_name, write_str, ToolCall, DSML};
use nrob::{json::Json, Result};

struct Tag<'a> { name: &'a str, close: bool, dsml: bool, attrs: Vec<(&'a str, &'a str)> }

fn tag<'a>(text: &'a str, i: &mut usize) -> Result<Tag<'a>> {
    *i += text[*i..].len() - text[*i..].trim_start().len();
    let raw = text[*i..].strip_prefix('<').ok_or_else(|| format_err("expected DSML tag"))?;
    let end = raw.find('>').ok_or_else(|| format_err("unfinished DSML tag"))?;
    let mut body = &raw[..end];
    *i += end + 2;
    let close = body.starts_with('/');
    if close { body = &body[1..]; }
    let dsml = body.starts_with(DSML);
    if dsml {
        body = &body[DSML.len()..];
        if !body.starts_with(char::is_whitespace) { return Err(format_err("DSML tag separator missing")); }
        body = body.trim_start();
    }
    let n = body.find(char::is_whitespace).unwrap_or(body.len());
    let name = &body[..n];
    let mut rest = &body[n..];
    let mut attrs = Vec::new();
    while !rest.trim().is_empty() {
        if !rest.starts_with(char::is_whitespace) { return Err(format_err("attribute separator missing")); }
        rest = rest.trim_start();
        let eq = rest.find('=').ok_or_else(|| format_err("attribute value missing"))?;
        let key = rest[..eq].trim();
        if key.is_empty() || key.contains(char::is_whitespace) || attrs.iter().any(|(k,_)| *k==key) {
            return Err(format_err("invalid/duplicate DSML attribute"));
        }
        rest = rest[eq+1..].trim_start().strip_prefix('"').ok_or_else(|| format_err("quoted attribute required"))?;
        let q = rest.find('"').ok_or_else(|| format_err("unfinished attribute"))?;
        attrs.push((key,&rest[..q]));rest = &rest[q+1..];
    }
    if close && !attrs.is_empty() { return Err(format_err("attributes on closing tag")); }
    Ok(Tag {name,close,dsml,attrs})
}

fn attr<'a>(t: &Tag<'a>, name: &str) -> Result<&'a str> {
    t.attrs.iter().find(|(k,_)| *k==name).map(|(_,v)| *v).ok_or_else(|| format_err("DSML attribute missing"))
}

pub(super) fn parse(text: &str) -> Result<(usize,Vec<ToolCall>)> {
    let mut i=0;let root=tag(text,&mut i)?;
    if root.name!="calls" || root.close || !root.dsml || !root.attrs.is_empty() { return Err(format_err("explicit DSML calls envelope required")); }
    let mut calls=Vec::new();
    loop {
        let invoke=tag(text,&mut i)?;
        if invoke.name=="calls" && invoke.close && invoke.dsml { return Ok((i,calls)); }
        if invoke.name!="invoke" || invoke.close || invoke.attrs.len()!=1 { return Err(format_err("expected invoke")); }
        let name=attr(&invoke,"name")?;
        if name.is_empty() { return Err(format_err("empty tool name")); }
        let mut params:Vec<(String,String,bool)>=Vec::new();
        loop {
            let p=tag(text,&mut i)?;
            if p.name=="invoke" && p.close { break; }
            if p.name!="parameter" || p.close || p.attrs.len()!=2 { return Err(format_err("expected parameter")); }
            let pname=attr(&p,"name")?;
            let string=match attr(&p,"string")? { "true"=>true,"false"=>false,_=>return Err(format_err("invalid string attribute")) };
            if params.iter().any(|(n,_,_)| n==pname) { return Err(format_err("duplicate parameter")); }
            // Match this parameter's namespace. A literal </parameter> inside a
            // canonical DSML string is data, not a reason to alter that string.
            let closing=if p.dsml {format!("</{DSML} parameter")} else {"</parameter".into()};
            let start=i;let mut search=i;
            let end=loop {
                let at=search+text[search..].find(&closing).ok_or_else(|| format_err("unclosed parameter"))?;
                let mut after=at;
                match tag(text,&mut after) {
                    Ok(t) if t.name=="parameter" && t.close && t.dsml==p.dsml => {i=after;break at;},
                    _ => search=at+closing.len(),
                }
            };
            let value=&text[start..end];
            if !string { Json::parse(value.as_bytes()).map_err(|_|format_err("invalid JSON parameter"))?; }
            params.push((pname.into(),value.into(),string));
        }
        let mut arguments=String::from("{");
        for (n,(key,value,string)) in params.iter().enumerate() {
            if n>0 {arguments.push_str(", ");}
            write_str(key,&mut arguments);arguments.push_str(": ");
            if *string {write_str(value,&mut arguments);} else {arguments.push_str(value);}
        }
        arguments.push('}');
        let (namespace,name)=split_tool_name(name,None).map_err(|e|format_err(e.to_string()))?;
        calls.push(ToolCall {name,namespace,arguments});
    }
}
