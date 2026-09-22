//! Parse a saved response only; does not execute any tools or load a model.
use dsv41::chat::{Mode,StreamParser};
use nrob::json::Json;
fn main() -> nrob::Result<()> {
    let path=std::env::args().nth(1).expect("TEXT_FILE");
    let text=std::fs::read_to_string(path)?;
    let mut p=StreamParser::new(Mode::Chat);
    for ch in text.chars() {p.push(&ch.to_string());}
    assert!(p.tool_calls_ready());
    let (_,calls)=p.finish();
    assert_eq!(calls.len(),2);
    assert_eq!(calls[0].name,"workspace_info");
    assert_eq!(calls[0].arguments,"{}");
    assert_eq!(calls[1].name,"list_files");
    assert_eq!(Json::parse(calls[1].arguments.as_bytes())?.get("path").and_then(Json::as_str),Some("."));
    println!("{}",Json::obj([("passed",Json::Bool(true)),("calls",Json::Arr(calls.into_iter().map(|c|Json::obj([
        ("name",Json::Str(c.name)),("arguments",Json::Str(c.arguments))])).collect()))]).to_json());
    Ok(())
}
