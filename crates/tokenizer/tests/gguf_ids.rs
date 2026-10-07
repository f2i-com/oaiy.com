//! A GGUF's tokenizer against llama.cpp's own ids for the same file (`TOKENIZER_GGUF` the model, `TOKENIZER_IDS` a
//! file of lines `text<TAB>ids`, the text JSON-escaped and the ids llama.cpp's `/tokenize` gave for it without
//! special tokens added): each line's text encodes to its ids. What a pre-tokenizer's split gets wrong shows here and
//! nowhere else: a model still answers from the wrong pieces of an indented line.
use gguf::GgufFile;
use tokenizer::Tokenizer;

fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                out.push(char::from_u32(u32::from_str_radix(&hex, 16).unwrap()).unwrap());
            }
            Some(other) => out.push(other),
            None => {}
        }
    }
    out
}

#[test]
#[ignore = "needs a GGUF (TOKENIZER_GGUF) and llama.cpp's ids for it (TOKENIZER_IDS)"]
fn a_ggufs_ids_are_llama_cpps() {
    let (Ok(model), Ok(ids)) = (std::env::var("TOKENIZER_GGUF"), std::env::var("TOKENIZER_IDS")) else { return };
    let tok = Tokenizer::from_gguf(&GgufFile::open_streaming(&model).unwrap()).unwrap();
    let mut wrong = 0;
    let lines = std::fs::read_to_string(ids).unwrap();
    for line in lines.lines().filter(|l| !l.trim().is_empty()) {
        let (text, want) = line.split_once('\t').unwrap();
        let text = unescape(text);
        let want: Vec<u32> = want.split_whitespace().map(|v| v.parse().unwrap()).collect();
        let got = tok.encode(&text, false).unwrap();
        if got != want {
            wrong += 1;
            eprintln!("{text:?}:\n  ours      {got:?}\n  llama.cpp {want:?}");
        }
    }
    eprintln!("{} texts, {wrong} with other ids than llama.cpp's", lines.lines().filter(|l| !l.trim().is_empty()).count());
    assert_eq!(wrong, 0);
}
