fn main() {
    let p = r"D:\glm5.3_flash\Q4_K_M\GLM-5.3-Flash-Q4_K_M-00001-of-00005.gguf";
    let g = gguf::GgufFile::open_streaming(p).expect("open");
    let t = tokenizer::Tokenizer::from_gguf(&g).expect("tokenizer");
    for s in ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<think>", "</think>", "<|observation|>"] {
        let ids = t.encode(s, false).expect("encode");
        let id = t.token_id(s);
        println!("{:16} -> {:?}   token_id={:?}", s, ids, id);
    }
    let prompt = "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>Hey<|assistant|><think>";
    let ids = t.encode(prompt, false).expect("encode");
    println!("\nwhole prompt -> {} ids: {:?}", ids.len(), ids);
    println!("decoded back: {:?}", t.decode(&ids));
}
