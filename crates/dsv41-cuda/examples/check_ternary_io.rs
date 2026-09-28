use oaiy_engine::store::WeightStore;
fn main() {
    let arg=std::env::args().nth(1).expect("SAMPLE_DIR");let root=std::path::Path::new(&arg);
    let store=dsv41::ternary::TernaryStore::sample(root,40,384).unwrap().with_direct(true).unwrap();
    assert!(store.direct_io(),"Expected direct IO on this Windows test host");
    for layer in [0,19,39] {
        let mut bytes=vec![0;store.record_bytes()];store.fetch(layer,0,&mut bytes).unwrap();
        assert_eq!(bytes,std::fs::read(root.join(format!("layer{layer:02}.expert000.w2"))).unwrap());
    }
    println!("PASS: all three ternary records match buffered bytes through direct IO");
}
