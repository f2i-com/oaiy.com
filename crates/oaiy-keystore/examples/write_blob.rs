//! Writes one secret with the DPAPI provider, for the other-user test of `tests/keystore.rs` (`another_users_blob_fails`).
//!
//! Run it as a Windows user other than the one that will run the test:
//!
//! ```text
//! cargo run -p oaiy-keystore --example write_blob -- <data folder> foreign.secret
//! ```
//!
//! It stores a fixed, harmless value under the given name in `<data folder>\keys`. Nothing else is written.

use oaiy_keystore::{open, Name, ProviderChoice};

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(folder), Some(name)) = (args.next(), args.next()) else {
        eprintln!("usage: write_blob <data folder> <name>");
        std::process::exit(2);
    };
    let store = match open(std::path::Path::new(&folder), ProviderChoice::DpapiFile) {
        Ok(store) => store,
        Err(error) => {
            eprintln!("cannot open the keystore: {error}");
            std::process::exit(1);
        }
    };
    let name = match Name::new(&name) {
        Ok(name) => name,
        Err(error) => {
            eprintln!("bad name: {error}");
            std::process::exit(2);
        }
    };
    match store.put(&name, b"a test value that belongs to the user who ran this") {
        Ok(()) => println!("stored under {name} in {folder}\\keys"),
        Err(error) => {
            eprintln!("cannot store: {error}");
            std::process::exit(1);
        }
    }
}
