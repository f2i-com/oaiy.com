//! The HTTP/1.1 subset lives in the core now (`nrob::http`), shared with
//! nrob-studio, which proxies this server; re-exported under the old path.

pub use nrob::http::*;
