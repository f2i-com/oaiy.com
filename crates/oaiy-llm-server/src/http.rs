//! The HTTP/1.1 subset lives in the core now (`oaiy_engine::http`), shared with
//! oaiy-studio, which proxies this server; re-exported under the old path.

pub use oaiy_engine::http::*;
