//! The local AI gateway: configured providers (credentials held on this machine)
//! plus an OpenAI-compatible chat/models proxy that injects the key server-side.
//!
//! - `providers` — the credential-hidden provider store (full/public split).
//! - `egress`    — SSRF/redirect guard for outbound provider calls.
//! - `gateway`   — the chat/models proxy + OpenAI↔Anthropic normalization.
//! - `routes`    — the `/api/ai/*` axum sub-router (merged in `http::serve`).

pub mod egress;
pub mod gateway;
pub mod providers;
pub mod routes;

pub use providers::{open_handle, ProviderStoreHandle};
pub use routes::{router as ai_router, AiState};
