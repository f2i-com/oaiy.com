//! The local AI gateway: configured providers (credentials held on this machine)
//! plus an OpenAI-compatible chat/models proxy that injects the key server-side.
//!
//! - `providers`  — the credential-hidden provider store (full/public split).
//! - `egress`     — SSRF/redirect guard for outbound provider calls.
//! - `gateway`    — the chat/models proxy + OpenAI↔Anthropic normalization.
//! - `routes`     — the `/api/ai/*` axum sub-router (merged in `http::serve`).
//! - `e2e`        — the sealed envelope a provider's web app chats through.
//! - `tunnel`     — answering the sealed turns it relays here.
//! - `chat_tools` — letting one of those turns act on the linked account.

pub mod chat_tools;
pub mod codex;
pub mod e2e;
pub mod egress;
pub mod gateway;
pub mod providers;
pub mod routes;
pub mod tunnel;

pub use codex::{CodexHandle, CODEX_PROVIDER_ID};
pub use providers::{open_handle, ProviderStoreHandle};
pub use routes::{router as ai_router, AiState};
