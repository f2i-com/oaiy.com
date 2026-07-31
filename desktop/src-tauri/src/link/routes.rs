//! HTTP surface for the account link.
//!
//! Privileged: starting a link opens the user's browser and ends in a stored
//! credential, and unlinking throws that credential away. Neither is something
//! a local web page should be able to do by being on loopback.

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};

use super::{LinkHandle, LinkStatus};

fn fail(status: StatusCode, message: String) -> axum::response::Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

async fn status(State(store): State<LinkHandle>) -> Json<LinkStatus> {
    Json(store.status())
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StartBody {
    connector_id: String,
    base_url: String,
}

async fn start(
    State(store): State<LinkHandle>,
    Json(body): Json<StartBody>,
) -> axum::response::Response {
    // The browser is opened from the worker thread, so a slow launcher cannot
    // hold this request open.
    let opener = |url: &str| {
        if let Err(e) = crate::link::routes::open_in_browser(url) {
            log::warn!("could not open the browser for linking: {e}");
        }
    };
    match super::start_link(store, &body.connector_id, &body.base_url, opener) {
        // The URL comes back so a UI can offer "open it again" when a launcher
        // silently did nothing — a blank screen is otherwise unrecoverable.
        Ok(url) => (StatusCode::ACCEPTED, Json(serde_json::json!({ "authorizeUrl": url })))
            .into_response(),
        Err(e) => fail(StatusCode::BAD_REQUEST, e),
    }
}

/// Forget the link on THIS machine.
///
/// Deliberately local-only. Revoking the credential server-side is the
/// provider's business and needs a route this descriptor does not describe;
/// pretending otherwise would tell the user their key was dead when it was not.
/// The UI says so.
async fn unlink(State(store): State<LinkHandle>) -> Json<LinkStatus> {
    Json(store.unlink())
}

/// Hand an authorize URL to the user's browser.
///
/// Deliberately NOT the GUI's `open_url` command. That one rejects `&` as a
/// shell metacharacter, which would reject every OAuth URL ever built, and it
/// is gui-gated so the headless server could not reach it.
///
/// Safe because nothing here goes through a shell: the URL is one argv element,
/// so `&` has no special meaning. What IS still checked is the scheme — this
/// must never become a way to launch `file:` or a custom protocol handler.
pub fn open_in_browser(url: &str) -> Result<(), String> {
    let allowed = url.starts_with("https://")
        || url.starts_with("http://127.0.0.1")
        || url.starts_with("http://localhost");
    if !allowed {
        return Err("refusing to open a non-https URL".into());
    }
    // Control characters and quotes could still break out of argv quoting on
    // Windows, where the command line is a single string the callee re-parses.
    if url.chars().any(|c| c.is_ascii_control() || c == '"' || c == '\'') {
        return Err("the authorize URL contains characters that cannot be passed safely".into());
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        std::process::Command::new("rundll32.exe")
            .args(["url.dll,FileProtocolHandler", url])
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .map_err(|e| format!("could not open the browser: {e}"))?;
    }
    #[cfg(target_os = "macos")]
    std::process::Command::new("open")
        .arg(url)
        .spawn()
        .map_err(|e| format!("could not open the browser: {e}"))?;
    #[cfg(target_os = "linux")]
    std::process::Command::new("xdg-open")
        .arg(url)
        .spawn()
        .map_err(|e| format!("could not open the browser: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_https_and_loopback_may_be_opened() {
        // This spawns a URL handler, so the scheme is the whole defence: file:
        // or a custom protocol would launch something else entirely.
        assert!(open_in_browser("file:///C:/Windows/System32/calc.exe").is_err());
        assert!(open_in_browser("javascript:alert(1)").is_err());
        assert!(open_in_browser("http://evil.example/authorize").is_err());
        assert!(open_in_browser("ms-settings:").is_err());
    }

    #[test]
    fn a_url_with_control_characters_or_quotes_is_refused() {
        assert!(open_in_browser("https://x.example/a\"b").is_err());
        assert!(open_in_browser("https://x.example/a
b").is_err());
    }
}

pub fn router(store: LinkHandle) -> Router {
    Router::new()
        .route("/api/link", get(status).delete(unlink))
        .route("/api/link/start", post(start))
        .with_state(store)
}
