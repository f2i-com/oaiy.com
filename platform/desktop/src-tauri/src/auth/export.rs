//! The table as data: what `cargo test` writes to `<target>/access/routes.json` for the scripts that
//! check clients against it (`platform/desktop/scripts/check-access.mjs`, and later the client-surface
//! scan). The file is a copy, never an input: the Rust table is the source of truth.

use std::path::PathBuf;

use serde_json::{json, Value};

use super::routes::{Class, DeskRole, Only, Route, ROUTES};

/// One row as the exported file spells it.
fn row_json(r: &Route) -> Value {
    let mut row = json!({
        "method": r.method.as_str(),
        "pattern": r.pattern,
        "since": r.since,
        "only": match r.only { Only::Everywhere => "everywhere", Only::Server => "server" },
    });
    let (class, extra) = match r.class {
        Class::Public => ("public", Value::Null),
        Class::AnyCredential => ("any-credential", Value::Null),
        Class::Console => ("console", Value::Null),
        Class::Session { elevate } => ("session", json!({ "elevate": elevate })),
        Class::Desk { roles } => (
            "desk",
            json!({ "roles": roles.iter().map(|r| match r { DeskRole::Dashboard => "dashboard", DeskRole::Agent => "agent", DeskRole::Flows => "flows" }).collect::<Vec<_>>() }),
        ),
        Class::Scope(scope) => ("scope", json!({ "scope": scope })),
        Class::Unclassified => ("unclassified", Value::Null),
    };
    row["class"] = json!(class);
    if let Value::Object(extra) = extra {
        for (k, v) in extra {
            row[k] = v;
        }
    }
    row
}

/// The exported document.
pub fn routes_json() -> Value {
    json!({
        "schema": "oaiy-access-routes/1",
        "routes": ROUTES.iter().map(row_json).collect::<Vec<_>>(),
    })
}

/// Where the tests write it: `$OAIY_ACCESS_OUT`, else `$CARGO_TARGET_DIR/access`, else `access` beside
/// the profile folder of the running test binary (`target/access`). Never inside the source tree.
pub fn output_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("OAIY_ACCESS_OUT") {
        return dir.into();
    }
    if let Some(target) = std::env::var_os("CARGO_TARGET_DIR") {
        return PathBuf::from(target).join("access");
    }
    // <target>/<profile>/deps/<test binary>
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.ancestors().nth(3).map(|target| target.join("access")))
        .unwrap_or_else(|| std::env::temp_dir().join("oaiy-access"))
}

/// Write the exported file and return where it went.
pub fn write_routes_json() -> std::io::Result<PathBuf> {
    let dir = output_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("routes.json");
    let mut text = serde_json::to_string_pretty(&routes_json()).map_err(std::io::Error::other)?;
    text.push('\n');
    std::fs::write(&path, text)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cargo_test_writes_the_routes_file_the_scripts_read() {
        let path = write_routes_json().expect("the routes file is written");
        let read: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(read["schema"], "oaiy-access-routes/1");
        let rows = read["routes"].as_array().unwrap();
        assert_eq!(rows.len(), ROUTES.len());
        let config = rows
            .iter()
            .find(|r| r["method"] == "GET" && r["pattern"] == "/api/config")
            .unwrap();
        assert_eq!(config["class"], "scope");
        assert_eq!(config["scope"], "system.read");
        assert_eq!(config["since"], 1);
        let health = rows.iter().find(|r| r["pattern"] == "/api/health").unwrap();
        assert_eq!(health["class"], "public");
        let server_only = rows.iter().filter(|r| r["only"] == "server").count();
        assert!(server_only > 0);
        // The file lands in the build's target folder, never in the source tree.
        assert!(
            !path.starts_with(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")),
            "{}",
            path.display()
        );
    }

    #[test]
    fn a_row_says_what_it_takes() {
        let rows = routes_json();
        let rows = rows["routes"].as_array().unwrap();
        let find = |m: &str, p: &str| {
            rows.iter()
                .find(|r| r["method"] == m && r["pattern"] == p)
                .unwrap()
                .clone()
        };
        assert_eq!(find("POST", "/api/auth/logout")["class"], "session");
        assert_eq!(find("POST", "/api/auth/logout")["elevate"], false);
        assert_eq!(find("GET", "/api/auth/whoami")["class"], "any-credential");
        assert_eq!(
            find("POST", "/api/auth/console/setup-code")["class"],
            "console"
        );
        assert_eq!(
            find("GET", "/api/local-protection/webview-key")["roles"],
            json!(["dashboard", "agent", "flows"])
        );
        assert_eq!(
            find("ANY", "/api/ai/engine/gateway/*path")["scope"],
            "ai.use"
        );
    }
}
