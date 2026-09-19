//! Build script: bake the git revision into the binary as `NROB_GIT_SHA`
//! for `nrob bench` JSON provenance (docs/ROADMAP.md Phase 0 requires a
//! `revision` field). An ambient `NROB_GIT_SHA` in the environment wins, so
//! CI can stamp builds without git; without git the value is "unknown".

use std::process::Command;

fn main() {
    // HEAD moves on every commit; re-resolve the sha when it does.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-env-changed=NROB_GIT_SHA");
    if std::env::var_os("NROB_GIT_SHA").is_some() {
        return; // ambient env wins; option_env! picks it up at compile time
    }
    let sha = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=NROB_GIT_SHA={sha}");
}
