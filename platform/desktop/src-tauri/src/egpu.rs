//! The eGPU module: a plugin that carries what an NVIDIA card in a Thunderbolt
//! enclosure needs on a Mac (its own Python, tinygrad, the card's server, its
//! kernels compiled ahead of time and its firmware: `tools/tinygpu/plugin`),
//! handed to the engines. With it a Mac needs neither Docker nor a compiler
//! for the card.
//!
//! While the module is on, the plugin providing it is asked `card.runtime`
//! each time it starts, and what it says goes to the engines
//! ([`crate::engines::provide_egpu`]): their eGPU runs the plugin's Python and
//! launcher from its next start (an idle one is started again on them at once).
//! The plugin's own process holds nothing: the engines hold the card, as they
//! stop it cleanly. A plugin is killed within seconds of being stopped, and a
//! card's holder killed has taken its Thunderbolt link down.
//!
//! The paths it gives must be inside its own folder, so the engines run only
//! files its trust covers. Stopped or crashed, what it gave goes on being used
//! (its files are what runs, not its process); turned off or removed, the
//! engines go back to their settings' Python (`llm.egpu.python`).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::modules::{self, Provider};
use crate::plugins::{PluginHost, PluginRegistryHandle, PluginState};

/// How long the plugin has to say what runs the card.
const ASK_TIMEOUT: Duration = Duration::from_secs(15);
/// How soon a plugin that did not say is asked again (sooner, when the modules change).
const ASK_AGAIN: Duration = Duration::from_secs(30);

/// Follow the eGPU module for as long as the desktop runs. Only where the engines have an eGPU (a Mac).
pub fn start(host: Arc<PluginHost>, plugins: PluginRegistryHandle) {
    if !oaiy_studio::egpu_available() {
        return;
    }
    tokio::spawn(async move {
        let mut changes = modules::subscribe();
        // The provider and its state when it last said (it is asked again once either changes), whether the engines
        // were given what it said, and the last reason it did not say (logged once).
        let (mut asked, mut given, mut said): (Option<(String, PluginState)>, bool, String) = (None, false, String::new());
        loop {
            let provider = changes.borrow_and_update().module(modules::EGPU).filter(|m| m.enabled).and_then(|m| m.provider.clone());
            match provider {
                None => {
                    asked = None;
                    if given {
                        log::info!("egpu: no plugin provides the eGPU now: the engines use their settings' Python");
                        crate::engines::provide_egpu(None);
                        given = false;
                    }
                }
                Some(p) if matches!(p.state, PluginState::Running | PluginState::Unhealthy) => {
                    let now = (p.plugin_id.clone(), p.state);
                    if asked.as_ref() != Some(&now) {
                        match ask(&host, &plugins, &p).await {
                            Ok(runtime) => {
                                log::info!("egpu: {} runs the eGPU (its Python {})", runtime.by, runtime.python.display());
                                crate::engines::provide_egpu(Some(runtime));
                                (asked, given) = (Some(now), true);
                                said.clear();
                            }
                            Err(why) => {
                                if why != said {
                                    log::warn!("egpu: {} did not say what runs the eGPU: {why}", p.name);
                                    said = why;
                                }
                            }
                        }
                    }
                }
                Some(_) => asked = None,
            }
            tokio::select! {
                changed = changes.changed() => if changed.is_err() { return },
                _ = tokio::time::sleep(ASK_AGAIN) => {}
            }
        }
    });
}

/// What `provider` says runs the card, its paths checked to be inside its folder.
async fn ask(host: &Arc<PluginHost>, plugins: &PluginRegistryHandle, provider: &Provider) -> Result<oaiy_studio::EgpuRuntime, String> {
    let dir = plugins
        .lock()
        .map_err(|_| "the plugin registry is unavailable".to_string())?
        .get(&provider.plugin_id)
        .map(|r| r.dir.clone())
        .ok_or_else(|| format!("{} is not installed", provider.plugin_id))?;
    let (host, connector) = (Arc::clone(host), provider.connector.clone().unwrap_or_else(|| modules::EGPU.to_string()));
    let call = tokio::task::spawn_blocking(move || host.forward_connector(&connector, "card.runtime", None, None, ASK_TIMEOUT));
    let data = match call.await {
        Ok(Ok(reply)) => crate::setup::unwrap_reply(reply)?,
        Ok(Err(e)) => return Err(crate::setup::describe_forward_error(e)),
        Err(e) => return Err(format!("the question was not asked: {e}")),
    };
    runtime(&data, &dir, &provider.name)
}

/// The runtime in a plugin's answer: its Python (a file), the folder for that Python's path and the folder with the
/// launcher, each inside `dir` (the plugin's own), as they are on disk (no link leads out of it).
fn runtime(data: &Value, dir: &Path, name: &str) -> Result<oaiy_studio::EgpuRuntime, String> {
    let dir = std::fs::canonicalize(dir).map_err(|e| format!("its folder {}: {e}", dir.display()))?;
    let path = |key: &str, file: bool| -> Result<PathBuf, String> {
        let raw = data.get(key).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).ok_or_else(|| format!("it gave no {key}"))?;
        let at = std::fs::canonicalize(raw).map_err(|e| format!("its {key} {raw}: {e}"))?;
        if !at.starts_with(&dir) {
            return Err(format!("its {key} {} is not in its own folder ({})", at.display(), dir.display()));
        }
        if file != at.is_file() {
            return Err(format!("its {key} {} is not a {}", at.display(), if file { "file" } else { "folder" }));
        }
        Ok(at)
    };
    let version = data.get("version").and_then(Value::as_str).unwrap_or("").trim();
    Ok(oaiy_studio::EgpuRuntime {
        python: path("python", true)?,
        pythonpath: path("pythonpath", false)?,
        scripts: path("scripts", false)?,
        by: if version.is_empty() { name.to_string() } else { format!("{name} {version}") },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_egpu_plugins_manifest_loads_and_provides_the_egpu_through_its_connector() {
        let dir = std::env::temp_dir().join(format!("oaiy-egpu-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shipped = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tools/tinygpu/plugin/manifest.json");
        std::fs::copy(&shipped, dir.join("manifest.json")).unwrap();
        let manifest = crate::plugins::PluginManifest::load(&dir).unwrap();
        let claims = modules::claims(&manifest);
        assert!(claims.provides(modules::EGPU), "{claims:?}");
        assert_eq!(claims.get(modules::EGPU).and_then(|c| c.connector.as_deref()), Some("egpu"));
        assert!(manifest.resolved_capabilities().iter().any(|c| c == "connector.egpu.card.runtime"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn what_a_plugin_says_runs_the_card_is_taken_only_from_inside_its_own_folder() {
        let root = std::env::temp_dir().join(format!("oaiy-egpu-module-{}", std::process::id()));
        let dir = root.join("egpu");
        for d in ["python/bin", "lib", "card"] {
            std::fs::create_dir_all(dir.join(d)).unwrap();
        }
        std::fs::write(dir.join("python/bin/python3.12"), "").unwrap();
        std::fs::write(root.join("elsewhere"), "").unwrap();
        let said = |python: &Path| {
            json!({"python": python, "pythonpath": dir.join("lib"), "scripts": dir.join("card"), "version": "0.1.0"})
        };
        let runtime = runtime(&said(&dir.join("python/bin/python3.12")), &dir, "eGPU").unwrap();
        assert_eq!(runtime.by, "eGPU 0.1.0");
        assert!(runtime.python.ends_with("python/bin/python3.12") && runtime.scripts.ends_with("card"), "{runtime:?}");
        let refused = |data: Value| super::runtime(&data, &dir, "eGPU").unwrap_err();
        assert!(refused(said(&root.join("elsewhere"))).contains("not in its own folder"), "a program outside it is not run");
        assert!(refused(said(&dir.join("python/bin/../../../elsewhere"))).contains("not in its own folder"), "nor one reached from it");
        assert!(refused(said(&dir.join("lib"))).contains("is not a file"));
        assert!(refused(said(&dir.join("python/bin/python3"))).contains("python"), "one that is not there");
        assert!(refused(json!({"python": dir.join("python/bin/python3.12")})).contains("no pythonpath"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
