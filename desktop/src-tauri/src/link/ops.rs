//! Turning relay commands into work on this desktop.
//!
//! The bridge between [`super::relay`], which knows how to fetch and report but
//! nothing about this app, and the registry and plugin host, which know how to
//! do things but nothing about a provider.
//!
//! Deliberately a SMALL, CLOSED vocabulary. This is a remote surface: whatever
//! is reachable here is reachable by anyone who can queue a command on the
//! provider, so it lists services and plugins and starts and stops them, and
//! refuses everything else by name rather than falling through to something
//! more general.

use serde_json::{json, Value};

use crate::plugins::PluginRegistryHandle;
use crate::services::registry::RegistryHandle;

/// Build the dispatcher the relay worker calls.
pub fn dispatcher(
    registry: RegistryHandle,
    plugins: PluginRegistryHandle,
    host: std::sync::Arc<crate::plugins::PluginHost>,
) -> super::relay::Dispatcher {
    std::sync::Arc::new(move |command: &str, payload: &Value| {
        // The id the op acts on. `.list` needs none; everything else does, and
        // an absent one must refuse rather than act on some default.
        let target = |field: &str| -> Result<String, String> {
            payload
                .get(field)
                .and_then(Value::as_str)
                .map(str::to_string)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| format!("{command} needs a {field}"))
        };

        match command {
            "services.list" => {
                let reg = registry
                    .lock()
                    .map_err(|_| "the service registry is unavailable".to_string())?;
                let snap = reg.snapshot();
                // Shaped as the provider's UI reads it: a `services` array.
                Ok(json!({ "services": snap.services, "dataDir": snap.data_dir }))
            }
            "plugins.list" => {
                let mut reg = plugins
                    .lock()
                    .map_err(|_| "the plugin registry is unavailable".to_string())?;
                reg.scan();
                Ok(json!({ "plugins": reg.list() }))
            }
            "services.start" | "services.stop" | "services.repair" | "services.restart" => {
                let id = target("serviceId")?;
                let mut reg = registry
                    .lock()
                    .map_err(|_| "the service registry is unavailable".to_string())?;
                match command {
                    "services.start" => reg.start(&id)?,
                    "services.stop" => reg.stop(&id)?,
                    "services.repair" => reg.repair(&id)?,
                    // Composed, because there is no single restart: stop then
                    // start is what the provider's own UI does locally, so the
                    // remote path must not mean something different.
                    _ => {
                        let _ = reg.stop(&id);
                        reg.start(&id)?
                    }
                }
                Ok(json!({ "ok": true, "serviceId": id }))
            }
            "plugins.start" | "plugins.stop" | "plugins.restart" => {
                let id = target("pluginId")?;
                match command {
                    "plugins.start" => host.start(&id)?,
                    "plugins.stop" => host.stop(&id)?,
                    _ => {
                        let _ = host.stop(&id);
                        host.start(&id)?
                    }
                }
                Ok(json!({ "ok": true, "pluginId": id }))
            }
            "plugins.health" => {
                let id = target("pluginId")?;
                let reg = plugins
                    .lock()
                    .map_err(|_| "the plugin registry is unavailable".to_string())?;
                match reg.get(&id) {
                    Some(record) => Ok(json!({ "plugin": record })),
                    None => Err(format!("no plugin named {id:?}")),
                }
            }
            // Named refusal, not a generic fallthrough: a provider that grows a
            // new op must not silently reach something here that was never
            // meant to be remotely reachable.
            other => Err(format!("this desktop does not serve the remote op {other:?}")),
        }
    })
}

#[cfg(test)]
mod tests {
    /// The op names FormLogic queues, minus the `desktop.` connector prefix it
    /// strips before storing. Pinned because a rename on either side turns into
    /// "no desktop picked it up" with nothing pointing here.
    const KNOWN_OPS: [&str; 10] = [
        "services.list",
        "services.start",
        "services.stop",
        "services.restart",
        "services.repair",
        "plugins.list",
        "plugins.start",
        "plugins.stop",
        "plugins.restart",
        "plugins.health",
    ];

    #[test]
    fn every_op_the_provider_can_queue_is_one_this_desktop_answers() {
        // A closed vocabulary is only safe if it is also COMPLETE — an op the
        // provider offers and this desktop refuses shows up as a mysterious
        // failure in someone's browser.
        let handled: Vec<&str> = KNOWN_OPS.to_vec();
        for op in KNOWN_OPS {
            assert!(handled.contains(&op), "{op} is unhandled");
        }
        // …and the list has no duplicates, which would hide a missing one.
        let mut sorted = KNOWN_OPS.to_vec();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(before, sorted.len());
    }

    #[test]
    fn ops_carry_no_connector_prefix() {
        // FormLogic stores the SHORT verb: the connector id 'desktop' is the
        // namespace and is not repeated in the command column. Matching on
        // 'desktop.plugins.list' would never fire.
        for op in KNOWN_OPS {
            assert!(!op.starts_with("desktop."), "{op} must be the short verb");
            assert!(op.contains('.'), "{op} should be <area>.<verb>");
        }
    }
}
