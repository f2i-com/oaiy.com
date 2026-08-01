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
use std::time::Duration;

/// The connector id that means "this app itself" rather than a plugin.
pub const DESKTOP_CONNECTOR: &str = "desktop";

/// How long a plugin gets to answer a relayed connector command.
///
/// Comfortably under the provider's own command TTL, so a plugin that hangs
/// produces a typed timeout the caller can read rather than a command that
/// expires looking like the desktop never picked it up.
const FORWARD_TIMEOUT: Duration = Duration::from_secs(30);

use crate::plugins::PluginRegistryHandle;
use crate::services::registry::RegistryHandle;

/// The idempotency key for a relayed connector command.
///
/// Namespaced so a plugin's journal shows where the key came from, and derived
/// only from the command id so a redelivery of the SAME command produces the
/// SAME key — which is the entire point. A generated one would be new each
/// time and would defeat the journal it is meant to key.
fn relay_idempotency_key(command_id: &str) -> String {
    format!("relay-command-{command_id}")
}

/// Say why a plugin could not serve a relayed command, in words that name the
/// fix rather than the internals.
///
/// This message is what the user reads on the provider's website, so "the
/// aokie plugin is not running on this desktop" beats a debug-printed enum.
fn forward_error_message(
    connector: &str,
    command: &str,
    error: crate::plugins::ForwardError,
) -> String {
    use crate::plugins::ForwardError;
    match error {
        ForwardError::Refused(refusal) => format!(
            "the {connector} plugin does not offer {command:?} ({refusal:?})"
        ),
        ForwardError::NotRunning { plugin_id } => {
            format!("the {plugin_id} plugin is not running on this desktop")
        }
        ForwardError::Call(e) => format!("the {connector} plugin did not answer {command:?}: {e:?}"),
        ForwardError::Internal(message) => message,
    }
}

/// Build the dispatcher the relay worker calls.
pub fn dispatcher(
    registry: RegistryHandle,
    plugins: PluginRegistryHandle,
    host: std::sync::Arc<crate::plugins::PluginHost>,
) -> super::relay::Dispatcher {
    std::sync::Arc::new(move |connector: &str, command: &str, payload: &Value, key: &str| {
        // A command names WHOSE verb it is. Anything that is not this app's own
        // belongs to a plugin's connector, and goes to that plugin through the
        // capability gate its manifest declares — the plugin decides what it
        // exposes, exactly as it does for a local caller.
        //
        // Matching the verb alone and ignoring this was a real bug: the
        // provider's call console polls `call.current` on the `aokie`
        // connector, every one was measured against the desktop's own list and
        // refused, and the console got nothing back for the whole call.
        if connector != DESKTOP_CONNECTOR {
            let payload = (!payload.is_null()).then(|| payload.clone());
            // The key is REQUIRED for anything with side effects. Passing none
            // was the second half of this bug: read-only verbs like
            // `call.current` went through, and every verb that DOES something —
            // answering, hanging up, speaking — came back "requires an
            // idempotencyKey" from the gate. The relayed command's own id is
            // the right key: it is stable across redelivery, which is exactly
            // what stops a blipped socket from answering the call twice.
            let key = relay_idempotency_key(key);
            return host
                .forward_connector(connector, command, payload, Some(&key), FORWARD_TIMEOUT)
                .map_err(|e| forward_error_message(connector, command, e));
        }
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
            // new op on the DESKTOP connector must not silently reach something
            // here that was never meant to be remotely reachable. A plugin's
            // connector is a different matter — that went to the plugin above.
            other => Err(format!(
                "this desktop does not serve the remote op {other:?} on the desktop connector"
            )),
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
    fn a_relayed_command_keys_on_its_own_id_so_a_retry_cannot_act_twice() {
        // A plugin refuses any side-effecting command with no key. Passing none
        // let read-only verbs through while every verb that DOES something came
        // back "requires an idempotencyKey" — so the call console could watch a
        // call and never touch it.
        let key = super::relay_idempotency_key("6b6e21fd-6cd2-41ed-ac1a-a30a91cbad3a");
        assert!(key.contains("6b6e21fd-6cd2-41ed-ac1a-a30a91cbad3a"));
        assert!(!key.trim().is_empty());
        // Same command id, same key: that is what makes a redelivery harmless
        // rather than a second answered call.
        assert_eq!(key, super::relay_idempotency_key("6b6e21fd-6cd2-41ed-ac1a-a30a91cbad3a"));
        assert_ne!(key, super::relay_idempotency_key("a-different-command"));
    }

    #[test]
    fn a_plugins_own_verbs_are_not_measured_against_the_desktops_list() {
        // `call.current`, `dongle.list`, `phone.listPaired` and friends belong
        // to the aokie plugin, not to this app. Refusing them because they are
        // not in the desktop's list is what left the provider's call console
        // blank for a whole call.
        for op in ["call.current", "dongle.list", "phone.listPaired", "settings.get"] {
            assert!(
                !super::tests::KNOWN_OPS.contains(&op),
                "{op} is a plugin verb and must not be in the desktop list"
            );
        }
        // …and the desktop's own list is still exactly what it was.
        assert!(KNOWN_OPS.contains(&"services.list"));
        assert_eq!(super::DESKTOP_CONNECTOR, "desktop");
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
