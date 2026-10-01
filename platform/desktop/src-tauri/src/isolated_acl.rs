//! The isolated dashboard needs only native navigation event subscriptions.
//! Tauri dispatches `plugin:*` commands before the application's invoke handler,
//! so its compiled-in core/plugin grants must be replaced as well.

use tauri::ipc::RuntimeAuthority;
use tauri::utils::acl::{
    resolved::{Resolved, ResolvedCommand},
    ExecutionContext,
};

fn permissions() -> Resolved {
    let mut resolved = Resolved::default();
    // Application commands remain subject to isolated_policy::ipc_allowed in
    // the application's handler. Plugin/core commands always go through ACL.
    for command in ["plugin:event|listen", "plugin:event|unlisten"] {
        resolved.allowed_commands.insert(
            command.into(),
            vec![ResolvedCommand {
                context: ExecutionContext::Local,
                // Tauri matches window OR webview grants. A window grant here would
                // authorize every embedded sibling sharing the dashboard's window.
                webviews: vec!["main".parse().expect("a literal webview label")],
                ..Default::default()
            }],
        );
    }
    resolved
}

fn authority() -> RuntimeAuthority {
    // Context's authority replacement API is public but marked unstable by
    // Tauri. Keep it covered against the version pinned in Cargo.lock. This
    // macro also handles Tauri's debug/dynamic-acl constructor configuration.
    tauri::runtime_authority!(Default::default(), permissions())
}

/// Call before building any isolated window; ordinary launches keep their ACL.
pub fn restrict(context: &mut tauri::Context<tauri::Wry>) {
    *context.runtime_authority_mut() = authority();
}

#[cfg(test)]
mod tests {
    use super::*;
    use tauri::ipc::Origin;

    #[test]
    fn only_the_local_dashboard_can_subscribe_to_navigation_events() {
        let authority = authority();
        for command in ["plugin:event|listen", "plugin:event|unlisten"] {
            assert!(authority
                .resolve_access(command, "main", "main", &Origin::Local)
                .is_some());
            for webview in ["embed-agent", "embed-flows", "embed-engines", "main-2", ""] {
                assert!(
                    authority
                        .resolve_access(command, "main", webview, &Origin::Local)
                        .is_none(),
                    "{command}: {webview}"
                );
            }
            let remote = Origin::Remote {
                url: "https://example.test".parse().unwrap(),
            };
            assert!(authority
                .resolve_access(command, "main", "main", &remote)
                .is_none());
        }
    }

    #[test]
    fn core_and_plugin_operations_have_no_grant() {
        let authority = authority();
        for command in [
            "plugin:image|from_path",
            "plugin:path|resolve_directory",
            "plugin:event|emit",
            "plugin:event|emit_to",
            "plugin:webview|create_webview",
            "plugin:webview|create_webview_window",
            "plugin:window|create",
            "plugin:tray|new",
            "plugin:menu|new",
            "plugin:shell|execute",
            "plugin:shell|open",
            "plugin:dialog|open",
            "plugin:notification|notify",
            "plugin:updater|check",
            "plugin:future|execute",
        ] {
            assert!(
                authority
                    .resolve_access(command, "main", "main", &Origin::Local)
                    .is_none(),
                "{command}"
            );
        }
    }

    #[test]
    fn application_commands_still_reach_the_application_refusal_policy() {
        let permissions = permissions();
        assert!(!permissions.has_app_acl);
        assert_eq!(permissions.allowed_commands.len(), 2);
        assert!(crate::isolated_policy::ipc_allowed("get_config"));
        for command in [
            "show_embedded",
            "set_theme",
            "update_check",
            "open_path",
            "unknown",
        ] {
            assert!(!crate::isolated_policy::ipc_allowed(command), "{command}");
        }
    }
}
