//! WebView2 may replace an explicit user-data directory with inherited overrides.
//! Refuse them before creating the isolated root or any WebView. This only reads
//! inherited override variables and registry metadata; no values are reported
//! and no machine settings are changed.

use std::ffi::OsString;

const POLICY_NAMES: [&str; 6] = [
    "UserDataFolder",
    "AdditionalBrowserArguments",
    "BrowserExecutableFolder",
    "ChannelSearchKind",
    "ReleaseChannels",
    "ReleaseChannelPreference",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hive {
    Machine,
    User,
}

impl Hive {
    fn label(self) -> &'static str {
        match self {
            Self::Machine => "HKLM",
            Self::User => "HKCU",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum View {
    Bits64,
    Bits32,
}

impl View {
    fn label(self) -> &'static str {
        match self {
            Self::Bits64 => "64-bit",
            Self::Bits32 => "32-bit",
        }
    }
}

/// Run only for an explicit isolated launch, before any WebView is constructed.
pub fn preflight() -> Result<(), String> {
    #[cfg(windows)]
    {
        check(std::env::vars_os(), policy_has_values)
    }
    #[cfg(not(windows))]
    {
        Err("isolated WebView2 qualification requires Windows".into())
    }
}

// Inputs are injected so refusal behavior is tested without modifying the
// process environment, touching live policy values or writing the registry.
fn check(
    environment: impl IntoIterator<Item = (OsString, OsString)>,
    mut policy_has_values: impl FnMut(Hive, View, &str) -> Result<bool, ()>,
) -> Result<(), String> {
    for (name, value) in environment {
        let name = name.to_string_lossy();
        if !value.is_empty() && name.to_ascii_uppercase().starts_with("WEBVIEW2_") {
            let name: String = name
                .chars()
                .map(|c| if c.is_control() { '?' } else { c })
                .collect();
            return Err(format!(
                "isolated WebView2 profile cannot be guaranteed: environment override {name} is set"
            ));
        }
    }
    for hive in [Hive::Machine, Hive::User] {
        for view in [View::Bits64, View::Bits32] {
            for name in POLICY_NAMES {
                let location = format!("{} {} WebView2/{name}", hive.label(), view.label());
                match policy_has_values(hive, view, name) {
                    Ok(false) => {}
                    Ok(true) => {
                        // Do not guess whether an AppId/executable-name policy
                        // applies. Any configured value makes this qualification
                        // environment unsuitable; no policy data is read or shown.
                        return Err(format!(
                            "isolated WebView2 profile cannot be guaranteed: {location} has a configured override"
                        ));
                    }
                    Err(()) => {
                        return Err(format!(
                            "isolated WebView2 profile cannot be guaranteed: cannot inspect {location}"
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(windows)]
fn policy_has_values(hive: Hive, view: View, name: &str) -> Result<bool, ()> {
    use std::ptr::{null, null_mut};
    use windows_sys::Win32::{
        Foundation::{ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND, ERROR_SUCCESS},
        System::Registry::{
            RegCloseKey, RegOpenKeyExW, RegQueryInfoKeyW, HKEY, HKEY_CURRENT_USER,
            HKEY_LOCAL_MACHINE, KEY_QUERY_VALUE, KEY_WOW64_32KEY, KEY_WOW64_64KEY,
        },
    };

    struct Key(HKEY);
    impl Drop for Key {
        fn drop(&mut self) {
            // SAFETY: this handle was returned by a successful RegOpenKeyExW.
            unsafe { RegCloseKey(self.0) };
        }
    }

    let root = match hive {
        Hive::Machine => HKEY_LOCAL_MACHINE,
        Hive::User => HKEY_CURRENT_USER,
    };
    let view = match view {
        View::Bits64 => KEY_WOW64_64KEY,
        View::Bits32 => KEY_WOW64_32KEY,
    };
    let path: Vec<u16> = format!(r"Software\Policies\Microsoft\Edge\WebView2\{name}")
        .encode_utf16()
        .chain(Some(0))
        .collect();
    let mut handle = null_mut();
    // SAFETY: path is NUL-terminated and handle points to writable storage.
    // KEY_QUERY_VALUE requests only the access needed for metadata inspection.
    let status =
        unsafe { RegOpenKeyExW(root, path.as_ptr(), 0, KEY_QUERY_VALUE | view, &mut handle) };
    match status {
        ERROR_FILE_NOT_FOUND | ERROR_PATH_NOT_FOUND => return Ok(false),
        ERROR_SUCCESS => {}
        _ => return Err(()),
    }
    let key = Key(handle);
    let mut value_count = 0;
    // SAFETY: key is open; only value_count is requested. All optional outputs
    // are null. Registry value names and contents are deliberately not fetched.
    let status = unsafe {
        RegQueryInfoKeyW(
            key.0,
            null_mut(),
            null_mut(),
            null(),
            null_mut(),
            null_mut(),
            null_mut(),
            &mut value_count,
            null_mut(),
            null_mut(),
            null_mut(),
            null_mut(),
        )
    };
    if status == ERROR_SUCCESS {
        Ok(value_count != 0)
    } else {
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variable(name: &str, value: &str) -> (OsString, OsString) {
        (OsString::from(name), OsString::from(value))
    }

    #[test]
    fn isolated_webview_clear_overrides_check_both_hives_and_views() {
        let mut inspected = Vec::new();
        check(
            [
                variable("WEBVIEW2_USER_DATA_FOLDER", ""),
                variable("UNRELATED_VARIABLE", "fixture-value"),
            ],
            |hive, view, name| {
                inspected.push((hive, view, name.to_owned()));
                Ok(false)
            },
        )
        .unwrap();
        for hive in [Hive::Machine, Hive::User] {
            for view in [View::Bits64, View::Bits32] {
                for name in POLICY_NAMES {
                    assert!(inspected.contains(&(hive, view, name.to_owned())));
                }
            }
        }
        assert_eq!(inspected.len(), 24);
    }

    #[test]
    fn isolated_webview_environment_override_refusal_never_exposes_values() {
        for name in [
            "WEBVIEW2_USER_DATA_FOLDER",
            "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
            "WEBVIEW2_BROWSER_EXECUTABLE_FOLDER",
            "WEBVIEW2_WAIT_FOR_SCRIPT_DEBUGGER",
            "webview2_release_channels",
            "WEBVIEW2_FUTURE_OVERRIDE",
        ] {
            let error = check([variable(name, "private-fixture-value")], |_, _, _| {
                panic!("environment refusal must precede policy inspection")
            })
            .unwrap_err();
            assert!(error.contains(name));
            assert!(!error.contains("private-fixture-value"));
        }
    }

    #[test]
    fn isolated_webview_each_policy_override_refuses_without_reading_values() {
        for blocked_hive in [Hive::Machine, Hive::User] {
            for blocked_view in [View::Bits64, View::Bits32] {
                for blocked_name in POLICY_NAMES {
                    let error = check([], |hive, view, name| {
                        Ok(hive == blocked_hive && view == blocked_view && name == blocked_name)
                    })
                    .unwrap_err();
                    assert!(error.contains(blocked_hive.label()));
                    assert!(error.contains(blocked_view.label()));
                    assert!(error.contains(blocked_name));
                    assert!(error.contains("configured override"));
                }
            }
        }
    }

    #[test]
    fn isolated_webview_policy_inspection_failure_refuses() {
        for blocked_hive in [Hive::Machine, Hive::User] {
            for blocked_view in [View::Bits64, View::Bits32] {
                let error = check([], |hive, view, _| {
                    if hive == blocked_hive && view == blocked_view {
                        Err(())
                    } else {
                        Ok(false)
                    }
                })
                .unwrap_err();
                assert!(error.contains(blocked_hive.label()));
                assert!(error.contains(blocked_view.label()));
                assert!(error.contains("cannot inspect"));
            }
        }
    }
}
