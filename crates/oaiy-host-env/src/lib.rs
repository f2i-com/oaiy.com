//! The computer's own environment for a program OAIY starts, where OAIY runs from an AppImage.
//!
//! An AppImage's launcher points the environment at what the AppImage carries before it starts OAIY: AppImageKit's
//! AppRun puts `$APPDIR/usr/lib` and the rest first in LD_LIBRARY_PATH, `$APPDIR/usr/bin` in PATH, and the same in
//! PYTHONPATH, XDG_DATA_DIRS, QT_PLUGIN_PATH and GStreamer's; linuxdeploy's GTK hook points GIO_EXTRA_MODULES,
//! GTK_PATH, GDK_PIXBUF_MODULE_FILE, GSETTINGS_SCHEMA_DIR and others into `$APPDIR`. A program OAIY starts inherited
//! them all, and on a newer system than the one the AppImage was made on, the system's own programs loaded the
//! AppImage's older libraries and stopped: OAIY 0.1.4's AppImage (made on Ubuntu 22.04) on Ubuntu 26.04 had every model
//! download fail with "curl: symbol lookup error: /usr/lib/x86_64-linux-gnu/libcurl.so.4: undefined symbol:
//! nghttp2_option_set_no_rfc9113_leading_and_trailing_ws_validation".
//!
//! OAIY's own process keeps them (the WebKit processes it starts find the AppImage's libraries only through
//! LD_LIBRARY_PATH: their RUNPATH is their own folder), so each program OAIY starts is given the computer's instead
//! ([`apply`], [`vars`]): every variable loses its entries inside `$APPDIR`, and one left with none is removed; the
//! AppImage's own variables (APPDIR, APPIMAGE, ARGV0, OWD) are removed; and a program started in no folder of its own
//! starts where OAIY was started from, not inside the AppImage, where its launcher runs OAIY. Outside an AppImage
//! nothing changes. (GTK_THEME and GDK_BACKEND, which the GTK hook sets without keeping what they were, are left.)

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The variables an AppImage's runtime and launcher set about the AppImage itself.
const APPIMAGE_OWN: [&str; 4] = ["APPDIR", "APPIMAGE", "ARGV0", "OWD"];

/// The folder the AppImage this process runs from is mounted at (`$APPDIR`, which its launcher sets, an extracted
/// one's too); None elsewhere.
pub fn appdir() -> Option<PathBuf> {
    appdir_of(std::env::var_os("APPDIR"))
}

/// [`appdir`] of a value of `APPDIR`: an absolute folder, and not `/` (whose entries are every one).
fn appdir_of(value: Option<OsString>) -> Option<PathBuf> {
    let dir = PathBuf::from(value?);
    (dir.is_absolute() && dir.parent().is_some()).then_some(dir)
}

/// A program `program` to start, with the computer's environment ([`apply`]).
pub fn command(program: impl AsRef<OsStr>) -> Command {
    let mut command = Command::new(program);
    apply(&mut command);
    command
}

/// `command` given the computer's environment where this process runs from an AppImage, and left as it is elsewhere.
/// Before a command's own `env`s, so that a value it sets on purpose stays (this sets those the AppImage changed).
/// Not for a command whose environment is cleared and built by hand: that one is built from [`vars`].
pub fn apply(command: &mut Command) -> &mut Command {
    let Some(appdir) = appdir() else { return command };
    for (key, value) in changes(&appdir, std::env::vars_os()) {
        match value {
            Some(value) => command.env(key, value),
            None => command.env_remove(key),
        };
    }
    if command.get_current_dir().is_none() && std::env::current_dir().is_ok_and(|dir| dir.starts_with(&appdir)) {
        let started_in = std::env::var_os("OWD").map(PathBuf::from).filter(|dir| dir.is_dir());
        if let Some(dir) = started_in.or_else(|| std::env::var_os("HOME").map(PathBuf::from)) {
            command.current_dir(dir);
        }
    }
    command
}

/// This process's environment as a program it starts is given it ([`apply`]'s), for one built by hand from it (a
/// plugin's allow-list). A variable that is not Unicode is left out, as `std::env::vars` cannot give it.
pub fn vars() -> Vec<(String, String)> {
    let all: Vec<(OsString, OsString)> = std::env::vars_os().collect();
    let mut changed: HashMap<OsString, Option<OsString>> = match appdir() {
        Some(appdir) => changes(&appdir, all.iter().map(|(k, v)| (k, v))).into_iter().collect(),
        None => HashMap::new(),
    };
    all.into_iter()
        .filter_map(|(key, value)| match changed.remove(&key) {
            Some(Some(value)) => Some((key, value)),
            Some(None) => None,
            None => Some((key, value)),
        })
        .filter_map(|(key, value)| Some((key.into_string().ok()?, value.into_string().ok()?)))
        .collect()
}

/// What an environment `vars` becomes for a program started from an AppImage mounted at `appdir`: each variable that
/// changes, with its new value, or None where it goes. A variable it leaves alone is not in it.
pub fn changes<I, K, V>(appdir: &Path, vars: I) -> Vec<(OsString, Option<OsString>)>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<OsStr>,
    V: AsRef<OsStr>,
{
    let mut out = Vec::new();
    for (key, value) in vars {
        let (key, value) = (key.as_ref(), value.as_ref());
        if APPIMAGE_OWN.iter().any(|own| key == OsStr::new(own)) {
            out.push((key.to_os_string(), None));
        } else if let Some(rest) = without(appdir, value) {
            out.push((key.to_os_string(), rest));
        }
    }
    out
}

/// `value` (a list, `:` between its entries, or one entry) without its entries inside `appdir`: None where it has
/// none there (it stays as it is), Some(None) where that leaves nothing, else Some(the rest, in their order).
fn without(appdir: &Path, value: &OsStr) -> Option<Option<OsString>> {
    // (a value that is not Unicode is no path the launcher wrote: those are its mount's, which is)
    let (value, dir) = (value.to_str()?, appdir.to_str()?.trim_end_matches('/'));
    let inside = |entry: &str| entry == dir || entry.strip_prefix(dir).is_some_and(|rest| rest.starts_with('/'));
    if !value.split(':').any(inside) {
        return None;
    }
    // (an empty entry, which a launcher's `...:$LD_LIBRARY_PATH` leaves where there was none, is no folder: dropped)
    let rest: Vec<&str> = value.split(':').filter(|entry| !entry.is_empty() && !inside(entry)).collect();
    Some((!rest.is_empty()).then(|| OsString::from(rest.join(":"))))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOUNT: &str = "/tmp/.mount_OAIYab12";

    fn changed(vars: &[(&str, &str)]) -> Vec<(String, Option<String>)> {
        changes(Path::new(MOUNT), vars.iter().copied())
            .into_iter()
            .map(|(k, v)| (k.into_string().unwrap(), v.map(|v| v.into_string().unwrap())))
            .collect()
    }

    #[test]
    fn the_appimages_entries_go_and_the_computers_stay_in_their_order() {
        // As OAIY 0.1.4's AppImage started it on Ubuntu 26.04 (AppRun, then linuxdeploy's GTK hook).
        let got = changed(&[
            ("LD_LIBRARY_PATH", "/tmp/.mount_OAIYab12/usr/lib/:/tmp/.mount_OAIYab12/usr/lib/x86_64-linux-gnu/:"),
            ("PATH", "/tmp/.mount_OAIYab12/usr/bin/:/tmp/.mount_OAIYab12/usr/sbin/:/home/me/.local/bin:/usr/local/bin:/usr/bin"),
            ("XDG_DATA_DIRS", "/tmp/.mount_OAIYab12/usr/share/:/tmp/.mount_OAIYab12/usr/share:/usr/share:/usr/share/mate:/usr/local/share"),
            ("GDK_PIXBUF_MODULE_FILE", "/tmp/.mount_OAIYab12//usr/lib/gdk-pixbuf-2.0/2.10.0/loaders.cache"),
            ("GIO_EXTRA_MODULES", "/tmp/.mount_OAIYab12/usr/lib/x86_64-linux-gnu/gio/modules"),
            ("PYTHONPATH", "/tmp/.mount_OAIYab12/usr/share/pyshared/:/home/me/lib"),
        ]);
        assert_eq!(
            got,
            [
                ("LD_LIBRARY_PATH".into(), None),
                ("PATH".into(), Some("/home/me/.local/bin:/usr/local/bin:/usr/bin".into())),
                ("XDG_DATA_DIRS".into(), Some("/usr/share:/usr/share/mate:/usr/local/share".into())),
                ("GDK_PIXBUF_MODULE_FILE".into(), None),
                ("GIO_EXTRA_MODULES".into(), None),
                ("PYTHONPATH".into(), Some("/home/me/lib".into())),
            ]
        );
    }

    #[test]
    fn the_appimages_own_variables_go_and_everything_else_is_left_alone() {
        let got = changed(&[
            ("APPDIR", MOUNT),
            ("APPIMAGE", "/home/me/Downloads/OAIY.AppImage"),
            ("ARGV0", "./OAIY.AppImage"),
            ("OWD", "/home/me"),
            ("HOME", "/home/me"),
            ("DISPLAY", ":0"),
            ("GTK_THEME", "Adwaita:light"),
            ("OAIY_PUBLIC_URL", "https://example.com:8443/oaiy"),
            // another folder whose name begins with the mount's is not inside it
            ("SOMEWHERE", "/tmp/.mount_OAIYab12x/lib"),
        ]);
        assert_eq!(got, [("APPDIR".into(), None), ("APPIMAGE".into(), None), ("ARGV0".into(), None), ("OWD".into(), None)]);
    }

    #[test]
    fn an_appdir_is_an_absolute_folder_and_not_the_root() {
        assert_eq!(appdir_of(Some(MOUNT.into())), Some(PathBuf::from(MOUNT)));
        assert_eq!(appdir_of(None), None);
        assert_eq!(appdir_of(Some("relative/dir".into())), None);
        assert_eq!(appdir_of(Some("/".into())), None);
        // a mount named with a slash at its end is the same folder
        assert_eq!(changed(&[("PATH", "/tmp/.mount_OAIYab12/usr/bin:/usr/bin")]), changes(Path::new("/tmp/.mount_OAIYab12/"), [("PATH", "/tmp/.mount_OAIYab12/usr/bin:/usr/bin")]).into_iter().map(|(k, v)| (k.into_string().unwrap(), v.map(|v| v.into_string().unwrap()))).collect::<Vec<_>>());
    }

    #[test]
    fn outside_an_appimage_a_command_is_left_as_it_is() {
        // (the tests do not run from an AppImage)
        if appdir().is_some() {
            return;
        }
        let mut command = Command::new("true");
        apply(&mut command);
        assert_eq!(command.get_envs().count(), 0);
        assert!(command.get_current_dir().is_none());
        let names: Vec<String> = vars().into_iter().map(|(k, _)| k).collect();
        let expected: Vec<String> = std::env::vars_os().filter_map(|(k, v)| v.into_string().ok().and(k.into_string().ok())).collect();
        assert_eq!(names, expected);
    }
}
