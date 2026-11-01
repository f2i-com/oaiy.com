//! Folders the tests make under the system temp folder, and when each goes. Before, every run left one behind for
//! each such test: sixteen thousand of them, gigabytes, on a computer that runs the suite often.
//!
//! libtest runs each test on a thread named for it, and [`temp_path`] keeps its list on that thread, so a test's folders
//! go when it ends, passed or failed. Two kinds of folder outlive a test and go when the process ends: one that went
//! into a store of the whole process ([`process_path`]), which the next test may be using, and one asked for on a
//! helper thread, which can end in the middle of the test that started it. A worker a test left running can write its
//! folder back after it has gone, so every folder is tried again then. A run that is killed removes nothing, and on
//! Windows nor does the end of a run that failed: libtest then leaves by `ExitProcess`, which skips the C library's exit.

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::{Mutex, Once};
use std::time::{Duration, Instant};

/// What goes when the process ends.
static AT_EXIT: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

extern "C" {
    /// The C library's: it is what runs when `main` returns and when libtest calls `exit`, and Rust has no hook of its own.
    fn atexit(callback: extern "C" fn()) -> std::ffi::c_int;
}

extern "C" fn remove_what_is_left() {
    for path in std::mem::take(&mut *AT_EXIT.lock().unwrap_or_else(|e| e.into_inner())) {
        let _ = std::fs::remove_dir_all(path);
    }
}

fn remove_at_exit(path: PathBuf) {
    static REGISTERED: Once = Once::new();
    // SAFETY: `atexit` keeps a plain function pointer, and this one neither unwinds nor needs anything of a thread.
    REGISTERED.call_once(|| unsafe {
        atexit(remove_what_is_left);
    });
    AT_EXIT.lock().unwrap_or_else(|e| e.into_inner()).push(path);
}

struct Made(Vec<PathBuf>);

impl Drop for Made {
    /// Windows removes no file that is open, and a worker of the test's may let go just after the test ends: so a
    /// second of tries.
    fn drop(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            self.0.retain(|path| std::fs::remove_dir_all(path).is_err() && path.exists());
            if self.0.is_empty() || Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }
}

thread_local! {
    static MADE: RefCell<Made> = const { RefCell::new(Made(Vec::new())) };
}

/// `<temp>/<prefix>-<uuid>`, not created here: removed when this test ends.
pub(crate) fn temp_path(prefix: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("{prefix}-{}", uuid::Uuid::new_v4().simple()));
    remove_at_exit(path.clone());
    // A test's thread has the test's path for its name. Any other is a helper's, which ends when it likes, or the main
    // thread (`--test-threads=1`), whose locals are never dropped.
    if std::thread::current().name().is_some_and(|name| name.contains("::")) {
        MADE.with(|made| made.borrow_mut().0.push(path.clone()));
    }
    path
}

/// `<temp>/<name>-<pid>`, the same for every test of this process, for a store the process keeps once it is opened
/// (`calendar::init`): removed when the process ends, since a test's end would take it from the tests still using it.
pub(crate) fn process_path(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
    remove_at_exit(path.clone());
    path
}

#[test]
fn a_tests_folder_goes_when_its_thread_ends_and_a_helpers_when_the_process_does() {
    // (a thread named as a test's: this file's own test cannot watch itself end)
    let mine = std::thread::Builder::new()
        .name("test_dirs::tests::a_test".into())
        .spawn(|| {
            let path = temp_path("oaiy-test-dirs");
            std::fs::create_dir_all(path.join("inside")).unwrap();
            path
        })
        .unwrap()
        .join()
        .unwrap();
    assert!(!mine.exists(), "gone with the test's thread");

    let helper = std::thread::spawn(|| {
        let path = temp_path("oaiy-test-dirs");
        std::fs::create_dir_all(&path).unwrap();
        path
    })
    .join()
    .unwrap();
    assert!(helper.exists(), "the test that started the helper is still using it");
    assert!(AT_EXIT.lock().unwrap().contains(&helper));
    std::fs::remove_dir_all(helper).unwrap();
}
