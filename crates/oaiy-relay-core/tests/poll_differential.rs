//! The poll rules against the repository's own readers of README 5.1.1 on random input: `tests/poll_diff_gen.py` writes a table of random cases (a float where an integer goes, a
//! seven-digit `Retry-After`, a `hold` that is not an object, an item without a `seq`, a date that has passed ...) with the expected answer of `verify_poll_client.py`, this
//! crate's `decide` is checked against every one, and `verify_poll_client.mjs` is run on the same table, so that a disagreement between the repository's two readers, or between
//! either of them and this code, on an input the hand-written table does not hold, is found here. On a machine without Python (or Node, for the second test) the test says SKIPPED in a banner and passes, unless `OAIY_REQUIRE_TOOLS=1` (the CI lane that has them), where it fails.

mod common;

use std::path::PathBuf;
use std::process::Command;

use common::{load_path, protocol_dir, run_poll_cases};

fn python() -> Option<&'static str> {
    ["python", "python3", "py"].into_iter().find(|p| Command::new(p).arg("--version").output().is_ok_and(|o| o.status.success()))
}

fn out_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
}

fn generate(py: &str, seed: u64, count: u64, no_integral_floats: bool) -> PathBuf {
    let out = out_dir().join(format!("poll-random-{seed}{}.json", if no_integral_floats { "-nif" } else { "" }));
    let mut cmd = Command::new(py);
    cmd.arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/poll_diff_gen.py")).arg(&out).arg(count.to_string()).arg(seed.to_string());
    if no_integral_floats {
        cmd.arg("no-integral-floats");
    }
    let status = cmd.status().expect("python");
    assert!(status.success(), "the generator failed");
    out
}

#[test]
fn random_cases_agree_with_the_python_reader() {
    let Some(py) = python() else {
        common::tool_missing("random_cases_agree_with_the_python_reader", "python is not installed, so no random table could be written");
        return;
    };
    let mut total = 0;
    for seed in [1, 2, 3] {
        let table = load_path(&generate(py, seed, 6000, false));
        total += run_poll_cases(&table);
    }
    assert!(total > 18_000, "{total}");
}

#[test]
fn the_node_reader_agrees_with_the_python_reader_on_the_same_random_cases() {
    let (Some(py), true) = (python(), Command::new("node").arg("--version").output().is_ok_and(|o| o.status.success())) else {
        common::tool_missing("the_node_reader_agrees_with_the_python_reader_on_the_same_random_cases", "python or node is not installed");
        return;
    };
    // Without the whole-number floats (`1.0`, `100.0`): JavaScript cannot tell that spelling from the integer, so the Node reader reads them as integers where the Python
    // reader and this crate read a float (a finding of this crate's README, "the readers disagree on"). The test below pins that it is the only thing they disagree on.
    let table = generate(py, 7, 6000, true);
    let out = Command::new("node")
        .arg(protocol_dir().join("fixtures/poll-client/verify_poll_client.mjs"))
        .arg("--file")
        .arg(&table)
        .output()
        .expect("node");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success() && text.contains(" 0 mismatches"), "{text}\n{}", String::from_utf8_lossy(&out.stderr));
}
