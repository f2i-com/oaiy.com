//! The NVIDIA engine: the CUDA build of the language-model server, an optional download.
//!
//! The installer carries the portable engine (WebGPU, else the CPU). On a computer with an NVIDIA card the CUDA
//! build of the same release is faster, and runs the models only it can (DeepSeek V4.1). It needs nothing from
//! NVIDIA but the driver (its kernels are compiled when it is built, and its GEMM is its own), so it is a small
//! download: `oaiy-cuda-engine-<version>-<platform>`, an asset of this release on GitHub.
//!
//! Before it is kept it is checked. Its minisign signature, with the key updates are checked with, must have been made
//! for exactly that file name, which names this version and this platform: an older release's engine, or another
//! platform's, is refused, whatever the bytes. The archive must be the kind its name says, and hold the program. It is
//! unpacked into `<data>/engines/cuda/<version>/` and the studio told to run it (`llm.server`): through its control
//! API when it is running (which restarts a running model on it), else in its configuration file.
//!
//! An engine takes the command line of its own release, so at each start [`reconcile`] keeps the studio on this
//! version's engine, and sets another version's aside (the folder is removed) until this version's is fetched.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use base64::Engine as _;
use futures_util::StreamExt;
use minisign_verify::{PublicKey, Signature};
use serde::Serialize;

/// The largest engine archive, and program, taken: far above the real one (about 10 MB packed, 26 MB unpacked).
const LIMIT: u64 = 256 << 20;
/// The most redirects a download follows (GitHub sends a release asset through two), and only to https.
const MAX_REDIRECTS: usize = 5;

/// The release asset of `version` for `os` (`std::env::consts::OS`) on x86-64: None where no CUDA build is made.
fn asset_for(version: &str, os: &str, arch: &str) -> Option<String> {
    match (os, arch) {
        ("windows", "x86_64") => Some(format!("oaiy-cuda-engine-{version}-windows-x64.zip")),
        ("linux", "x86_64") => Some(format!("oaiy-cuda-engine-{version}-linux-x86_64.tar.gz")),
        _ => None,
    }
}

/// This platform's asset for `version`.
pub fn asset(version: &str) -> Option<String> {
    asset_for(version, std::env::consts::OS, std::env::consts::ARCH)
}

fn program() -> &'static str {
    if cfg!(windows) {
        "oaiy-llm-server.exe"
    } else {
        "oaiy-llm-server"
    }
}

/// The folder every version's engine is kept under.
fn root(data_dir: &Path) -> PathBuf {
    data_dir.join("engines").join("cuda")
}

/// Where `version`'s engine is kept.
pub fn folder(data_dir: &Path, version: &str) -> PathBuf {
    root(data_dir).join(version)
}

/// `version`'s engine program, when it has been fetched.
pub fn installed(data_dir: &Path, version: &str) -> Option<PathBuf> {
    let p = folder(data_dir, version).join(program());
    p.is_file().then_some(p)
}

fn decode_text(b64: &str) -> Option<String> {
    String::from_utf8(base64::engine::general_purpose::STANDARD.decode(b64.trim()).ok()?).ok()
}

/// The `file:` field of a trusted comment (tab-separated `key:value` pairs): None when absent or there twice.
fn signed_file(comment: &str) -> Option<&str> {
    let mut found = comment.split('\t').filter_map(|part| part.strip_prefix("file:"));
    let first = found.next()?;
    found.next().is_none().then_some(first)
}

/// Check `bytes` against `signature` (the content of the `.sig` file) with `pubkey` (both base64, as the Tauri CLI
/// writes them), and that the signature was made for `asset` and the bytes are the archive its name says.
pub fn verify(bytes: &[u8], signature: &str, pubkey: &str, asset: &str) -> Result<(), String> {
    let key = decode_text(pubkey).and_then(|t| PublicKey::decode(&t).ok()).ok_or("This copy of OAIY has no usable key to check downloads with, so the NVIDIA engine was refused.")?;
    let sig = decode_text(signature).and_then(|t| Signature::decode(&t).ok()).ok_or("The NVIDIA engine's signature is not in a form that can be read, so it was refused.")?;
    key.verify(bytes, &sig, true).map_err(|_| "The downloaded NVIDIA engine does not match its signature, so it was thrown away. It may be damaged, or not made by OAIY.".to_string())?;
    // The trusted comment is covered by the signature just checked.
    let file = signed_file(sig.trusted_comment()).ok_or("The NVIDIA engine's signature does not say which file it was made for, so it was refused.")?;
    if file != asset {
        return Err(format!("The NVIDIA engine's signature was made for {file}, not {asset}, so it was refused."));
    }
    let kind_fits = if asset.ends_with(".zip") { bytes.starts_with(b"PK\x03\x04") } else { bytes.starts_with(&[0x1f, 0x8b]) };
    if !kind_fits {
        return Err(format!("The downloaded NVIDIA engine is not the archive {asset} should be, so it was thrown away."));
    }
    Ok(())
}

/// The program out of the archive `asset` (a zip, else a .tar.gz), written into `into` (a file beside it first, then
/// renamed, so a half-written program is never there). The path of the program.
fn unpack(bytes: &[u8], asset: &str, into: &Path) -> Result<PathBuf, String> {
    let bad = |e: &dyn std::fmt::Display| format!("The NVIDIA engine's archive could not be read ({e}).");
    let mut data = Vec::new();
    if asset.ends_with(".zip") {
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| bad(&e))?;
        let entry = zip.by_name(program()).map_err(|_| format!("The NVIDIA engine's archive has no {}.", program()))?;
        entry.take(LIMIT + 1).read_to_end(&mut data).map_err(|e| bad(&e))?;
    } else {
        let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(bytes));
        let mut found = false;
        for entry in archive.entries().map_err(|e| bad(&e))? {
            let entry = entry.map_err(|e| bad(&e))?;
            let path = entry.path().map_err(|e| bad(&e))?.into_owned();
            if entry.header().entry_type().is_file() && path.strip_prefix(".").unwrap_or(&path) == Path::new(program()) {
                entry.take(LIMIT + 1).read_to_end(&mut data).map_err(|e| bad(&e))?;
                found = true;
                break;
            }
        }
        if !found {
            return Err(format!("The NVIDIA engine's archive has no {}.", program()));
        }
    }
    if data.len() as u64 > LIMIT {
        return Err("The NVIDIA engine in the archive is far larger than it should be, so it was refused.".into());
    }
    std::fs::create_dir_all(into).map_err(|e| format!("Could not make {}: {e}", into.display()))?;
    let dest = into.join(program());
    let part = into.join(format!("{}.part", program()));
    std::fs::write(&part, &data).map_err(|e| format!("Could not write {}: {e}", part.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&part, std::fs::Permissions::from_mode(0o755)).map_err(|e| format!("Could not make {} runnable: {e}", part.display()))?;
    }
    std::fs::rename(&part, &dest).map_err(|e| format!("Could not put the NVIDIA engine in place: {e}"))?;
    Ok(dest)
}

/// What [`reconcile`] does with the studio's `llm.server`, from it and this version's engine: None leaves it.
fn wanted_server(server: &str, current: Option<&Path>, root: &Path) -> Option<String> {
    let ours = Path::new(server).starts_with(root);
    match current {
        // This version's engine, unless the studio runs it already.
        Some(p) if Path::new(server) != p => Some(p.to_string_lossy().into_owned()),
        Some(_) => None,
        // Another version's (or one removed): the default again.
        None if ours => Some("oaiy-llm-server".into()),
        None => None,
    }
}

/// At a start, before the studio is: keep it on this version's engine when there is one, take it off another
/// version's, and remove other versions' folders.
pub fn reconcile(config: &Path, data_dir: &Path, version: &str) -> Result<(), String> {
    let root = root(data_dir);
    let current = installed(data_dir, version);
    let server = oaiy_studio::llm_server(config)?;
    if let Some(value) = wanted_server(&server, current.as_deref(), &root) {
        oaiy_studio::set_llm_server(config, &value)?;
        log::info!("engines: the CUDA language-model server is now {value}");
    }
    if let Ok(entries) = std::fs::read_dir(&root) {
        for e in entries.flatten() {
            if e.file_name() != version && e.path().is_dir() {
                match std::fs::remove_dir_all(e.path()) {
                    Ok(()) => log::info!("engines: removed another version's NVIDIA engine, {}", e.path().display()),
                    Err(err) => log::warn!("engines: could not remove {}: {err}", e.path().display()),
                }
            }
        }
    }
    Ok(())
}

/// A fetch's progress, for the window to show.
#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Fetch {
    /// `idle`, `downloading`, `checking`, `done` or `failed`.
    pub state: String,
    pub got: u64,
    pub total: Option<u64>,
    pub error: Option<String>,
}

static FETCH: Mutex<Option<Fetch>> = Mutex::new(None);

fn fetch_state() -> Fetch {
    FETCH.lock().unwrap_or_else(|p| p.into_inner()).clone().unwrap_or_else(|| Fetch { state: "idle".into(), ..Default::default() })
}

fn set_fetch(f: impl FnOnce(&mut Fetch)) {
    let mut g = FETCH.lock().unwrap_or_else(|p| p.into_inner());
    let mut cur = g.clone().unwrap_or_else(|| Fetch { state: "idle".into(), ..Default::default() });
    f(&mut cur);
    *g = Some(cur);
}

/// What the window shows of the NVIDIA engine.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    /// A CUDA build is made for this kind of computer.
    pub offered: bool,
    /// This computer has an NVIDIA GPU with its driver.
    pub nvidia: bool,
    /// This version's engine has been fetched.
    pub installed: bool,
    pub version: String,
    pub fetch: Fetch,
}

/// Whether `nvidia-smi` lists a GPU, asked once: it can take a second.
fn nvidia() -> bool {
    static SEEN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SEEN.get_or_init(oaiy_studio::nvidia_gpu)
}

pub fn status(data_dir: &Path) -> Status {
    let version = env!("CARGO_PKG_VERSION");
    Status { offered: asset(version).is_some(), nvidia: nvidia(), installed: installed(data_dir, version).is_some(), version: version.into(), fetch: fetch_state() }
}

fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(concat!("oaiy-desktop/", env!("CARGO_PKG_VERSION")))
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= MAX_REDIRECTS {
                attempt.error("too many redirects")
            } else if attempt.url().scheme() != "https" {
                attempt.error("a redirect away from https")
            } else {
                attempt.follow()
            }
        }))
        .build()
        .map_err(|e| format!("Could not start a download: {e}"))
}

/// Tell the studio to run `value` as its CUDA language-model server: through its control API when it is running,
/// else in its configuration file.
async fn point_studio(data_dir: &Path, value: &str) -> Result<(), String> {
    match crate::http::engines_ui() {
        Some(ui) => {
            let mut cfg = crate::http::studio_json(&ui, reqwest::Method::GET, "/api/config", None).await.map_err(|(_, e)| e)?;
            let llm = cfg.as_object_mut().ok_or("the engines' configuration is not an object")?.entry("llm").or_insert_with(|| serde_json::json!({}));
            llm.as_object_mut().ok_or("the engines' llm settings are not an object")?.insert("server".into(), serde_json::Value::String(value.into()));
            crate::http::studio_json(&ui, reqwest::Method::PUT, "/api/config", Some(cfg)).await.map_err(|(_, e)| e)?;
            Ok(())
        }
        None => oaiy_studio::set_llm_server(&crate::engines::config_path(data_dir), value),
    }
}

/// Fetch, check and install this version's NVIDIA engine (one fetch at a time), and point the studio at it.
pub async fn fetch(data_dir: PathBuf, pubkey: String) -> Result<PathBuf, String> {
    let version = env!("CARGO_PKG_VERSION");
    let asset = asset(version).ok_or("There is no NVIDIA engine for this kind of computer.")?;
    {
        let mut g = FETCH.lock().unwrap_or_else(|p| p.into_inner());
        if g.as_ref().is_some_and(|f| f.state == "downloading" || f.state == "checking") {
            return Err("The NVIDIA engine is being fetched already.".into());
        }
        *g = Some(Fetch { state: "downloading".into(), ..Default::default() });
    }
    let result: Result<PathBuf, String> = async {
        let base = format!("https://github.com/{}/releases/download/v{version}/", crate::update::REPO);
        let client = client()?;
        let failed = |e: reqwest::Error| format!("The NVIDIA engine could not be downloaded ({e}). Check the internet connection and try again.");
        let signature = client.get(format!("{base}{asset}.sig")).send().await.and_then(|r| r.error_for_status()).map_err(failed)?.text().await.map_err(failed)?;
        let response = client.get(format!("{base}{asset}")).send().await.and_then(|r| r.error_for_status()).map_err(failed)?;
        let total = response.content_length();
        if total.is_some_and(|n| n > LIMIT) {
            return Err("The NVIDIA engine on offer is far larger than it should be, so it was not downloaded.".into());
        }
        set_fetch(|f| f.total = total);
        let mut bytes = Vec::with_capacity(total.unwrap_or(0) as usize);
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            bytes.extend_from_slice(&chunk.map_err(failed)?);
            if bytes.len() as u64 > LIMIT {
                return Err("The NVIDIA engine on offer is far larger than it should be, so it was thrown away.".into());
            }
            let got = bytes.len() as u64;
            set_fetch(|f| f.got = got);
        }
        set_fetch(|f| f.state = "checking".into());
        verify(&bytes, &signature, &pubkey, &asset)?;
        let program = unpack(&bytes, &asset, &folder(&data_dir, version))?;
        point_studio(&data_dir, &program.to_string_lossy()).await?;
        log::info!("engines: the NVIDIA engine {version} is in {}", program.display());
        Ok(program)
    }
    .await;
    set_fetch(|f| match &result {
        Ok(_) => f.state = "done".into(),
        Err(e) => {
            f.state = "failed".into();
            f.error = Some(e.clone());
        }
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real signatures by a throwaway key (testdata/README.txt): a zip and a .tar.gz holding a stand-in program,
    // signed under the asset names of version 0.1.0.
    const PUBKEY: &str = include_str!("nvidia_engine/testdata/throwaway.key.pub");
    const ZIP: &[u8] = include_bytes!("nvidia_engine/testdata/engine-windows.zip.bin");
    const ZIP_SIG: &str = include_str!("nvidia_engine/testdata/engine-windows.zip.bin.sig");
    const TGZ: &[u8] = include_bytes!("nvidia_engine/testdata/engine-linux.tar.gz.bin");
    const TGZ_SIG: &str = include_str!("nvidia_engine/testdata/engine-linux.tar.gz.bin.sig");
    const WIN: &str = "oaiy-cuda-engine-0.1.0-windows-x64.zip";
    const LINUX: &str = "oaiy-cuda-engine-0.1.0-linux-x86_64.tar.gz";

    #[test]
    fn each_platform_with_a_cuda_build_has_an_asset_named_for_the_version() {
        assert_eq!(asset_for("0.2.0", "windows", "x86_64").as_deref(), Some("oaiy-cuda-engine-0.2.0-windows-x64.zip"));
        assert_eq!(asset_for("0.2.0", "linux", "x86_64").as_deref(), Some("oaiy-cuda-engine-0.2.0-linux-x86_64.tar.gz"));
        assert_eq!(asset_for("0.2.0", "macos", "aarch64"), None);
        assert_eq!(asset_for("0.2.0", "linux", "aarch64"), None);
    }

    #[test]
    fn a_signed_engine_for_this_name_is_accepted() {
        verify(ZIP, ZIP_SIG, PUBKEY, WIN).unwrap();
        verify(TGZ, TGZ_SIG, PUBKEY, LINUX).unwrap();
    }

    #[test]
    fn an_engine_signed_for_another_version_or_platform_is_refused() {
        let other_version = verify(ZIP, ZIP_SIG, PUBKEY, "oaiy-cuda-engine-0.2.0-windows-x64.zip").unwrap_err();
        assert!(other_version.contains("made for oaiy-cuda-engine-0.1.0-windows-x64.zip"), "{other_version}");
        let other_platform = verify(TGZ, TGZ_SIG, PUBKEY, WIN).unwrap_err();
        assert!(other_platform.contains("not oaiy-cuda-engine-0.1.0-windows-x64.zip"), "{other_platform}");
    }

    #[test]
    fn changed_bytes_or_another_key_are_refused() {
        let mut changed = ZIP.to_vec();
        let last = changed.len() - 1;
        changed[last] ^= 1;
        assert!(verify(&changed, ZIP_SIG, PUBKEY, WIN).unwrap_err().contains("does not match its signature"));
        let updates_key = include_str!("../tauri.conf.json");
        let conf: serde_json::Value = serde_json::from_str(updates_key).unwrap();
        let production = conf["plugins"]["updater"]["pubkey"].as_str().unwrap();
        assert!(verify(ZIP, ZIP_SIG, production, WIN).unwrap_err().contains("does not match its signature"));
        assert!(verify(ZIP, "not a signature", PUBKEY, WIN).unwrap_err().contains("not in a form that can be read"));
    }

    #[test]
    fn the_program_comes_out_of_either_archive_and_nothing_half_written_is_left() {
        let dir = std::env::temp_dir().join(format!("oaiy-nvidia-engine-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // The archives hold the stand-in under both platforms' names, so either platform's test reads its own.
        for (bytes, asset) in [(ZIP, WIN), (TGZ, LINUX)] {
            let into = dir.join(asset);
            let program = unpack(bytes, asset, &into).unwrap();
            assert_eq!(program, into.join(super::program()));
            assert_eq!(std::fs::read(&program).unwrap(), b"a stand-in for oaiy-llm-server\n");
            assert!(!into.join(format!("{}.part", super::program())).exists());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_studio_is_kept_on_this_versions_engine_and_taken_off_another_versions() {
        let root = Path::new("/data/engines/cuda");
        let mine = root.join("0.2.0").join(program());
        // Fetched for this version: run it, unless it is run already.
        assert_eq!(wanted_server("oaiy-llm-server", Some(&mine), root), Some(mine.to_string_lossy().into_owned()));
        assert_eq!(wanted_server(&root.join("0.1.0").join(program()).to_string_lossy(), Some(&mine), root), Some(mine.to_string_lossy().into_owned()));
        assert_eq!(wanted_server(&mine.to_string_lossy(), Some(&mine), root), None);
        // Not fetched for this version: off another version's, and anything else left as it is.
        assert_eq!(wanted_server(&root.join("0.1.0").join(program()).to_string_lossy(), None, root).as_deref(), Some("oaiy-llm-server"));
        assert_eq!(wanted_server("oaiy-llm-server", None, root), None);
        assert_eq!(wanted_server("C:/OAIY/engines/oaiy-llm-server.exe", None, root), None);
    }

    #[test]
    fn reconcile_points_the_configuration_at_this_versions_engine_and_removes_others() {
        let data = std::env::temp_dir().join(format!("oaiy-nvidia-reconcile-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data);
        let config = crate::engines::config_path(&data);
        std::fs::create_dir_all(config.parent().unwrap()).unwrap();
        for v in ["0.1.0", "0.2.0"] {
            std::fs::create_dir_all(folder(&data, v)).unwrap();
            std::fs::write(folder(&data, v).join(program()), b"x").unwrap();
        }
        reconcile(&config, &data, "0.2.0").unwrap();
        assert_eq!(oaiy_studio::llm_server(&config).unwrap(), folder(&data, "0.2.0").join(program()).to_string_lossy());
        assert!(!folder(&data, "0.1.0").exists(), "another version's engine is removed");
        // This version's removed too (a later version's start did it): back to the default.
        std::fs::remove_dir_all(folder(&data, "0.2.0")).unwrap();
        reconcile(&config, &data, "0.2.0").unwrap();
        assert_eq!(oaiy_studio::llm_server(&config).unwrap(), "oaiy-llm-server");
        let _ = std::fs::remove_dir_all(&data);
    }
}
