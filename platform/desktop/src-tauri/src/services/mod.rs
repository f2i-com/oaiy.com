//! Service management — JSON templates + process runner + in-memory
//! registry, plus model downloads and embedded Python.
//!
//! The flow:
//!   1. On startup, registry::init() seeds the user's config dir with
//!      built-in templates if none exist, then loads every *.json under
//!      that dir into the in-memory registry.
//!   2. HTTP handlers in http.rs read from / mutate the registry,
//!      downloads, and python helpers via the shared state.
//!   3. runner::Runner owns spawned child processes + ring-buffer log
//!      capture — used uniformly for service main processes, install
//!      scripts, and python venv setup so the UI's LogsViewer just
//!      points at "logs of this thing" and works for all three.
//!   4. downloads::Downloads handles HF / direct-URL model fetches with
//!      pause/resume; files land under `${dataDir}/models/`.
//!   5. python::Python manages a bundled python-build-standalone runtime
//!      + named, reusable venvs under `${dataDir}/venvs/`.

pub mod catalog;
pub mod downloads;
pub mod node_runtime;
pub mod python;
pub mod registry;
pub mod runner;
pub mod template;

// Re-exports kept under #[allow] so a future Tauri command outside this
// module can pull them without touching the module-paths everywhere.
// Today the HTTP layer and lib.rs use the explicit `registry::Registry`
// path, so these are technically unused in-crate.
#[allow(unused_imports)]
pub use registry::{Registry, RegistrySnapshot, ServiceRuntime, ServiceStatus};
#[allow(unused_imports)]
pub use template::ServiceTemplate;
