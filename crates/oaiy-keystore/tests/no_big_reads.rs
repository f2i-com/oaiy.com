//! "A file that is far larger than any secret is not read into memory": the promise of `read_bounded`, observed. The test in `keystore.rs` shows that
//! such a file is an error; this one shows that the error comes before the allocation. A counting global allocator records the largest request made
//! while `get` looks at a sparse 300 MiB file that costs the disk nothing; for the keyfile and for DPAPI it must be under a megabyte.
//!
//! This is the only test in this binary, so no other thread's allocations are counted. The `unsafe` is test code: a `GlobalAlloc` wrapper that forwards every
//! call to the system allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use oaiy_keystore::{open_at, KeyError, Name, ProviderChoice};

struct Counting;

static LARGEST: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the arguments it was given and only records the requested size.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LARGEST.fetch_max(new_size, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn a_file_far_larger_than_any_secret_is_refused_before_it_is_read_into_memory() {
    let mut providers = Vec::new();
    #[cfg(unix)]
    providers.push((ProviderChoice::Keyfile, "kf"));
    #[cfg(all(windows, feature = "unsafe-keyfile"))]
    providers.push((ProviderChoice::KeyfileUnsafe, "kf"));
    if cfg!(windows) {
        providers.push((ProviderChoice::DpapiFile, "ks"));
    }
    for (choice, ext) in providers {
        let dir = std::env::temp_dir().join(format!("oaiy-keystore-bigread-{ext}-{}", std::process::id()));
        let store = open_at(dir.join("keys"), choice).unwrap();
        let path = dir.join("keys").join(format!("huge.file.{ext}"));
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(300 * 1024 * 1024).unwrap(); // sparse: 300 MiB that costs nothing
        drop(file);
        #[cfg(unix)]
        {
            // the mode is checked before the size: make it a permissible file so that it is the size that is refused
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        LARGEST.store(0, Ordering::Relaxed);
        let result = store.get(&Name::new("huge.file").unwrap());
        let asked = LARGEST.load(Ordering::Relaxed);
        assert!(matches!(result, Err(KeyError::Corrupt(_))), "{ext}: {result:?}");
        assert!(asked < 1024 * 1024, "{ext}: reading a 300 MiB file asked the allocator for {asked} bytes in one piece");
        drop(store); // the store holds its folder open, and Windows will not remove an open folder
        let _ = std::fs::remove_dir_all(&dir);
    }
}
