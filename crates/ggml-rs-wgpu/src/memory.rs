//! What memory there is: a card's own and the system's budget for this process, and the computer's.

use super::*;

/// The adapter's own memory where its API says: Vulkan's largest device-local heap (a discrete card's VRAM). None on
/// Direct3D 12 and Metal.
pub(super) fn device_memory(adapter: &wgpu::Adapter) -> Option<u64> {
    #[cfg(any(windows, target_os = "linux"))]
    {
        // SAFETY: the adapter's handles are only read, by a query that creates and frees nothing.
        let a = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
        let props = unsafe { a.shared_instance().raw_instance().get_physical_device_memory_properties(a.raw_physical_device()) };
        let heaps = &props.memory_heaps[..(props.memory_heap_count as usize).min(props.memory_heaps.len())];
        // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
        heaps.iter().filter(|h| h.flags.as_raw() & 1 != 0).map(|h| h.size).max()
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        let _ = adapter;
        None
    }
}

/// Vulkan's budget for this process on the adapter's largest device-local heap (VK_EXT_memory_budget: what the OS
/// lets it keep there, the rest of the computer's use of the card taken off) and its use of it now. None on other
/// APIs.
pub(super) fn heap_budget(adapter: &wgpu::Adapter) -> Option<(u64, u64)> {
    #[cfg(any(windows, target_os = "linux"))]
    {
        // SAFETY: the adapter's handles are only read, by a query that creates and frees nothing.
        let a = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
        let instance = a.shared_instance().raw_instance();
        let mut budget = ash::vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let heaps = {
            let mut props = ash::vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            unsafe { instance.get_physical_device_memory_properties2(a.raw_physical_device(), &mut props) };
            props.memory_properties
        };
        let n = (heaps.memory_heap_count as usize).min(heaps.memory_heaps.len());
        (0..n).filter(|&i| heaps.memory_heaps[i].flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL)).map(|i| (budget.heap_budget[i], budget.heap_usage[i])).max_by_key(|&(b, _)| b)
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        let _ = adapter;
        None
    }
}

/// The weights a discrete card with `memory` bytes holds by default: all but 4 GiB (the cache, the work buffers and
/// the rest of the computer's use of it), or half of a card under 8 GiB.
pub(super) fn discrete_budget(memory: u64) -> u64 {
    memory.saturating_sub(4 * GIB).max(memory / 2)
}

/// Whether the adapter's memory is the computer's own and the adapter its main GPU: an Apple-silicon Mac's (Metal
/// calls a GPU "integrated" where `hasUnifiedMemory` says so, and there every GPU of the computer's is). A PC's
/// integrated GPU keeps the small default: it is the slower of what such a computer has, beside a card or not.
pub(super) fn unified(info: &wgpu::AdapterInfo) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64")) && info.backend == wgpu::Backend::Metal && info.device_type == wgpu::DeviceType::IntegratedGpu
}

/// The weights a GPU that shares the computer's memory holds by default: what a card with two thirds of that memory
/// would ([`discrete_budget`]), the other third the system's, its programs' and the model's own use of the CPU. Where
/// the computer's owner has said how much the GPU may have (`wired_limit`: a Mac's `sysctl iogpu.wired_limit_mb`, 0
/// or absent where they have not), a card with that much, never more than the computer has.
pub(super) fn unified_budget(total: u64, wired_limit: Option<u64>) -> u64 {
    discrete_budget(wired_limit.filter(|&w| w > 0).unwrap_or(total / 3 * 2).min(total))
}

/// [`unified_budget`] of this computer; None where its memory cannot be read.
pub(super) fn host_unified_budget() -> Option<u64> {
    let (_, total) = host_memory()?;
    Some(unified_budget(total as u64, wired_limit()))
}

/// What the Mac's owner allows its GPU, in bytes (`iogpu.wired_limit_mb`; 0: the system's own choice).
#[cfg(target_os = "macos")]
fn wired_limit() -> Option<u64> {
    said("/usr/sbin/sysctl", &["-n", "iogpu.wired_limit_mb"])?.trim().parse::<u64>().ok().map(|mb| mb << 20)
}

#[cfg(not(target_os = "macos"))]
fn wired_limit() -> Option<u64> {
    None
}

/// `(available, total)` bytes of the computer's memory, for a build without CUDA: what the portable engine sizes its
/// expert cache from (a streamed MoE model such as GLM-5.3-Flash), as the CUDA build does from
/// `ggml_rs_cuda::host_memory`. Without it the portable build assumed 32 GB on any computer, so a model of 190 GB read
/// most of each token's experts from the disk on a computer with 192 GB.
///
/// The same reader as `ggml_rs_cuda::host_memory`, here because this is the crate every portable build links and
/// `ggml-rs` and `llama-rs` keep no `unsafe` at all: on Windows it is one documented call, on Linux a file, on a Mac
/// two of its own programs' answers. `None` on a platform it has no reader for, so a caller keeps a fallback.
pub fn host_memory() -> Option<(usize, usize)> {
    host_memory_impl()
}

#[cfg(windows)]
pub(super) fn host_memory_impl() -> Option<(usize, usize)> {
    // MEMORYSTATUSEX, exactly as documented: `length` must be set by the caller and every other field is written by
    // the call.
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut s = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    // SAFETY: `s` is a fully initialised MEMORYSTATUSEX of the size its own `length` field declares, and the call only
    // writes into it. The pointer is valid for the duration of the call and nothing retains it.
    let ok = unsafe { GlobalMemoryStatusEx(&mut s) };
    (ok != 0).then_some((s.avail_phys as usize, s.total_phys as usize))
}

#[cfg(target_os = "linux")]
pub(super) fn host_memory_impl() -> Option<(usize, usize)> {
    // /proc/meminfo reports both, in kB. MemAvailable is the kernel's own estimate of what a new allocation can have;
    // MemFree undercounts badly because of the page cache.
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| -> Option<usize> {
        text.lines().find(|l| l.starts_with(name)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<usize>().ok()).map(|kb| kb * 1024)
    };
    Some((field("MemAvailable:")?, field("MemTotal:")?))
}

/// What one of the programs every Mac has printed (`sysctl`, `vm_stat`), or None. By its full path: a program started
/// from the Finder has a short PATH.
#[cfg(target_os = "macos")]
fn said(program: &str, args: &[&str]) -> Option<String> {
    use std::process::{Command, Stdio};
    let out = Command::new(program).args(args).stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The bytes a new allocation can have on a Mac, from `vm_stat`'s text: its free, speculative and inactive pages
/// (the last the file cache the system drops when asked), at the page size the first line says (16 KB on Apple
/// silicon). "Pages free" alone undercounts badly, as Linux's MemFree does.
#[cfg(any(target_os = "macos", test))]
fn mac_available(vm_stat: &str) -> Option<usize> {
    let page = vm_stat.lines().next()?.split("page size of ").nth(1)?.split_whitespace().next()?.parse::<usize>().ok()?;
    let pages = |name: &str| vm_stat.lines().find_map(|l| l.strip_prefix(name)).and_then(|v| v.trim().trim_end_matches('.').parse::<usize>().ok());
    Some((pages("Pages free:")? + pages("Pages inactive:")? + pages("Pages speculative:").unwrap_or(0)) * page)
}

#[cfg(target_os = "macos")]
pub(super) fn host_memory_impl() -> Option<(usize, usize)> {
    // std has no memory query and this crate's other readers are one call each: here two programs' text, so nothing
    // of the system's headers is declared by hand for a computer no test of this crate has run on.
    let total = said("/usr/sbin/sysctl", &["-n", "hw.memsize"])?.trim().parse::<usize>().ok()?;
    let available = mac_available(&said("/usr/bin/vm_stat", &[])?)?;
    Some((available.min(total), total))
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
pub(super) fn host_memory_impl() -> Option<(usize, usize)> {
    None
}

#[cfg(test)]
mod host_memory_tests {
    use super::{mac_available, unified_budget, GIB};

    #[test]
    fn a_gpu_that_shares_the_computers_memory_holds_weights_as_a_card_with_two_thirds_of_it() {
        // 24 GB: a card of 16, less 4 for the cache and the work buffers; 48 GB: a card of 32
        assert_eq!(unified_budget(24 * GIB, None), 12 * GIB);
        assert_eq!(unified_budget(48 * GIB, Some(0)), 28 * GIB, "a limit of 0 is the system's own choice");
        // a small one: half of its two thirds, as a card under 8 GiB
        assert_eq!(unified_budget(9 * GIB, None), 3 * GIB);
        // the owner's own limit for the GPU, where they set one, and never more than there is
        assert_eq!(unified_budget(32 * GIB, Some(28 * GIB)), 24 * GIB);
        assert_eq!(unified_budget(32 * GIB, Some(100 * GIB)), 28 * GIB);
        assert_eq!(unified_budget(0, None), 0);
    }

    #[test]
    fn a_macs_available_memory_is_read_from_vm_stat() {
        // (the lines `vm_stat` prints, their counts made up: 16 KB pages, as Apple silicon's)
        let text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
                    Pages free:                               20000.\n\
                    Pages active:                            400000.\n\
                    Pages inactive:                          300000.\n\
                    Pages speculative:                         5000.\n\
                    Pages throttled:                              0.\n\
                    Pages wired down:                        150000.\n\
                    Pages purgeable:                           1000.\n\
                    \"Translation faults\":                 123456789.\n\
                    File-backed pages:                       250000.\n\
                    Anonymous pages:                         455000.\n\
                    Pages stored in compressor:               90000.\n\
                    Pages occupied by compressor:             30000.\n";
        assert_eq!(mac_available(text), Some((20000 + 300000 + 5000) * 16384));
        // an Intel Mac's 4 KB pages, and a system that prints no speculative line
        assert_eq!(mac_available("Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 10.\nPages inactive: 5.\n"), Some(15 * 4096));
        // text that is not vm_stat's
        assert_eq!(mac_available(""), None);
        assert_eq!(mac_available("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: many.\n"), None);
        assert_eq!(mac_available("Pages free: 10.\nPages inactive: 5.\n"), None);
    }

    /// Whatever the platform, the answer must be self-consistent or absent.
    #[test]
    fn host_memory_is_plausible_or_absent() {
        if let Some((free, total)) = super::host_memory() {
            assert!(total > (1 << 30), "total {total} is implausibly small");
            assert!(free <= total, "free {free} exceeds total {total}");
        }
    }
}
