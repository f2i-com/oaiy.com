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

/// `(available, total)` bytes of the computer's memory, for a build without CUDA: what the portable engine sizes its
/// expert cache from (a streamed MoE model such as GLM-5.3-Flash), as the CUDA build does from
/// `ggml_rs_cuda::host_memory`. Without it the portable build assumed 32 GB on any computer, so a model of 190 GB read
/// most of each token's experts from the disk on a computer with 192 GB.
///
/// The same reader as `ggml_rs_cuda::host_memory`, here because this is the crate every portable build links and
/// `ggml-rs` and `llama-rs` keep no `unsafe` at all: on Windows it is one documented call. `None` on a platform it has
/// no reader for, so a caller keeps a fallback.
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

#[cfg(not(any(windows, target_os = "linux")))]
pub(super) fn host_memory_impl() -> Option<(usize, usize)> {
    None
}

#[cfg(test)]
mod host_memory_tests {
    /// Whatever the platform, the answer must be self-consistent or absent.
    #[test]
    fn host_memory_is_plausible_or_absent() {
        if let Some((free, total)) = super::host_memory() {
            assert!(total > (1 << 30), "total {total} is implausibly small");
            assert!(free <= total, "free {free} exceeds total {total}");
        }
    }
}
