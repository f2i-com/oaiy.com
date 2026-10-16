//! What a computer's GPUs are doing now: each one's load, the memory it has in use and its temperature, read from what
//! the system itself says, with no program run and nothing asked of the person ([`now`]):
//!
//! - **macOS** (Apple silicon): the GPU's driver in the IORegistry and the SMC's sensors (`mac`);
//! - **Linux**: the kernel's DRM devices in sysfs: AMD's driver's load and memory, every driver's hwmon sensors (`linux`);
//! - **Windows**: the graphics kernel's statistics (D3DKMT), Task Manager's, for every vendor's driver (`windows`).
//!
//! NVIDIA's own driver says all of it through nvidia-smi as well, which `oaiy-studio` asks first; this is how every
//! other GPU is read (AMD's, Intel's, an Apple-silicon Mac's, an NVIDIA card without NVIDIA's driver), and what a
//! system does not say is None. `oaiy-studio` forbids `unsafe`: macOS says these only through IOKit and CoreFoundation,
//! and Windows through gdi32, C calls that are this crate (each `unsafe` block with its `SAFETY:`); Linux's are files.

/// One GPU as it is now. Each reading is None where its system did not say it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Gpu {
    /// Its PCI vendor id (0x1002 AMD, 0x8086 Intel, 0x10de NVIDIA), as the system's list of display adapters names it;
    /// None for an Apple-silicon Mac's own GPU, which is on no PCI bus.
    pub vendor_id: Option<u32>,
    /// Its PCI device id.
    pub device_id: Option<u32>,
    /// The share of the time it was busy, in percent: as its driver last measured it, or (Windows) its busiest
    /// engine's since the last reading.
    pub load: Option<i64>,
    /// The bytes of memory it has in use: of its own (a card's), or of the computer's (an Apple-silicon Mac's).
    pub memory_in_use: Option<i64>,
    /// °C: the GPU's own sensor, or the mean of its sensors (a Mac's).
    pub temperature: Option<f64>,
}

/// The computer's GPUs as they are now, in its own order of them (Linux: its DRM cards'; Windows: its graphics
/// kernel's adapters'; a Mac: its one GPU). Empty where the system says nothing of them.
pub fn now() -> Vec<Gpu> {
    #[cfg(target_os = "macos")]
    return mac::read();
    #[cfg(target_os = "linux")]
    return linux::read(std::path::Path::new("/sys/class/drm"));
    #[cfg(windows)]
    return windows::read();
    #[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
    Vec::new()
}

/// The busiest engine's share of `seconds`, in percent, from each engine's running time (100 ns units) at two
/// readings `seconds` apart; None where they are not the same engines or no time has passed.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn busiest(before: &[i64], after: &[i64], seconds: f64) -> Option<i64> {
    if before.len() != after.len() || after.is_empty() || seconds <= 0.0 {
        return None;
    }
    let most = before.iter().zip(after).map(|(b, a)| (a - b).max(0)).max()?;
    Some(((most as f64 * 1e-7 / seconds) * 100.0).round().clamp(0.0, 100.0) as i64)
}

#[cfg(target_os = "macos")]
mod mac;

#[cfg(any(target_os = "linux", test))]
mod linux;

#[cfg(windows)]
mod windows;

#[cfg(test)]
mod tests {
    use super::*;

    /// Prints what this computer's GPUs say (`cargo test -p oaiy-gpu-stats -- --nocapture`); on an Apple-silicon Mac its
    /// one GPU says all three, in range.
    #[test]
    fn the_gpus_say_their_load_memory_and_temperature() {
        let gpus = now();
        println!("{gpus:?}");
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            let gpu = gpus[0];
            assert_eq!((gpus.len(), gpu.vendor_id), (1, None));
            assert!(gpu.load.is_some_and(|p| (0..=100).contains(&p)), "{gpu:?}");
            assert!(gpu.memory_in_use.is_some_and(|b| b > 0), "{gpu:?}");
            assert!(gpu.temperature.is_some_and(|t| (10.0..130.0).contains(&t)), "{gpu:?}");
            // (asked again: the SMC kept open, its keys found once)
            assert!(now()[0].temperature.is_some());
        }
        for gpu in &gpus {
            assert!(gpu.load.is_none_or(|p| (0..=100).contains(&p)) && gpu.temperature.is_none_or(|t| (1.0..130.0).contains(&t)), "{gpu:?}");
        }
    }

    /// Windows' load: the busiest engine's running time over the time between two readings (100 ns units).
    #[test]
    fn a_load_is_its_busiest_engines_share_of_the_time_between_readings() {
        // (over half a second: the 3D engine ran 0.3 s, the copy engine 0.05 s, video none)
        assert_eq!(busiest(&[1_000_000, 50, 7], &[4_000_000, 500_050, 7], 0.5), Some(60));
        assert_eq!(busiest(&[0], &[20_000_000], 1.0), Some(100), "never past 100");
        assert_eq!(busiest(&[5], &[3], 1.0), Some(0), "a counter that went back is no time");
        assert_eq!(busiest(&[1, 2], &[3], 1.0), None, "not the same engines");
        assert_eq!(busiest(&[1], &[3], 0.0), None);
        assert_eq!(busiest(&[], &[], 1.0), None);
    }
}
