//! What an Apple-silicon Mac's GPU is doing, read as Activity Monitor and the menu bar's meters read it, with no
//! program run and nothing asked of the user ([`now`]):
//!
//! - its load and the memory it has in use, from its driver: `PerformanceStatistics` in the IORegistry ("Device
//!   Utilization %", "In use system memory"; a 2 GB buffer raised the latter by 2.3 GB, and it fell back as the
//!   program ended);
//! - its temperature, from the SMC (the system management controller, where the Mac's sensors are read): the mean of
//!   the GPU's sensors, its `Tg` keys. On an M5 Pro there are 42, at 34 °C idle and 47 °C after 30 s of compute, while
//!   the CPU's (`Tp`) reached 41 °C.
//!
//! What a Mac does not say is None, and on every other system all of it is: `oaiy-studio`, which forbids `unsafe`,
//! shows what there is. macOS says these only through IOKit and CoreFoundation, whose C calls are this crate.

/// The GPU as it is now. Each is None where the Mac did not say it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Gpu {
    /// The share of the time its driver last measured that it was busy, in percent.
    pub load: Option<i64>,
    /// The bytes of the computer's memory it has in use.
    pub memory_in_use: Option<i64>,
    /// °C, the mean of its sensors.
    pub temperature: Option<f64>,
}

/// The GPU as it is now (on a Mac with Apple silicon; elsewhere nothing is known).
pub fn now() -> Gpu {
    #[cfg(target_os = "macos")]
    {
        let (load, memory_in_use) = mac::load_and_memory();
        Gpu { load, memory_in_use, temperature: mac::temperature() }
    }
    #[cfg(not(target_os = "macos"))]
    Gpu::default()
}

#[cfg(target_os = "macos")]
mod mac {
    use std::ffi::{c_char, c_void};
    use std::sync::{Mutex, OnceLock};

    type CFTypeRef = *const c_void;

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFStringCreateWithCString(alloc: CFTypeRef, text: *const c_char, encoding: u32) -> CFTypeRef;
        fn CFDictionaryGetValue(dict: CFTypeRef, key: CFTypeRef) -> CFTypeRef;
        fn CFNumberGetValue(number: CFTypeRef, kind: isize, value: *mut c_void) -> u8;
        fn CFGetTypeID(cf: CFTypeRef) -> usize;
        fn CFDictionaryGetTypeID() -> usize;
        fn CFNumberGetTypeID() -> usize;
        fn CFRelease(cf: CFTypeRef);
    }

    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IOServiceMatching(name: *const c_char) -> CFTypeRef;
        fn IOServiceGetMatchingService(main_port: u32, matching: CFTypeRef) -> u32;
        fn IORegistryEntryCreateCFProperty(entry: u32, key: CFTypeRef, alloc: CFTypeRef, options: u32) -> CFTypeRef;
        fn IOServiceOpen(service: u32, task: u32, kind: u32, connect: *mut u32) -> i32;
        fn IOConnectCallStructMethod(connect: u32, selector: u32, input: *const c_void, input_size: usize, output: *mut c_void, output_size: *mut usize) -> i32;
        fn IOObjectRelease(object: u32) -> i32;
    }

    extern "C" {
        static mach_task_self_: u32;
    }

    const UTF8: u32 = 0x0800_0100;
    const SINT64: isize = 4;

    /// The first IOKit service of class `class` (NUL-terminated), released by the caller; 0 where there is none.
    fn service(class: &[u8]) -> u32 {
        debug_assert_eq!(class.last(), Some(&0));
        // SAFETY: `class` is a NUL-terminated C string that outlives the call. IOServiceMatching returns a new
        // dictionary (or null, which IOServiceGetMatchingService takes and answers with 0), and
        // IOServiceGetMatchingService consumes that one reference, so nothing is leaked or released twice.
        unsafe { IOServiceGetMatchingService(0, IOServiceMatching(class.as_ptr().cast())) }
    }

    /// A number in the CFDictionary `dict` under `name` (NUL-terminated), where it is one.
    fn number(dict: CFTypeRef, name: &[u8]) -> Option<i64> {
        debug_assert_eq!(name.last(), Some(&0));
        // SAFETY: `dict` is a live CF object the caller holds a reference to; its type is checked before it is used as
        // a dictionary. The key string is created here and released after the lookup; CFDictionaryGetValue returns
        // a borrowed value (not released here), whose type is checked before CFNumberGetValue writes an i64 into
        // `out`, a properly aligned local.
        unsafe {
            if CFGetTypeID(dict) != CFDictionaryGetTypeID() {
                return None;
            }
            let key = CFStringCreateWithCString(std::ptr::null(), name.as_ptr().cast(), UTF8);
            if key.is_null() {
                return None;
            }
            let value = CFDictionaryGetValue(dict, key);
            CFRelease(key);
            let mut out = 0i64;
            let got = !value.is_null() && CFGetTypeID(value) == CFNumberGetTypeID() && CFNumberGetValue(value, SINT64, (&mut out as *mut i64).cast()) != 0;
            got.then_some(out)
        }
    }

    /// The GPU's load (percent) and the bytes of memory it has in use, as its driver says them.
    pub fn load_and_memory() -> (Option<i64>, Option<i64>) {
        let accelerator = service(b"IOAccelerator\0");
        if accelerator == 0 {
            return (None, None);
        }
        // SAFETY: `accelerator` is a service this function holds (released below, once). The key string is created
        // and released here. IORegistryEntryCreateCFProperty returns a new reference (or null), released below
        // after `number` has read from it.
        unsafe {
            let key = CFStringCreateWithCString(std::ptr::null(), c"PerformanceStatistics".as_ptr(), UTF8);
            let stats = if key.is_null() { std::ptr::null() } else { IORegistryEntryCreateCFProperty(accelerator, key, std::ptr::null(), 0) };
            if !key.is_null() {
                CFRelease(key);
            }
            IOObjectRelease(accelerator);
            if stats.is_null() {
                return (None, None);
            }
            let got = (number(stats, b"Device Utilization %\0"), number(stats, b"In use system memory\0"));
            CFRelease(stats);
            got
        }
    }

    /// The SMC, opened once, and its GPU sensors' keys (with their sizes), found once among its keys.
    struct Smc {
        connect: u32,
        gpu: Vec<(u32, u32)>,
    }

    // AppleSMC's one method, and the commands it takes (the struct's `data8`)
    const METHOD: u32 = 2;
    const READ_KEY: u8 = 5;
    const READ_INDEX: u8 = 8;
    const KEY_INFO: u8 = 9;
    const FLOAT: u32 = u32::from_be_bytes(*b"flt ");

    impl Smc {
        /// One request in the SMC's 80-byte struct (its key at 0, its key info's size at 28, the command at 42, an
        /// index at 44; in the answer, the result at 40 and the value's bytes from 48): the answer, where it says the
        /// request was done.
        fn call(&self, key: u32, command: u8, index: u32, size: u32) -> Option<[u8; 80]> {
            let mut input = [0u8; 80];
            input[0..4].copy_from_slice(&key.to_ne_bytes());
            input[28..32].copy_from_slice(&size.to_ne_bytes());
            input[42] = command;
            input[44..48].copy_from_slice(&index.to_ne_bytes());
            let (mut output, mut size) = ([0u8; 80], 80usize);
            // SAFETY: both buffers are 80 bytes, the size of AppleSMC's struct, as the sizes passed say; the kernel
            // writes at most `size` bytes into `output`. `connect` is an open connection to AppleSMC (`open`).
            let r = unsafe { IOConnectCallStructMethod(self.connect, METHOD, input.as_ptr().cast(), 80, output.as_mut_ptr().cast(), &mut size) };
            (r == 0 && output[40] == 0).then_some(output)
        }

        fn word(bytes: &[u8]) -> u32 {
            u32::from_ne_bytes(bytes[0..4].try_into().unwrap())
        }

        fn open() -> Option<Smc> {
            let smc_service = service(b"AppleSMC\0");
            if smc_service == 0 {
                return None;
            }
            let mut smc = Smc { connect: 0, gpu: Vec::new() };
            // SAFETY: `smc_service` is held here and released once, after the open; `mach_task_self_` is this task's
            // port, which libSystem sets before any code runs; `smc.connect` is a local the call writes.
            let opened = unsafe {
                let opened = IOServiceOpen(smc_service, mach_task_self_, 0, &mut smc.connect);
                IOObjectRelease(smc_service);
                opened
            };
            if opened != 0 {
                return None;
            }
            let count_key = u32::from_be_bytes(*b"#KEY");
            let info = smc.call(count_key, KEY_INFO, 0, 0)?;
            let count = smc.call(count_key, READ_KEY, 0, Self::word(&info[28..]))?;
            let count = u32::from_be_bytes(count[48..52].try_into().unwrap());
            for index in 0..count.min(20_000) {
                let Some(found) = smc.call(0, READ_INDEX, index, 0) else { continue };
                let key = Self::word(&found);
                if &key.to_be_bytes()[..2] != b"Tg" {
                    continue;
                }
                if let Some(info) = smc.call(key, KEY_INFO, 0, 0) {
                    let (size, kind) = (Self::word(&info[28..]), Self::word(&info[32..]));
                    if kind == FLOAT && size == 4 {
                        smc.gpu.push((key, size));
                    }
                }
            }
            Some(smc)
        }

        /// The mean of the GPU's sensors that read as a temperature (one not in use reads 0).
        fn gpu_temperature(&self) -> Option<f64> {
            let read: Vec<f64> = self
                .gpu
                .iter()
                .filter_map(|&(key, size)| self.call(key, READ_KEY, 0, size))
                .map(|out| f32::from_le_bytes(out[48..52].try_into().unwrap()) as f64)
                .filter(|t| (10.0..130.0).contains(t))
                .collect();
            (!read.is_empty()).then(|| read.iter().sum::<f64>() / read.len() as f64)
        }
    }

    /// The GPU's temperature (°C), where the Mac has sensors for it. The SMC is opened at the first asking and kept
    /// (one connection, its requests one at a time).
    pub fn temperature() -> Option<f64> {
        static SMC: OnceLock<Mutex<Option<Smc>>> = OnceLock::new();
        let smc = SMC.get_or_init(|| Mutex::new(Smc::open())).lock().unwrap_or_else(|p| p.into_inner());
        smc.as_ref()?.gpu_temperature()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prints what this computer's GPU says (`cargo test -p oaiy-mac-gpu -- --nocapture`); on an Apple-silicon Mac all
    /// three are there and in range.
    #[test]
    fn the_gpu_says_its_load_memory_and_temperature_on_an_apple_silicon_mac() {
        let gpu = now();
        println!("{gpu:?}");
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert!(gpu.load.is_some_and(|p| (0..=100).contains(&p)), "{gpu:?}");
            assert!(gpu.memory_in_use.is_some_and(|b| b > 0), "{gpu:?}");
            assert!(gpu.temperature.is_some_and(|t| (10.0..130.0).contains(&t)), "{gpu:?}");
            // (asked again: the SMC kept open, its keys found once)
            assert!(now().temperature.is_some());
        } else if !cfg!(target_os = "macos") {
            assert_eq!(gpu, Gpu::default());
        }
    }
}
