//! SSD -> RAM -> GPU weight residency for block-structured image models.
//!
//! The same three tiers the LLM side uses (and `ltx::store` for video), for the
//! Qwen Image transformer and its Qwen3-VL text encoder: each block lives where
//! the budget allows, in this order of preference:
//!
//! * **GPU**: resident for the whole job, as every block was before this module.
//! * **RAM**: a host copy, uploaded before each use and dropped after it. Costs one
//!   PCIe transfer per block per pass; saves its VRAM.
//! * **SSD**: nothing kept; the block is read from the published weights again for
//!   each pass. Costs the read (and, for ConvRot checkpoints, the host decode);
//!   saves both VRAM and RAM.
//!
//! Residency never changes the arithmetic: a block runs on the device with the
//! same weights whichever tier it came from, so the output is bit-identical.
//! Blocks are loaded onto the device first and demoted from there, so quantized
//! and ConvRot decoding always happens once, where it is cheapest.
use candle_core::{Device, Result};
use nrob::json::Json;

pub const GIB: u64 = 1 << 30;

/// VRAM kept free beside resident weights: one streamed block, the attention and
/// MLP activations of a 2048x2048 pass, and the VAE that loads after them.
#[cfg(feature = "cuda")]
const HEADROOM: u64 = 6 * GIB;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Memory {
    /// GPU while the VRAM budget lasts, then RAM, then SSD.
    Auto,
    /// Everything on the GPU (the behaviour before tiering; fails if it does not fit).
    Gpu,
    /// Nothing resident on the GPU but the block in use; host copies up to the RAM
    /// budget, SSD for the rest.
    Ram,
    /// Only the block in use is anywhere but the disk.
    Ssd,
}

impl Memory {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "gpu" => Ok(Self::Gpu),
            "ram" => Ok(Self::Ram),
            "ssd" => Ok(Self::Ssd),
            _ => Err("memory must be auto, gpu, ram or ssd".into()),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Gpu => "gpu",
            Self::Ram => "ram",
            Self::Ssd => "ssd",
        }
    }
}

/// Where a job's weights may live.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub memory: Memory,
    /// Weight VRAM; `None` means "whatever is free, less the headroom".
    pub vram_bytes: Option<u64>,
    /// Host copies of blocks that do not stay on the GPU.
    pub ram_bytes: u64,
}

impl Default for Budget {
    fn default() -> Self {
        Self { memory: Memory::Auto, vram_bytes: None, ram_bytes: 32 * GIB }
    }
}

impl Budget {
    /// `memory` (auto|gpu|ram|ssd), `vram_gb` and `ram_gb` from a worker request.
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        let memory = match j.get("memory") {
            None | Some(Json::Null) => Memory::Auto,
            Some(v) => Memory::parse(v.as_str().ok_or("memory must be a string")?)?,
        };
        let gib = |key: &str, max: i64| -> std::result::Result<Option<u64>, String> {
            match j.get(key) {
                None | Some(Json::Null) => Ok(None),
                Some(v) => {
                    let n = v.as_i64().ok_or_else(|| format!("{key} must be an integer"))?;
                    if !(0..=max).contains(&n) {
                        return Err(format!("{key} must be between 0 and {max}"));
                    }
                    Ok(Some(n as u64 * GIB))
                }
            }
        };
        Ok(Self {
            memory,
            vram_bytes: gib("vram_gb", 192)?,
            ram_bytes: gib("ram_gb", 512)?.unwrap_or(32 * GIB),
        })
    }

    /// The VRAM weights may take on `dev`: the configured budget, capped by what
    /// the device has free now less the headroom.
    pub fn vram_limit(&self, dev: &Device) -> Result<u64> {
        let configured = self.vram_bytes.unwrap_or(u64::MAX);
        #[cfg(feature = "cuda")]
        if let Ok(cuda) = dev.as_cuda_device() {
            let free = cuda
                .cuda_stream()
                .context()
                .mem_get_info()
                .map_err(candle_core::Error::wrap)?
                .0 as u64;
            return Ok(configured.min(free.saturating_sub(HEADROOM)));
        }
        let _ = dev;
        Ok(configured)
    }

    pub fn to_json(&self) -> Json {
        Json::obj([
            ("memory", Json::str(self.memory.name())),
            ("vram_gb", self.vram_bytes.map_or(Json::Null, |b| Json::Int((b / GIB) as i64))),
            ("ram_gb", Json::Int((self.ram_bytes / GIB) as i64)),
        ])
    }
}

/// A block of weights that can be measured and copied between devices.
pub trait Resident: Sized {
    fn bytes(&self) -> u64;
    fn to_device(&self, dev: &Device) -> Result<Self>;
}

enum Slot<B> {
    Device(B),
    Host(B),
    Disk,
}

/// `count` blocks, each on the GPU, in RAM or on the SSD.
pub struct Tiered<B> {
    slots: Vec<Slot<B>>,
    dev: Device,
    pub gpu_bytes: u64,
    pub host_bytes: u64,
    /// Bytes re-read from the weights for SSD blocks during passes.
    pub streamed_bytes: u64,
}

impl<B: Resident> Tiered<B> {
    /// Load every block with `load` (onto `dev`) and settle it in the tier the
    /// budget allows. SSD mode does not read a block until its first use.
    pub fn load(
        count: usize,
        budget: &Budget,
        dev: &Device,
        mut load: impl FnMut(usize) -> Result<B>,
        mut progress: impl FnMut(usize),
    ) -> Result<Self> {
        let vram = match budget.memory {
            Memory::Auto => budget.vram_limit(dev)?,
            _ => 0,
        };
        let ram = match budget.memory {
            Memory::Auto | Memory::Ram => budget.ram_bytes,
            _ => 0,
        };
        let mut this = Self { slots: Vec::with_capacity(count), dev: dev.clone(), gpu_bytes: 0, host_bytes: 0, streamed_bytes: 0 };
        for i in 0..count {
            if budget.memory == Memory::Ssd {
                this.slots.push(Slot::Disk);
                continue;
            }
            let block = load(i)?;
            let size = block.bytes();
            let slot = if budget.memory == Memory::Gpu || this.gpu_bytes + size <= vram {
                this.gpu_bytes += size;
                Slot::Device(block)
            } else if this.host_bytes + size <= ram {
                this.host_bytes += size;
                let host = block.to_device(&Device::Cpu)?;
                drop(block);
                Slot::Host(host)
            } else {
                Slot::Disk
            };
            this.slots.push(slot);
            progress(i + 1);
        }
        Ok(this)
    }

    /// Run `f` with block `i` on the device, uploading or re-reading it (with
    /// `load`) when it does not live there.
    pub fn with<R>(
        &mut self,
        i: usize,
        load: impl FnOnce(usize) -> Result<B>,
        f: impl FnOnce(&B) -> Result<R>,
    ) -> Result<R> {
        match &self.slots[i] {
            Slot::Device(b) => f(b),
            Slot::Host(b) => f(&b.to_device(&self.dev)?),
            Slot::Disk => {
                let b = load(i)?;
                self.streamed_bytes += b.bytes();
                f(&b)
            }
        }
    }

    /// Blocks per tier: (gpu, ram, ssd).
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut n = (0, 0, 0);
        for s in &self.slots {
            match s {
                Slot::Device(_) => n.0 += 1,
                Slot::Host(_) => n.1 += 1,
                Slot::Disk => n.2 += 1,
            }
        }
        n
    }

    pub fn report(&self) -> Json {
        let (gpu, ram, ssd) = self.counts();
        Json::obj([
            ("gpu_blocks", Json::Int(gpu as i64)),
            ("ram_blocks", Json::Int(ram as i64)),
            ("ssd_blocks", Json::Int(ssd as i64)),
            ("gpu_bytes", Json::Int(self.gpu_bytes as i64)),
            ("ram_bytes", Json::Int(self.host_bytes as i64)),
            ("streamed_bytes", Json::Int(self.streamed_bytes as i64)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    struct Fake(Tensor);
    impl Resident for Fake {
        fn bytes(&self) -> u64 {
            (self.0.elem_count() * self.0.dtype().size_in_bytes()) as u64
        }
        fn to_device(&self, dev: &Device) -> Result<Self> {
            Ok(Fake(self.0.to_device(dev)?))
        }
    }

    fn block(i: usize) -> Result<Fake> {
        // 1 KiB of f32 per block, valued by index so every tier can be checked.
        Ok(Fake(Tensor::full(i as f32, 256, &Device::Cpu)?))
    }

    fn run(budget: Budget) -> Result<(Tiered<Fake>, Vec<f32>)> {
        let mut t = Tiered::load(4, &budget, &Device::Cpu, block, |_| {})?;
        let mut sums = Vec::new();
        for i in 0..4 {
            sums.push(t.with(i, block, |b| b.0.sum_all()?.to_scalar::<f32>())?);
        }
        Ok((t, sums))
    }

    #[test]
    fn budgets_fill_gpu_then_ram_then_ssd_without_changing_results() -> Result<()> {
        let expected: Vec<f32> = (0..4).map(|i| i as f32 * 256.).collect();
        let (t, sums) = run(Budget { memory: Memory::Auto, vram_bytes: Some(2048), ram_bytes: 1024 })?;
        assert_eq!(t.counts(), (2, 1, 1));
        assert_eq!(t.streamed_bytes, 1024);
        assert_eq!(sums, expected);
        let (t, sums) = run(Budget { memory: Memory::Ram, vram_bytes: None, ram_bytes: 2048 })?;
        assert_eq!(t.counts(), (0, 2, 2));
        assert_eq!(sums, expected);
        let (t, sums) = run(Budget { memory: Memory::Ssd, vram_bytes: None, ram_bytes: 1 << 40 })?;
        assert_eq!(t.counts(), (0, 0, 4));
        assert_eq!(t.streamed_bytes, 4096);
        assert_eq!(sums, expected);
        let (t, sums) = run(Budget { memory: Memory::Gpu, vram_bytes: Some(0), ram_bytes: 0 })?;
        assert_eq!(t.counts(), (4, 0, 0));
        assert_eq!(sums, expected);
        Ok(())
    }

    #[test]
    fn requests_parse_memory_and_bounded_budgets() {
        let parse = |s: &str| Budget::parse(&Json::parse(s.as_bytes()).unwrap());
        let b = parse(r#"{"memory":"ssd","ram_gb":4,"vram_gb":8}"#).unwrap();
        assert_eq!(b.memory, Memory::Ssd);
        assert_eq!((b.ram_bytes, b.vram_bytes), (4 * GIB, Some(8 * GIB)));
        let d = parse("{}").unwrap();
        assert_eq!((d.memory, d.vram_bytes, d.ram_bytes), (Memory::Auto, None, 32 * GIB));
        for bad in [r#"{"memory":"disk"}"#, r#"{"memory":1}"#, r#"{"ram_gb":-1}"#, r#"{"vram_gb":193}"#, r#"{"ram_gb":"4"}"#] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }
}
