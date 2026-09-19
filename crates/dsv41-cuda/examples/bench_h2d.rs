//! Host-to-device bandwidth for one expert record (18.8 MB), per device:
//! from pageable memory (a `Vec`, what the RAM tier holds), from pinned
//! memory, through `Gpu::write`'s pinned staging buffers, and the host copy
//! into a pinned buffer that staging adds; device-to-host both ways.
//!
//!   cargo run -p dsv41-cuda --release --example bench_h2d -- [devices]

use std::time::Instant;

use dsv41::expert::RECORD_BYTES;
use dsv41_cuda::Gpu;

fn main() -> nrob::Result<()> {
    let devices: Vec<usize> = std::env::args().nth(1).as_deref().unwrap_or("1,0").split(',').map(|v| v.trim().parse().expect("device ordinal")).collect();
    let n = 40;
    let host: Vec<u8> = (0..RECORD_BYTES).map(|i| (i * 31 % 251) as u8).collect();
    for d in devices {
        let g = Gpu::new(d)?;
        let mut dev = g.zeros::<u8>(RECORD_BYTES)?;
        let gb = |s: f64| (n * RECORD_BYTES) as f64 / s / 1e9;
        // pageable (the driver stages it)
        nrob_cu(g.stream.memcpy_htod(&host, &mut dev.slice_mut(..)))?;
        g.sync()?;
        let t = Instant::now();
        for _ in 0..n {
            nrob_cu(g.stream.memcpy_htod(&host, &mut dev.slice_mut(..)))?;
        }
        g.sync()?;
        let pageable = gb(t.elapsed().as_secs_f64());
        // pinned
        // SAFETY: filled completely below before any copy reads it.
        let mut pinned = nrob_cu(unsafe { g.context().alloc_pinned::<u8>(RECORD_BYTES) })?;
        nrob_cu(pinned.as_mut_slice())?.copy_from_slice(&host);
        nrob_cu(g.stream.memcpy_htod(&pinned, &mut dev.slice_mut(..)))?;
        g.sync()?;
        let t = Instant::now();
        for _ in 0..n {
            nrob_cu(g.stream.memcpy_htod(&pinned, &mut dev.slice_mut(..)))?;
        }
        g.sync()?;
        let pinned_bw = gb(t.elapsed().as_secs_f64());
        // host copy into the pinned (write-combined) buffer
        let t = Instant::now();
        for _ in 0..n {
            nrob_cu(pinned.as_mut_slice())?.copy_from_slice(&host);
        }
        let host_copy = gb(t.elapsed().as_secs_f64());
        // one staged write alone, then its pieces
        g.sync()?;
        let t = Instant::now();
        g.write(&host, &mut dev.slice_mut(..))?;
        let issued = t.elapsed().as_secs_f64();
        g.sync()?;
        let done = t.elapsed().as_secs_f64();
        println!("cuda:{d}: one staged write: issued in {:.2} ms, landed in {:.2} ms", issued * 1e3, done * 1e3);
        for round in 0..2 {
            let t = Instant::now();
            for _ in 0..n {
                g.write(&host, &mut dev.slice_mut(..))?;
            }
            g.sync()?;
            println!("cuda:{d}: staged (Gpu::write), round {round}: {:.1} GB/s", gb(t.elapsed().as_secs_f64()));
        }
        // device to host: pageable, and into pinned
        let mut back = vec![0u8; RECORD_BYTES];
        let t = Instant::now();
        for _ in 0..n {
            nrob_cu(g.stream.memcpy_dtoh(&dev, &mut back))?;
        }
        g.sync()?;
        let d2h_pageable = gb(t.elapsed().as_secs_f64());
        let t = Instant::now();
        for _ in 0..n {
            nrob_cu(g.stream.memcpy_dtoh(&dev, &mut pinned))?;
        }
        g.sync()?;
        let d2h_pinned = gb(t.elapsed().as_secs_f64());
        println!(
            "cuda:{d}: to device: pageable {pageable:.1} GB/s, pinned {pinned_bw:.1} GB/s, host copy into pinned {host_copy:.1} GB/s; to host: pageable {d2h_pageable:.1} GB/s, pinned {d2h_pinned:.1} GB/s"
        );
    }
    Ok(())
}

fn nrob_cu<T, E: std::fmt::Debug>(r: std::result::Result<T, E>) -> nrob::Result<T> {
    r.map_err(|e| nrob::Error::Unsupported(format!("cuda: {e:?}")))
}
