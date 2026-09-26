//! Embed assets/nrob-studio.ico as the executable's icon, so Explorer, the
//! taskbar and "Start with Windows" show NROB's mark instead of a blank one.
//!
//! No resource compiler is needed: the MSVC linker accepts a compiled `.res`
//! file as an input, and that format is simple enough to write here. Each image
//! of the .ico becomes an RT_ICON resource, and one RT_GROUP_ICON (id 1)
//! indexes them; Windows picks the first group icon as the file's icon.

use std::io::Write;

const RT_ICON: u16 = 3;
const RT_GROUP_ICON: u16 = 14;
const LANG_EN_US: u16 = 0x0409;
const MOVEABLE_PURE_DISCARDABLE: u16 = 0x1030;

/// One resource: a 32-byte header with ordinal type and name, then the data
/// padded to a 4-byte boundary.
fn resource(out: &mut Vec<u8>, kind: u16, id: u16, flags: u16, data: &[u8]) {
    out.extend((data.len() as u32).to_le_bytes()); // DataSize
    out.extend(32u32.to_le_bytes()); // HeaderSize
    out.extend([0xFF, 0xFF]);
    out.extend(kind.to_le_bytes()); // TYPE (ordinal)
    out.extend([0xFF, 0xFF]);
    out.extend(id.to_le_bytes()); // NAME (ordinal)
    out.extend(0u32.to_le_bytes()); // DataVersion
    out.extend(flags.to_le_bytes()); // MemoryFlags
    out.extend(if kind == 0 { 0u16 } else { LANG_EN_US }.to_le_bytes()); // LanguageId
    out.extend(0u32.to_le_bytes()); // Version
    out.extend(0u32.to_le_bytes()); // Characteristics
    out.extend(data);
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

fn main() {
    println!("cargo:rerun-if-changed=assets/nrob-studio.ico");
    println!("cargo:rerun-if-changed=build.rs");
    let windows_msvc = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    if !windows_msvc {
        return;
    }
    let ico = std::fs::read("assets/nrob-studio.ico").expect("assets/nrob-studio.ico");
    assert!(ico.len() > 6 && u16_at(&ico, 2) == 1, "not an .ico file");
    let count = u16_at(&ico, 4) as usize;

    let mut res = Vec::new();
    resource(&mut res, 0, 0, 0, &[]); // the empty entry every .res starts with
    let mut group = Vec::new();
    group.extend(0u16.to_le_bytes());
    group.extend(1u16.to_le_bytes());
    group.extend((count as u16).to_le_bytes());
    for i in 0..count {
        let e = 6 + i * 16;
        let (size, offset) = (u32_at(&ico, e + 8) as usize, u32_at(&ico, e + 12) as usize);
        let id = (i + 1) as u16;
        resource(&mut res, RT_ICON, id, MOVEABLE_PURE_DISCARDABLE, &ico[offset..offset + size]);
        // GRPICONDIRENTRY: the ICONDIRENTRY's first 12 bytes, then the id.
        group.extend(&ico[e..e + 12]);
        group.extend(id.to_le_bytes());
    }
    resource(&mut res, RT_GROUP_ICON, 1, MOVEABLE_PURE_DISCARDABLE, &group);

    let path = std::path::Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR")).join("nrob-studio-icon.res");
    std::fs::File::create(&path).and_then(|mut f| f.write_all(&res)).expect("writing the icon resource");
    println!("cargo:rustc-link-arg-bins={}", path.display());
}
