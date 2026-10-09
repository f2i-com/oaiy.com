//! The usage profile: every routed expert's count of uses, kept from one run to the next.

/// A usage profile's first bytes: every routed expert's count of uses ([`dsv41::moe::Uses::counts`]) after them, as
/// little-endian u32s, behind the layers and the experts a layer.
const USAGE_MAGIC: &[u8; 8] = b"OAIYUSE1";

/// Read the usage profile at `path` into `uses`: whether it was one of this model's shape (a missing file, another
/// model's, or one cut short changes nothing).
pub(crate) fn read_usage(path: &std::path::Path, uses: &dsv41::moe::Uses) -> bool {
    let Ok(bytes) = std::fs::read(path) else { return false };
    let (layers, per_layer) = uses.shape();
    let word = |at: usize| bytes.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    if bytes.get(..8) != Some(&USAGE_MAGIC[..]) || word(8) != Some(layers as u32) || word(12) != Some(per_layer as u32) || bytes.len() != 16 + 4 * layers * per_layer {
        return false;
    }
    let counts: Vec<u32> = bytes[16..].chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    uses.seed(&counts)
}

/// Write `uses` as the usage profile at `path` (beside it first, then in its place: a reader never sees half of one).
pub(crate) fn write_usage(path: &std::path::Path, uses: &dsv41::moe::Uses) -> std::io::Result<()> {
    let (layers, per_layer) = uses.shape();
    let mut bytes = Vec::with_capacity(16 + 4 * layers * per_layer);
    bytes.extend_from_slice(USAGE_MAGIC);
    bytes.extend_from_slice(&(layers as u32).to_le_bytes());
    bytes.extend_from_slice(&(per_layer as u32).to_le_bytes());
    for n in uses.counts() {
        bytes.extend_from_slice(&n.to_le_bytes());
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let beside = path.with_extension("writing");
    std::fs::write(&beside, &bytes)?;
    std::fs::rename(&beside, path)
}
