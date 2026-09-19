//! The seam between an engine and weight storage.
//!
//! An engine never opens a weight file for its experts itself; it asks a
//! `WeightStore`, usually through [`crate::ecache::Ecache`]. The stores in
//! the workspace:
//!
//! - `llama_rs::expert_stream::GgufExpertStore`: a GGUF MoE model's
//!   experts, read with positioned reads straight from the `.gguf` file.
//! - `dsv41::expert::SafetensorsExpertStore`: DeepSeek-V4.1's MXFP4 experts, read
//!   from the safetensors shards.
//!
//! Records are uniform-size and addressed by (layer, expert), which is why
//! the trait is this small.

/// A source of expert records.
pub trait WeightStore: Send + Sync {
    /// Bytes per expert record (uniform across the store).
    fn record_bytes(&self) -> usize;
    /// Number of (layer, expert) slots.
    fn shape(&self) -> (u32, u32);
    /// Copy the record for (layer, expert) into `dst` (len == record_bytes()).
    fn fetch(&self, layer: u32, expert: u32, dst: &mut [u8]) -> crate::error::Result<()>;
    /// Whether reads bypass the OS page cache.
    fn direct_io(&self) -> bool {
        false
    }
}
