//! The Qwen3-style decoder, its linear layers and KV cache live in `oaiy-tts`
//! (the realtime speech engine); speech, music and sound here share them.
//! The LTX weight store feeds them as a `TensorSource`, with its quantized
//! formats decoded as usual.
use crate::ltx::store::Store;
use candle_core::{DType, Device, Result, Tensor};
pub use oaiy_tts::model::{Cache, Decoder, Linear};
use oaiy_tts::weights::TensorSource;

impl TensorSource for Store {
    fn load(&mut self, name: &str, dev: &Device) -> Result<Tensor> {
        self.tensor(name, dev, false)
    }

    fn load_f32(&mut self, name: &str, dev: &Device) -> Result<Tensor> {
        self.tensor_f32(name, dev)
    }

    fn load_rows(&mut self, name: &str, ids: &[u32], dev: &Device) -> Result<Tensor> {
        self.rows(name, ids, dev)?.to_dtype(DType::BF16)
    }

    fn has(&self, name: &str) -> bool {
        self.index.get(name).is_some()
    }

    fn tensor_names(&self) -> Vec<String> {
        self.index.names().map(str::to_owned).collect()
    }
}
