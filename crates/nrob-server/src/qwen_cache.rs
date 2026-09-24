//! Host checkpoints for Qwen's hybrid attention/recurrent cache. Rewinding only
//! the attention length is incorrect: the delta-net and convolution state must
//! also be restored to exactly the same token boundary.
use ggml_rs::{Backend, Tensor};
use llama_rs::{KvCache, qwen35::SsmConfig};
use std::sync::Arc;

pub struct Snapshot {
    pub pos: usize,
    layers: Vec<[Option<Tensor>; 4]>, // K, V, delta-net, convolution
}

impl Snapshot {
    pub fn capture(kv: &KvCache, attention: &[bool], backend: &dyn Backend) -> Self {
        let layers = attention.iter().enumerate().map(|(i, &attn)| {
            if attn {
                [Some(backend.slice_axis0(&kv.k[i], kv.len).to_host()),
                 Some(backend.slice_axis0(&kv.v[i], kv.len).to_host()), None, None]
            } else {
                [None, None, kv.ssm_state[i].as_ref().map(Tensor::to_host),
                 kv.ssm_conv[i].as_ref().map(Tensor::to_host)]
            }
        }).collect();
        Self { pos: kv.len, layers }
    }

    pub fn restore(&self, kv: &mut KvCache, attention: &[bool], ssm: SsmConfig, backend: &dyn Backend) -> Result<(), String> {
        if self.pos == 0 || self.pos > kv.max_len || self.layers.len() != attention.len() {
            return Err("incompatible Qwen checkpoint dimensions".into());
        }
        let vd = ssm.inner_size / ssm.time_step_rank;
        let conv = 2 * ssm.state_size * ssm.group_count + ssm.inner_size;
        // Validate every tensor before touching the live state.
        for (i, (layer, &attn)) in self.layers.iter().zip(attention).enumerate() {
            let expected = if attn {
                let shape = vec![self.pos, kv.n_kv_heads_per_layer[i], kv.head_dims[i]];
                [Some(shape.clone()), Some(shape), None, None]
            } else {
                [None, None, Some(vec![ssm.time_step_rank, vd, vd]), Some(vec![ssm.conv_kernel - 1, conv])]
            };
            if layer.iter().zip(&expected).any(|(t, s)| t.as_ref().map(Tensor::shape) != s.as_deref()) {
                return Err("incompatible Qwen checkpoint tensor shape".into());
            }
        }
        kv.reset();
        for (i, layer) in self.layers.iter().enumerate() {
            if let (Some(k), Some(v)) = (&layer[0], &layer[1]) {
                backend.copy_axis0_into(&mut kv.k[i], 0, &backend.to_device(k.clone()));
                backend.copy_axis0_into(&mut kv.v[i], 0, &backend.to_device(v.clone()));
            }
            kv.ssm_state[i] = layer[2].as_ref().map(|t| backend.to_device(t.clone()));
            kv.ssm_conv[i] = layer[3].as_ref().map(|t| backend.to_device(t.clone()));
        }
        kv.len = self.pos;
        Ok(())
    }
}

impl crate::disk::PromptState for Arc<Snapshot> {
    fn pos(&self) -> usize { self.pos }
    fn encoded_len(&self) -> usize {
        16 + self.layers.iter().flatten().map(|t| 8 + t.as_ref().map_or(0, |t| 8 * t.rank() + 4 * t.numel())).sum::<usize>()
    }
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend((self.pos as u64).to_le_bytes());
        out.extend((self.layers.len() as u64).to_le_bytes());
        for t in self.layers.iter().flatten() {
            out.extend((t.as_ref().map_or(0, Tensor::rank) as u64).to_le_bytes());
            if let Some(t) = t {
                for &dim in t.shape() { out.extend((dim as u64).to_le_bytes()); }
                for &v in t.data() { out.extend(v.to_le_bytes()); }
            }
        }
    }
    fn decode(mut bytes: &[u8]) -> nrob::Result<Self> {
        fn bad() -> nrob::Error { nrob::Error::Format("damaged Qwen prompt state".into()) }
        fn number(bytes: &mut &[u8]) -> nrob::Result<usize> {
            let raw = bytes.get(..8).ok_or_else(bad)?;
            let n = u64::from_le_bytes(raw.try_into().map_err(|_| bad())?);
            *bytes = &bytes[8..];
            usize::try_from(n).map_err(|_| bad())
        }
        let pos = number(&mut bytes)?;
        let count = number(&mut bytes)?;
        if pos == 0 || count > 256 { return Err(bad()); }
        let mut layers = Vec::with_capacity(count);
        for _ in 0..count {
            let mut layer = [None, None, None, None];
            for t in &mut layer {
                let rank = number(&mut bytes)?;
                if rank == 0 { continue; }
                if rank > 3 { return Err(bad()); }
                let shape = (0..rank).map(|_| number(&mut bytes)).collect::<nrob::Result<Vec<_>>>()?;
                let size = shape.iter().try_fold(4usize, |a, &b| a.checked_mul(b)).ok_or_else(bad)?;
                let raw = bytes.get(..size).ok_or_else(bad)?;
                let data = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                *t = Some(Tensor::from_vec(data, shape));
                bytes = &bytes[size..];
            }
            layers.push(layer);
        }
        if !bytes.is_empty() { return Err(bad()); }
        Ok(Arc::new(Snapshot { pos, layers }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk::PromptState;

    #[test]
    fn round_trip_restores_attention_recurrence_and_conv_after_divergence() {
        let backend = ggml_rs::CpuBackend::new();
        let mut kv = KvCache::new(&backend, 2, 8, 1, 2);
        kv.len = 3;
        kv.k[0].data_mut()[..6].fill(2.0);
        kv.v[0].data_mut()[..6].fill(3.0);
        kv.ssm_state[1] = Some(Tensor::from_vec(vec![4.0; 4], vec![1, 2, 2]));
        kv.ssm_conv[1] = Some(Tensor::from_vec(vec![5.0; 12], vec![2, 6]));
        let snap = Arc::new(Snapshot::capture(&kv, &[true, false], &backend));
        let mut bytes = Vec::new(); snap.encode(&mut bytes);
        assert_eq!(bytes.len(), snap.encoded_len());
        let decoded = Arc::<Snapshot>::decode(&bytes).unwrap();
        kv.k[0].data_mut().fill(99.0); kv.ssm_state[1].as_mut().unwrap().data_mut().fill(99.0);
        kv.len = 7;
        let config = SsmConfig { conv_kernel: 3, group_count: 1, inner_size: 2, state_size: 2, time_step_rank: 1 };
        decoded.restore(&mut kv, &[true, false], config, &backend).unwrap();
        assert_eq!(kv.len, 3);
        assert_eq!(&kv.k[0].data()[..6], &[2.0; 6]);
        assert_eq!(&kv.v[0].data()[..6], &[3.0; 6]);
        assert_eq!(kv.ssm_state[1].as_ref().unwrap().data(), &[4.0; 4]);
        assert_eq!(kv.ssm_conv[1].as_ref().unwrap().data(), &[5.0; 12]);
        // A changed model layout must fail before resetting the live cache.
        assert!(decoded.restore(&mut kv, &[false, true], config, &backend).is_err());
        assert_eq!(kv.len, 3);
        for n in 0..bytes.len() { assert!(Arc::<Snapshot>::decode(&bytes[..n]).is_err()); }
        bytes.extend([0]); assert!(Arc::<Snapshot>::decode(&bytes).is_err());
    }
}
