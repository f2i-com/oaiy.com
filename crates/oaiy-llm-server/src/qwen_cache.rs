//! Host checkpoints for Qwen's hybrid attention/recurrent cache. Rewinding only
//! the attention length is incorrect: the delta-net and convolution state must
//! also be restored to exactly the same token boundary.
use ggml_rs::{Backend, Tensor};
use llama_rs::{KvCache, qwen35::SsmConfig};
use std::sync::Arc;

/// `slots`' tensors on the host: those a layer's device holds as its chain's vectors read back together, one wait a
/// device for them all and their bytes copied on every core (a tensor at a time was a wait each: the 27B's 96
/// recurrent tensors, 149 MB, 34 ms a checkpoint where 29 now, the transfer's own), the rest each its own.
fn hosted(kv: &KvCache, slots: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
    let mut out: Vec<Option<Tensor>> = slots.iter().map(|_| None).collect();
    // (the layers by their devices, in turn)
    let mut left: Vec<usize> = (0..slots.len()).filter(|&i| slots[i].is_some()).collect();
    while let Some(&first) = left.first() {
        let backend = kv.layer_backends.get(first);
        let same = |i: usize| match (backend, kv.layer_backends.get(i)) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        let (these, rest): (Vec<usize>, Vec<usize>) = left.iter().partition(|&&i| i == first || same(i));
        left = rest;
        let chain = backend.and_then(|b| b.chain());
        let vectors: Vec<(usize, ggml_rs::DeviceVec)> = these.iter().filter_map(|&i| Some((i, chain?.aliased(slots[i].as_ref()?)?))).collect();
        if let (Some(chain), true) = (chain, vectors.len() > 1) {
            let mut rec = chain.begin();
            for (_, v) in &vectors {
                rec.read(v);
            }
            for ((i, _), values) in vectors.iter().zip(rec.finish()) {
                out[*i] = Some(Tensor::from_vec(values, slots[*i].as_ref().expect("a tensor").shape().to_vec()));
            }
        }
        for i in these {
            if out[i].is_none() {
                out[i] = slots[i].as_ref().map(Tensor::to_host);
            }
        }
    }
    out
}

#[derive(Clone)]
pub struct Snapshot {
    pub pos: usize,
    layers: Vec<[Option<Tensor>; 4]>, // K, V, delta-net, convolution
}

/// Conversation checkpoints reuse the append-only attention prefix already on
/// the GPUs. Only recurrent/conv state needs a separate copy at each boundary.
/// Unlike a disk snapshot, this is usable only while that prefix is still live.
/// (Cloned when the engine sets a conversation aside and keeps using its checkpoints.)
#[derive(Clone)]
pub struct RecurrentSnapshot(Snapshot);

impl RecurrentSnapshot {
    pub fn from_snapshot(snapshot: &Snapshot) -> Self {
        Self(Snapshot { pos: snapshot.pos, layers: snapshot.layers.iter()
            .map(|layer| [None, None, layer[2].clone(), layer[3].clone()]).collect() })
    }
    pub fn capture(kv: &KvCache) -> Self {
        let mut states = hosted(kv, &kv.ssm_state);
        let mut convs = hosted(kv, &kv.ssm_conv);
        Self(Snapshot { pos: kv.len, layers: (0..kv.ssm_state.len()).map(|i| [None, None, states[i].take(), convs[i].take()]).collect() })
    }
    pub fn bytes(&self) -> usize {
        self.0.layers.iter().flatten().flatten().map(|t| t.numel()*4).sum()
    }
    /// Restore onto a cache whose attention (and other per-token) buffers still hold the
    /// checkpoint's prefix: every recurrent slot goes back on its own layer's device, whatever
    /// its shape (Qwen3.8-Flash-Next: the delta-net states, then the n-gram layer's).
    pub fn restore_slots(&self, kv: &mut KvCache, common_prefix: usize) -> Result<(), String> {
        if self.0.pos > common_prefix || self.0.pos > kv.len || self.0.layers.len() != kv.ssm_state.len() {
            return Err("recurrent checkpoint no longer matches the live cache".into());
        }
        for (i, layer) in self.0.layers.iter().enumerate() {
            let backend = kv.layer_backends[i].clone();
            kv.ssm_state[i] = layer[2].as_ref().map(|t| backend.to_device(t.clone()));
            kv.ssm_conv[i] = layer[3].as_ref().map(|t| backend.to_device(t.clone()));
        }
        kv.len = self.0.pos;
        Ok(())
    }
    pub fn restore(&self, kv: &mut KvCache, common_prefix: usize, attention: &[bool], ssm: SsmConfig, backend: &dyn Backend) -> Result<(), String> {
        if self.0.pos > common_prefix || self.0.pos > kv.len {
            return Err("Qwen recurrent checkpoint no longer has a live attention prefix".into());
        }
        self.0.restore_inner(kv, attention, ssm, backend, true)
    }
}

impl Snapshot {
    pub fn capture(kv: &KvCache, attention: &[bool], backend: &dyn Backend) -> Self {
        let layers = attention.iter().enumerate().map(|(i, &attn)| {
            if attn {
                let backend = kv.layer_backends.get(i).map(|b| b.as_ref()).unwrap_or(backend);
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
        self.restore_inner(kv, attention, ssm, backend, false)
    }

    fn restore_inner(&self, kv: &mut KvCache, attention: &[bool], ssm: SsmConfig, backend: &dyn Backend, reuse_attention: bool) -> Result<(), String> {
        if self.pos == 0 || self.pos > kv.max_len || self.layers.len() != attention.len() {
            return Err("incompatible Qwen checkpoint dimensions".into());
        }
        let vd = ssm.inner_size / ssm.time_step_rank;
        let conv = 2 * ssm.state_size * ssm.group_count + ssm.inner_size;
        // Validate every tensor before touching the live state.
        for (i, (layer, &attn)) in self.layers.iter().zip(attention).enumerate() {
            let expected = if attn && reuse_attention {
                [None, None, None, None]
            } else if attn {
                let shape = vec![self.pos, kv.n_kv_heads_per_layer[i], kv.head_dims[i]];
                [Some(shape.clone()), Some(shape), None, None]
            } else {
                [None, None, Some(vec![ssm.time_step_rank, vd, vd]), Some(vec![ssm.conv_kernel - 1, conv])]
            };
            if layer.iter().zip(&expected).any(|(t, s)| t.as_ref().map(Tensor::shape) != s.as_deref()) {
                return Err("incompatible Qwen checkpoint tensor shape".into());
            }
        }
        if !reuse_attention { kv.reset(); }
        for (i, layer) in self.layers.iter().enumerate() {
            if let (Some(k), Some(v)) = (&layer[0], &layer[1]) {
                let placed = kv.layer_backends.get(i).cloned();
                let backend = placed.as_ref().map(|b| b.as_ref()).unwrap_or(backend);
                kv.reserve_layer(backend, i, self.pos);
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
    fn write_to(&self, out: &mut dyn std::io::Write) -> std::io::Result<()> {
        out.write_all(&(self.pos as u64).to_le_bytes())?;
        out.write_all(&(self.layers.len() as u64).to_le_bytes())?;
        let mut buffer = Vec::with_capacity(65536);
        for t in self.layers.iter().flatten() {
            out.write_all(&(t.as_ref().map_or(0, Tensor::rank) as u64).to_le_bytes())?;
            if let Some(t) = t {
                for &dim in t.shape() { out.write_all(&(dim as u64).to_le_bytes())?; }
                for chunk in t.data().chunks(16384) {
                    buffer.clear();
                    for &v in chunk { buffer.extend(v.to_le_bytes()); }
                    out.write_all(&buffer)?;
                }
            }
        }
        Ok(())
    }
    fn decode(mut bytes: &[u8]) -> oaiy_engine::Result<Self> {
        let len = bytes.len() as u64;
        Self::read_from(&mut bytes, len)
    }
    fn read_from(input: &mut dyn std::io::Read, mut remaining: u64) -> oaiy_engine::Result<Self> {
        fn bad() -> oaiy_engine::Error { oaiy_engine::Error::Format("damaged Qwen prompt state".into()) }
        fn number(input: &mut dyn std::io::Read, remaining: &mut u64) -> oaiy_engine::Result<usize> {
            *remaining = remaining.checked_sub(8).ok_or_else(bad)?;
            let mut raw = [0u8; 8];
            input.read_exact(&mut raw)?;
            usize::try_from(u64::from_le_bytes(raw)).map_err(|_| bad())
        }
        let pos = number(input, &mut remaining)?;
        let count = number(input, &mut remaining)?;
        if pos == 0 || count > 256 { return Err(bad()); }
        let mut layers = Vec::with_capacity(count);
        let mut buffer = vec![0u8; 65536];
        for _ in 0..count {
            let mut layer = [None, None, None, None];
            for t in &mut layer {
                let rank = number(input, &mut remaining)?;
                if rank == 0 { continue; }
                if rank > 3 { return Err(bad()); }
                let shape = (0..rank).map(|_| number(input, &mut remaining)).collect::<oaiy_engine::Result<Vec<_>>>()?;
                let size = shape.iter().try_fold(4usize, |a, &b| a.checked_mul(b)).ok_or_else(bad)?;
                remaining = remaining.checked_sub(size as u64).ok_or_else(bad)?;
                let mut data = Vec::with_capacity(size / 4);
                let mut left = size;
                while left > 0 {
                    let n = left.min(buffer.len());
                    input.read_exact(&mut buffer[..n])?;
                    data.extend(buffer[..n].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
                    left -= n;
                }
                *t = Some(Tensor::from_vec(data, shape));
            }
            layers.push(layer);
        }
        if remaining != 0 { return Err(bad()); }
        Ok(Arc::new(Snapshot { pos, layers }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk::PromptState;

    #[test]
    fn recurrent_checkpoint_rewinds_live_attention_but_rejects_overwritten_prefix() {
        let backend = ggml_rs::CpuBackend::new();
        let mut kv = KvCache::new(&backend, 2, 16, 1, 2);
        kv.len = 3;
        kv.k[0].data_mut()[..6].fill(2.0);
        kv.v[0].data_mut()[..6].fill(3.0);
        kv.ssm_state[1] = Some(Tensor::from_vec(vec![4.0; 4], vec![1, 2, 2]));
        kv.ssm_conv[1] = Some(Tensor::from_vec(vec![5.0; 12], vec![2, 6]));
        let snap = RecurrentSnapshot::capture(&kv);
        assert_eq!(snap.bytes(), 64, "attention storage is not copied into RAM checkpoints");
        kv.len = 7;
        kv.ssm_state[1].as_mut().unwrap().data_mut().fill(99.0);
        let config = SsmConfig { conv_kernel: 3, group_count: 1, inner_size: 2, state_size: 2, time_step_rank: 1 };
        assert!(snap.restore(&mut kv, 2, &[true, false], config, &backend).is_err());
        assert_eq!(kv.len, 7);
        assert_eq!(kv.ssm_state[1].as_ref().unwrap().data(), &[99.0; 4]);
        snap.restore(&mut kv, 3, &[true, false], config, &backend).unwrap();
        assert_eq!(kv.len, 3);
        assert_eq!(&kv.k[0].data()[..6], &[2.0; 6]);
        assert_eq!(&kv.v[0].data()[..6], &[3.0; 6]);
        assert_eq!(kv.ssm_state[1].as_ref().unwrap().data(), &[4.0; 4]);
        assert_eq!(kv.ssm_conv[1].as_ref().unwrap().data(), &[5.0; 12]);
    }

    #[test]
    fn lazy_growth_and_restore_preserve_prefix() {
        let backend: Arc<dyn Backend> = Arc::new(ggml_rs::CpuBackend::new());
        let mut kv = KvCache::new_lazy_per_layer_kv(vec![backend.clone()], 1024, &[1], &[2]);
        let first = Tensor::from_vec(vec![2.0; 400], vec![200, 1, 2]);
        let second = Tensor::from_vec(vec![3.0; 400], vec![200, 1, 2]);
        kv.append(backend.as_ref(), 0, &first, &first); kv.commit(200);
        kv.append(backend.as_ref(), 0, &second, &second); kv.commit(200);
        assert_eq!(kv.k[0].dim(0), 512);
        assert_eq!(&kv.k[0].data()[..400], first.data());
        let snapshot = Snapshot::capture(&kv, &[true], backend.as_ref());
        let mut restored = KvCache::new_lazy_per_layer_kv(vec![backend.clone()], 1024, &[1], &[2]);
        let ssm = SsmConfig { conv_kernel: 3, group_count: 1, inner_size: 2, state_size: 2, time_step_rank: 1 };
        snapshot.restore(&mut restored, &[true], ssm, backend.as_ref()).unwrap();
        assert_eq!(restored.len, 400);
        assert_eq!(restored.k[0].dim(0), 512);
        assert_eq!(&restored.k[0].data()[..800], &kv.k[0].data()[..800]);
    }

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
        let mut streamed = Vec::new(); snap.write_to(&mut streamed).unwrap();
        assert_eq!(bytes, streamed);
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
