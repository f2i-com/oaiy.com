//! Host checkpoints for Qwen's hybrid attention/recurrent cache. Rewinding only
//! the attention length is incorrect: the delta-net and convolution state must
//! also be restored to exactly the same token boundary.
use ggml_rs::{Backend, Tensor};
use crate::llm::{KvCache, arch::qwen35::SsmConfig};
use std::sync::Arc;

/// `t` for a cache to take as its own: a checkpoint's tensor still on a device is the checkpoint's vector there, which
/// a run would write over, so its values from the host.
fn its_own(t: &Tensor) -> Tensor {
    if t.is_device() { t.to_host() } else { t.clone() }
}

/// The layers of `left` on the first of them's device (a layer's the cache's own for it, else `fallback`), that
/// device, and the rest.
fn by_device<'a>(kv: &'a KvCache, left: &[usize], fallback: Option<&'a Arc<dyn Backend>>) -> (Vec<usize>, Option<&'a Arc<dyn Backend>>, Vec<usize>) {
    let of = |i: usize| kv.layer_backends.get(i).or(fallback);
    let backend = of(left[0]);
    let same = |i: usize| match (backend, of(i)) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    };
    let (these, rest): (Vec<usize>, Vec<usize>) = left.iter().partition(|&&i| same(i));
    (these, backend, rest)
}

/// `slots`' tensors as they are now, apart from the cache's: those a layer's device holds as its chain's vectors
/// copied there into new vectors (submitted, not waited for), the rest read to the host.
fn copied(kv: &KvCache, slots: &[Option<Tensor>], fallback: Option<&Arc<dyn Backend>>) -> Vec<Option<Tensor>> {
    let mut out: Vec<Option<Tensor>> = slots.iter().map(|_| None).collect();
    let mut left: Vec<usize> = (0..slots.len()).filter(|&i| slots[i].is_some()).collect();
    while !left.is_empty() {
        let (these, backend, rest) = by_device(kv, &left, fallback);
        left = rest;
        if let Some(chain) = backend.and_then(|b| b.chain()) {
            let vectors: Vec<(usize, ggml_rs::DeviceVec)> = these.iter().filter_map(|&i| Some((i, chain.aliased(slots[i].as_ref()?)?))).collect();
            if !vectors.is_empty() {
                let mut rec = chain.begin();
                for (i, v) in &vectors {
                    let copy = chain.vec(v.len);
                    rec.copy(v, 0, &copy, 0, v.len);
                    out[*i] = Some(chain.alias(&copy, slots[*i].as_ref().expect("a tensor").shape().to_vec()));
                }
                // (gone to the GPU after what is there already; nothing of it read here)
                rec.flush();
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

/// `slots`' tensors on the host: those a layer's device holds as its chain's vectors read back together, one wait a
/// device for them all and their bytes copied on every core, the rest each its own.
fn hosted(kv: &KvCache, slots: &[Option<Tensor>], fallback: Option<&Arc<dyn Backend>>) -> Vec<Option<Tensor>> {
    let mut out: Vec<Option<Tensor>> = slots.iter().map(|_| None).collect();
    let mut left: Vec<usize> = (0..slots.len()).filter(|&i| slots[i].is_some()).collect();
    while !left.is_empty() {
        let (these, backend, rest) = by_device(kv, &left, fallback);
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

/// A whole state ([`Snapshot::from_checkpoint`]) made a layer at a time: [`WholeBuilder::step`] copies the next
/// attention layer's keys and values up to its position from the cache, which must hold the same rows there at each
/// step (the cache appended to after, between steps, holds them still).
pub struct WholeBuilder {
    pos: usize,
    attention: Vec<bool>,
    backend: Arc<dyn Backend>,
    layers: Vec<[Option<Tensor>; 4]>,
    next: usize,
}

impl WholeBuilder {
    /// The next attention layer's rows copied: the whole state once the last is in, else None.
    pub fn step(&mut self, kv: &KvCache) -> Option<Snapshot> {
        while self.next < self.attention.len() && !self.attention[self.next] {
            self.next += 1;
        }
        if let Some(i) = (self.next < self.attention.len()).then_some(self.next) {
            let backend = kv.layer_backends.get(i).map(|b| b.as_ref()).unwrap_or(self.backend.as_ref());
            // (the host's rows, and the rows a chained run's device holds alone read from it)
            let (k, v) = kv.layer_prefix(backend, i, self.pos);
            self.layers[i][0] = Some(k);
            self.layers[i][1] = Some(v);
            self.next += 1;
            return None;
        }
        Some(Snapshot { pos: self.pos, layers: std::mem::take(&mut self.layers) })
    }

    pub fn pos(&self) -> usize {
        self.pos
    }
}

impl RecurrentSnapshot {
    pub fn from_snapshot(snapshot: &Snapshot) -> Self {
        Self(Snapshot { pos: snapshot.pos, layers: snapshot.layers.iter()
            .map(|layer| [None, None, layer[2].clone(), layer[3].clone()]).collect() })
    }
    /// The cache's recurrent tensors, on the host.
    pub fn capture(kv: &KvCache) -> Self {
        let mut states = hosted(kv, &kv.ssm_state, None);
        let mut convs = hosted(kv, &kv.ssm_conv, None);
        Self(Snapshot { pos: kv.len, layers: (0..kv.ssm_state.len()).map(|i| [None, None, states[i].take(), convs[i].take()]).collect() })
    }
    /// [`Self::capture`] with no wait for the GPU: the tensors a device's chain holds (a layer's device the cache's
    /// own for it, else `backend`) are copied there into vectors of their own, and [`Self::settle`] brings them to
    /// the host once the reply is out (the 27B's 149 MB read to the host was 30 ms of a prompt, a checkpoint or two a
    /// prompt); the rest are read to the host here.
    pub fn capture_later(kv: &KvCache, backend: Option<&Arc<dyn Backend>>) -> Self {
        let mut states = copied(kv, &kv.ssm_state, backend);
        let mut convs = copied(kv, &kv.ssm_conv, backend);
        Self(Snapshot { pos: kv.len, layers: (0..kv.ssm_state.len()).map(|i| [None, None, states[i].take(), convs[i].take()]).collect() })
    }
    /// A checkpoint at `pos` of the recurrent tensors a run kept from inside it (`crate::llm::Tapped`'s, a layer each:
    /// on their device as [`Self::capture_later`]'s copies are, and settled as they are).
    pub fn tapped(pos: usize, states: Vec<Option<Tensor>>, convs: Vec<Option<Tensor>>) -> Self {
        Self(Snapshot { pos, layers: states.into_iter().zip(convs).map(|(state, conv)| [None, None, state, conv]).collect() })
    }
    /// The tensors still on a device brought to the host (the cards' memory is the models'): for when the GPU has
    /// nothing else to do.
    pub fn settle(&mut self, kv: &KvCache, backend: Option<&Arc<dyn Backend>>) {
        if !self.0.layers.iter().flatten().flatten().any(Tensor::is_device) || self.0.layers.len() != kv.ssm_state.len() {
            return;
        }
        for slot in 2..4 {
            let tensors: Vec<Option<Tensor>> = self.0.layers.iter().map(|l| l[slot].clone()).collect();
            for (layer, t) in self.0.layers.iter_mut().zip(hosted(kv, &tensors, backend)) {
                layer[slot] = t;
            }
        }
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
            kv.ssm_state[i] = layer[2].as_ref().map(|t| backend.to_device(its_own(t)));
            kv.ssm_conv[i] = layer[3].as_ref().map(|t| backend.to_device(its_own(t)));
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
                let (k, v) = kv.layer_prefix(backend, i, kv.len);
                [Some(k), Some(v), None, None]
            } else {
                [None, None, kv.ssm_state[i].as_ref().map(Tensor::to_host),
                 kv.ssm_conv[i].as_ref().map(Tensor::to_host)]
            }
        }).collect();
        Self { pos: kv.len, layers }
    }

    /// The whole state at a checkpoint, for the disk, taken after the run went past it: the checkpoint's recurrent
    /// tensors, and the attention cache's keys and values up to its position, which later tokens only added to. So a
    /// prompt's run copies a checkpoint's recurrent state alone (on its device), and the attention cache is read out
    /// once the reply is out, not between the prompt's last tokens: the 27B's keys and values for 2,800 tokens are
    /// some 350 MB, which a card in a Thunderbolt enclosure gave back in a quarter of a second each. `None` where the
    /// checkpoint is not of this cache's shape, or past what it holds.
    /// The bytes [`Self::from_checkpoint`]'s state takes at `pos` positions, known before it is made: the attention
    /// layers' keys and values (the recurrent tensors are small beside them).
    pub fn whole_bytes(pos: usize, kv: &KvCache, attention: &[bool]) -> u64 {
        attention.iter().enumerate().filter(|&(_, &held)| held).map(|(i, _)| 8 * pos as u64 * (kv.n_kv_heads_per_layer[i] * kv.head_dims[i]) as u64).sum()
    }

    pub fn from_checkpoint(checkpoint: &RecurrentSnapshot, kv: &KvCache, attention: &[bool], backend: &dyn Backend) -> Option<Self> {
        let pos = checkpoint.0.pos;
        if pos == 0 || pos > kv.len || checkpoint.0.layers.len() != attention.len() {
            return None;
        }
        let layers = attention.iter().enumerate().map(|(i, &attn)| {
            if attn {
                let backend = kv.layer_backends.get(i).map(|b| b.as_ref()).unwrap_or(backend);
                let (k, v) = kv.layer_prefix(backend, i, pos);
                [Some(k), Some(v), None, None]
            } else {
                let layer = &checkpoint.0.layers[i];
                [None, None, layer[2].as_ref().map(its_own), layer[3].as_ref().map(its_own)]
            }
        }).collect();
        Some(Self { pos, layers })
    }

    /// [`Self::from_checkpoint`] a layer at a time, so the work can stop between layers ([`WholeBuilder`]).
    pub fn builder(checkpoint: &RecurrentSnapshot, kv: &KvCache, attention: &[bool], backend: Arc<dyn Backend>) -> Option<WholeBuilder> {
        let pos = checkpoint.0.pos;
        if pos == 0 || pos > kv.len || checkpoint.0.layers.len() != attention.len() {
            return None;
        }
        // (the recurrent tensors now, small and the checkpoint's: it may be let go before the keys and values are in)
        let layers = attention.iter().enumerate().map(|(i, &attn)| {
            if attn { [None, None, None, None] } else { let layer = &checkpoint.0.layers[i]; [None, None, layer[2].as_ref().map(its_own), layer[3].as_ref().map(its_own)] }
        }).collect();
        Some(WholeBuilder { pos, attention: attention.to_vec(), backend, layers, next: 0 })
    }

    pub fn restore(&self, kv: &mut KvCache, attention: &[bool], ssm: SsmConfig, backend: &dyn Backend) -> Result<(), String> {
        self.restore_inner(kv, attention, ssm, backend, false)
    }

    /// [`Self::restore`] onto a cache whose slots are not the Qwen3.5 hybrid's (Qwen3.8-Flash-Next's: its layers, its
    /// n-gram layer's and its indexers'): `rows` says which slots hold keys and values a token ([`Self::
    /// from_checkpoint`] made the state with the same), each of the cache's own heads and width; the other slots'
    /// recurrent tensors go back whatever their shape, each on its slot's device. The cache is emptied first, and
    /// nothing is touched where the state is not of its shape.
    pub fn restore_whole(&self, kv: &mut KvCache, rows: &[bool]) -> Result<(), String> {
        if self.pos == 0 || self.pos > kv.max_len || self.layers.len() != rows.len() || rows.len() != kv.ssm_state.len() {
            return Err("a prompt state of another cache's shape".into());
        }
        for (i, (layer, &held)) in self.layers.iter().zip(rows).enumerate() {
            let shape = [self.pos, kv.n_kv_heads_per_layer[i], kv.head_dims[i]];
            let fits = |t: &Option<Tensor>| if held { t.as_ref().is_some_and(|t| t.shape() == shape) } else { t.is_none() };
            if !fits(&layer[0]) || !fits(&layer[1]) || (held && (layer[2].is_some() || layer[3].is_some())) {
                return Err("a prompt state whose keys and values are not this cache's".into());
            }
        }
        kv.reset();
        for (i, layer) in self.layers.iter().enumerate() {
            let backend = kv.layer_backends[i].clone();
            if let (Some(k), Some(v)) = (&layer[0], &layer[1]) {
                kv.reserve_layer(backend.as_ref(), i, self.pos);
                backend.copy_axis0_into(&mut kv.k[i], 0, &backend.to_device(k.clone()));
                backend.copy_axis0_into(&mut kv.v[i], 0, &backend.to_device(v.clone()));
            }
            kv.ssm_state[i] = layer[2].as_ref().map(|t| backend.to_device(its_own(t)));
            kv.ssm_conv[i] = layer[3].as_ref().map(|t| backend.to_device(its_own(t)));
        }
        kv.len = self.pos;
        kv.host_rows = self.pos;
        Ok(())
    }

    fn restore_inner(&self, kv: &mut KvCache, attention: &[bool], ssm: SsmConfig, backend: &dyn Backend, reuse_attention: bool) -> Result<(), String> {
        if self.pos == 0 || self.pos > kv.max_len || self.layers.len() != attention.len() {
            return Err("incompatible Qwen checkpoint dimensions".into());
        }
        // Validate every tensor before touching the live state.
        for (i, (layer, &attn)) in self.layers.iter().zip(attention).enumerate() {
            let expected = if attn && reuse_attention {
                [None, None, None, None]
            } else if attn {
                let shape = vec![self.pos, kv.n_kv_heads_per_layer[i], kv.head_dims[i]];
                [Some(shape.clone()), Some(shape), None, None]
            } else {
                [None, None, Some(ssm.state_shape().to_vec()), Some(ssm.conv_shape().to_vec())]
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
            kv.ssm_state[i] = layer[2].as_ref().map(|t| backend.to_device(its_own(t)));
            kv.ssm_conv[i] = layer[3].as_ref().map(|t| backend.to_device(its_own(t)));
        }
        kv.len = self.pos;
        // (the rows written here the host's, until a chained run takes them to its device)
        if !reuse_attention {
            kv.host_rows = self.pos;
        }
        Ok(())
    }
}

impl crate::engines::disk::PromptState for Arc<Snapshot> {
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
    use crate::engines::disk::PromptState;

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

    /// A whole state of a cache whose slots are Flash-Next's (a layer that attends, a delta-net layer, the n-gram
    /// layer's slot, an indexer's) goes to the disk's bytes and back onto an empty cache as it was at its checkpoint:
    /// the keys and values up to it, the recurrent tensors whatever their shapes; and not onto a cache of another shape.
    #[test]
    fn a_whole_state_of_a_cache_of_slots_goes_to_the_disk_and_back() {
        let backend: Arc<dyn Backend> = Arc::new(ggml_rs::CpuBackend::new());
        let make = || KvCache::new_lazy_per_layer_kv(vec![backend.clone(); 4], 64, &[2, 1, 1, 1], &[3, 1, 1, 4]);
        let rows = [true, false, false, true];
        let mut kv = make();
        let fill = |seq: usize, heads: usize, dim: usize, from: f32| Tensor::from_vec((0..seq * heads * dim).map(|i| from + i as f32).collect(), vec![seq, heads, dim]);
        kv.append(backend.as_ref(), 0, &fill(5, 2, 3, 100.0), &fill(5, 2, 3, 200.0));
        kv.append(backend.as_ref(), 3, &fill(5, 1, 4, 300.0), &fill(5, 1, 4, 300.0));
        kv.commit(5);
        kv.ssm_state[1] = Some(Tensor::from_vec(vec![4.0; 8], vec![2, 2, 2]));
        kv.ssm_conv[1] = Some(Tensor::from_vec(vec![5.0; 6], vec![1, 6]));
        kv.ssm_state[2] = Some(Tensor::from_vec(vec![11.0, 12.0], vec![2]));
        kv.ssm_conv[2] = Some(Tensor::from_vec(vec![6.0; 9], vec![3, 3]));
        let checkpoint = RecurrentSnapshot::capture(&kv);
        // the run goes on past the checkpoint
        kv.append(backend.as_ref(), 0, &fill(2, 2, 3, 900.0), &fill(2, 2, 3, 950.0));
        kv.append(backend.as_ref(), 3, &fill(2, 1, 4, 970.0), &fill(2, 1, 4, 970.0));
        kv.commit(2);
        kv.ssm_state[1].as_mut().unwrap().data_mut().fill(97.0);
        kv.ssm_state[2] = Some(Tensor::from_vec(vec![13.0, 14.0], vec![2]));
        let whole = Snapshot::from_checkpoint(&checkpoint, &kv, &rows, backend.as_ref()).unwrap();
        let mut bytes = Vec::new();
        Arc::new(whole).encode(&mut bytes);
        let back = <Arc<Snapshot> as PromptState>::decode(&bytes).unwrap();
        assert_eq!(back.pos, 5);
        let mut fresh = make();
        back.restore_whole(&mut fresh, &rows).unwrap();
        assert_eq!(fresh.len, 5);
        assert_eq!(backend.slice_axis0(&fresh.k[0], 5).to_host().data(), fill(5, 2, 3, 100.0).data());
        assert_eq!(backend.slice_axis0(&fresh.v[0], 5).to_host().data(), fill(5, 2, 3, 200.0).data());
        assert_eq!(backend.slice_axis0(&fresh.k[3], 5).to_host().data(), fill(5, 1, 4, 300.0).data());
        assert_eq!(fresh.ssm_state[1].as_ref().unwrap().data(), &[4.0; 8]);
        assert_eq!(fresh.ssm_conv[1].as_ref().unwrap().shape(), &[1, 6]);
        assert_eq!(fresh.ssm_state[2].as_ref().unwrap().data(), &[11.0, 12.0]);
        assert_eq!(fresh.ssm_conv[2].as_ref().unwrap().data(), &[6.0; 9]);
        assert!(fresh.ssm_state[0].is_none() && fresh.ssm_state[3].is_none());
        // a cache of other slots, or of other heads: refused, and left as it was
        let mut other = KvCache::new_lazy_per_layer_kv(vec![backend.clone(); 4], 64, &[1, 1, 1, 1], &[3, 1, 1, 4]);
        other.len = 1;
        assert!(back.restore_whole(&mut other, &rows).is_err());
        assert_eq!(other.len, 1);
        assert!(back.restore_whole(&mut make(), &[true, false, true, true]).is_err());
        assert!(back.restore_whole(&mut make(), &[true, false, false]).is_err());
    }

    #[test]
    fn a_disk_state_made_after_the_run_is_the_one_taken_at_its_checkpoint() {
        let backend = ggml_rs::CpuBackend::new();
        let mut kv = KvCache::new(&backend, 2, 16, 1, 2);
        kv.len = 3;
        kv.k[0].data_mut()[..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        kv.v[0].data_mut()[..6].copy_from_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        kv.ssm_state[1] = Some(Tensor::from_vec(vec![4.0; 4], vec![1, 2, 2]));
        kv.ssm_conv[1] = Some(Tensor::from_vec(vec![5.0; 12], vec![2, 6]));
        let then = Snapshot::capture(&kv, &[true, false], &backend);
        let checkpoint = RecurrentSnapshot::capture(&kv);
        // the run goes on: more keys and values after the checkpoint's, the recurrent state moved on
        kv.k[0].data_mut()[6..14].fill(99.0);
        kv.v[0].data_mut()[6..14].fill(98.0);
        kv.len = 7;
        kv.ssm_state[1].as_mut().unwrap().data_mut().fill(97.0);
        let later = Snapshot::from_checkpoint(&checkpoint, &kv, &[true, false], &backend).unwrap();
        // its size known before it is made: the keys and values, which the rest is little beside
        let rows: usize = later.layers.iter().flat_map(|l| &l[..2]).flatten().map(|t| 4 * t.numel()).sum();
        assert_eq!(Snapshot::whole_bytes(3, &kv, &[true, false]), rows as u64);
        assert_eq!(rows, 2 * 3 * 2 * 4);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        Arc::new(then).encode(&mut a);
        Arc::new(later).encode(&mut b);
        assert_eq!(a, b);
        // not of this cache: a layout of another model, or a position past what it holds
        assert!(Snapshot::from_checkpoint(&checkpoint, &kv, &[true, false, true], &backend).is_none());
        kv.len = 2;
        assert!(Snapshot::from_checkpoint(&checkpoint, &kv, &[true, false], &backend).is_none());
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

    /// An LFM2 model's state (its short convs': a placeholder of one value and the conv window over the model's width,
    /// `SsmConfig` with no delta net) goes back as a delta net's does; one of a delta net's shape is refused.
    #[test]
    fn a_short_convs_state_restores() {
        let backend = ggml_rs::CpuBackend::new();
        let mut kv = KvCache::new(&backend, 2, 8, 1, 2);
        kv.len = 3;
        kv.k[1].data_mut()[..6].fill(2.0);
        kv.v[1].data_mut()[..6].fill(3.0);
        kv.ssm_state[0] = Some(Tensor::zeros(vec![1, 1, 1]));
        kv.ssm_conv[0] = Some(Tensor::from_vec((0..8).map(|v| v as f32).collect(), vec![2, 4]));
        let snap = Snapshot::capture(&kv, &[false, true], &backend);
        kv.ssm_conv[0].as_mut().unwrap().data_mut().fill(99.0);
        kv.len = 6;
        let lfm2 = SsmConfig { conv_kernel: 3, group_count: 0, inner_size: 4, state_size: 0, time_step_rank: 0 };
        assert_eq!((lfm2.state_shape(), lfm2.conv_shape()), ([1, 1, 1], [2, 4]));
        snap.restore(&mut kv, &[false, true], lfm2, &backend).unwrap();
        assert_eq!(kv.len, 3);
        assert_eq!(kv.ssm_conv[0].as_ref().unwrap().data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let delta_net = SsmConfig { conv_kernel: 3, group_count: 1, inner_size: 2, state_size: 1, time_step_rank: 1 };
        assert!(snap.restore(&mut kv, &[false, true], delta_net, &backend).is_err());
    }
}
