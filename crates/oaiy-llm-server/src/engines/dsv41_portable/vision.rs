//! DeepSeek-V4.1's pictures: its vision tower (the ViT and its aligner, [`crate::dsv41::vision::VisionTower`]) and the
//! spans it makes of a prompt's images, each its delimiters' learned rows around the aligner's rows, in reading order
//! ([`crate::dsv41::vision::span_rows`]). A checkpoint's tower is its own; a GGUF, which carries none, is given the
//! folder of the same model's safetensors checkpoint (`--vision-projector`), whose tower and image bias it takes.

use std::path::Path;
use std::sync::Arc;

use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec};
use ggml_rs_wgpu::WgpuBackend;

use crate::dsv41::config::{Config, VisionConfig};
use crate::dsv41::vision::{rope_tables, unfold, Linear, Prepared, VisionTower, TOWER_NORM_EPS};
use crate::engines::job::JobImage;

/// The spans of the last pictures read, by their bytes' hash: a conversation sends its pictures again with each turn,
/// and an agent's screenshot of a page that did not change is the same picture.
const SEEN: usize = 8;

pub(crate) struct Vision {
    cfg: VisionConfig,
    /// The tower on the host (the reference's arithmetic, bf16 roundings and all), until one is made on a card.
    host: Option<VisionTower>,
    card: Option<CardTower>,
    /// The span's delimiters' learned rows: `image_start`, `image_newline`, `image_end`.
    delimiters: [Vec<f32>; 3],
    seen: Vec<(u64, Arc<Vec<f32>>)>,
}

impl Vision {
    /// The tower of the checkpoint at `dir` (its `config.json`'s `vision_config`, its `vision.*`, `aligner.*` and
    /// `image_*` tensors), for a model of `model`'s width.
    pub(crate) fn load(dir: &Path, model: &Config) -> oaiy_engine::Result<Vision> {
        let cfg = Config::load(dir)?;
        if cfg.dim != model.dim || cfg.vocab_size != model.vocab_size || cfg.image_token_id != model.image_token_id {
            return Err(oaiy_engine::Error::Arg(format!(
                "{}: its vision tower is for a model {} wide with {} tokens (image token {}), where this one is {} wide with {} (image token {})",
                dir.display(),
                cfg.dim,
                cfg.vocab_size,
                cfg.image_token_id,
                model.dim,
                model.vocab_size,
                model.image_token_id
            )));
        }
        let tower = VisionTower::load(dir, &cfg)?;
        let delimiters = [tower.start.clone(), tower.newline.clone(), tower.end.clone()];
        Ok(Vision { cfg: tower.cfg.clone(), host: Some(tower), card: None, delimiters, seen: Vec::new() })
    }

    /// How its pictures are sized: at most `max_tokens` tokens a picture where that is given and fewer than the
    /// checkpoint's own most (`max_image_tokens`, 1,024 in V4.1: some 25 s of a 2-bit GGUF's prompt reading).
    pub(crate) fn config(&self, max_tokens: Option<usize>) -> VisionConfig {
        let mut vc = self.cfg.clone();
        if let Some(most) = max_tokens {
            vc.max_tokens = vc.max_tokens.min(most.max(MIN_TOKENS));
        }
        vc
    }

    /// The tower made on `card` (its matrices f16 there, and room for a picture of `max_tokens` tokens' passes taken
    /// from its budget first, so the experts that fill the rest leave it), the host's copy let go: what it took, or
    /// why it stays on the host. A picture's tower on the host took minutes where on a card it is a second.
    pub(crate) fn on_card(&mut self, card: Arc<WgpuBackend>, max_tokens: Option<usize>) -> Result<u64, String> {
        let host = self.host.as_ref().ok_or("its tower is on a card already")?;
        let vc = self.config(max_tokens);
        // (the most patches a picture of so many tokens has: each token a 3 x 3 of them)
        let patches = vc.max_tokens * vc.downsample * vc.downsample;
        let passes = CardTower::pass_bytes(&vc, host.weights().aligner[1].n, patches);
        let weights = CardTower::weight_bytes(host);
        if !card.reserve(weights + passes) {
            return Err(format!("{:.1} GB for it are not left of the card's budget", (weights + passes) as f64 / 1e9));
        }
        self.card = Some(CardTower::of(host, &vc, card).ok_or("a matrix of it does not go to f16")?);
        self.host = None;
        Ok(weights + passes)
    }

    /// `img`'s span of rows (`[img.n_tokens()][dim]`), made by the tower, or kept from when the same picture (`hash`)
    /// was last read.
    pub(crate) fn span(&mut self, img: &Prepared, hash: u64) -> Arc<Vec<f32>> {
        let d = self.delimiters[0].len();
        if let Some(at) = self.seen.iter().position(|(h, rows)| *h == hash && rows.len() == img.n_tokens() * d) {
            let kept = self.seen.remove(at);
            self.seen.push(kept.clone());
            return kept.1;
        }
        let aligned = match (&self.card, &self.host) {
            (Some(card), _) => card.encode(img),
            (None, Some(host)) => host.encode(img),
            (None, None) => unreachable!("a tower on the host or a card"),
        };
        let [start, newline, end] = &self.delimiters;
        let rows = Arc::new(crate::dsv41::vision::span_rows(img, &aligned, start, newline, end));
        if self.seen.len() == SEEN {
            self.seen.remove(0);
        }
        self.seen.push((hash, Arc::clone(&rows)));
        rows
    }

    /// Nothing kept of the pictures read (an incognito request's state is wiped).
    pub(crate) fn forget(&mut self) {
        self.seen.clear();
    }
}

/// A linear layer on the card: its matrix f16, and its bias where it has one.
struct Lin {
    w: DeviceVec,
    b: Option<DeviceVec>,
    n: usize,
    k: usize,
}

impl Lin {
    fn of(chain: &dyn DeviceChain, l: &Linear) -> Option<Lin> {
        Some(Lin { w: chain.vec_f16_rounded(&l.w)?, b: l.b.as_ref().map(|b| vector(chain, b)), n: l.n, k: l.k })
    }

    /// `y[r] = W x[r] + b` for `rows` rows, the inputs f16 once scaled into its range (the tensor cores' kernel).
    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        r.matmul_f16_rows_ranged(&self.w, self.n, self.k, x, y, rows);
        if let Some(b) = &self.b {
            r.add_bias_rows(y, b, rows, self.n);
        }
    }
}

fn vector(chain: &dyn DeviceChain, values: &[f32]) -> DeviceVec {
    let v = chain.vec(values.len().max(1));
    chain.upload(&v, values);
    v
}

struct Block {
    norm1: DeviceVec,
    wqkv: Lin,
    wo: Lin,
    norm2: DeviceVec,
    w1: Lin,
    w2: Lin,
}

/// The tower on a card ([`VisionTower`]'s arithmetic in f32, its matrices f16): the patch projection, 32 blocks (RMS
/// norms, the fused q, k and v, full attention over the picture's patches rotated by each one's row and column, the
/// SwiGLU MLP), the last norm, then each 3 x 3 patches' features side by side (on the host: `unfold`) through the
/// aligner's erf-GELU MLP. A recording a block, each a few milliseconds: none near Windows' two seconds.
struct CardTower {
    card: Arc<WgpuBackend>,
    patch: Lin,
    blocks: Vec<Block>,
    norm: DeviceVec,
    aligner: [Lin; 2],
    cfg: VisionConfig,
}

impl CardTower {
    fn of(host: &VisionTower, cfg: &VisionConfig, card: Arc<WgpuBackend>) -> Option<CardTower> {
        let chain = ggml_rs::Backend::chain(card.as_ref())?;
        let w = host.weights();
        let blocks = w
            .blocks
            .iter()
            .map(|b| Some(Block { norm1: vector(chain, b.norm1), wqkv: Lin::of(chain, b.wqkv)?, wo: Lin::of(chain, b.wo)?, norm2: vector(chain, b.norm2), w1: Lin::of(chain, b.w1)?, w2: Lin::of(chain, b.w2)? }))
            .collect::<Option<Vec<_>>>()?;
        Some(CardTower { card: Arc::clone(&card), patch: Lin::of(chain, w.patch)?, blocks, norm: vector(chain, w.norm), aligner: [Lin::of(chain, w.aligner[0])?, Lin::of(chain, w.aligner[1])?], cfg: cfg.clone() })
    }

    /// Its matrices' and vectors' bytes on a card.
    fn weight_bytes(host: &VisionTower) -> u64 {
        let w = host.weights();
        let lin = |l: &Linear| (l.n * l.k * 2 + l.b.as_ref().map_or(0, |b| b.len() * 4)) as u64;
        let blocks: u64 = w.blocks.iter().map(|b| lin(b.wqkv) + lin(b.wo) + lin(b.w1) + lin(b.w2) + ((b.norm1.len() + b.norm2.len()) * 4) as u64).sum();
        lin(w.patch) + blocks + (w.norm.len() * 4) as u64 + lin(w.aligner[0]) + lin(w.aligner[1])
    }

    /// The bytes a picture's passes take on the card: its `patches` rows' vectors through the blocks, and its tokens'
    /// through the aligner (`width` the model's).
    fn pass_bytes(cfg: &VisionConfig, width: usize, patches: usize) -> u64 {
        let (d, ff) = (cfg.dim, cfg.inter_dim);
        let in_dim = 3 * cfg.patch_size * cfg.patch_size;
        // patches in; x, normed, q, k, v, o and attention's out; qkv; kv; the MLP's fused and its product
        let blocks = patches * (in_dim + 7 * d + 3 * d + 2 * d + 2 * ff + ff);
        let tokens = patches.div_ceil(cfg.downsample * cfg.downsample);
        let aligner = tokens * (d * cfg.downsample * cfg.downsample + 3 * width);
        // (attention's out padded to 64 rows)
        ((blocks + aligner + 64 * d) * 4) as u64
    }

    /// `img`'s aligner rows, `[n_llm_h * n_llm_w][width]`.
    fn encode(&self, img: &Prepared) -> Vec<f32> {
        let chain = ggml_rs::Backend::chain(self.card.as_ref()).expect("the tower's card has a chain");
        let cfg = &self.cfg;
        let (n, d, heads) = (img.n_patches(), cfg.dim, cfg.n_heads);
        let (hd, ff) = (d / heads, cfg.inter_dim);
        let v = |len: usize| chain.vec(len.max(1));
        // each patch's rotation: the first half of a head's pairs by its row, the next by its column, as (sine, cosine)
        let (cos, sin) = rope_tables(img.n_vit_h, img.n_vit_w, hd / 2, cfg.rope_theta);
        let table: Vec<f32> = cos.iter().zip(&sin).flat_map(|(&c, &s)| [s, c]).collect();
        let (patches, table) = (vector(chain, &img.patches), vector(chain, &table));
        let (x, normed, qkv, q, k, vv, kv, o) = (v(n * d), v(n * d), v(n * 3 * d), v(n * d), v(n * d), v(n * d), v(n * 2 * d), v(n * d));
        let att = v(chain.attention_rows_full_out_len(n, heads, hd, n));
        let (f, fg) = (v(n * 2 * ff), v(n * ff));
        let scale = 1.0 / (hd as f32).sqrt();
        {
            let mut rec = chain.begin();
            rec.keep_groups(false);
            self.patch.run(rec.as_mut(), &patches, &x, n);
            rec.finish();
        }
        for b in &self.blocks {
            let mut rec = chain.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            r.rmsnorm_rows(&x, &b.norm1, &normed, n, TOWER_NORM_EPS);
            b.wqkv.run(r, &normed, &qkv, n);
            r.copy_cols(&qkv, &q, n, d, 3 * d, 0);
            r.copy_cols(&qkv, &k, n, d, 3 * d, d);
            r.copy_cols(&qkv, &vv, n, d, 3 * d, 2 * d);
            r.rope_rows(&q, n, heads, hd, &table, true);
            r.rope_rows(&k, n, heads, hd, &table, true);
            r.store_rows(&k, &kv, n, d, 0, 2 * d, 0);
            r.store_rows(&vv, &kv, n, d, 0, 2 * d, d);
            r.attention_rows_full(&q, &kv, &att, n, heads, heads, hd, n, scale);
            b.wo.run(r, &att, &o, n);
            r.add(&x, &o);
            r.rmsnorm_rows(&x, &b.norm2, &normed, n, TOWER_NORM_EPS);
            b.w1.run(r, &normed, &f, n);
            r.silu_mul_split_rows(&f, &fg, n);
            b.w2.run(r, &fg, &o, n);
            r.add(&x, &o);
            rec.finish();
        }
        let feats = {
            let mut rec = chain.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            r.rmsnorm_rows(&x, &self.norm, &normed, n, TOWER_NORM_EPS);
            r.read(&normed);
            rec.finish().pop().expect("the features read")
        };
        // each 3 x 3 patches' features side by side (zeros past the edge), then the aligner
        let r3 = cfg.downsample;
        let u = unfold(&feats, img.n_vit_h, img.n_vit_w, d, r3);
        let rows = u.len() / (d * r3 * r3);
        let [a1, a2] = &self.aligner;
        let (u, h, g, out) = (vector(chain, &u), v(rows * a1.n), v(rows * a1.n), v(rows * a2.n));
        let mut rec = chain.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        a1.run(r, &u, &h, rows);
        r.gelu_erf(&h, &g, rows * a1.n);
        a2.run(r, &g, &out, rows);
        r.read(&out);
        rec.finish().pop().expect("the aligner's rows read")
    }
}

/// The fewest tokens a picture is let down to (`--image-tokens`): a 13 x 13 grid, the smallest the checkpoint's
/// `min_pixels` makes of a square picture.
pub(crate) const MIN_TOKENS: usize = 13 * 14 + 2;

/// `prompt` as states know it: each image's tokens (all the image token) told apart by its bytes' hash and their place
/// in its span, so a state is read on from only for the same pictures in the same places. Ids past the vocabulary's.
pub(crate) fn keyed(prompt: &[u32], images: &[JobImage]) -> Vec<u32> {
    let mut keys = prompt.to_vec();
    for img in images {
        let n = img.prep.n_tokens();
        for (k, key) in keys.iter_mut().skip(img.start).take(n).enumerate() {
            *key = 0x8000_0000 | (mix(img.hash ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) as u32 & 0x7FFF_FFFF);
        }
    }
    keys
}

/// splitmix64's finalizer.
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::job::ImagePrep;
    use std::path::PathBuf;

    /// A picture: DSV41_IMAGE's (a PNG or JPEG), else one made of bands and a gradient, `width` x `height`.
    fn image(width: usize, height: usize) -> oaiy_image::Image {
        if let Some(file) = std::env::var_os("DSV41_IMAGE") {
            return oaiy_image::decode(&std::fs::read(file).expect("DSV41_IMAGE")).expect("a PNG or JPEG");
        }
        let rgb = (0..width * height).flat_map(|i| {
            let (x, y) = (i % width, i / width);
            [(x * 255 / width) as u8, if (y / 40) % 2 == 0 { 230 } else { 30 }, ((x + y) % 256) as u8]
        });
        oaiy_image::Image { width, height, rgb: rgb.collect() }
    }

    /// The tower on a card gives the host's rows (the reference's, which rounds every value to bf16 as the checkpoint's
    /// own arithmetic does) but for those roundings: after one block within 1% (relative L2), and after all 32, where
    /// the roundings have grown through them (0.99998 in cosine after 8 blocks, 0.9989 after 32, the same with the
    /// matrices bf16 or the inputs f32), within the 6% the CUDA tower was held to; and in a hundredth of the host's
    /// time. Then a picture of the most tokens, on the card alone. DSV41_VISION: a DeepSeek-V4.1 checkpoint's folder (its
    /// tower); on a GPU.
    #[test]
    #[ignore]
    fn the_tower_on_a_card_gives_the_hosts_rows() {
        let Some(dir) = std::env::var_os("DSV41_VISION").map(PathBuf::from) else { return };
        let Ok(card) = WgpuBackend::new(Some(8 << 30)) else { return };
        let cfg = Config::load(&dir).unwrap();
        let vc = cfg.vision.clone().expect("a checkpoint with a vision tower");
        let host = VisionTower::load(&dir, &cfg).unwrap();
        let card = Arc::new(card);
        let tower = CardTower::of(&host, &vc, Arc::clone(&card)).expect("the tower on the card");
        let small = crate::dsv41::vision::prepare(&image(320, 240), &vc);
        let rel = |got: &[f32], want: &[f32]| {
            assert_eq!(got.len(), want.len());
            let diff: f64 = got.iter().zip(want).map(|(a, b)| f64::from(a - b).powi(2)).sum();
            (diff / want.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>()).sqrt()
        };
        // one block, then the last norm and the aligner, both ways
        let (cos, sin) = rope_tables(small.n_vit_h, small.n_vit_w, vc.dim / vc.n_heads / 2, vc.rope_theta);
        let x = host.block(0, &host.patch_embed(&small), small.n_patches(), &cos, &sin);
        let want = host.align(&crate::dsv41::ops::rmsnorm(&x, host.weights().norm, TOWER_NORM_EPS), &small);
        let mut one = CardTower::of(&host, &vc, Arc::clone(&card)).unwrap();
        one.blocks.truncate(1);
        let first = rel(&one.encode(&small), &want);
        // all of it
        let began = std::time::Instant::now();
        let want = host.encode(&small);
        let host_s = began.elapsed().as_secs_f64();
        let _ = tower.encode(&small);
        let began = std::time::Instant::now();
        let got = tower.encode(&small);
        let card_s = began.elapsed().as_secs_f64();
        let whole = rel(&got, &want);
        eprintln!("{} patches, {} tokens: the host {host_s:.1}s, the card {card_s:.2}s; relative L2 after a block {first:.1e}, after all {whole:.1e}", small.n_patches(), small.n_tokens());
        assert!(first < 1e-2 && whole < 6e-2, "the card's rows are not the host's: {first} after a block, {whole} after all");
        // the most tokens a picture takes, on the card alone
        let big = crate::dsv41::vision::prepare(&image(1920, 1200), &vc);
        let began = std::time::Instant::now();
        let rows = tower.encode(&big);
        eprintln!("{} patches, {} tokens on the card in {:.2}s", big.n_patches(), big.n_tokens(), began.elapsed().as_secs_f64());
        let d = host.weights().aligner[1].n;
        assert!(rows.iter().all(|v| v.is_finite()) && rows.len() == big.n_llm_h * big.n_llm_w * d);
    }

    /// The tower of a checkpoint (DSV41_VISION: its folder) loads with every layer's image bias beside it, and its
    /// pictures are sized as asked: no more tokens than `--image-tokens`, and none let down below [`MIN_TOKENS`].
    #[test]
    fn the_tower_and_its_image_bias_load() {
        let Some(dir) = std::env::var_os("DSV41_VISION").map(PathBuf::from) else { return };
        let cfg = Config::load(&dir).unwrap();
        let vision = Vision::load(&dir, &cfg).unwrap();
        let idx = crate::dsv41::safetensors::StIndex::open(&dir).unwrap();
        for l in 0..cfg.n_layers {
            assert_eq!(crate::dsv41::linear::load_vec(&idx, &format!("layers.{l}.ffn.gate.bias_vl")).unwrap().len(), cfg.n_routed_experts);
        }
        let own = vision.config(None);
        assert_eq!(own.max_tokens, cfg.vision.as_ref().unwrap().max_tokens);
        assert_eq!(vision.config(Some(600)).max_tokens, 600);
        assert_eq!(vision.config(Some(10)).max_tokens, MIN_TOKENS);
        let fewer = crate::dsv41::vision::prepare(&image(1920, 1080), &vision.config(Some(600)));
        assert!(fewer.n_tokens() <= 600, "{} tokens", fewer.n_tokens());
        let all = crate::dsv41::vision::prepare(&image(1920, 1080), &own);
        assert!(all.n_tokens() > 600 && all.n_tokens() <= own.max_tokens, "{} tokens", all.n_tokens());
    }

    /// A GGUF of the model writes a picture's place as its image token (DSV41_GGUF: its folder): what the tower's span
    /// takes the rows of.
    #[test]
    fn a_ggufs_tokenizer_has_the_image_token() {
        let Some(dir) = std::env::var_os("DSV41_GGUF").map(PathBuf::from) else { return };
        let tok = crate::dsv41::tokenizer::Tokenizer::load(&dir).unwrap();
        let id = tok.special(crate::dsv41::chat::IMAGE).expect("the image placeholder a token of its own");
        let cfg = crate::dsv41::gguf_file::open(&dir).unwrap().config;
        assert_eq!(id, cfg.image_token_id);
        assert_eq!(tok.encode(&format!("a{}b", crate::dsv41::chat::IMAGE)).iter().filter(|&&t| t == id).count(), 1);
    }

    fn picture(start: usize, hash: u64) -> JobImage {
        let prep = Prepared { patches: Vec::new(), n_vit_h: 6, n_vit_w: 6, n_llm_h: 2, n_llm_w: 2 };
        JobImage { start, prep: ImagePrep::Deepseek(prep), hash }
    }

    #[test]
    fn a_pictures_tokens_are_told_apart_by_its_hash_and_place() {
        // 2 x 2 tokens and their delimiters: 2 * 3 + 2 = 8 a picture, all the image token
        let prompt: Vec<u32> = [1, 2].into_iter().chain([9; 8]).chain([3]).collect();
        let a = keyed(&prompt, &[picture(2, 7)]);
        let b = keyed(&prompt, &[picture(2, 8)]);
        assert_eq!(a[..2], prompt[..2]);
        assert_eq!(a[10], 3);
        assert!(a[2..10].iter().all(|&k| k >= 0x8000_0000), "past the vocabulary");
        let mut distinct = a[2..10].to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 8, "each place its own key");
        assert_eq!(a, keyed(&prompt, &[picture(2, 7)]), "the same picture, the same keys");
        assert_eq!(a.iter().zip(&b).take_while(|(x, y)| x == y).count(), 2, "another picture: the states part at its first token");
    }
}
