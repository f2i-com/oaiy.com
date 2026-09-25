// Ported from the user-owned F2I plugin-diffusion SDXL implementation.
//! SDXL configuration constants — hyperparameters frozen by the
//! reference SDXL 1.0 release (and inherited by every Civitai
//! derivative including Illustrious-XL, Pony Diffusion XL, etc.).
//!
//! Values cross-checked against Stability AI's
//! `sd_xl_base_1.0.safetensors` `model.diffusion_model.*` and
//! `first_stage_model.*` tensor shapes.

/// Top-level SDXL config bundle. Sub-configs are inlined so a future
/// "tweaked SDXL" variant can override one component without copying
/// every field.
#[derive(Debug, Clone)]
pub struct SdxlConfig {
    pub vae: VaeConfig,
    pub unet: UNetConfig,
    pub text_encoder: TextEncoderConfig,
}

impl SdxlConfig {
    /// Stock SDXL 1.0 — the config Illustrious-XL / Pony / etc. all
    /// inherit unchanged.
    pub fn sdxl_1_0() -> Self {
        Self {
            vae: VaeConfig::sdxl_default(),
            unet: UNetConfig::sdxl_1_0(),
            text_encoder: TextEncoderConfig::sdxl_dual_clip(),
        }
    }
}

// ---------------------------------------------------------------------------
// AutoencoderKL — the VAE
// ---------------------------------------------------------------------------

/// VAE config. The SDXL VAE is structurally identical to SD 1.5's
/// (4-channel latent, 8× spatial downsample, 4-stage symmetric encoder/
/// decoder); only the trained weights differ.
#[derive(Debug, Clone)]
pub struct VaeConfig {
    /// Output channels of the initial encoder conv (= input channels
    /// of the final decoder conv). SDXL ships 128.
    pub base_channels: usize,
    /// Channel multipliers per resolution. SDXL = [1, 2, 4, 4] so the
    /// encoder produces 128, 256, 512, 512 channels at successive
    /// resolutions. 4 entries = 4 down stages = 8× spatial downsample.
    pub channel_mults: Vec<usize>,
    /// ResBlocks per stage. SDXL = 2.
    pub layers_per_block: usize,
    /// Latent channel count after the encoder's split-into-mean-and-logvar.
    /// On-disk `conv_out` produces `2 * latent_channels` (mu + logvar
    /// stacked), and `quant_conv` is a 1×1 that projects that 8-ch to
    /// 8-ch (acts as a 1×1 KL-projection bias). SDXL = 4.
    pub latent_channels: usize,
    /// Scaling factor applied to the latent before/after the UNet. The
    /// SDXL VAE outputs roughly-unit-variance latents that the UNet was
    /// trained against scaled by 0.13025. (SD 1.5 used 0.18215.)
    pub scaling_factor: f64,
    /// Input image channels (3 for RGB).
    pub in_channels: usize,
    /// Output image channels (3 for RGB).
    pub out_channels: usize,
    /// GroupNorm groups in every Norm layer. SDXL = 32.
    pub norm_groups: usize,
}

impl VaeConfig {
    pub fn sdxl_default() -> Self {
        Self {
            base_channels: 128,
            channel_mults: vec![1, 2, 4, 4],
            layers_per_block: 2,
            latent_channels: 4,
            scaling_factor: 0.13025,
            in_channels: 3,
            out_channels: 3,
            norm_groups: 32,
        }
    }

    /// Resolved per-stage output channels: [128, 256, 512, 512] for the
    /// stock config.
    pub fn channels_per_stage(&self) -> Vec<usize> {
        self.channel_mults
            .iter()
            .map(|m| m * self.base_channels)
            .collect()
    }

    /// Number of down/up stages (== `channel_mults.len()`).
    pub fn num_stages(&self) -> usize {
        self.channel_mults.len()
    }
}

// ---------------------------------------------------------------------------
// UNet — the diffusion model
// ---------------------------------------------------------------------------

/// SDXL UNet config. Values match `sd_xl_base_1.0.safetensors`.
#[derive(Debug, Clone)]
pub struct UNetConfig {
    /// Input latent channels (= VAE latent_channels). 4 for SDXL.
    pub in_channels: usize,
    /// Output channels of the final conv. = in_channels (the UNet
    /// predicts noise in latent space). 4 for SDXL.
    pub out_channels: usize,
    /// Block-output channel counts per resolution.
    /// SDXL = [320, 640, 1280].
    pub block_out_channels: Vec<usize>,
    /// Resnet blocks per up/down stage. 2 for SDXL.
    pub layers_per_block: usize,
    /// Number of transformer layers inside each `CrossAttnDownBlock` /
    /// `CrossAttnUpBlock` resolution. SDXL = [0, 2, 10] — the lowest
    /// resolution has NO attention (just plain DownBlock), the middle
    /// has 2 transformer blocks per ResNet, the deepest has 10. Length
    /// must equal `block_out_channels.len()`.
    pub transformer_layers_per_block: Vec<usize>,
    /// Head dimension of every cross-attention head. 64 for SDXL.
    pub attention_head_dim: usize,
    /// Time-embedding inner dimension. 1280 (= 4 × block_out[0]).
    pub time_embed_dim: usize,
    /// ADM/label-embedding input dimension. SDXL concatenates
    /// (pooled_clip_g [1280], micro-conditioning [256 × 6 = 1536]) →
    /// 2816 total dims.
    pub label_emb_in_dim: usize,
    /// Cross-attention context dimension (concatenated CLIP-L 768 +
    /// CLIP-G 1280 → 2048).
    pub cross_attention_dim: usize,
    /// GroupNorm groups in every Norm layer. 32 for SDXL.
    pub norm_groups: usize,
}

impl UNetConfig {
    pub fn sdxl_1_0() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            block_out_channels: vec![320, 640, 1280],
            layers_per_block: 2,
            transformer_layers_per_block: vec![0, 2, 10],
            attention_head_dim: 64,
            time_embed_dim: 1280,
            label_emb_in_dim: 2816,
            cross_attention_dim: 2048,
            norm_groups: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// Dual text encoder
// ---------------------------------------------------------------------------

/// SDXL text-encoder config. Two CLIP-family encoders are run in
/// parallel; their last-hidden-state outputs are concatenated along the
/// channel dim to form the cross-attention context. CLIP-G's *pooled*
/// output is separately fed into the ADM (`label_emb`) conditioning
/// stream alongside the micro-conditioning (original size + crop).
#[derive(Debug, Clone)]
pub struct TextEncoderConfig {
    pub clip_l: CLIPConfig,
    pub clip_g: CLIPConfig,
}

impl TextEncoderConfig {
    pub fn sdxl_dual_clip() -> Self {
        Self {
            clip_l: CLIPConfig::clip_l_14_336(),
            clip_g: CLIPConfig::open_clip_g_14_laion2b(),
        }
    }
}

/// CLIP text encoder hyperparameters. Same struct serves both CLIP-L
/// (OpenAI CLIP-ViT-L/14, 768-d, 12 layers) and CLIP-G (OpenCLIP
/// G/14 trained on LAION-2B, 1280-d, 32 layers).
#[derive(Debug, Clone)]
pub struct CLIPConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    /// True for the CLIP-G last-layer hidden state (which SDXL
    /// **doesn't** apply final_layer_norm to before passing to the
    /// UNet — verified against the SDXL paper + reference impl).
    /// False for CLIP-L (gets the standard penultimate-layer state).
    pub use_pooled_output: bool,
}

impl CLIPConfig {
    /// OpenAI CLIP-ViT-L/14 — the SD 1.x text encoder, kept in SDXL.
    pub fn clip_l_14_336() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            max_position_embeddings: 77,
            use_pooled_output: false,
        }
    }

    /// OpenCLIP ViT-G/14 trained on LAION-2B — the second SDXL encoder.
    pub fn open_clip_g_14_laion2b() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 20,
            max_position_embeddings: 77,
            use_pooled_output: true,
        }
    }
}
