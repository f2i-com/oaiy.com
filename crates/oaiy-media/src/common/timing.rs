//! Timings of the picture models' stages on this computer, for speed work (each test `#[ignore]`d: it needs the model
//! files, named by the environment, and a GPU). Prints, asserts nothing.

/// A caption as an enhancer writes one: some 250 words, what a picture's prompt usually is by the time it is encoded.
#[cfg(test)]
pub const CAPTION: &str = "A close-up, eye-level photograph of a white ceramic coffee mug standing on a light oak desk beside a tall \
window, soft morning light falling from the left and casting a long, gentle shadow across the wood grain. The mug is glossy, with \
a slightly rounded rim and a curved handle on the right, and on its side the words \"GOOD MORNING\" are printed in bold red capital \
letters, centred and evenly spaced. A thin wisp of steam rises from the dark coffee inside. Behind the mug the window is out of \
focus, showing a pale blue sky and the blurred green leaves of a potted plant on the sill; a closed notebook with a black pen lies \
at the lower left of the frame, its cover a muted grey. The colour palette is warm and calm: cream, honey-brown wood, soft greens, \
with the red lettering the strongest accent. The depth of field is shallow, the mug sharp from rim to base, the background softly \
blurred into round highlights. The composition places the mug slightly right of centre, the handle turned towards the light, the \
desk's edge running diagonally across the lower third. Realistic textures throughout: the faint glaze reflections on the ceramic, \
fine scratches in the varnish of the desk, the paper grain of the notebook. The mood is quiet and inviting, the first cup of the \
day before work begins.";

#[cfg(test)]
mod tests {
    use super::CAPTION;
    use std::time::Instant;

    fn var(name: &str) -> Option<std::path::PathBuf> {
        std::env::var_os(name).map(std::path::PathBuf::from)
    }

    /// FLUX 3 Image Klein's text encoder (`OAIY_KLEIN_TEXT_ENCODER`, `OAIY_KLEIN_TOKENIZER`): loading, then the caption
    /// three times, as a picture's job asks.
    #[test]
    #[ignore = "needs FLUX 3 Klein's text encoder (OAIY_KLEIN_TEXT_ENCODER, OAIY_KLEIN_TOKENIZER) and a WebGPU adapter"]
    fn klein_text_encoder() {
        let (Some(te), Some(tok)) = (var("OAIY_KLEIN_TEXT_ENCODER"), var("OAIY_KLEIN_TOKENIZER")) else { panic!("set the paths") };
        for round in 0..2 {
            let t = Instant::now();
            let mut encoder = crate::klein::webgpu::text::WgpuKleinText::load_spec(&te, &tok, 0, crate::klein::webgpu::text::FLUX3).unwrap();
            let loaded = t.elapsed().as_secs_f64();
            let mut each = Vec::new();
            for _ in 0..3 {
                let t = Instant::now();
                let out = encoder.encode_all(&[CAPTION]).unwrap();
                each.push(format!("{:.2} s ({:?})", t.elapsed().as_secs_f64(), out[0].dims()));
            }
            eprintln!("klein text encoder, round {round}: loaded in {loaded:.2} s; encoded in {}", each.join(", "));
        }
    }

    /// MOSS-Image's text encoder (`OAIY_MOSS_TEXT_ENCODER`, with `OAIY_MOSS_TOKENIZER`): loading, then the caption three
    /// times with its layers sent each time (as on a Mac now) and three kept.
    #[test]
    #[ignore = "needs MOSS-Image's text encoder (OAIY_MOSS_TEXT_ENCODER, OAIY_MOSS_TOKENIZER) and a WebGPU adapter"]
    fn moss_text_encoder() {
        let Some(te) = var("OAIY_MOSS_TEXT_ENCODER") else { panic!("set the path") };
        let tok = var("OAIY_MOSS_TOKENIZER");
        let t = Instant::now();
        let mut encoder = crate::moss_image::webgpu::text::WgpuMossText::load(&te, tok.as_deref(), 0).unwrap();
        eprintln!("moss text encoder: loaded in {:.2} s; {} tokens", t.elapsed().as_secs_f64(), encoder.tokens(CAPTION).unwrap().len());
        for keep in [false, true] {
            encoder.keep_layers(keep);
            let mut each = Vec::new();
            for _ in 0..3 {
                let t = Instant::now();
                encoder.encode_all(&[CAPTION]).unwrap();
                each.push(format!("{:.2} s", t.elapsed().as_secs_f64()));
            }
            eprintln!("moss text encoder, layers {}: {}", if keep { "kept" } else { "sent each time" }, each.join(", "));
        }
    }
}
