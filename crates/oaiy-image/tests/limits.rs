//! What a small file may ask of the decoders: a header is a claim, and a 200-byte JPEG claiming 179 million pixels in
//! four components asked for 2.7 GB, on a request's thread.

/// A JPEG up to its first scan: a quantization table, one-code Huffman tables and a frame of `n` components.
fn jpeg(width: u16, height: u16, n: u8) -> Vec<u8> {
    let mut f = vec![0xFF, 0xD8];
    f.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]);
    f.extend_from_slice(&[1; 64]);
    for class in [0x00, 0x10] {
        f.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, class, 1]);
        f.extend_from_slice(&[0; 15]);
        f.push(0);
    }
    f.extend_from_slice(&[0xFF, 0xC0, 0x00, 8 + 3 * n, 8]);
    f.extend_from_slice(&height.to_be_bytes());
    f.extend_from_slice(&width.to_be_bytes());
    f.push(n);
    for id in 1..=n {
        f.extend_from_slice(&[id, 0x11, 0]);
    }
    f
}

/// A scan of the first component, with no data after it.
const EMPTY_SCAN: [u8; 10] = [0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00];

fn scans(mut file: Vec<u8>, n: usize) -> Vec<u8> {
    for _ in 0..n {
        file.extend_from_slice(&EMPTY_SCAN);
    }
    file.extend_from_slice(&[0xFF, 0xD9]);
    file
}

fn refused_for_size(file: &[u8]) -> bool {
    oaiy_image::decode(file).is_err_and(|e| e.to_string().contains("image size"))
}

#[test]
fn a_jpeg_frame_past_the_limit_is_refused_before_its_coefficients_are_reserved() {
    assert!(refused_for_size(&jpeg(65_535, 2_730, 4)), "179 million pixels of four components");
    assert!(refused_for_size(&jpeg(8_001, 8_000, 1)), "just past it");
    assert!(!refused_for_size(&jpeg(8_000, 8_000, 1)), "a size this reads: its coefficients are only reserved, no scan reads them");
}

/// One SOS is ten bytes, and each walks every MCU of the frame even with no data: a 64 MiB body held six million of them.
#[test]
fn a_jpeg_of_endless_scans_is_refused() {
    assert!(oaiy_image::decode(&scans(jpeg(8, 8, 1), 100)).is_ok(), "more than a real file has, and it is read");
    let e = oaiy_image::decode(&scans(jpeg(8, 8, 1), 101)).unwrap_err().to_string();
    assert!(e.contains("scans"), "{e}");
}

fn png_header(width: u32, height: u32) -> Vec<u8> {
    let mut f = oaiy_image::png::SIGNATURE.to_vec();
    f.extend_from_slice(&13u32.to_be_bytes());
    f.extend_from_slice(b"IHDR");
    f.extend_from_slice(&width.to_be_bytes());
    f.extend_from_slice(&height.to_be_bytes());
    f.extend_from_slice(&[16, 6, 0, 0, 0]); // 16-bit RGBA
    f.extend_from_slice(&[0; 4]); // its CRC, which is not read
    f.extend_from_slice(&0u32.to_be_bytes());
    f.extend_from_slice(b"IEND");
    f.extend_from_slice(&[0; 4]);
    f
}

#[test]
fn a_png_past_the_limit_is_refused_on_its_header() {
    assert!(refused_for_size(&png_header(9_000, 9_000)));
    assert!(refused_for_size(&png_header(u32::MAX, u32::MAX)), "a product that wraps in 32 bits");
}
