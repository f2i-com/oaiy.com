"""Golden files for DeepSeek-V4.1 vision on nrob: what Pillow and the reference produce, for the
Rust image decoders, preprocessing and vision tower to match.

  decode/     test images (PNG in every colour type / bit depth / interlace / filter / deflate block
              kind, JPEG baseline and progressive in every subsampling Pillow writes, restart markers,
              optimized tables, odd sizes) + manifest.json; `<name>.rgb` is Pillow's
              `Image.open(f).convert("RGB")`, the reference's first step
  plan.json   plan_image_grid over a sweep of sizes: [w, h, n_llm_h, n_llm_w, best_h, best_w]
  preprocess/ for a few images: the padded RGB the reference normalizes (`<name>.padded`) and the
              bf16 patches it feeds the ViT (`<name>.patches`, u16 LE) + manifest.json
  features.safetensors
              the reference ViT + aligner on the GPU (bf16) for a few images: patches, the
              patch-embed output, block 0's output, the ViT output and the aligner output

Usage: PYTHONUTF8=1 python vision_golden.py [--out E:\\deepseek\\golden\\vision] [--only decode,plan,...]
"""

import argparse
import io
import json
import math
import os
import struct
import sys
import types
import zlib

import numpy as np
from PIL import Image

REF = os.environ.get("DSV41_REF", r"E:\deepseek\reference")
MODEL = os.environ.get("DSV41_MODEL", r"E:\deepseek\model")
sys.path.insert(0, os.path.join(REF, "inference"))


# ---------------------------------------------------------------- PNG writer (for what Pillow can't)

ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def pack_row(samples: np.ndarray, depth: int) -> bytes:
    """One scanline of samples (flattened channels) at this bit depth, MSB first."""
    s = samples.astype(np.int64).ravel()
    if depth == 16:
        return b"".join(struct.pack(">H", int(v)) for v in s)
    if depth == 8:
        return bytes(int(v) for v in s)
    per = 8 // depth
    out = bytearray()
    for i in range(0, len(s), per):
        b = 0
        for j in range(per):
            v = int(s[i + j]) if i + j < len(s) else 0
            b |= v << (8 - depth * (j + 1))
        out.append(b)
    return bytes(out)


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def filter_rows(rows, bpp, pick):
    """Filter each scanline with the type `pick(i)` chooses; returns the filtered stream."""
    out = bytearray()
    prev = bytes(len(rows[0])) if rows else b""
    for i, row in enumerate(rows):
        f = pick(i)
        out.append(f)
        for x in range(len(row)):
            a = row[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            pred = [0, a, b, (a + b) // 2, paeth(a, b, c)][f]
            out.append((row[x] - pred) & 0xFF)
        prev = row
    return bytes(out)


def write_png(path, px, color_type, depth, interlace=False, pick=lambda i: i % 5, level=6, strategy=0,
              palette=None, trns=None, idat_split=None, wbits=15):
    h, w = px.shape[:2]
    ch = CHANNELS[color_type]
    bpp = max(1, ch * depth // 8)
    raw = bytearray()
    if interlace:
        for x0, y0, dx, dy in ADAM7:
            sub = px[y0::dy, x0::dx]
            if sub.shape[0] == 0 or sub.shape[1] == 0:
                continue
            raw += filter_rows([pack_row(r, depth) for r in sub], bpp, pick)
    else:
        raw += filter_rows([pack_row(r, depth) for r in px], bpp, pick)
    c = zlib.compressobj(level, zlib.DEFLATED, wbits, 9, strategy)
    data = c.compress(bytes(raw)) + c.flush()
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color_type, 0, 0, int(interlace)))
    if palette is not None:
        out += chunk(b"PLTE", bytes(int(v) for v in np.asarray(palette).ravel()))
    if trns is not None:
        out += chunk(b"tRNS", trns)
    parts = [data] if not idat_split else [data[i : i + idat_split] for i in range(0, len(data), idat_split)]
    for p in parts:
        out += chunk(b"IDAT", p)
    out += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(out)


def pattern(h, w, c, maxv, seed):
    """Smooth bands, edges and noise: exercises every filter and coefficient."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    chans = []
    for k in range(c):
        f = (np.sin(x / (2.5 + 1.7 * k) + y / (4.0 + k)) + 1) / 2 * 0.45
        f += (x + y * (k + 1)) / (w + h * (k + 1) + 1) * 0.35
        f += ((x // 7 + y // 5 + k) % 2) * 0.1
        f += rng.random((h, w)) * 0.1
        chans.append(f)
    v = np.clip(np.stack(chans, -1), 0, 0.999999)
    return np.floor(v * (maxv + 1)).astype(np.int64)


def make_pngs(d):
    cases = []

    def add(name, px, ct, depth, **kw):
        write_png(os.path.join(d, name + ".png"), px, ct, depth, **kw)
        cases.append(name + ".png")

    sizes = [(37, 23), (1, 1), (5, 3), (16, 16), (129, 7)]
    for (w, h) in sizes:
        for depth in (1, 2, 4, 8, 16):
            add(f"g{depth}_{w}x{h}", pattern(h, w, 1, (1 << depth) - 1, depth + w), 0, depth)
    for depth in (8, 16):
        add(f"rgb{depth}", pattern(29, 41, 3, (1 << depth) - 1, 3 + depth), 2, depth)
        add(f"ga{depth}", pattern(19, 33, 2, (1 << depth) - 1, 4 + depth), 4, depth)
        add(f"rgba{depth}", pattern(21, 31, 4, (1 << depth) - 1, 5 + depth), 6, depth)
    # 16-bit grey in the range Pillow does not clip, and at the edge of it
    lo = pattern(17, 23, 1, 300, 77)
    add("g16_low", lo, 0, 16)
    for depth in (1, 2, 4, 8):
        n = min(1 << depth, 200)
        rng = np.random.default_rng(depth)
        pal = rng.integers(0, 256, (n, 3))
        idx = pattern(27, 35, 1, n - 1, 11 + depth)
        add(f"p{depth}", idx, 3, depth, palette=pal)
        alpha = bytes(int(a) for a in rng.integers(0, 256, max(1, n // 2)))
        add(f"p{depth}_trns", idx, 3, depth, palette=pal, trns=alpha)
    add("g8_trns", pattern(13, 17, 1, 255, 21), 0, 8, trns=struct.pack(">H", 100))
    add("rgb8_trns", pattern(13, 17, 3, 255, 22), 2, 8, trns=struct.pack(">HHH", 10, 20, 30))
    # Adam7, including sizes where some passes are empty
    for (w, h) in [(1, 1), (2, 3), (5, 5), (9, 7), (33, 20)]:
        add(f"i_rgb8_{w}x{h}", pattern(h, w, 3, 255, 31 + w), 2, 8, interlace=True)
        add(f"i_g2_{w}x{h}", pattern(h, w, 1, 3, 32 + w), 0, 2, interlace=True)
    add("i_rgba16", pattern(23, 29, 4, 65535, 41), 6, 16, interlace=True)
    add("i_p4", pattern(19, 21, 1, 15, 42), 3, 4, interlace=True,
        palette=np.random.default_rng(9).integers(0, 256, (16, 3)))
    # deflate: stored, fixed Huffman, Huffman-only, RLE, small windows, split IDATs
    px = pattern(47, 53, 3, 255, 51)
    add("z_stored", px, 2, 8, level=0)
    add("z_fixed", px, 2, 8, strategy=zlib.Z_FIXED)
    add("z_huffonly", px, 2, 8, strategy=zlib.Z_HUFFMAN_ONLY)
    add("z_rle", px, 2, 8, strategy=zlib.Z_RLE)
    add("z_win9", px, 2, 8, wbits=9, level=9)
    add("z_split", px, 2, 8, idat_split=7)
    add("z_big_stored", pattern(300, 300, 3, 255, 52), 2, 8, level=0)  # > 64 KiB: several stored blocks
    for f in range(5):
        add(f"f{f}", pattern(15, 26, 3, 255, 60 + f), 2, 8, pick=lambda i, f=f: f)
    # what Pillow itself writes (and a flat image deflate turns into long matches)
    rng = np.random.default_rng(7)
    shot = np.full((120, 200, 3), 245, np.uint8)
    shot[10:30, 10:190] = (40, 90, 200)
    shot[40:110, 20:100] = rng.integers(0, 256, (70, 80, 3))
    Image.fromarray(shot).save(os.path.join(d, "pil_rgb.png"), optimize=True)
    cases.append("pil_rgb.png")
    Image.fromarray(np.dstack([shot, np.full((120, 200), 128, np.uint8)])).save(os.path.join(d, "pil_rgba.png"))
    cases.append("pil_rgba.png")
    Image.fromarray(shot).convert("P", palette=Image.Palette.ADAPTIVE, colors=37).save(os.path.join(d, "pil_p.png"))
    cases.append("pil_p.png")
    return cases


def make_jpegs(d):
    cases = []

    def add(name, px, **kw):
        mode = "L" if px.ndim == 2 else "RGB"
        Image.fromarray(px.astype(np.uint8), mode).save(os.path.join(d, name + ".jpg"), **kw)
        cases.append(name + ".jpg")

    def rgb(h, w, seed):
        return pattern(h, w, 3, 255, seed)

    for sub, tag in [(0, "444"), (1, "422"), (2, "420")]:
        for (w, h) in [(101, 67), (1, 1), (7, 3), (17, 9), (16, 16), (33, 32)]:
            add(f"b{tag}_{w}x{h}", rgb(h, w, w * 3 + sub), quality=85, subsampling=sub)
        add(f"p{tag}", rgb(77, 123, 90 + sub), quality=85, subsampling=sub, progressive=True)
        add(f"o{tag}", rgb(45, 71, 95 + sub), quality=80, subsampling=sub, optimize=True)
        add(f"rst{tag}", rgb(61, 83, 97 + sub), quality=75, subsampling=sub, restart_marker_blocks=3)
        add(f"prst{tag}", rgb(61, 83, 98 + sub), quality=75, subsampling=sub, progressive=True,
            restart_marker_blocks=5)
    for (w, h) in [(55, 33), (1, 1), (9, 17)]:
        add(f"gray_{w}x{h}", pattern(h, w, 1, 255, w)[..., 0], quality=80)
    add("pgray", pattern(49, 57, 1, 255, 5)[..., 0], quality=80, progressive=True)
    add("rstgray", pattern(49, 57, 1, 255, 6)[..., 0], quality=80, restart_marker_rows=1)
    add("q100", rgb(40, 50, 7), quality=100, subsampling=0)
    add("q5", rgb(40, 50, 8), quality=5, subsampling=2)
    add("q100p", rgb(40, 50, 9), quality=100, subsampling=2, progressive=True)
    add("big420", rgb(480, 640, 10), quality=85, subsampling=2)
    for tag, kw in [("cmyk", {}), ("pcmyk", {"progressive": True})]:
        px = pattern(37, 45, 4, 255, 12).astype(np.uint8)
        Image.fromarray(px, "CMYK").save(os.path.join(d, tag + ".jpg"), quality=85, **kw)
        cases.append(tag + ".jpg")
    mascot = np.asarray(Image.open(os.path.join(MODEL, "mascot.png")).convert("RGB"))
    add("mascot", mascot, quality=90)
    return cases


def gen_decode(out):
    d = os.path.join(out, "decode")
    os.makedirs(d, exist_ok=True)
    manifest = []
    for name in make_pngs(d) + make_jpegs(d):
        with Image.open(os.path.join(d, name)) as im:
            mode = im.mode
            rgb = im.convert("RGB")
        with open(os.path.join(d, name + ".rgb"), "wb") as f:
            f.write(rgb.tobytes())
        manifest.append({"file": name, "width": rgb.width, "height": rgb.height, "pil_mode": mode})
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"decode: {len(manifest)} images")


# ---------------------------------------------------------------- preprocessing

def vision_args():
    """The reference's ModelArgs fields for vision, from its inference/config.json."""
    cfg = json.load(open(os.path.join(REF, "inference", "config.json")))
    return types.SimpleNamespace(**cfg)


def gen_plan(out):
    from image_processor import plan_image_grid

    args = vision_args()
    rng = np.random.default_rng(0)
    sizes = {(w, h) for w in (1, 2, 13, 14, 15, 41, 42, 43, 544, 545, 546, 547) for h in (1, 14, 42, 544, 546, 1000)}
    sizes |= {(int(math.exp(a)), int(math.exp(b))) for a, b in rng.uniform(0, math.log(9000), (4000, 2))}
    sizes |= {(w, h) for w, h in rng.integers(1, 3000, (2000, 2))}
    sizes |= {(1, 20000), (20000, 1), (3, 5000), (5000, 3), (100, 3000), (3000, 100)}
    rows = []
    for w, h in sorted(sizes):
        w, h = int(w), int(h)
        rows.append([w, h, *plan_image_grid(w, h, args)])
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump(rows, f)
    print(f"plan: {len(rows)} sizes")


def preprocess_images(out):
    """The images preprocess/ and features cover: name -> encoded bytes."""
    d = os.path.join(out, "decode")
    imgs = {}
    with open(os.path.join(MODEL, "mascot.png"), "rb") as f:
        imgs["mascot"] = f.read()  # 256x256 RGBA: upscaled to the minimum size, alpha dropped

    def enc(px, fmt, **kw):
        b = io.BytesIO()
        Image.fromarray(px.astype(np.uint8)).save(b, fmt, **kw)
        return b.getvalue()

    imgs["wide"] = enc(pattern(400, 1500, 3, 255, 101), "JPEG", quality=90)  # padded top/bottom
    imgs["tall"] = enc(pattern(3000, 200, 3, 255, 102), "PNG")  # token cap, very tall
    imgs["large"] = enc(pattern(2000, 3000, 3, 255, 103), "JPEG", quality=85)  # downscale (antialias)
    imgs["tiny"] = enc(pattern(30, 20, 3, 255, 104), "PNG")  # big upscale
    imgs["odd"] = enc(pattern(701, 997, 3, 255, 105), "JPEG", quality=85, subsampling=1)
    imgs["gray"] = enc(pattern(600, 800, 1, 255, 106)[..., 0], "PNG")
    imgs["square"] = enc(pattern(640, 640, 3, 255, 107), "JPEG", quality=85)  # scaled down, no pad
    with open(os.path.join(d, "big420.jpg"), "rb") as f:
        imgs["photo"] = f.read()
    return imgs


def gen_preprocess(out):
    import torch
    from image_processor import load_image, plan_image_grid

    args = vision_args()
    d = os.path.join(out, "preprocess")
    os.makedirs(d, exist_ok=True)
    manifest = []
    for name, data in preprocess_images(out).items():
        fmt = "png" if data[:4] == b"\x89PNG" else "jpg"
        with open(os.path.join(d, f"{name}.{fmt}"), "wb") as f:
            f.write(data)
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image({"data": data}, args)
        with open(os.path.join(d, name + ".patches"), "wb") as f:
            f.write(patches.contiguous().view(torch.int16).numpy().astype("<i2").tobytes())
        # the image right before normalization, rebuilt the way load_image builds it
        from PIL import ImageOps

        with Image.open(io.BytesIO(data)) as src:
            image = src.convert("RGB")
        _, _, bh, bw = plan_image_grid(image.width, image.height, args)
        padded = ImageOps.pad(image, (bw, bh), color=(127, 127, 127))
        with open(os.path.join(d, name + ".padded"), "wb") as f:
            f.write(padded.tobytes())
        manifest.append({"name": name, "file": f"{name}.{fmt}", "width": image.width, "height": image.height,
                         "best_width": bw, "best_height": bh, "n_vit_h": n_vit_h, "n_vit_w": n_vit_w,
                         "n_llm_h": n_llm_h, "n_llm_w": n_llm_w})
        print(f"preprocess {name}: {image.width}x{image.height} -> {bw}x{bh}, llm {n_llm_h}x{n_llm_w}")
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


# ---------------------------------------------------------------- vision tower

def gen_features(out, names):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from image_processor import load_image
    from vision import Aligner, ViT

    args = vision_args()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    vit, aligner = ViT(args), Aligner(args)
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    state = {}
    for key, shard in idx.items():
        if key.startswith(("vision.", "aligner.")):
            with safe_open(os.path.join(MODEL, shard), "pt", device="cpu") as f:
                state[key] = f.get_tensor(key)
    vit.load_state_dict({k[len("vision."):]: v for k, v in state.items() if k.startswith("vision.")})
    aligner.load_state_dict({k[len("aligner."):]: v for k, v in state.items() if k.startswith("aligner.")})
    imgs = preprocess_images(out)
    tensors = {}
    with torch.inference_mode():
        for name in names:
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image({"data": imgs[name]}, args)
            patches = patches.cuda()
            x = vit.patch_embed(patches)
            tensors[f"{name}.patch_embed"] = x
            from vision import get_vision_cos_sin

            cos, sin = get_vision_cos_sin(n_vit_h, n_vit_w, vit.rope_dim, vit.rope_theta)
            b0 = vit.blocks[0](x, cos, sin)
            tensors[f"{name}.block0"] = b0
            if name == names[0]:
                # every block's output, for locating drift
                h = x
                for i, blk in enumerate(vit.blocks):
                    h = blk(h, cos, sin)
                    tensors[f"{name}.block{i:02d}.out"] = h
            feats = vit(patches, n_vit_h, n_vit_w)
            tensors[f"{name}.vit"] = feats
            tensors[f"{name}.aligner"] = aligner(feats, n_vit_h, n_vit_w)
            tensors[f"{name}.patches"] = patches
            tensors[f"{name}.dims"] = torch.tensor([n_vit_h, n_vit_w, n_llm_h, n_llm_w], dtype=torch.int64)
            print(f"features {name}: {n_vit_h}x{n_vit_w} patches -> {tuple(tensors[name + '.aligner'].shape)}")
    save_file({k: v.detach().contiguous().cpu() for k, v in tensors.items()}, os.path.join(out, "features.safetensors"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"E:\deepseek\golden\vision")
    ap.add_argument("--only", default="decode,plan,preprocess,features")
    ap.add_argument("--features", default="mascot,wide,odd")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only.split(","))
    if "decode" in only:
        gen_decode(a.out)
    if "plan" in only:
        gen_plan(a.out)
    if "preprocess" in only:
        gen_preprocess(a.out)
    if "features" in only:
        gen_features(a.out, a.features.split(","))


if __name__ == "__main__":
    main()
