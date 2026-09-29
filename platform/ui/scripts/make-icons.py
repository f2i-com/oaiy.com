"""Make the flow editor's app icons from the OAIY icon.

    python platform/ui/scripts/make-icons.py          # writes into platform/ui/public
    python platform/ui/scripts/make-icons.py --check  # exits 1 if the files there are not what this makes

The source is the desktop app's 512 px icon (platform/desktop/src-tauri/icons/icon.png): a
violet tile with rounded corners (transparent outside them) and the mark, the flows joining
in a hub, well inside it. From it:

  icon-192.png, icon-512.png   the tile as it is, for "purpose: any" (the corners stay transparent)
  icon-maskable-512.png        "purpose: maskable": a full-bleed square, because the system cuts its
                               own shape (a circle, a squircle) out of it. The tile is cropped in to
                               where its edges are straight, so nothing transparent is left, and the
                               mark stays inside the 80 % circle every mask keeps (checked below).
  apple-touch-icon.png         180 px, the same full-bleed square. iOS rounds it itself and fills
                               anything transparent with black, so it is opaque. (The file this
                               replaces was transparent but for a sliver at one edge.)

Needs Pillow (pip install Pillow). The output does not depend on the machine: the same source
gives the same bytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / "desktop" / "src-tauri" / "icons" / "icon.png"
OUT = HERE.parent / "public"

# How far the source is scaled up before its middle is kept, so the rounded corners fall outside
# the kept square. The corner radius is 112 px of 512; the kept square's corner must be inside the
# rounded shape: 256 * (1 - 1/s) >= 112 * (1 - 1/sqrt(2)), which needs s >= 1.147.
FULL_BLEED_SCALE = 1.2

# A mask keeps the circle across the middle 80 % of the icon. The mark must fit inside it.
SAFE_RADIUS = 0.4


def full_bleed(source: Image.Image, size: int) -> Image.Image:
    """The tile with its rounded corners cropped away, as a square of `size` px."""
    side = source.width
    kept = side / FULL_BLEED_SCALE
    left = (side - kept) / 2
    box = (left, left, left + kept, left + kept)
    square = source.resize((size, size), Image.LANCZOS, box=box)
    alpha = square.getchannel("A")
    if alpha.getextrema() != (255, 255):
        raise SystemExit("the cropped icon still has transparent pixels: raise FULL_BLEED_SCALE")
    return square


def mark_stays_in_safe_zone(square: Image.Image) -> bool:
    """Whether every pixel that is not the tile's gradient (the white mark) is inside the safe circle."""
    size = square.width
    pixels = square.convert("RGB").load()
    cx = cy = (size - 1) / 2
    limit = (SAFE_RADIUS * size) ** 2
    for y in range(size):
        for x in range(size):
            r, g, b = pixels[x, y]
            # The tile runs from indigo to violet: red stays below 130. White and its shades are the mark.
            if r > 150 and g > 150 and (x - cx) ** 2 + (y - cy) ** 2 > limit:
                return False
    return True


def make() -> dict[str, Image.Image]:
    source = Image.open(SOURCE).convert("RGBA")
    if source.size != (512, 512):
        raise SystemExit(f"{SOURCE} is {source.size}, expected 512x512")
    maskable = full_bleed(source, 512)
    if not mark_stays_in_safe_zone(maskable):
        raise SystemExit("the mark reaches outside the maskable safe zone: lower FULL_BLEED_SCALE")
    return {
        "icon-192.png": source.resize((192, 192), Image.LANCZOS),
        "icon-512.png": source,
        "icon-maskable-512.png": maskable,
        "apple-touch-icon.png": full_bleed(source, 180).convert("RGB"),
    }


def encode(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def main() -> int:
    check = "--check" in sys.argv[1:]
    stale = []
    for name, image in make().items():
        data = encode(image)
        target = OUT / name
        if check:
            if not target.exists() or target.read_bytes() != data:
                stale.append(name)
        else:
            target.write_bytes(data)
            print(f"{name}: {image.width}x{image.height}, {len(data)} bytes")
    if check and stale:
        print("out of date: " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
