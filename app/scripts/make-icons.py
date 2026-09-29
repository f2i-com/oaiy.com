#!/usr/bin/env python3
"""
Make the Agent web app's icons from OAIY's own icon.

    python app/scripts/make-icons.py          (from anywhere; needs Pillow)

Reads   platform/desktop/src-tauri/icons/icon.png   (512 px, rounded tile, transparent corners)
        platform/desktop/public/favicon.svg         (the same mark as an SVG)
Writes  app/public/icon.svg                         the SVG favicon (a copy, line endings normalised)
        app/public/favicon-32.png                   32 px, for tabs that do not take an SVG
        app/public/icon-192.png                     192 px, purpose "any" (transparent corners, as the mark is)
        app/public/icon-512.png                     512 px, purpose "any"
        app/public/icon-maskable-512.png            512 px, purpose "maskable": no transparency, artwork in the safe zone
        app/public/apple-touch-icon.png             180 px, opaque (iOS fills transparency with black and rounds the corners itself)

The maskable and touch icons are the tile's own gradient carried out to the edges, with the mark at its
size in the tile. The mark has to sit inside the safe zone (a circle of 80 percent of the icon's width
around its centre), because a launcher may cut the icon down to a circle: the script measures it and
stops if it does not. The output is the same every run (check with git status).
"""
import math
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PNG = ROOT / "platform" / "desktop" / "src-tauri" / "icons" / "icon.png"
SOURCE_SVG = ROOT / "platform" / "desktop" / "public" / "favicon.svg"
OUT = ROOT / "app" / "public"

SIZE = 512
SAFE_RADIUS = 0.4 * SIZE  # the maskable safe zone: a circle 80 percent as wide as the icon


def tile_gradient(tile: Image.Image) -> Image.Image:
    """The tile's diagonal gradient carried out over the whole square, so no corner is transparent."""
    px = tile.load()
    # Two points well inside the rounded tile (its corner radius is 112 px), on the gradient's diagonal.
    a, b = 48, 464
    c0, c1 = px[a, a][:3], px[b, b][:3]
    t0, t1 = 2 * a / (2 * (SIZE - 1)), 2 * b / (2 * (SIZE - 1))
    out = Image.new("RGB", (SIZE, SIZE))
    op = out.load()
    for y in range(SIZE):
        for x in range(SIZE):
            k = ((x + y) / (2 * (SIZE - 1)) - t0) / (t1 - t0)
            op[x, y] = tuple(max(0, min(255, round(c0[i] + (c1[i] - c0[i]) * k))) for i in range(3))
    return out


def mark_radius(tile: Image.Image) -> float:
    """How far from the centre the tile's white artwork reaches (the nodes and the hub)."""
    px = tile.load()
    far = 0.0
    for y in range(SIZE):
        for x in range(SIZE):
            r, g, b, a = px[x, y]
            if a > 200 and r > 200 and g > 200 and b > 200:
                far = max(far, math.hypot(x + 0.5 - SIZE / 2, y + 0.5 - SIZE / 2))
    return far


def save(image: Image.Image, name: str) -> None:
    path = OUT / name
    image.save(path, format="PNG", optimize=True)
    print(f"{path.relative_to(ROOT)}  {image.width}x{image.height}  {image.mode}")


def main() -> int:
    tile = Image.open(SOURCE_PNG).convert("RGBA")
    if tile.size != (SIZE, SIZE):
        print(f"{SOURCE_PNG} is {tile.size}, expected {SIZE}x{SIZE}", file=sys.stderr)
        return 1
    reach = mark_radius(tile)
    if reach > SAFE_RADIUS:
        print(f"the mark reaches {reach:.0f} px from the centre, outside the safe zone ({SAFE_RADIUS:.0f} px)", file=sys.stderr)
        return 1
    print(f"the mark reaches {reach:.0f} px from the centre; the safe zone is {SAFE_RADIUS:.0f} px")

    OUT.mkdir(parents=True, exist_ok=True)

    # The SVG favicon: the same mark as the desktop's, with LF line endings whatever the checkout used.
    (OUT / "icon.svg").write_bytes(SOURCE_SVG.read_bytes().replace(b"\r\n", b"\n"))
    print(f"{(OUT / 'icon.svg').relative_to(ROOT)}  copied from {SOURCE_SVG.relative_to(ROOT)}")

    # Purpose "any": the tile as it is, transparent corners and all.
    save(tile.resize((32, 32), Image.LANCZOS), "favicon-32.png")
    save(tile.resize((192, 192), Image.LANCZOS), "icon-192.png")
    save(tile, "icon-512.png")

    # Purpose "maskable" and the touch icon: the gradient to the edges, the mark where it is in the tile.
    full = tile_gradient(tile).convert("RGBA")
    full.alpha_composite(tile)
    full = full.convert("RGB")
    save(full, "icon-maskable-512.png")
    save(full.resize((180, 180), Image.LANCZOS), "apple-touch-icon.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
