"""Make the site's pictures: the screenshots on the landing page and the social card.

    python platform/ui/scripts/make-site-images.py          # writes into platform/ui/public
    python platform/ui/scripts/make-site-images.py --check  # exits 1 if the files there are not what this makes

Sources are docs/images (the README's screenshots of a demo setup: the business, the people and their
numbers are made up, and the numbers are ones the ACMA keeps for fiction) and the desktop's
512 px icon. Made here:

  public/images/*.webp   the four screenshots the landing page shows, as WebP at their own size, each
                         under 200 KB. The size in pixels is what src/landing/screenshots.ts says, and
                         tests/site-assets.mjs checks the two agree.
  public/og-image.png    the social card (Open Graph and Twitter), 1200 x 630: the mark, the line the
                         page opens with, and the Agent, cropped by the card's edge.

The card's type is Inter (the site's own face), read from the npm package the pages use, so run
`npm ci` in platform/ui first. Needs Pillow (pip install Pillow), whose FreeType reads WOFF2.
The same sources give the same bytes.
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
UI = HERE.parent
REPO = UI.parent.parent
IMAGES = REPO / "docs" / "images"
PUBLIC = UI / "public"
ICON = REPO / "platform" / "desktop" / "src-tauri" / "icons" / "icon.png"
INTER = UI / "node_modules" / "@fontsource-variable" / "inter" / "files" / "inter-latin-wght-normal.woff2"

# The landing page's screenshots: the file they come from, and the name they are published under.
SCREENSHOTS = {
    "agent": "hero-agent.png",
    "flows": "flows.png",
    "receptionist-call": "agent-call.png",
    "calendar": "calendar.png",
}
WEBP_QUALITY = 80
MAX_BYTES = 200 * 1024

# The marketing site's own palette (src/landing/marketing.css, Prism Lab).
BG = (14, 21, 23)
BORDER = (48, 65, 59)
TEXT = (239, 245, 240)
TEXT_SECONDARY = (172, 190, 182)
ACCENT = (105, 240, 183)


def screenshot(name: str) -> bytes:
    image = Image.open(IMAGES / SCREENSHOTS[name]).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, "WEBP", quality=WEBP_QUALITY, method=6)
    data = buffer.getvalue()
    if len(data) > MAX_BYTES:
        raise SystemExit(f"{name}.webp is {len(data)} bytes: lower WEBP_QUALITY or the size")
    return data


def inter(size: int, weight: int) -> ImageFont.FreeTypeFont:
    if not INTER.exists():
        raise SystemExit(f"{INTER} is missing: run `npm ci` in platform/ui first")
    font = ImageFont.truetype(str(INTER), size)
    font.set_variation_by_axes([weight])
    return font


def spaced(draw: ImageDraw.ImageDraw, at: tuple[float, float], text: str, font: ImageFont.FreeTypeFont, fill, tracking: float) -> float:
    """Draw `text` letter by letter with `tracking` px added after each; returns the x it ended at."""
    x, y = at
    for char in text:
        draw.text((x, y), char, font=font, fill=fill)
        x += font.getlength(char) + tracking
    return x


def wrap(text: str, font: ImageFont.FreeTypeFont, width: float) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if font.getlength(trial) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def card() -> bytes:
    scale = 2  # drawn at twice the size and brought down, so edges are smooth
    w, h = 1200 * scale, 630 * scale
    canvas = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(canvas)

    # A faint light from the top right, as the page has.
    glow = Image.new("RGB", (w, h), BG)
    gdraw = ImageDraw.Draw(glow)
    for i in range(60, 0, -1):
        t = i / 60
        radius = int(900 * scale * t)
        colour = tuple(int(BG[c] + (ACCENT[c] - BG[c]) * 0.055 * (1 - t) ** 2) for c in range(3))
        gdraw.ellipse((w - radius, -radius // 2, w + radius, radius // 2 + radius), fill=colour)
    canvas.paste(glow)
    draw = ImageDraw.Draw(canvas)

    # The mark and the wordmark.
    mark = Image.open(ICON).convert("RGBA").resize((88 * scale, 88 * scale), Image.LANCZOS)
    canvas.paste(mark, (64 * scale, 60 * scale), mark)
    spaced(draw, (172 * scale, 78 * scale), "OAIY", inter(44 * scale, 650), TEXT, 0.16 * 44 * scale)

    # The line the page opens with.
    big = inter(58 * scale, 640)
    draw.text((64 * scale, 200 * scale), "Less prompting.", font=big, fill=TEXT)
    draw.text((64 * scale, 200 * scale + 68 * scale), "More possibilities.", font=big, fill=ACCENT)

    # What it is.
    body = inter(26 * scale, 420)
    y = 372 * scale
    for line in wrap("An AI agent and flow builder for your browser and your computer.", body, 470 * scale):
        draw.text((64 * scale, y), line, font=body, fill=TEXT_SECONDARY)
        y += 38 * scale
    draw.text((64 * scale, 556 * scale), "oaiy.com", font=inter(24 * scale, 560), fill=ACCENT)

    # The Agent, cropped by the card's right and bottom edges.
    shot = Image.open(IMAGES / SCREENSHOTS["agent"]).convert("RGB")
    shot = shot.resize((880 * scale, 550 * scale), Image.LANCZOS)
    left, top = 632 * scale, 112 * scale
    pad = 2 * scale
    frame = Image.new("RGB", (shot.width + 2 * pad, shot.height + 2 * pad), BORDER)
    frame.paste(shot, (pad, pad))
    mask = Image.new("L", frame.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, frame.width * 2, frame.height * 2), radius=14 * scale, fill=255)
    canvas.paste(frame, (left - pad, top - pad), mask)

    out = canvas.resize((1200, 630), Image.LANCZOS)
    buffer = BytesIO()
    out.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def make() -> dict[Path, bytes]:
    made = {PUBLIC / "images" / f"{name}.webp": screenshot(name) for name in SCREENSHOTS}
    made[PUBLIC / "og-image.png"] = card()
    return made


def main() -> int:
    check = "--check" in sys.argv[1:]
    stale = []
    for target, data in make().items():
        if check:
            if not target.exists() or target.read_bytes() != data:
                stale.append(target.name)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            print(f"{target.relative_to(UI)}: {len(data)} bytes")
    if check and stale:
        print("out of date: " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
