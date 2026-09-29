"""Make the site's pictures: the screenshots on the landing page and the social card.

    python platform/ui/scripts/make-site-images.py          # writes into platform/ui/public
    python platform/ui/scripts/make-site-images.py --check  # exits 1 if the files there do not look like what this makes
    python platform/ui/scripts/make-site-images.py --verify-redactions  # exits 1 if a covered number is not covered

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
The same sources give the same pictures; `--check` compares how they look, not their bytes (another
Pillow or libwebp encodes to other bytes).
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

# The only phone numbers the site shows are the three the ACMA keeps for fiction that this project uses
# (0491 570 006, 0491 570 156, 0491 570 157). The demo screenshots also show others from the same reserved
# ranges, so a number that is not one of the three is covered in the site's copy of the picture, with a plain
# bar over just the number: (left, top, right, bottom) in the picture's own pixels. docs/images is not changed.
# `--verify-redactions` checks each box really differs from the source and that nothing else does.
REDACTIONS = {
    # The number in the second request ("Ella Morgan", asked by text): a phone number that is not one of the three above.
    "calendar": [(406, 306, 485, 322)],
}

ALLOWED_NUMBERS = ("0491 570 006", "0491 570 156", "0491 570 157")

# The numbers each of the site's pictures shows once redacted, read off the picture when it was chosen. A picture that
# is swapped for another has to be looked at again and this updated (tests/site-assets.mjs holds it to ALLOWED_NUMBERS).
NUMBERS_SHOWN = {
    "agent": [],
    "flows": [],
    "receptionist-call": ["0491 570 006"],
    "calendar": ["0491 570 006"],
}

# The marketing site's own palette (src/landing/marketing.css, Prism Lab).
BG = (14, 21, 23)
BORDER = (48, 65, 59)
TEXT = (239, 245, 240)
TEXT_SECONDARY = (172, 190, 182)
ACCENT = (105, 240, 183)


def source(name: str) -> Image.Image:
    return Image.open(IMAGES / SCREENSHOTS[name]).convert("RGB")


def redacted(name: str) -> Image.Image:
    """The screenshot with the numbers that may not be shown covered by a bar the colour of a darker page."""
    image = source(name)
    draw = ImageDraw.Draw(image)
    for left, top, right, bottom in REDACTIONS.get(name, []):
        page = image.getpixel((min(right, image.width - 1) - 40, max(top - 4, 0)))
        draw.rounded_rectangle((left, top, right, bottom), radius=3, fill=tuple(int(c * 0.74) for c in page))
    return image


def screenshot(name: str) -> bytes:
    image = redacted(name)
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


# --check compares what a picture looks like, not its bytes: another Pillow or libwebp encodes the same
# picture to different bytes, and that must not fail a check made on another machine.
TOLERANCE = 1.5  # mean difference, out of 255, per colour channel


def same_picture(existing: bytes, fresh: bytes) -> bool:
    from PIL import ImageChops, ImageStat

    a = Image.open(BytesIO(existing)).convert("RGBA")
    b = Image.open(BytesIO(fresh)).convert("RGBA")
    if a.size != b.size:
        return False
    return max(ImageStat.Stat(ImageChops.difference(a, b)).mean) <= TOLERANCE


def verify_redactions() -> int:
    """Each covered box differs from the source (the number is really gone), and the rest of the picture does not."""
    from PIL import ImageChops, ImageStat

    problems = []
    for name in SCREENSHOTS:
        shown = Image.open(PUBLIC / "images" / f"{name}.webp").convert("RGB")  # the file the site serves
        original = source(name)
        difference = ImageChops.difference(shown, original)
        boxes = REDACTIONS.get(name, [])
        for box in boxes:
            covered = ImageStat.Stat(difference.crop(box)).mean
            # A bar is a different colour from the page in every channel (WebP alone changes text by a few levels, and a
            # channel or two by up to about 12), and it is a flat colour where the number was writing.
            flat = max(ImageStat.Stat(shown.crop(box)).stddev)
            if min(covered) < 30 or flat > 10:
                problems.append(f"{name}: {box} is not covered (it differs by {min(covered):.1f}, and is {flat:.1f} from flat)")
            # A box has to be over writing in the source: if the screenshot changes and the number moves, this says so.
            ink = sum(original.crop(box).convert("L").histogram()[:150])
            if ink < 60:
                problems.append(f"{name}: {box} has only {ink} dark pixels of the source under it: is the number still there?")
        mask = Image.new("L", shown.size, 255)
        for box in boxes:
            ImageDraw.Draw(mask).rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill=0)
        elsewhere = ImageStat.Stat(difference, mask).mean
        if max(elsewhere) > 4:
            problems.append(f"{name}: the rest of the picture differs from its source by {max(elsewhere):.1f}")
    for problem in problems:
        print(problem)
    print("redactions ok" if not problems else "redactions NOT ok")
    return 1 if problems else 0


def main() -> int:
    if "--numbers" in sys.argv[1:]:
        import json

        print(json.dumps({"allowed": ALLOWED_NUMBERS, "shown": NUMBERS_SHOWN, "pictures": list(SCREENSHOTS)}))
        return 0
    if "--verify-redactions" in sys.argv[1:]:
        return verify_redactions()
    check = "--check" in sys.argv[1:]
    stale = []
    for target, data in make().items():
        if check:
            if not target.exists() or not same_picture(target.read_bytes(), data):
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
