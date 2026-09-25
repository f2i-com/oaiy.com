#!/usr/bin/env python3
"""Manual pixel-grounding checks against a running Nrob daemon (requires Pillow).

Answers appear only in generated pixels, never in request text or filenames.
The JSON report also includes blank/no-image controls; inspect those for guesses.
No daemon credentials are written into the report.
"""
import argparse
import base64
import io
import json
import pathlib
import random
import sys
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont


def main():
    # Windows redirected consoles may use a legacy codepage. Model replies can
    # contain arbitrary Unicode; keep the complete answer in the UTF-8 report.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=pathlib.Path, required=True,
                        help="project's Nrob daemon state JSON")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--font", required=True, help="path to a TrueType font")
    parser.add_argument("--seed", type=int, default=9252042)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    font = ImageFont.truetype(args.font, 76)
    records = []
    cases = []
    ocr = "What text is printed in this image?"
    colour = "What colour is the circle? What colour is the square?"

    def ask(label, question, pixels=None):
        content = [{"type": "text", "text": question}]
        if pixels is not None:
            buf = io.BytesIO()
            pixels.save(buf, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            content.insert(0, {"type": "image_url", "image_url": {"url": url}})
        body = dict(model="orcasaq-2-27b", temperature=0, max_tokens=96,
                    reasoning_effort="none", stream=False, messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": content}])
        request = urllib.request.Request(
            "http://" + state["addr"] + "/v1/chat/completions",
            data=json.dumps(body).encode(), headers={
                "Authorization": "Bearer " + state["key"],
                "Content-Type": "application/json"})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        record = dict(label=label, seconds=time.perf_counter()-started,
                      answer=result["choices"][0]["message"]["content"],
                      usage=result.get("usage"))
        records.append(record)
        print(f'{label}: {record["seconds"]:.3f}s {record["answer"]}', flush=True)
        return record

    for i, (circle, square) in enumerate([
            ("red", "blue"), ("green", "orange"),
            ("blue", "red"), ("orange", "green")]):
        code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3))
        code += " " + str(rng.randint(100, 999))
        pixels = Image.new("RGB", (768, 768), "white")
        draw = ImageDraw.Draw(pixels)
        draw.ellipse((80, 140, 310, 370), fill=circle)
        draw.rectangle((460, 140, 690, 370), fill=square)
        draw.text((100, 480), code, fill="black", font=font)
        pixels.save(args.output / f"card-{i}.png")
        cases.append((pixels, code))
        record = ask(f"ocr-{i}", ocr, pixels)
        record.update(expected=code, passed=code in record["answer"])
        record = ask(f"colours-{i}", colour, pixels)
        # Keyword presence is a smoke check. The report retains the answer so
        # a reviewer can verify which colour was assigned to which shape.
        answer = record["answer"].lower()
        record.update(expected=dict(circle=circle, square=square),
                      passed=circle in answer and square in answer)
    record = ask("return-first", ocr, cases[0][0])
    record.update(expected=cases[0][1], passed=cases[0][1] in record["answer"])
    ask("blank-control", ocr, Image.new("RGB", (768, 768), "white"))
    ask("no-image-control", ocr)
    report = dict(seed=args.seed, checks=records,
                  note="Manually review shape-colour assignments and blank/no-image controls.")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if any(record.get("passed") is False for record in records):
        raise SystemExit("Some image checks failed; see report.json")


if __name__ == "__main__":
    main()
