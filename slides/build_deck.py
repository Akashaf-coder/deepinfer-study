#!/usr/bin/env python3
"""Build a self-contained DeepInfer slide deck.

Embeds the repo's Pareto/architecture charts as base64 JPEG so the HTML works
offline and can be shared as a single file.

Usage:
    python3 slides/build_deck.py
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path.home() / "DeepInfer"
ASSETS = HERE / "assets"
OUT = HERE / "DeepInfer-深度解析.html"

IMAGES = {
    "__IMG_ARCH__": ("doc/images/architecture_overview.png", 1600, 84),
    "__IMG_PREFILL__": ("doc/images/prefill_pareto.png", 1400, 88),
    "__IMG_DECODE__": ("doc/images/decode_pareto.png", 1400, 88),
    "__IMG_WIDE__": ("doc/images/wide_ep_decode_pareto.png", 1400, 88),
}


def to_jpeg(src: Path, dst: Path, max_w: int, quality: int) -> None:
    subprocess.run(
        [
            "sips", "-Z", str(max_w),
            "-s", "format", "jpeg",
            "-s", "formatOptions", str(quality),
            str(src), "--out", str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def main() -> int:
    template = (HERE / "deck_template.html").read_text(encoding="utf-8")
    ASSETS.mkdir(exist_ok=True)

    replacements = {}
    for token, (rel, width, quality) in IMAGES.items():
        src = REPO / rel
        if not src.is_file():
            print(f"missing image: {src}", file=sys.stderr)
            return 1
        dst = ASSETS / (Path(rel).stem + ".jpg")
        to_jpeg(src, dst, width, quality)
        replacements[token] = data_uri(dst)
        print(f"embedded {rel} -> {dst.name} ({dst.stat().st_size/1024:.0f} KB)")

    html = template
    for token, uri in replacements.items():
        if token not in html:
            print(f"warning: placeholder {token} not in template", file=sys.stderr)
        html = html.replace(token, uri)

    leftovers = [t for t in IMAGES if t in html]
    if leftovers:
        print(f"unreplaced placeholders: {leftovers}", file=sys.stderr)
        return 1

    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024/1024:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
