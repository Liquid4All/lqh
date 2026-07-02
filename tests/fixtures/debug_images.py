"""Deterministic PIL-generated debug images for VLM tests.

Produces a small "bring your own image folder" dataset of simple synthetic
images — colored shapes (circle / square / triangle), 1–3 per image, with an
optional short text label — plus a ``labels.json`` ground-truth file. Used by
the sources/scoring unit tests, the ``vision_shapes`` e2e project, and the
VLM training smoke test.

Everything is seeded, so repeat runs produce byte-identical images and the
ground truth can be asserted against VLM outputs (e.g. "how many circles?").
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

SHAPES = ("circle", "square", "triangle")

# Name → RGB. Kept far apart in hue so a VLM can't reasonably confuse them.
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 50, 47),
    "blue": (38, 139, 210),
    "green": (60, 140, 60),
    "orange": (203, 105, 22),
    "purple": (108, 60, 196),
}

TEXTS = ("CAT", "DOG", "LQH", "42", "SUN", "BOX")

BACKGROUND = (245, 245, 240)


def generate_debug_images(
    dest: Path,
    *,
    count: int = 16,
    seed: int = 7,
    size: int = 256,
) -> list[dict[str, Any]]:
    """Generate *count* labeled debug images under *dest*.

    Each image contains 1–3 same-shape, same-color shapes in a row and,
    for roughly half the images, a short dark text label in the top-left
    corner. Returns the ground-truth records and writes them to
    ``dest/labels.json``. Filenames encode the ground truth too:
    ``{index:02d}_{count}_{color}_{shape}[_{text}].png``.
    """
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for i in range(count):
        shape = rng.choice(SHAPES)
        color_name = rng.choice(sorted(COLORS))
        rgb = COLORS[color_name]
        n_shapes = rng.randint(1, 3)
        text = rng.choice(TEXTS) if rng.random() < 0.5 else None

        img = Image.new("RGB", (size, size), BACKGROUND)
        draw = ImageDraw.Draw(img)

        cell = size // 3
        pad = max(6, cell // 8)
        y0 = size // 2 - cell // 2 + pad
        y1 = size // 2 + cell // 2 - pad
        for col in sorted(rng.sample(range(3), n_shapes)):
            x0 = col * cell + pad
            x1 = (col + 1) * cell - pad
            if shape == "circle":
                draw.ellipse((x0, y0, x1, y1), fill=rgb)
            elif shape == "square":
                draw.rectangle((x0, y0, x1, y1), fill=rgb)
            else:  # triangle
                draw.polygon(
                    [((x0 + x1) // 2, y0), (x0, y1), (x1, y1)], fill=rgb
                )

        if text is not None:
            draw.text((pad, pad), text, fill=(30, 30, 30))

        stem = f"{i:02d}_{n_shapes}_{color_name}_{shape}"
        if text is not None:
            stem += f"_{text}"
        fname = f"{stem}.png"
        img.save(dest / fname)

        records.append(
            {
                "file": fname,
                "shape": shape,
                "color": color_name,
                "count": n_shapes,
                "text": text,
            }
        )

    (dest / "labels.json").write_text(json.dumps(records, indent=2))
    return records
