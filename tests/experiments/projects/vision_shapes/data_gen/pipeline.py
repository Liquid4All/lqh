"""Vision shapes QA data generation pipeline (VLM e2e debug task).

Implements the multi-round VLM pattern from the ``vision_data_generation``
skill over a deterministic synthetic image folder:

1. **Understand** — a VLM inventories the image as JSON (objects, text,
   colors, scene).
2. **Ask** — a text-only LLM turns the inventory into one grounded question
   (counting, color naming, text reading, or description).
3. **Answer** — a VLM answers the question while looking at the image.

Output: user turn = [image, question], assistant turn = answer text.

The debug images (colored shapes + optional text, seeded PIL generation —
see ``tests/fixtures/debug_images.py`` for the canonical generator) are
created on first use into ``<project>/images/`` so nothing binary is
checked into the repo.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import lqh.sources as sources
from lqh.pipeline import (
    ChatMLMessage,
    Conversation,
    GenerationError,
    Pipeline,
    safe_content,
    step,
)

# ---------------------------------------------------------------------------
# Deterministic debug-image generation (mirrors tests/fixtures/debug_images.py,
# inlined because pipelines run standalone inside the project directory).
# ---------------------------------------------------------------------------

_SHAPES = ("circle", "square", "triangle")
_COLORS = {
    "red": (220, 50, 47),
    "blue": (38, 139, 210),
    "green": (60, 140, 60),
    "orange": (203, 105, 22),
    "purple": (108, 60, 196),
}
_TEXTS = ("CAT", "DOG", "LQH", "42", "SUN", "BOX")


def _ensure_debug_images(dest: Path, count: int = 16, seed: int = 7, size: int = 256) -> None:
    if dest.is_dir() and any(dest.glob("*.png")):
        return
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(count):
        shape = rng.choice(_SHAPES)
        color_name = rng.choice(sorted(_COLORS))
        n_shapes = rng.randint(1, 3)
        text = rng.choice(_TEXTS) if rng.random() < 0.5 else None

        img = Image.new("RGB", (size, size), (245, 245, 240))
        draw = ImageDraw.Draw(img)
        cell = size // 3
        pad = max(6, cell // 8)
        y0, y1 = size // 2 - cell // 2 + pad, size // 2 + cell // 2 - pad
        for col in sorted(rng.sample(range(3), n_shapes)):
            x0, x1 = col * cell + pad, (col + 1) * cell - pad
            if shape == "circle":
                draw.ellipse((x0, y0, x1, y1), fill=_COLORS[color_name])
            elif shape == "square":
                draw.rectangle((x0, y0, x1, y1), fill=_COLORS[color_name])
            else:
                draw.polygon([((x0 + x1) // 2, y0), (x0, y1), (x1, y1)], fill=_COLORS[color_name])
        if text is not None:
            draw.text((pad, pad), text, fill=(30, 30, 30))

        stem = f"{i:02d}_{n_shapes}_{color_name}_{shape}"
        if text is not None:
            stem += f"_{text}"
        img.save(dest / f"{stem}.png")
        records.append({"file": f"{stem}.png", "shape": shape, "color": color_name, "count": n_shapes, "text": text})
    (dest / "labels.json").write_text(json.dumps(records, indent=2))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class VisionShapesQA(Pipeline):
    @classmethod
    def source(cls, project_dir):
        images_dir = Path(project_dir) / "images"
        _ensure_debug_images(images_dir)
        return sources.image_folder(images_dir)

    async def generate(self, client, input: sources.ImageItem) -> Conversation:
        # Encode once; small synthetic images, so no downscaling happens.
        self.image_url = input.as_data_url()
        # Same seeded model for both VLM rounds of one sample; diverse
        # across samples.
        self.vlm = f"random:medium:{input.path.stem}"

        await self._understand(client)
        await self._ask(client)
        await self._answer(client)

        return [
            ChatMLMessage("user", [
                {"type": "image_url", "image_url": {"url": self.image_url}},
                {"type": "text", "text": self.question},
            ]),
            ChatMLMessage("assistant", self.answer),
        ]

    @step(retries=3)
    async def _understand(self, client):
        resp = await client.chat.completions.create(
            model=self.vlm,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self.image_url}},
                    {"type": "text", "text": (
                        "Inventory this image. Return a JSON object with keys: "
                        "shapes (list of strings), shape_count (integer), "
                        "colors (list of strings), visible_text (list of strings)."
                    )},
                ],
            }],
            response_format={"type": "json_object"},
        )
        data = json.loads(safe_content(resp).strip())
        if not isinstance(data.get("shapes"), list):
            raise GenerationError("understanding JSON missing 'shapes' list")
        self.understanding = data

    @step(retries=3)
    async def _ask(self, client):
        resp = await client.chat.completions.create(
            model="random:small",
            messages=[{
                "role": "user",
                "content": (
                    "Here is a machine-readable inventory of a simple image:\n"
                    f"{json.dumps(self.understanding)}\n\n"
                    "Write ONE short question about the image that can be answered "
                    "by looking at it: counting the shapes, naming the shape or "
                    "color, reading the visible text, or briefly describing the "
                    "image. It must be answerable from the inventory above. "
                    "Return only the question text."
                ),
            }],
        )
        self.question = safe_content(resp).strip().strip('"')
        if len(self.question) < 8:
            raise GenerationError("question too short")

    @step(retries=3)
    async def _answer(self, client):
        resp = await client.chat.completions.create(
            model=self.vlm,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self.image_url}},
                    {"type": "text", "text": (
                        f"{self.question}\n\n"
                        "Answer by looking at the image, in one or two short "
                        "sentences. Be direct and concrete; do not mention that "
                        "you are looking at an image."
                    )},
                ],
            }],
        )
        self.answer = safe_content(resp).strip()
        if not self.answer:
            raise GenerationError("empty answer")
