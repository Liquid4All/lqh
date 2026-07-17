# Skill: Vision Data Generation (VLM)

Turn a user-provided **image folder** into an image-question-answer dataset
suitable for fine-tuning the LFM2.5-VL vision models.

**Currently the user must supply the images themselves** — typically a local
folder of raw images. They can be **unlabelled** (no captions, no annotations);
the data pipeline's job is exactly to take those raw images and synthesize
structured training data from them: conversational (image-question-answer) turns
or input-output pairs. If the user has no images, this skill can't proceed —
ask them to point at a folder. (Future: we may add pulling images from Liquid's
own image sources so the user doesn't have to bring their own — not available
today.)

This skill **extends the `data_generation` skill** — load that one too if you
have not already. All of its rules apply unchanged: the pipeline interface
(`lqh.pipeline`), one `Pipeline` subclass per file, `@step` retries,
`safe_content`, the draft → judge → validate → filter workflow, and the
default-on filter gate before training. This file covers only what is
*different* for vision.

Cloud execution (`execution="cloud"`, see the data_generation skill) works
for vision pipelines too: the image folder read via `lqh.sources.image_folder`
during the validated local run is uploaded with the job (large folders go via
a direct-to-storage upload; the hard ceiling is 2 GB — downscale first if the
raw folder exceeds it). VLM generation is exactly the kind of long, heavy run
that belongs in the cloud.

## How vision requests work

Images ride inside normal chat messages as OpenAI multi-part content:

```python
{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": item.as_data_url()}},
        {"type": "text", "text": "Describe what you see in this image."},
    ],
}
```

- `as_data_url()` returns a `data:image/...;base64,...` string.
- The backend detects the `image_url` part and routes the request to the
  **vision pool** automatically. You keep using the same model strings
  (`random:small|medium|large`, `small`, `medium`, `large`,
  `random:<size>:<seed>`) — no vision-specific model names exist.
- Judges do the same: `judge:small|medium|large` with an image in the prompt
  routes to a vision-capable judge. Scoring pipelines need no change.
- Vision pools are smaller than text pools, so `random:<size>` gives less
  model diversity for image steps. Get diversity from prompts and
  `liquidrandom` instead.

## Image loading and preprocessing

Always load images with `lqh.sources.image_folder` (path-safe, tested,
deterministic ordering):

```python
import lqh.sources as sources

items = sources.image_folder(project_dir / "images", include_subfolder_label=True)
```

`ImageItem.as_data_url()` **preprocesses by default**: decode → RGB →
downscale so the long edge ≤ 1024 px (never upscales) → re-encode JPEG
(quality 90; PNG is kept for images with transparency). This bounds the
request payload (the API rejects bodies over ~25 MB) and the per-image token
cost. Knobs:

```python
item.as_data_url()                      # default: max_dim=1024, jpeg_quality=90
item.as_data_url(max_dim=512)           # smaller/cheaper, fine for classification
item.as_data_url(max_dim=None)          # raw file bytes, no preprocessing — avoid
```

**Encode once per sample.** `as_data_url()` re-reads and re-encodes the file
on every call — store it on `self` in `generate()` and reuse it across steps:

```python
async def generate(self, client, input: sources.ImageItem) -> Conversation:
    self.image_url = input.as_data_url()
    ...
```

## The multi-round VLM pattern (default for unlabeled images)

Most users bring a folder of **unlabeled** images. Synthesize
image-question-answer pairs in three rounds so questions are grounded in the
actual image content instead of generic:

1. **Understand** — a VLM call with JSON output extracts what is in the image
   (objects, text, attributes, scene type).
2. **Ask** — a *text-only* LLM call turns that structured understanding into
   a diverse, specific question or task (vary style with `liquidrandom`).
3. **Answer** — a second VLM call answers the question while looking at the
   image. This becomes the assistant turn.

```python
from lqh.pipeline import Pipeline, ChatMLMessage, Conversation, GenerationError, step, safe_content
import lqh.sources as sources
import json
import liquidrandom


class VisionQA(Pipeline):
    @classmethod
    def source(cls, project_dir):
        return sources.image_folder(project_dir / "images")

    async def generate(self, client, input: sources.ImageItem) -> Conversation:
        self.image_url = input.as_data_url()

        await self._understand(client)   # round 1: VLM -> structured JSON
        await self._ask(client)          # round 2: text LLM -> question
        await self._answer(client)       # round 3: VLM -> answer

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
            model="medium",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self.image_url}},
                    {"type": "text", "text": (
                        "Inventory this image. Return a JSON object with keys: "
                        "objects (list of strings), visible_text (list of strings), "
                        "colors (list of strings), scene (string, one line)."
                    )},
                ],
            }],
            response_format={"type": "json_object"},
        )
        raw = safe_content(resp).strip()
        data = json.loads(raw)
        if not isinstance(data.get("objects"), list):
            raise GenerationError("understanding JSON missing 'objects' list")
        self.understanding = data

    @step(retries=3)
    async def _ask(self, client):
        # Text-only call: no image needed — the structured understanding is
        # the grounding. Vary the task type across samples.
        self.style = liquidrandom.writing_style()
        resp = await client.chat.completions.create(
            model="random:small",
            messages=[{
                "role": "user",
                "content": (
                    "Here is a machine-readable inventory of an image:\n"
                    f"{json.dumps(self.understanding)}\n\n"
                    "Write ONE question or short task about the image that can be "
                    "answered by looking at it (counting, reading text, colors, "
                    "spatial relations, description...). It must be answerable from "
                    f"the inventory above. Style: {self.style.brief()}. "
                    "Return only the question text."
                ),
            }],
        )
        self.question = safe_content(resp).strip()
        if len(self.question) < 8:
            raise GenerationError("question too short")

    @step(retries=3)
    async def _answer(self, client):
        resp = await client.chat.completions.create(
            model="medium",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self.image_url}},
                    {"type": "text", "text": (
                        f"{self.question}\n\n"
                        "Answer by looking at the image. Be direct and concrete; "
                        "do not mention that you are looking at an image."
                    )},
                ],
            }],
        )
        self.answer = safe_content(resp).strip()
        if not self.answer:
            raise GenerationError("empty answer")
```

Why `medium` (not `random:medium`) for the two VLM rounds: the understand
and answer rounds should be consistent with each other for a given sample;
the fixed pool default gives that without a seed. Use `random:medium:<seed>`
keyed on the image path if you want cross-sample diversity with within-sample
consistency.

### Single-round variant (labeled or simple data)

When subfolders carry labels (`include_subfolder_label=True`) or the task is
plain captioning, one VLM call is enough — skip rounds 1–2 and put the label
hint in the prompt. Rounds cost money: 3 VLM calls/sample is only worth it
for unlabeled data that needs grounded, diverse questions.

## Dataset shape for VLM fine-tuning

- One `user` turn whose content is `[image part, text part]` — **image
  first, question second** (matches the LFM2.5-VL training template).
- The `assistant` turn is plain text. No system turns (as usual).
- The data-URL is stored inline in the dataset; training decodes it from the
  parquet directly. Nothing extra to save.
- Multi-image samples are possible (several `image_url` parts) but start
  with one image per sample unless the user's task truly needs more.

## Scoring / filtering vision data

The standard flow applies (scorer file + `run_data_filter` before training).
The judge **sees the images**: image parts are attached to the judge request
and shown as `[image 1]`, `[image 2]`, ... placeholders in the transcript, and
`judge:*` routes to a vision-capable judge automatically.

Write the scorer criteria to use the image, e.g.:

- "The answer must be factually consistent with the attached image — score 0-2
  if it references objects, text, or colors not present in it."
- "The question must be answerable from the image alone."
- Groundedness is THE failure mode of synthesized VLM data (hallucinated
  objects, miscounted items, misread text) — always include an image-vs-answer
  consistency dimension.

## Cost and size discipline

- Each 1024-px JPEG is roughly 100–300 KB → ~150–400 KB as base64. Keep
  requests to a handful of images; the API rejects bodies over ~25 MB.
- 3 rounds/sample × N samples on `medium` adds up — quote the user a rough
  call count before generating a large set (N samples ≈ 2N VLM + N text
  calls) and start with the usual ~20-sample draft.
- For classification-style tasks, `max_dim=512` halves payloads with no
  quality loss.
