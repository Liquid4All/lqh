# Specification: Vision Shapes QA

## Overview

A vision-language task over a synthetic "bring your own image folder"
dataset: deterministic PIL-generated images containing 1–3 colored shapes
(circle / square / triangle) and, on roughly half the images, a short text
label. The model answers free-form questions about an image — counting
shapes, naming colors, reading the rendered text, describing the scene.

This is the debug/e2e task for the VLM pipeline (VLM.md): it exercises

1. **Vision data generation** — the multi-round VLM pattern (understand the
   image via JSON, generate a question with a text LLM, answer with a VLM).
2. **Vision scoring** — the judge sees the image and checks groundedness
   (answers must match what is actually in the image; the ground truth is
   also encoded in the filename and `images/labels.json`).
3. **VLM fine-tuning** — LoRA SFT of `LiquidAI/LFM2.5-VL-450M` on the
   generated image-QA pairs.
4. **VLM inference** — checkpoint eval generates answers to held-out
   image questions.

The images are generated on first use by the pipeline's `source()` (seeded,
so repeat runs are identical) — nothing binary is checked in.

## Answer style

Short, direct, factual. One or two sentences; no meta-references to "the
image I was given".
