# Specification: Concise Article Summarization

## Overview

A short-form summarization model that takes a 1-3 paragraph
article-style input (news, blog, technical write-up) and produces a
**2-3 sentence neutral summary** in plain text. No JSON, no tool
calls — pure free-form text. The model prioritises factual fidelity
to the input, conciseness, and a neutral/objective tone.

This is the "open-ended" e2e task in our suite: it tests SFT and DPO
on a free-form generation problem where there is no single "correct"
output, multiple summaries are defensible, and quality is best
captured via an LLM judge that evaluates faithfulness, coverage, and
style.

## Input Format

- **Type**: Plain text — a 1-3 paragraph article excerpt
- **Length**: ~80-300 words
- **Domain**: News, blog posts, tech write-ups, business updates
- **Style**: First/third person; varied register

The user prompt is `"Summarize the following:\n\n<excerpt>"`. The
model may also see the same prefix without the colon — robustness
to small instruction variations is in scope.

## Output Format

- **Type**: Plain text (no JSON, no markdown, no preamble)
- **Length**: 2-3 sentences (~30-70 words)
- **Style**: Neutral, objective, declarative
- **Constraints**:
  - Must be factually grounded in the input (no hallucinations)
  - Must cover the main point(s) of the input
  - Must NOT include "Here is a summary:" or similar prefaces
  - Must NOT introduce information absent from the input

## Quality dimensions

The judge scorer (`evals/scorers/scorer.md`) evaluates each summary on:

1. **Factual fidelity** (most weight): no hallucinated facts, no
   contradictions, no claims not supported by the input.
2. **Coverage**: the main point of the input is present.
3. **Conciseness**: 2-3 sentences, no padding.
4. **Style**: neutral/objective tone, no preamble.

## Why DPO is included

Summarization is preference-shaped: multiple summaries can all be
"correct" but differ in length, focus, or phrasing. SFT teaches the
model the format; DPO sharpens it toward the preferred style by
contrasting on-policy generations with golden references from the
training set. We run **3 DPO iterations** after SFT.
