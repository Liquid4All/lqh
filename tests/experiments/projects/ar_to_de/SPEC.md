# Specification: Arabic → German Translation (Web & Conversational)

## Overview

A one-way Arabic→German translator focused on **web texts and
conversational content**. Free-form output (no JSON wrapper, no
schema). The model takes Arabic input from one of two domains —
web content (news, blogs, product descriptions, forum posts) or
conversational content (chat, reviews, comments) — and produces
a German translation that preserves meaning, register, and tone.

This is the **DPO-focused** task in the e2e suite. Translation is
preference-shaped: many translations of the same source are
defensible, but some are clearly better than others (more fluent,
closer to the source's tone, fewer mistakes). That's exactly the
shape DPO needs — chosen and rejected differ continuously, not
binarily.

## Input Format

User prompt:

```
Translate to German:

<Arabic source text>
```

Source text varies in length (one sentence to a few paragraphs)
and register (formal news, casual chat, product copy, social
post).

## Output Format

Plain German text. No JSON, no preamble like "Hier ist die
Übersetzung:", no markdown. Just the translation.

## Domains

The pipeline generates samples from two top-level domains:

**Web** (60% of samples):
- News headlines and short articles
- Blog posts and opinion pieces
- Product descriptions and listings
- Forum posts (Reddit-style)
- Social media posts (short-form)

**Conversational** (40% of samples):
- Chat dialogues (multi-turn snippets)
- Customer reviews
- Comment threads
- Casual messages

Topics are sampled across technology, news, food, travel, sports,
finance, health, education, entertainment, business, social issues,
and personal life — to give the model exposure to common
vocabulary across registers.

## Why DPO is the focus

Compared to spam_filter (binary, asymmetric DPO failure) and
email_triage (constrained enums, mode collapse on extreme actions),
free-form translation is the canonical shape DPO can shine on:

- **Continuous preference signal**: a translation can be 6/10 (rough
  but understandable) or 9/10 (natural, accurate) — not 0/1.
- **No enum to collapse to**: the model can't degenerate to "always
  pick option X" because there's no fixed option set.
- **Bilateral errors**: when the SFT model is wrong, it's wrong in
  *many* directions (literal translation, dropped nuance, wrong
  register, etc.) — no single direction for DPO to over-correct on.

## Variants

- **`e2e_config.json`** (default): SFT then DPO 3 iterations.
- **`e2e_config_dpo_only.json`**: DPO from baseline, no SFT.
  Comparison case to see whether SFT pre-training is needed for
  this task or whether on-policy DPO can teach translation from
  scratch.
