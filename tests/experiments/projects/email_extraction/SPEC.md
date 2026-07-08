# Specification: Customer-Support Email Extraction

## Overview

Given a single customer-support email (subject + body), the model
produces a structured JSON object with the sender's name, the
high-level intent, any products mentioned, an urgency score (1-5),
and a one-line summary. Strict JSON schema is enforced via constraint
decoding.

This is the **JSON + nested-fields** e2e task in our suite. It tests
that the model can:

1. Extract a person's name from a free-form sign-off / signature.
2. Classify intent into a closed enum.
3. Produce a JSON **array** (`mentioned_products`) — the harder
   constraint case where the model must choose count + items.
4. Produce an **integer** in a bounded range (urgency 1-5).
5. Produce a free-form `summary` string within the same JSON object.

## Input Format

User prompt is a verbatim email:

```
Subject: <subject line>

<body text>
```

Bodies are 1-3 paragraphs, sometimes with bullet points or a list of
issues, sometimes a single sentence rant.

## Output Format

JSON object with fields:

- `sender_name` (string) — extracted from sign-off or "From:"-like
  cues. Empty string if not present.
- `intent` (enum: `question` | `complaint` | `request` | `cancellation`) —
  the dominant intent.
- `mentioned_products` (array of strings) — proper-noun product names
  mentioned. Empty array if none.
- `urgency` (integer, 1-5) — 1 = casual, 5 = customer is angry / says
  legal / mentions safety.
- `summary` (string) — 1 sentence (≤ 25 words) summarising the email.

The schema is strict — additionalProperties is false; missing fields
must be filled with sensible defaults (`""`, `[]`, `1`).

## Why this task

`translation` covers basic enum + free-form-string JSON. This task
adds **arrays** (variable-length, items chosen by the model) and
**bounded integers**, plus the harder extraction problem of finding
a person's name in noisy text. Together they exercise more of
lm-format-enforcer's grammar coverage.

## Why SFT only

The output is well-specified and deterministic in shape. SFT against
high-quality references is the natural fit; DPO would only add value
if there were preference structure on `summary` phrasing, and we
don't expect that to be the bottleneck here. (If post-SFT scores
plateau noticeably below the API baseline, DPO can be added later.)
