# Specification: Generic Spam Filter

## Overview

A binary classifier that takes a **filter rule** (free-form natural
language, written by the user) and a **message** (email, SMS, chat,
DM), and outputs `{"match": "yes"}` if the message matches the rule
or `{"match": "no"}` otherwise.

The filter rule is the user's spam policy — anything they want
filtered. Examples:

- "Filter messages that want to sell me something"
- "Filter messages asking me to click a link"
- "Filter messages from senders pretending to be from my bank"
- "Filter messages about meetings"
- "Filter dating or romance-related messages"

This is the **simplest-possible JSON** e2e task in the suite (single
enum field, two values). It tests:

1. **Instruction-following over a free-form rule** — the model must
   actually read the rule, not classify by surface features.
2. **Balanced binary classification under constraint decoding** —
   the model can't degenerate to "always yes" or "always no" because
   accuracy is the metric.
3. **Distinction from confounding signals** — a "yes for sales" sample
   and a "no for meetings, but is a sales pitch" sample contain the
   same surface features; the rule must be the deciding factor.

## Input Format

User prompt:

```
Filter rule: <free-form rule, 1 sentence>

Message:
<message body, 1-4 paragraphs>
```

## Output Format

```json
{"match": "yes" | "no"}
```

Schema-enforced via constraint decoding. No reasoning, no extra
fields.

## Distribution

Per-sample label rolled deterministically 50/50, then a message is
generated to match the assigned label. The "no" case is split:

- **30%** innocent messages (no spam-like content at all — about
  family, work, friends, normal life), to anchor the negative class.
- **70%** decoy messages — spam-like content matching a *different*
  filter rule than the one being asked about. This forces the model
  to read the rule, not the message in isolation. ("This message is
  a sales pitch, but the rule was about meetings, so the answer is
  no.")

## Why this task

`translation`, `email_extraction`, and `email_triage` all test
multi-field structured output. This task tests the opposite end of
the JSON spectrum: a single binary decision under hard constraint.
The base model's accuracy here is interesting on its own — instruct
models are typically biased toward "yes" or "follow the implicit
ask"; SFT on balanced data should correct that.

## Training variants

Three configs ship with this task:

- **`e2e_config.json`** (default): SFT only.
- **`e2e_config_with_dpo.json`**: SFT followed by 3 DPO iterations.
- **`e2e_config_dpo_only.json`**: DPO from baseline, no SFT.

Binary classification IS correctness-driven, but DPO can still help
because every wrong-answer sample produces a strong preference pair:

- **chosen** = the labelled correct answer (`{"match": "yes"}` or
  `{"match": "no"}`), scoring ≈ 10 by the judge.
- **rejected** = the model's wrong answer (the opposite enum value),
  scoring ≈ 1 by the judge.
- **gap** ≈ 9 — well above the `min_gap=0.5` floor, all wrong-answer
  samples qualify under the gap-quantile selector.

In practice this means DPO does targeted token-flipping on the
`yes`/`no` token wherever SFT was wrong. It might be marginal over
a well-trained SFT model, but it's worth measuring — that's why we
ship both variants. The SFT-only config is the default because it's
cheaper and usually enough; pick a DPO variant if SFT plateaus
visibly below the API baseline. If both plateau, scale data per
`data_generation` SKILL.md Phase 3.5.4.

## How to run each variant

```bash
# SFT only (default)
python -m tests.remote.experiment_e2e_pipeline --task spam_filter

# SFT + DPO 3 iter
python -m tests.remote.experiment_e2e_pipeline --task spam_filter \
    --config-name with_dpo

# DPO from baseline (no SFT)
python -m tests.remote.experiment_e2e_pipeline --task spam_filter \
    --config-name dpo_only
```

The harness's per-iter held-out eval + early-abort apply to both DPO
variants, so a degenerate run aborts after iter 0.
