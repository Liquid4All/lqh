# Specification: Customer Feedback Bot

## Overview

A customer-feedback-bot agent for an e-commerce / SaaS company. The bot
takes a single customer message (complaint, question, request, escalation)
and uses one of **5 tools** to act on it: look up an order, pull customer
history, log feedback for analytics, send a canned acknowledgment, or
escalate to a human agent. After the tool returns a result, the bot
writes a friendly customer-facing reply.

This is the **tool-calling** e2e task in our suite. It tests that
SFT can teach the LFM to:

1. Pick the right tool for the customer's intent.
2. Extract the right arguments from natural language (order IDs,
   severity levels, escalation reasons, customer IDs).
3. Interpret a tool result into a helpful reply.

## Tools

| name | purpose | required args |
|------|---------|---------------|
| `lookup_order(order_id)` | Retrieve an order's status, items, and tracking info | `order_id` |
| `get_customer_history(customer_id, time_window)` | Pull the customer's past orders / interactions | `customer_id`, `time_window` |
| `log_feedback(category, severity, summary)` | Record feedback for analytics | `category`, `severity`, `summary` |
| `send_acknowledgment(template_id, customer_id)` | Send a canned reply (refund_initiated, receipt_resend, apology, etc.) | `template_id`, `customer_id` |
| `escalate_to_human(reason, urgency)` | Hand off to a human agent | `reason`, `urgency` |

Argument formats are JSON-schema enforced server-side. The training
data exercises edge cases: missing customer IDs, ambiguous order
references, multi-issue messages, escalation triggers (legal threats,
"speak to manager"), language other than English (rare).

## Conversation shape

Each sample is a 5-message ChatML conversation with tools attached
to the system message (so the chat template renders them):

```
system   — "You are a customer feedback bot…" (with tools=[…])
user     — the customer's message (1-3 sentences)
assistant — brief prelude + 1 tool call
tool     — JSON-string mock result
assistant — final, customer-facing reply (2-4 sentences)
```

This matches `tests/remote/tool_calling_e2e_pipeline.py`'s shape and
flows through `lqh.infer` and `lqh.train` unchanged.

## What the eval actually scores

`lqh.scoring._strip_trailing_assistant` removes the last assistant
turn before inference, so the model is asked to **regenerate the
final reply** given system+user+assistant(tool_calls)+tool. The judge
sees the whole conversation including the tool call and scores the
final reply on:

- Helpfulness — does the reply address the customer's stated issue?
- Accuracy — does it reflect what the tool actually returned (not
  invent details the tool didn't provide)?
- Tone — friendly, professional, not robotic.
- No hallucinated promises ("I'll process this immediately" only if
  the tool actually said so).

This eval format does NOT score tool *selection* (the tool call is
fixed in the dataset). To stress-test selection separately, run the
trained model against a handcrafted set of system+user-only prompts
and inspect the produced tool_calls — that's outside this harness's
current scope.

## Why SFT only (no DPO)

DPO needs preference pairs assembled via on-policy generation +
scoring. For this task, "good vs bad" reply quality is well-captured
by SFT against high-quality reference replies (from the data-gen
pipeline + judge filter). Adding DPO is a future extension if the
final-reply quality plateaus below the API baseline.
