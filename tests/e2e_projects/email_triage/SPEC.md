# Specification: Customer-Support Email Triage

## Overview

Given a customer-support email, the model produces a *triage plan*:
which category, what priority, what immediate actions to take, and
why. This is the **JSON + preference-shaped** e2e task in the suite.
Multiple triage plans can be defensible for the same email
(e.g. "acknowledge + log" vs "escalate + offer_refund"), so DPO is
included on top of SFT to sharpen toward the preferred policy.

## Input Format

A single customer-support email (subject + body), same shape as
`email_extraction`. Bodies are 1-3 paragraphs.

## Output Format

JSON object with fields:

- `category` (enum): the support category — `billing`, `shipping`,
  `product_quality`, `account`, `feature_request`, or `other`.
- `priority` (enum): `p0` (critical, drop everything), `p1` (high,
  same business day), `p2` (normal, within 48h), `p3` (low, batched).
- `next_actions` (array of enum, 1-3 items): actions to take
  immediately. From: `acknowledge`, `escalate`, `offer_refund`,
  `loop_in_team`, `request_more_info`, `close`.
- `rationale` (string, 1-2 sentences): why this triage plan, grounded
  in the email's content.

Strict JSON schema; constraint decoding enforces the enums and array
membership.

## Why DPO

`acknowledge` vs `escalate` is a judgment call when the email is in
the middle of the urgency spectrum. Two plans can both be valid:

- `["acknowledge", "request_more_info"]` (low-friction, polite)
- `["escalate", "offer_refund"]` (proactive, more expensive)

SFT converges to the *median* preference. DPO contrasts on-policy
generations against the SFT-data-distribution preferences and
sharpens toward more confident, decisive plans for the cases that
warrant them (e.g. legal threats, safety issues).

We run **3 DPO iterations** after SFT.

## Why this is distinct from `email_extraction`

`email_extraction` does objective field-extraction from text — there
is one right answer per field, and SFT is sufficient. `email_triage`
is decision-shaped — the same email can warrant different actions
and the choice has business-cost trade-offs. DPO is the natural fit.

The two pipelines deliberately share the email-generation pattern
(same persona/product/tone rolls) so a future cross-task experiment
could compare extraction vs triage on shared inputs.
