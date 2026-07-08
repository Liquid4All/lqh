# Scorer: Customer-Support Email Triage

## Task

Score the triage JSON on a 1-10 scale. The schema is constraint-decoded
so format is enforced; you are scoring **judgment** — did the triage
plan match what a competent ops manager would do?

## Dimensions

Roughly: **priority (30%)**, **next_actions (30%)**, **category
(20%)**, **rationale (20%)**.

### 1. Priority

- p0 ↔ critical: safety issue, legal threat, data exposure,
  paying-customer outage. Wrong p0 (over-escalating) is a notable
  penalty; missing a real p0 is a major penalty.
- p1 ↔ clear harm or active failure but not life-threatening.
- p2 ↔ real issue, no immediate harm.
- p3 ↔ minor, feature request, positive feedback, casual question.
- Off by 1 = small penalty, off by 2 = significant.

### 2. Next actions

- For p0, `escalate` MUST be present. Missing it is a hard penalty
  (drops by 3+).
- `offer_refund` requires concrete evidence in the email that the
  company failed. Reflexively refunding on every complaint is
  wasteful (penalty); withholding when the company clearly broke
  something is unhelpful (penalty).
- `request_more_info` is appropriate when the email is too vague to
  act on; using it when the email is clearly actionable is a delay
  smell.
- 1-3 items; 0 or >3 is a schema violation (handled by constraint
  decoding so shouldn't happen, but penalise hard if it does).
- Multiple defensible plans may exist for ambiguous emails — both
  acknowledge-style and escalate-style should be acceptable for
  middle-of-the-road urgency. Don't over-penalise alternative
  defensible choices.

### 3. Category

- One of the 6 enum values; `other` is a fallback.
- `billing` ↔ charges, refunds, invoices.
- `shipping` ↔ delivery, tracking, lost packages.
- `product_quality` ↔ defects, malfunction, descriptions don't match.
- `account` ↔ login, password, profile, account management.
- `feature_request` ↔ asking for something the product doesn't do.
- `other` ↔ positive feedback, off-topic, ambiguous.
- Choosing `other` when a specific category fits = small penalty.

### 4. Rationale

- 1-2 sentences referencing specifics from the email.
- "Customer is unhappy" without a specific reference = penalty.
- Hallucinating details not in the email = major penalty.

## Score guide

- **10**: Priority + actions + category all correct (or alternative
  defensible plan). Rationale references concrete email content.
- **8-9**: Minor issue (off-by-1 priority, slightly redundant
  action, slightly generic rationale).
- **6-7**: One real misjudgment — wrong category that's still
  partially defensible, missing a useful action, priority off by 2.
- **4-5**: One major misjudgment — missed a p0, refunded for no
  cause, escalated a p3 thank-you note.
- **2-3**: Multiple major issues, or one critical (e.g. missed a
  legal threat, classified a safety issue as a feature_request).
- **1**: Schema violation despite constraint decoding (rare,
  investigate) OR triage plan is so wrong it would actively harm
  the operation.

## Output format

Return JSON with `reasoning` and `score`.
