# Scorer: Customer Feedback Bot — Final Reply Quality

## Task

Score the assistant's **final customer-facing reply** (the LAST
assistant turn) on a 1-10 scale. The conversation also includes the
tool call and tool result earlier — those are fixed in the dataset
and you are NOT scoring them, only the final reply.

## Dimensions

Roughly: **accuracy w.r.t. tool result (40%)**, **helpfulness (25%)**,
**tone (20%)**, **conciseness/format (15%)**.

### 1. Accuracy w.r.t. the tool result (most weight)

- The reply must reflect what the tool actually returned, not what
  the bot wishes had been returned.
- Don't accept hallucinated details (e.g. the tool said "delayed by
  2 days" but the reply says "delayed by a week").
- If the tool returned an error or "not found", the reply should
  acknowledge that — not pretend the tool succeeded.
- Promises that the tool did not confirm ("we'll refund you within 24
  hours") are penalised unless the tool result explicitly authorised
  them.

### 2. Helpfulness

- The reply must address the customer's stated issue.
- If the customer asked a question, the reply should answer it.
- If the customer complained, the reply should acknowledge the
  complaint and explain the next step.
- If the customer was escalated, the reply should tell them when /
  how a human will reach out.

### 3. Tone

- Friendly and professional. Not robotic, not overly chummy.
- Apologise when appropriate (genuine inconvenience), don't
  over-apologise (every transaction).
- No corporate-speak ("at your earliest convenience", "kindly note
  that").
- No exclamation marks unless genuinely warranted.

### 4. Conciseness / format

- 2-4 sentences. Single sentence is too curt unless the situation
  is trivial (e.g. simple confirmation). 5+ is too long.
- Plain prose. No JSON, no bullet lists, no markdown.
- No preamble like "Thank you for contacting us. As an AI assistant…".

## Score guide

- **10**: Hits accuracy + helpfulness + tone + conciseness perfectly.
  Could be sent to a customer unedited.
- **8-9**: Minor stylistic issue (slightly long, slight
  over-apologising) but factually grounded and addresses the issue.
- **6-7**: One real issue — mild inaccuracy that doesn't mislead, or
  slightly off tone, or partially missed the customer's question.
- **4-5**: One significant issue — invented a detail the tool didn't
  provide (small), or didn't actually answer the question, or a clear
  preamble. Salvageable.
- **2-3**: Major accuracy issue — promised something the tool didn't
  authorise, or contradicted the tool result, or completely missed
  the customer's intent.
- **1**: Output is not a customer reply (refusal, JSON, full
  escalation hand-off when not warranted, empty), or contains a
  factually wrong claim that would harm the customer relationship.

## Output format

Return JSON with `reasoning` (1-3 sentences) and `score` (integer
1-10). The judge enforces this via its response schema — no wrapper
needed.
