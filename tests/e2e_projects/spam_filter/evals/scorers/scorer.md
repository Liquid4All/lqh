# Scorer: Generic Spam Filter

## Task

The user gives a filter rule and a message; the assistant outputs
`{"match": "yes"}` or `{"match": "no"}`. The schema is constraint-
decoded so format is enforced. You are scoring **whether the
yes/no answer is correct given the rule and the message**.

## How to score

Read the rule, read the message, and decide what a careful human
would answer. Compare to the assistant's answer.

- **10**: assistant's answer matches the obviously correct answer.
  Rule is unambiguous, message clearly does or doesn't match.
- **8-9**: answer is correct but the case is borderline. A
  reasonable human could see the other side; the assistant picked
  the better of the two. Or: answer is correct, but the message
  has a weak signal that could mislead.
- **6-7**: answer is **defensible but suboptimal**. The case is
  genuinely ambiguous (e.g. a colleague's sales pitch — could be
  filtered as "sales" or kept as "from a colleague"). Either yes
  or no is reasonable; the assistant picked one of them with the
  reasoning the rule supports.
- **3-5**: answer is **clearly wrong** but the wrongness has a
  surface-level excuse — e.g. message contains spam-like surface
  features (URLs, urgent language) and the rule was about something
  unrelated. The assistant fell for the surface signal instead of
  reading the rule.
- **1-2**: answer is **clearly wrong** with no excuse. Rule
  unambiguous, message unambiguous, assistant flipped the polarity.

## Examples

**Filter rule**: "Filter messages that want to sell me something"
- Message: "BUY NOW! 50% off, today only!" → correct answer: yes
- Message: "Hey, can we move tomorrow's standup to 3pm?" → correct
  answer: no
- Message: "Free meeting room booking for our team!" — borderline.
  "Free" is a sales-y word but "meeting room booking" is operational.
  Either yes or no is defensible (score 6-8 either way).

**Filter rule**: "Filter messages about meetings"
- Message: "Can we move the standup to 3pm?" → correct: yes
- Message: "BUY NOW! Limited offer!" — message looks spammy, but the
  rule is about MEETINGS. Correct answer: **no**. If the assistant
  said "yes" because the message looks spammy, that's a clear case
  of not reading the rule (score 1-3).

## Output format

Return JSON with `reasoning` (1-2 sentences) and `score` (integer
1-10). Schema enforced.
