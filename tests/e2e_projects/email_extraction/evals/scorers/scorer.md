# Scorer: Customer-Support Email Extraction

## Task

Score the assistant's JSON-extracted email summary on a 1-10 scale.
The schema (`prompts/schema.json`) is strict — additionalProperties
is false; the JSON parse / shape is enforced by constraint decoding.
You are scoring **content correctness**, not format.

## Dimensions

Roughly: **intent (25%)**, **sender_name (15%)**, **mentioned_products
(20%)**, **urgency (20%)**, **summary (20%)**.

### 1. Intent (`intent` field)

- Must match the email's dominant intent. Multi-issue emails: pick
  the dominant one (the one the customer most wants action on).
- "Cancellation" is its own bucket — only when the customer is
  explicitly cancelling something (subscription, order, account).
- "Complaint" vs "request": if the customer is unhappy AND wants
  something specific, prefer `request` if the action is the
  emphasis, `complaint` if the negative emotion is the emphasis.

### 2. Sender name (`sender_name`)

- Must match the name in the sign-off or signature, exactly as
  written (capitalisation matters; honorifics like "Dr."/"Mr." may
  be included or stripped — both acceptable).
- Empty string `""` is correct when no name is recoverable.
- Hallucinating a name (when none was given) is a major penalty.

### 3. Mentioned products (`mentioned_products`)

- Must include each proper-noun product mentioned, in any order.
  Generic words ("the product", "your service") should NOT appear.
- Missing a clearly-named product = penalty proportional to how
  central it is to the email.
- Inventing a product not in the email = major penalty.
- An empty array is correct when no products are named.

### 4. Urgency (`urgency`, 1-5)

- 1: casual ("just curious, no rush"), 3: clearly wants action
  but polite, 5: angry, threatens legal action, mentions a safety
  issue, says "URGENT" or all-caps complaints.
- Off by 1 = small penalty, off by 2+ = significant.

### 5. Summary (`summary`)

- One sentence, ≤ ~25 words, captures the email's main point.
- Must be factually grounded in the email.
- Tone neutral; no opinions or hedging not in the source.

## Score guide

- **10**: All five fields correct (or off-by-1 on urgency in a
  defensible direction). Could be used directly downstream.
- **8-9**: One field has a minor issue (slight summary phrasing,
  off-by-1 urgency, missed a secondary product).
- **6-7**: One real issue — wrong intent that's defensible, missed
  the central product, urgency off by 2.
- **4-5**: One major issue — invented a sender name, classified a
  cancellation as a complaint, or summary contradicts the email.
- **2-3**: Multiple major issues, or one critical (e.g. completely
  missed the email's intent).
- **1**: Schema violation despite constraint decoding (rare —
  usually indicates a constraint-decoding regression to investigate)
  OR the JSON is right but every field is wrong.

## Output format

Return JSON with `reasoning` and `score`.
