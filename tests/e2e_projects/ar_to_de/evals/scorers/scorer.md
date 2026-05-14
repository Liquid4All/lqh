# Scorer: Arabic → German Translation

## Task

Score the assistant's German translation of an Arabic source text on
a 1-10 scale. The conversation has the source text in the user turn
and the German translation in the assistant turn.

## Dimensions

Roughly: **accuracy (50%)**, **fluency (30%)**, **register/tone (15%)**, **format (5%)**.

### 1. Accuracy (most weight)

- Every factual element in the source — entities, numbers, dates,
  claims — must appear in the translation, semantically equivalent.
- No hallucinated content the source doesn't contain.
- No dropped clauses or sentences (unless the source clearly had
  filler the German equivalent omits idiomatically).
- Negations, conditionals, and quantifiers must be preserved
  faithfully.

### 2. Fluency (German naturalness)

- The translation should read like native German prose, not
  word-for-word rendering.
- Correct grammar (case, tense, gender agreement).
- Idiomatic phrasing — German equivalents of Arabic idioms, not
  literal carryover.
- Natural word order and sentence flow.

### 3. Register and tone

- Casual chat → casual German (informal "du", short sentences).
- Formal news → formal German (third person, full sentences).
- Product copy → marketing tone with product name preserved.
- Customer review → first-person conversational with appropriate
  emotion (positive/negative/mixed).
- Mistakenly using formal "Sie" in a casual chat (or vice versa)
  is a noticeable register error.

### 4. Format

- Plain text only. Penalise any preamble ("Hier ist die Übersetzung:")
  or quotes around the output.
- Penalise repeating the Arabic source.
- Penalise outputting JSON or markdown.

## Score guide

- **10**: Could ship this translation as-is. Faithful, fluent,
  right register, clean format.
- **8-9**: One minor issue — slightly stiff phrasing in one place,
  a small word-choice quibble, or one missing nuance that doesn't
  change meaning.
- **6-7**: Translation is mostly correct and understandable but has
  a real fluency or register issue (literal translation of an idiom,
  inconsistent tone) or a small missing/added detail.
- **4-5**: Significant fluency issue (broken German, wrong word
  order in a way that disrupts reading), OR a meaningful accuracy
  error (got a number wrong, dropped a clause), but the gist is
  preserved.
- **2-3**: Major accuracy error — said the opposite, hallucinated a
  detail, or fundamentally garbled. Or completely wrong register
  (tweet translated as legal text).
- **1**: Output is not a translation at all (refusal, source echoed
  back, JSON wrapper, empty), or it's in the wrong language.

## Output format

Return JSON with `reasoning` (1-2 sentences) and `score` (integer
1-10). Schema enforced.
