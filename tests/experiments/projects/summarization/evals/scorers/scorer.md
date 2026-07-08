# Scorer: Concise Article Summarization

## Task

The model takes a 1-3 paragraph article excerpt and produces a 2-3
sentence neutral summary. Score the assistant's output on a 1-10 scale.

## Dimensions

Weight roughly: **fidelity (50%)**, **coverage (25%)**, **style (15%)**, **conciseness (10%)**.

### 1. Factual fidelity (most weight)

- Every claim in the summary must be grounded in the input.
- No hallucinated facts, dates, names, numbers, quotes, or causal links.
- No reversed polarity (e.g. summary says "increased" when source says "decreased").

### 2. Coverage

- The main point of the input must be present.
- If the input has multiple equally-weighted points, at least the most
  prominent should be covered.
- Missing the main thrust of the source is a major penalty.

### 3. Style

- Neutral, declarative, objective.
- No subjective adjectives or hedging that isn't in the source
  ("important", "shocking", "groundbreaking" added by the model = penalty).
- No preamble: "Here is a summary:", "In summary:", "The article says:".

### 4. Conciseness

- 2-3 sentences, roughly 30-70 words.
- One sentence: penalise unless the source is genuinely tiny.
- 4+ sentences or padding ("which is interesting because…"): penalise.

### 5. Format

- Plain prose only — no JSON, markdown, headings, or bullet points.
- No quoted blocks, no source attribution like "(source: ...)".

## Score guide

- **10**: Perfect fidelity + coverage + style + conciseness. Could be
  used directly without edit.
- **8-9**: Minor stylistic issues (slightly long, slightly subjective
  word) but factually solid and covers the main point.
- **6-7**: One real issue — missed a secondary point, slightly off
  style, mild over-summarisation, or one borderline-but-defensible
  added word.
- **4-5**: One significant issue — added a small unsupported detail,
  partially missed the main point, or has a clear preamble. Salvageable.
- **2-3**: Major fidelity issue — hallucinated a fact, reversed a
  polarity, or completely missed the main point.
- **1**: Output is not a summary (refusal, JSON, full-rewrite, empty,
  off-topic), or contains a fundamental factual error that would
  mislead a reader.

## Output format

Return JSON with `reasoning` (1-3 sentences explaining the score) and
`score` (integer 1-10). The judge enforces this via its response
schema; you do not need to add any wrapper.
