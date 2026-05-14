# Translation: English → German + French

## Task
Translate any English input text into both German and French, returning a
JSON object with two keys: `de` and `fr`.

## Input
- One to five sentences of English text.
- May include informal language, slang, and short phrases.

## Output format
A single JSON object exactly of the shape:

```json
{"de": "<German translation>", "fr": "<French translation>"}
```

No commentary, no extra keys, no markdown fencing.

## Quality criteria
- Both translations preserve the source meaning.
- Tone matches: casual stays casual, formal stays formal.
- Output is parseable as JSON.

## Base model
Use the smallest available LFM (~1.2B parameters).
