# Specification: Business Chat Translation (EN ↔ DE ↔ JA)

## Overview

A translation model for business and professional messaging contexts (Slack, Teams, email) that translates between English, German, and Japanese in all four bidirectional directions: EN→DE, DE→EN, EN→JA, and JA→EN. The model outputs structured JSON including the detected source language, target language, and the translation. It prioritizes accuracy above all, uses polite/formal register consistently, and preserves common English business loanwords in German and Japanese output as is customary in real-world business communication.

## Input Format

- **Type**: Plain text — single chat messages or short conversation threads
- **Domain**: Business / professional messaging (Slack, Teams, email)
- **Typical length**: Variable — single sentence to a few sentences (1–5 messages in a thread)
- **Languages**: English, German, Japanese
- **Preprocessing**: None expected; input is raw chat text

The input may be a single message or a short thread. When a thread is provided, the model should use the full context to resolve pronouns and references, but translate only the most recent message (or the entire thread if no specific message is indicated).

## Output Format

- **Type**: JSON object
- **Structure**: Three required fields:
  - `source_lang`: ISO 639-1 code of the detected source language (`"en"`, `"de"`, or `"ja"`)
  - `target_lang`: ISO 639-1 code of the target language
  - `translation`: The translated text as a string
- **Typical length**: Proportional to input (1–5 sentences)
- **Reasoning**: No reasoning or explanation — just the structured translation output

Example output structure:
```json
{
  "source_lang": "en",
  "target_lang": "de",
  "translation": "Können Sie mir den Q3-Bericht bis Ende des Tages schicken?"
}
```

## Requirements

1. The model MUST translate accurately between all four direction pairs: EN→DE, DE→EN, EN→JA, JA→EN.
2. The model MUST output valid JSON with exactly three fields: `source_lang`, `target_lang`, and `translation`.
3. The model MUST correctly detect the source language and report it using ISO 639-1 codes (`"en"`, `"de"`, `"ja"`).
4. The model MUST always use polite/formal register:
   - German: Use "Sie" form (not "du")
   - Japanese: Use "desu/masu" form (not casual/plain form)
5. The model MUST preserve common English business loanwords in German and Japanese translations (e.g., "Meeting", "Deadline", "Feedback", "Agenda", "Presentation", "Project", "Budget").
6. The model MUST translate as-is without flagging ambiguity or asking for clarification, even when the input is ambiguous or lacks context.
7. The model SHOULD preserve proper nouns and product names in most cases, but adapt them when contextually appropriate (e.g., transliterate to katakana for Japanese when the term is widely used in that form).
8. The model MUST NOT add commentary, explanations, or translation notes in the output.
9. The model MUST NOT hallucinate or infer content that is not present in the input.
10. The model SHOULD handle conversation threads by using context to resolve references, producing a contextually accurate translation.

## Examples

### Example 1: EN→DE (Single message)

**Input:**
```
Translate to German: Can you send me the Q3 report by end of day?
```

**Expected Output:**
```json
{
  "source_lang": "en",
  "target_lang": "de",
  "translation": "Können Sie mir den Q3-Bericht bis Ende des Tages schicken?"
}
```

### Example 2: DE→EN (Business message with loanwords)

**Input:**
```
Translate to English: Ich habe das Meeting mit dem Kunden auf nächste Woche verschoben. Die Präsentation muss noch aktualisiert werden.
```

**Expected Output:**
```json
{
  "source_lang": "de",
  "target_lang": "en",
  "translation": "I've rescheduled the Meeting with the client to next week. The presentation still needs to be updated."
}
```

### Example 3: EN→JA (Polite request)

**Input:**
```
Translate to Japanese: Please review the contract and let me know if there are any issues.
```

**Expected Output:**
```json
{
  "source_lang": "en",
  "target_lang": "ja",
  "translation": "契約書をご確認いただき、問題がありましたらお知らせください。"
}
```

### Example 4: JA→EN (Message with katakana loanwords)

**Input:**
```
Translate to English: 明日のミーティングのアジェンダを送付しました。ご確認をお願いします。
```

**Expected Output:**
```json
{
  "source_lang": "ja",
  "target_lang": "en",
  "translation": "I've sent the agenda for tomorrow's Meeting. Please review it."
}
```

### Example 5: EN→DE (Conversation thread with context)

**Input:**
```
Translate to German:
[Thread]
Alex: The client wants to change the project timeline.
Alex: They asked if we can move the deadline to March.
Alex: Can you check with the team?
```

**Expected Output:**
```json
{
  "source_lang": "en",
  "target_lang": "de",
  "translation": "Können Sie mit dem Team überprüfen, ob wir die Deadline auf März verschieben können?"
}
```

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| Input contains mixed languages (e.g., German message with English phrases) | Detect the dominant language as source; translate the full message including embedded English terms appropriately |
| Input is very short (1–2 words, e.g., "Thanks!", "OK") | Translate as-is; maintain polite register in target language |
| Input contains technical jargon or acronyms (e.g., "ROI", "KPI", "SaaS") | Preserve acronyms as-is; translate surrounding text normally |
| Input contains a proper noun / person name | Preserve in original script; transliterate to katakana for Japanese target if appropriate |
| Input is ambiguous (e.g., "Let's schedule it" with no clear referent) | Translate as-is without flagging or asking for clarification |
| Input contains numbers, dates, or currencies | Adapt formatting to target language conventions (e.g., German "3.000,50 €" vs English "€3,000.50"; Japanese "3月15日" for dates) |
| Input is in a language other than EN/DE/JA | Detect the language in `source_lang` but still attempt translation to the requested target; output may be lower quality |
| Input contains emojis or emoticons | Preserve emojis as-is in the translation |
| Input is a question | Translate as a question; maintain question markers appropriate to target language (e.g., Japanese "か" or "でしょうか") |

## Quality Criteria

- **Accuracy**: The translation must be factually and semantically correct. Every meaning in the source must be preserved in the target. This is the top priority.
- **Completeness**: No information from the source should be dropped or omitted in the translation.
- **Conciseness**: The translation should not be unnecessarily verbose. Match the length and density of the source.
- **Tone/Style**: Always polite and professional. Formal register (Sie, desu/masu). Business-appropriate language.
- **Faithfulness**: The translation must be grounded in the source text. No hallucinated content, no added information, no omitted information.
- **Naturalness**: While accuracy is primary, the output should still read naturally in the target language. Avoid awkward literal translations when a natural equivalent exists.
- **Loanword handling**: English business loanwords (Meeting, Deadline, Feedback, etc.) should be preserved in German and Japanese output as is customary in business communication.
- **JSON validity**: The output must always be valid JSON with the three required fields. No extra fields, no malformed JSON.
