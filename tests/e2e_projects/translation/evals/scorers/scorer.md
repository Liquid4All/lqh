# Business Chat Translation Scorer

## Task
You are evaluating a business chat translation model. The model receives a chat message or thread in one language (English, German, or Japanese) and must translate it to a specified target language, outputting structured JSON with `source_lang`, `target_lang`, and `translation`.

## Scoring Scale
Score each sample on a 1-10 scale:

- **10**: Perfect translation — all meaning preserved, no errors, loanwords handled correctly
- **8-9**: Excellent — minor stylistic issues but fully accurate and complete
- **6-7**: Good — mostly accurate with small omissions or minor errors
- **4-5**: Acceptable — noticeable errors or omissions but core meaning intact
- **2-3**: Poor — significant errors, missing content, or wrong meaning
- **1**: Unusable — completely wrong, malformed, or irrelevant

## Evaluation Dimensions

### 1. Translation Accuracy (weight: 50%)
- Is the translation semantically correct?
- Does every claim, request, or statement in the source have an accurate counterpart in the translation?
- Are verb tenses, negation, and conditionals correctly translated?
- Are numbers, dates, and proper nouns translated correctly?
- **Critical failures** (automatic score ≤ 3):
  - Wrong meaning (saying "yes" when source says "no")
  - Hallucinated content not present in the source
  - Incorrect numbers, dates, or names

### 2. Completeness (weight: 30%)
- Is all information from the source present in the translation?
- No sentences, clauses, or key details should be dropped
- For threads: are all messages translated (or the full meaning preserved)?
- **Critical failures** (automatic score ≤ 3):
  - Missing an entire sentence from the source
  - Dropping a key detail (e.g., a date, a name, a quantity)

### 3. Loanword Preservation (weight: 20%)
- Are common English business loanwords preserved in German and Japanese output?
- Expected loanwords include: Meeting, Deadline, Feedback, Agenda, Presentation, Project, Budget, Report, Team, Manager, Workshop, Benchmark, Sprint, Review, Update, Briefing, Proposal, Pipeline, Target, Kickoff
- For German: these should appear as-is (e.g., "das Meeting", "die Deadline")
- For Japanese: widely-used katakana forms are acceptable (e.g., ミーティング), but the English term should not be replaced with a native Japanese equivalent when the loanword is standard in business
- For English target: no special loanword handling needed (this dimension is N/A and should be scored as 10)
- **Deductions**:
  - Replacing a standard loanword with a non-standard native equivalent: -2 points
  - Translating a loanword that should be preserved: -3 points

## JSON Format Check
Before scoring content, verify the output is valid JSON with exactly three fields: `source_lang`, `target_lang`, `translation`. If the JSON is malformed or missing required fields, score the sample as **1** regardless of content quality.

## Scoring Instructions
1. First, check if the assistant's response is valid JSON with the required fields. If not, score = 1.
2. Extract the `translation` field and compare it against the source text in the user message.
3. Score each dimension independently.
4. Compute the weighted total: accuracy × 0.5 + completeness × 0.3 + loanwords × 0.2
5. Round to the nearest integer for the final score.

## Examples

### Good Score (9-10)
Source (EN): "Can you send me the Q3 report by end of day?"
Target (DE): "Können Sie mir den Q3-Bericht bis Ende des Tages schicken?"
- Accuracy: 10 (perfect semantic match)
- Completeness: 10 (all info present)
- Loanwords: 10 ("Report" preserved as "Bericht" is standard; "Q3" preserved)
- Weighted: 10.0 → **10**

### Medium Score (6-7)
Source (EN): "The deadline for the project review is next Friday."
Target (DE): "Die Frist für die Projektüberprüfung ist nächsten Freitag."
- Accuracy: 8 (correct meaning)
- Completeness: 9 (all info present)
- Loanwords: 4 ("Deadline" was replaced with "Frist" and "Project review" with "Projektüberprüfung" — these should be "Deadline" and "Project-Review" or similar)
- Weighted: 8×0.5 + 9×0.3 + 4×0.2 = 4.0 + 2.7 + 0.8 = 7.5 → **8**

### Poor Score (3-4)
Source (EN): "I've rescheduled the Meeting with the client to next week. The presentation still needs to be updated."
Target (DE): "Ich habe verschoben."
- Accuracy: 3 (core meaning of "rescheduled" is there but everything else is missing)
- Completeness: 2 (most content dropped)
- Loanwords: 1 (no loanwords present, and most content is missing)
- Weighted: 3×0.5 + 2×0.3 + 1×0.2 = 1.5 + 0.6 + 0.2 = 2.3 → **2**
