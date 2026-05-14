# Skill: Data Validation

You are now in **data validation** mode. Your goal is to create validation criteria and evaluate generated datasets against the project specification.

## Overview

Data validation ensures that generated training data actually matches the spec before it is used for fine-tuning. You will:

1. Read the project spec and understand the requirements
2. Examine the generated dataset(s) to understand what was produced
3. Create a validation criteria document
4. Sample and review data against those criteria
5. Report findings and suggest fixes

## Rules

1. **Always read the spec first.** Use `read_file` on `SPEC.md` and any additional spec files. The validation criteria must be derived from the spec.
2. **Examine the dataset before writing criteria.** Use `read_file` on the generated parquet files in `datasets/` to understand the data format and content. Look at at least 5-10 samples.
3. **Create the validation criteria as a text file** in the project root or a dedicated location (e.g., `evals/validation_criteria.md` or `validation_instructions.txt`).
4. **Be specific.** Vague criteria like "output should be good" are useless. Every criterion should be objectively assessable.
5. **Use `ask_user` to confirm** the validation criteria with the user before finalizing.
6. **Sample broadly.** Don't just look at the first few rows. Use `read_file` with different `offset` values to sample from different parts of the dataset.

## Validation Criteria Categories

Your validation criteria document should cover these areas:

### 1. Format Compliance

Check that each sample matches the expected structure:

- **Conversation structure**: Correct roles in correct order (system, user, assistant, tool, etc.)
- **Required fields**: All fields that should be present are present and non-empty
- **Length constraints**: Outputs within the specified word/token/character ranges
- **Format**: JSON is valid JSON, bullet points are actual bullet points, sections are properly labeled, etc.
- **Language**: Content is in the expected language(s)

Example criteria:
```
- Each sample MUST have exactly 3 messages: system, user, assistant
- The assistant message MUST contain 4 sections: Problem, Method, Results, Limitations
- Each section MUST have 2-4 bullet points
- Total assistant message length MUST be between 150-500 words
- All content MUST be in English
```

### 2. Content Quality

Check that the content is accurate and useful:

- **Relevance**: Does the output address the input? Is the response on-topic?
- **Accuracy**: Are facts, names, dates, numbers plausible and internally consistent?
- **Faithfulness**: Does the output stay grounded in the input (no hallucination)?
- **Completeness**: Are all aspects of the input addressed in the output?
- **Specificity**: Is the output specific enough, or is it vague and generic?

Example criteria:
```
- The summary MUST reference specific methods, results, or findings from the input
- The summary MUST NOT contain information not present in the input text
- Named entities (people, organizations, methods) MUST be spelled correctly
- Numbers and statistics MUST match the source material
```

### 3. Diversity

Check that the dataset has sufficient variety:

- **Topic diversity**: Are different subjects/domains represented?
- **Style diversity**: Are there different tones, formats, levels of complexity?
- **Input diversity**: Are inputs varied in length, structure, and difficulty?
- **Persona diversity**: If personas are used, are they varied in demographics and backgrounds?
- **Edge case coverage**: Does the dataset include the edge cases specified in the spec?

Example criteria:
```
- No two consecutive samples should have the same topic
- At least 10% of samples should cover edge cases (out-of-scope inputs, ambiguous queries)
- Personas should span at least 3 different age groups and 5 different occupations
- Input lengths should range from short (1-2 sentences) to long (3+ paragraphs)
```

### 4. Conversation Naturalness

For conversational data:

- **Flow**: Does the conversation flow naturally? Are follow-ups relevant to prior turns?
- **Persona consistency**: Does each speaker maintain a consistent voice and knowledge level?
- **Turn length**: Are turn lengths realistic for the scenario?
- **Variety of interaction patterns**: Not every conversation should follow the same question-answer-followup template

### 5. Spec Compliance

Direct checks against the spec's requirements:

- Go through each numbered requirement in the spec and create a corresponding validation criterion
- Check the spec's examples: does the generated data look like the examples?
- Check the spec's edge cases: are they represented in the data?
- Check the spec's quality criteria: does the data meet them?

## Validation Criteria Document Format

Create the document with this structure:

```markdown
# Validation Criteria for <Project Name>

Based on: SPEC.md (and any additional specs)
Dataset: datasets/<dataset_name>/

## Format Checks

1. [ ] <criterion>
2. [ ] <criterion>
...

## Content Quality Checks

1. [ ] <criterion>
2. [ ] <criterion>
...

## Diversity Checks

1. [ ] <criterion>
2. [ ] <criterion>
...

## Spec Compliance Checks

1. [ ] <criterion> (from Requirement #1)
2. [ ] <criterion> (from Requirement #2)
...

## Edge Case Coverage

1. [ ] <edge case from spec> - present in dataset: YES/NO
2. [ ] <edge case from spec> - present in dataset: YES/NO
...

## Sample Review Notes

### Sample 1 (row N)
- Pass/Fail: ...
- Notes: ...

### Sample 2 (row N)
- Pass/Fail: ...
- Notes: ...

...

## Summary

- Total criteria: N
- Passing: N
- Failing: N
- Issues found: ...
- Recommendations: ...
```

## Workflow

### Step 1: Read the Spec

Use `read_file` on `SPEC.md`. Extract:
- All numbered requirements
- Input/output format expectations
- Example input/output pairs
- Edge cases
- Quality criteria

### Step 2: Examine the Dataset

Use `read_file` on the parquet file(s) in `datasets/`. Look at:
- The schema (column names and types)
- The first few rows (initial impression)
- Random samples from the middle and end (use `offset`)
- Total row count

### Step 3: Draft Validation Criteria

Based on the spec and your examination of the data, write the validation criteria document. Use `create_file` or `write_file`.

### Step 4: Review with the User

Use `show_file` to display the criteria document, then `ask_user`:
- "Are these criteria complete?"
- "Anything you want to add or adjust?"
- "Should any criterion be stricter or more lenient?"

### Step 5: Evaluate

Go through a sample of the data (at least 10-20 rows from different parts of the dataset) and evaluate against the criteria. Update the document with your findings.

### Step 6: Report

Present findings to the user:
- How many criteria pass vs. fail
- Specific examples of failures
- Recommendations (re-run the pipeline with fixes? adjust the spec? generate more data?)

Use `ask_user` to offer next steps:
- "Fix the pipeline and regenerate"
- "Adjust the spec"
- "The data looks good, proceed to training"
- "Generate more data to improve diversity"

## Next Steps

After validation, use `ask_user` to offer:

1. **"Fix the pipeline and regenerate"** — If significant issues were found, go back to `/datagen` and fix the pipeline.
2. **"Start fine-tuning"** (recommended if data looks good) — Load `/train` to fine-tune a model on the validated data.
3. **"Generate more data to improve diversity"** — If the data is correct but lacks diversity, generate additional samples.
4. **"Adjust the spec"** — If validation revealed spec gaps, go back to `/spec` and update.
5. **"I'm done for now"** — The user can download and use the data externally.

After fine-tuning, the next step is to re-run model eval (`/eval` with mode=model_eval) to compare the fine-tuned model against baselines and the prompt-optimized model. If scores aren't good enough, generate on-policy failure data and fine-tune again.

## Tips

- **Be constructively critical.** The goal is to improve the data, not just approve it. Find real issues.
- **Quantify when possible.** "3 out of 15 samples had this issue" is better than "some samples had issues."
- **Check for systematic patterns.** If many samples share the same structure or phrasing, that's a diversity problem worth flagging.
- **Compare to spec examples.** The examples in SPEC.md are the ground truth. If the generated data diverges from them, flag it.
- **Look for subtle issues.** Common problems: all outputs start the same way, diversity in one dimension but not another, technically correct but unhelpful responses.
