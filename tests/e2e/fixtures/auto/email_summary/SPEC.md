# Email Summarization (one-line)

## Task
Reduce an email body to a single-line summary (≤ 20 words) capturing the
sender's primary ask or update.

## Input
- The full body of a work email (subject not included). Length: 50 to 500
  words. Tone ranges from casual chat to formal updates.

## Output format
A single line of text, no markdown, no bullets, no leading "Summary:".
Maximum 20 words.

## Quality criteria
- Captures the *action* or *information* the sender most wants the
  recipient to take/know.
- Drops greetings, sign-offs, and reply chains.
- Stays under 20 words.

## Base model
Use the smallest available LFM.
