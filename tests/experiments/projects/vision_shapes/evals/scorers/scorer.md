# Scorer: Vision Shapes QA

## Task

The user turn contains an image (attached to this request — see the
`[image N]` marker in the transcript) and a question about it. The
assistant answers in free text. You are scoring **whether the answer is
correct and grounded in the attached image**.

The images are simple and synthetic: 1–3 same-colored shapes (circle,
square, or triangle) in a row on a light background, sometimes with a short
dark text label (e.g. "CAT", "42") in the top-left corner. There is no
ambiguity in them — counts, colors, shapes, and text are objectively
checkable by looking.

## How to score

Look at the attached image, answer the question yourself, then compare.

- **10**: answer is fully correct (right count / color / shape / text /
  description) and concise.
- **8-9**: correct on the asked fact but with harmless extra detail, or a
  correct answer phrased vaguely (e.g. "a few shapes" when asked to
  describe, not to count).
- **6-7**: partially correct — e.g. right shape but wrong color when both
  were asked, or a description that gets the scene right but one attribute
  wrong.
- **3-5**: the central fact is wrong (wrong count, wrong color, misread
  text) but the answer is still about the visible content.
- **1-2**: ungrounded — references objects, colors, or text that are not
  in the image, or answers a different question entirely.

Meta-references ("in the image provided...") cost one point. An answer
that refuses or claims it cannot see the image scores 1.

Output JSON with keys: reasoning, score.
