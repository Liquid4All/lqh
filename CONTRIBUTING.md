# Contributing to lqh

Thanks for your interest in Liquid Harness. Please read this document before opening a pull request.

## Discuss before you PR

**Random PRs will be rejected.** We want every change to be aligned on direction, scope, and design before code is written. If you have an idea, a bug report, or a feature request, please:

1. Open a [GitHub issue](https://github.com/Liquid4All/lqh/issues) or a [discussion](https://github.com/Liquid4All/lqh/discussions) first.
2. Wait for a maintainer to weigh in and agree on the approach.
3. Only then proceed to implementation.

The single exception is **security hot fixes**. If you have identified a security-sensitive bug and have a minimal, targeted patch, you may open a PR directly. Please also email security@liquid.ai so we can coordinate disclosure.

## This is a vibe-coded repo

lqh is itself built and maintained with coding agents. To keep the codebase coherent across contributors and over time, most discussions do **not** conclude with a hand-written patch. They conclude with a **prompt** that a coding agent will then execute against the repo.

In practice, this means:

- Most issues/discussions will end with an agreed-upon prompt rather than a "go ahead and send a PR".
- A maintainer (or you, with the agreed prompt) runs the prompt through a coding agent, reviews the diff, and lands the result.
- If you want to drive the implementation yourself, propose the **prompt** in the discussion thread, not the diff. We will iterate on the prompt together, and the prompt becomes the source of truth for the change.

This sounds unusual, but it keeps style, structure, and patterns consistent — small, mechanical, "by hand" cleanups tend to drift the codebase away from what the agents produce, which makes future agent-driven changes harder.

## What we do accept as direct PRs

- Security hot fixes (see above).
- Typo fixes and obvious documentation corrections.
- Trivial, mechanical fixes that a reviewer can verify at a glance.

Anything larger — features, refactors, dependency changes, behavior changes, new tools or skills — goes through the discussion-then-prompt flow.

## Reporting bugs

Please include:

- Your OS, Python version, and `lqh` version.
- The exact command and (if applicable) the project directory layout.
- The full error output and, if relevant, the relevant `.lqh/conversations/*.jsonl` excerpt.

## Reporting security issues

Email **security@liquid.ai**. Do not open a public issue for security vulnerabilities.

---

Made with care by [Liquid AI](https://www.liquid.ai/).
