# LQH cli mode for third-party agentic harnesses

## Context:

lqh is a terminal user interface (TUI) app where the user interacts with an agent for customizing LFMs.
Typical steps are:
1. Agent helps user derive SPEC.md with specifications of the task to be solved
2. Agent writes scoring criteria together with the user
3. Agent builds a data generation pipeline
4. Agent checks the data generation pipeline works (creating n=3 dummy samples) and also that the data quality is good (creating n=20 samples and reading them)
5. Agent generates a validation and a training dataset
6. Agent scores both dataset to filter out low quality samples
7. Agent runs zero-shot evaluation on the validation dataset to get a baseline performance
8. Agent runs a training loop to fine-tune the model on the training dataset
9. Agent runs on-policy preference optimization to improve the model based on user feedback
10. Agent either spins up an API endpoint with the fine-tuned model or exports it to gguf for edge inference with llama.cpp
11. User runs actual evaluation and testing of the deployed model (API or edge)
12. User comes back with feedback in the form of A) failure cases or B) changes in the specifications. -> this is where we go back to one of the previous steps and iterate.

Important: This is the ideal scenario where the loop is only at step 12. In practice, this is **itself** an iterative process where, in addition to the entire loop at step 12, the agent or user may also go back to any of the previous steps (1-11), eg. if the generated data is poor, the model does not learn etc.
Note: there is also a `--auto` mode where the agent can run the entire process from 2 onwards without user interaction.

In essence: lqh is an agentic harness with built-in tools for orchestrating the process of customizing LFMs. 

## Objective:

We users to use lqh using Codex, Claude Code, Hermes Agent, or other third-party agentic harnesses to orchestrate the process, and they may want to call individual steps of the process programmatically.

We want this because the user may be more familiar with these harnesses or include some personal context or workflows.
To achieve this my plan is to add a "cli mode" to lqh that instead of using the TUI, allows running individual steps or tools of this process via command line interface (CLI) commands. 
This way Codex or the other harnesses can call these commands to orchestrate the process.

## Design tradeoffs:

The question for me is whether we expose high-level primitives (eg. "run the data generation pipeline") or directly expose the individual tools that the "orchestration" agent in lqh has access to.
My thought is that we want to expose the individual tools, because this gives more flexibility to the user and allows them to orchestrate the process in a more fine-grained way.
The question is then whether we also want to expose the high-level primitives as well. This might be useful for the third-party harness to use lqh as a sub-agent

In short, my though is that we need:
- Direct tool access: for flexibility and fine-grained control by the third-party harness
- Sub-agent mode: for allowing the third-party harness to keep a more high-level orchestration and delegate some of the work to lqh

An open question is whether there is a `cli` mode or a `tool`, `subagent` mode or so and how to structure the commands.

## Skills and docs exposure

The lqh python app already ships with SKILLS.md files, instructions, and documentation. Thus, ideally we allow exposing this docs via the cli.
The third-party harness should be able to know what lqh can do, what modes it supports, and what it is made for via cli.

There is a conflict as well: Imagine a user or a third-party harness running `lqh --help` -> what should we show? Info for the user (simple, the user may only need the --auto mode), or info for the third-party harness (more detailed, exposing all the tools and modes)?

## Entry point

We want to have a single entry point `lqh` but when invoked with the cli mode, we don't want to load the TUI but use stdin/stdout to communicate with the third-party harness. This is because the TUI is not suitable for programmatic access.
We also want the cli mode to be snappy, thus we need to make sure the telemetry calls, UI rendering, etc are not enabled on startup.

## Direct tool access

We don't need all tools, eg. writing, files, etc. But some for persistency management (summary) or so maybe.
The question is how we manage this in case additional tools will be added in the future.

## Sub-agent mode

In the sub-agent mode, we probably want to:
- Have different or additional system prompt instructions to tell the orchestration model that this is sub-agent mode
- It shares some concepts with the `--auto` mode in the sense that no user input is expected, but it is different in the sense that the UI rendering is not used
- The sub-agent should terminate with a clear exit code and summary of what it did and resulting artifacts (eg. files, models, etc) so that the third-party harness can continue its orchestration.

