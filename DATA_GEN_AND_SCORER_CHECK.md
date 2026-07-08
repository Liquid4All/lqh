# Data generation and scorer check

The data gen skill should include the following steps and specifications:

- Starting from the SPEC.md that was captured using a dedicated skill
- The agent (= orchestration model) creates a data generation pipeline
- The agent runs it with 3 samples to make sure it does not crash
- If the python code or LLM calls crash -> agent should fix it iteratively until it works
- Once it works, we run is again on a demo dataset of 20 samples
- The agent should inspect these samples and if good show it to the user to get feedback and approval
- If the data is bad (either decided by the agent or by the user) -> improve and iterate on the data generation pipeline and repeat. Typically, this means breaking down each data sample generation into smaller and easier steps that combined yield a strong sample
- If even after several iterations the data is still bad or if the data looks off by hallucination, the agent should move the data generation from the "random:small" pool to "random:medium" or "random:large" pool to get better data generation capabilities and repeat the process
- Once the orchestration model and the user has confirmed the data is good, we create a scorer/judging prompt/spec based on the specification and rubrics we want to capture.
- The agent should then test the scorer on the 20 demo samples and inspect the scorer outputs to make sure it is working as intended. Optionally we can also show the scoring result to the user to confirm.
- If the scorer underperforms, again either determined by the agent or by the user, the agent should iterate on the scorer prompt/spec to take the feedback or failure mode into account
- If even after several iterations the scorer is still underperforming, the agent should move the scorer generation from the "judge:small" to the "judge:medium" or "judge:large" pool to get better scorer generation capabilities and repeat the process


Note: Make sure the data gen skill include a python code snippet of a valid and working data generation pipeline (correct imports, return values, LLM calls, etc).
Note: Make sure that the agent/orchestration model should always inspect the output of a generated samples or scored samples to make sure we have a good pipeline or prompt as well as a right sized model.
Note: Typically, the first steps is to break down the data generation into smaller steps, eg. in a multi-turn conversation generate the steps separately in sequence instead of one-shot the entire conversation. Same goes for multi-output, multi-cases, or multi-objective data. For scorer improvements, typically we can boost the performance by providing in addition to clear instructions and rubrics also clear examples in both directions or coverage of the spectrum, eg. good, bad and mid.

Note: Make sure it is clear to the agent that the data generation also includes the scorer prompt and the checking if the scorer works.
Recap: Make sure it is clear to the agent that the sequence is typically to 1. get the first pipeline working (syntax, import, format errors fixing), then 2. inspect the generated data, 3. improve the data gen pipeline by breaking it down (different cases, combining single turns to multi-turns, multiple outputs separately etc), 4. move to a bigger model if needed, 5. create a scorer based on the same specification and rubrics, 6. check the scorer performance on the demo data, 7. iterate on the scorer if needed by improving the prompt/spec and providing examples, 8. move to a bigger model for the scorer if needed.

