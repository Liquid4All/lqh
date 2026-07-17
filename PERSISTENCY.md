# LQH persistency

lqh is a agent cli/tui for customizing LFMs.
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

## Objective

We need some form of persistency across multiple lqh sessions.
We made the design choice to use folders/git repo as our persistency mechanism, ie. folder=project, this means we can assume the user runs the lqh command in the same path across multiple sessions.
Note: We want later on to support "cloud sessions" where lqh runs in a Modal Sandbox but even in this case we can assume the user runs the lqh command in the same path across multiple sessions.

Within a session we have the conversation history to understand what has been done so far and this works fine.
However, for more complex tasks or feedback, the user may require multiple sessions to complete the task.

## Scenarios

### Scenario 1: User work gets interrupted and needs to continue later

- User finishes only partial of this process in a session, eg. stop in step 2-11
- User closes session and gets assigned another tasks by their boss
- User comes back later (a few days or weeks later) and wants to continue the previous task

My proposal: resume session

### Scenario 2: Very complex tasks require multiple sessions

- User works on a complex tasks that require several iterations of data generation (long running) and fine-tuning (long running) and evaluation (long running).
- A single session does not provide enough context length capacity to hold all the conversation history
- Note: there is a compaction/summarization mechanism to reduce the conversation history to a smaller size, but this is a lossy process
- This is scattered over several days or weeks

### Scenario 3: User wants to compare multiple approaches

- User completed the entire process
- User wants to try a different approach to the same task, eg. different scoring criteria, different data generation pipeline, different model architecture, etc.

### Scenario 4: User comes back with failure cases from deployed model

- User completed the entire process and deployed the model
- After several days or weeks, user comes back with failure cases from the deployed model and wants to improve the model based on these failure cases
- This could either be changes in the specifications, or additional data generation pipeline to cover the failure cases, or both.
- Typically, it is a combination of both: we update the specs but don't throw away the data but rather generate additional data to cover the failure cases.

### Scenario 5: User comes back with changes in the specifications

- User completed the entire process and deployed the model
- Similar to the failure cases scenario, but this time the user comes back with changes in the specifications, eg. new requirements, new constraints, instead of clear wrong input-output pairs.


## Notes:

- Data generation is very expensive, thus we want to avoid throwing away data. Instead a typical process would be generate additional data to cover the new requirements or failure cases. This means we need to keep track of the previous data generation pipeline and the generated data, so that we can generate additional data without throwing away the previous data.
- I think we have `remote_jobs.json` or something like this where we make sure lqh can reconnect to remote jobs (data gen in the cloud, fine-tuning etc). I think this is great. The main question of this document is whether we have something similar for the high-level concept of status of the project.

