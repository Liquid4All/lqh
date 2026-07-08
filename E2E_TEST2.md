# LQH end to end tests

We want to have better end to end testing of LQH.

## Current status

What we currently have:
- Some older e2e tests that are a bit dated
- Some "e2e" tests that test some specific functionality but are not really end to end tests in the holistic app sense
- Some unit tests
- Some benchmarks

All of these things are relatively unstructured right now. Your task is to refactor and improve the testing framework of LQH in a systematic way.

## Goal

- Keep unit tests
- The github unit tests seem broken btw. Let's fix them if possible
- Keep the function specific e2e tests but maybe let's rename them to "function tests" or something like that
- Rework the existing e2e tests to be more holistic and cover the entire app flow
- Keep and extend the benchmarks

After the refactoring, we should have the unit tests, the function tests, the e2e tests and the benchmarks in a more structured way.

## Unit tests

Already working well. Should test for python code that executes correctly.
I think there is little things to do but let's double check that we have a decent coverage of the code base.

Bonus: let's try to get the github unit tests working again. I think they are broken because of some dependency issues.

TLDR: Unit tests cover python code.

## Function tests

Should test some functions such as SFT cloud fine-tuning working.
These tests should cover entire workflows, eg. spec capture, data gen, scoring, fine-tuning, model export, etc.
Note: These tests may run a bit (few minutes to hours) and require to be logged in to the platform, however, should be mostly smoke tests in the sense that for instance the dataset size is tiny.

TLDR: Function tests cover workflows.

## E2E tests

These are actual tests where we simulate a user using an orchestration model of LQH.
The only thing that is not covered here is the actual UI.

Examples: simulate a user who wants to create a observability model that monitors processes and CPU and memory usage, and informs the user or kills the process depending on the instructions. The model size should be LFM2.5 350M and be exported to gguf with Q4_K

Note: These tests may run for a few hours to days and require to be logged in to the platform.

TLDR: E2E tests cover the entire app flow.

## Benchmarks

Benchmarks should compare the functionality of different components of LQH.
We currently have already the orchestration benchmark that compares the performance of different orchestration models on the different tasks.
We also have a benchmark that compares the base vs instruct vs old base models, and models of different sizes on a task but it lives currently as "e2e" test.

We want to refactor this to have maybe a benchmark folder with different benchmarks for different components of LQH.
In the future we may add more such as different scoring methods, prompts or models, or different quantization methods, etc.

For now, we should have the orchestration one, and the base vs instrcut vs old base one, and maybe more if they already exist in the code base.

TLDR: Benchmarks compare the performance of different components of LQH.
