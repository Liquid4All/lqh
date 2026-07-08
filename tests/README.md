# LQH test suite

Four test categories plus shared infrastructure. Rule of thumb: `tests/unit/`
runs everywhere and is free *in CI and for logged-out environments*; the
other suites talk to the live platform and need `/login` (or
`LQH_DEBUG_API_KEY`) — and the expensive suites have additional opt-in gates
so nothing spends serious money by accident.

| Suite | What it covers | Cost / duration | How to run |
|---|---|---|---|
| `unit/` | Python code executes correctly | free without auth; a few small API calls when logged in | `pytest` (default) |
| `function/` | One workflow per test (spec capture, datagen, scoring, local/cloud training, GGUF) | small LLM/GPU spend, minutes–hours | `pytest tests/function` |
| `e2e/` | Entire app flow with an LLM-simulated user | real training money, hours | `LQH_E2E=1 pytest tests/e2e` |
| `benchmarks/` | Compare LQH components against each other | large, hours–days | `python -m` runners (below) |

## unit/ — unit tests

Python correctness tests. `pytest` (bare) runs exactly this suite —
`testpaths` points at `tests/unit` so a dev can't accidentally trigger the
expensive suites.

Markers (auto-skip via `tests/conftest.py` when the environment is missing):

- `@pytest.mark.integration` — hits api.lqh.ai; skipped without auth. When
  you *are* logged in these run and spend a small amount on LLM calls.
- `@pytest.mark.gpu` — needs CUDA + `pip install lqh[train]`; skipped
  without. On a GPU box these run real (tiny) training steps.

CI (`.github/workflows/ci.yml`) runs `pytest tests/unit` on Python
3.11–3.13 (markers skip there: no auth, no GPU), a
`pytest tests --collect-only` import check of every pytest suite, and
`--help` import checks of the standalone benchmark/experiment runners.

## function/ — workflow smoke tests

Each test drives one workflow end to end at smoke scale (tiny datasets).
Many use the `tests/harness/` simulated-human agent loop for a single stage
(`test_spec_only`, `test_spec_and_datagen`, `test_translation`, …); others
exercise a workflow directly (`test_cloud_sft_smoke`, `test_training_e2e`).
`test_full_pipeline.py` / `test_full_pipeline_tools.py` /
`test_cloud_finetune_agent_e2e.py` run the agent loop against deterministic
fakes — free and always-on agent-contract checks.

Requirements vary per file (documented in each docstring):

- Platform login for the live-API tests (`/login` or `LQH_DEBUG_API_KEY`).
- `LQH_E2E=1` for the cloud training smokes (`test_cloud_*`) — they spend
  real GPU money.
- CUDA + `lqh[train]` for the local-GPU training smokes
  (`test_training_e2e`, `test_vlm_training_e2e`).
- `--remote-host=<ssh-host>` or `LQH_TEST_REMOTE_HOST` for the
  `test_remote_*` / `test_sft_and_upload` tests (real SSH GPU box).
- `LQH_TEST_DATABASE_URL` for `test_job_token_scope`.
- A stored HF token (`/hf_login`) for upload tests.

Without the gates the tests skip, so `pytest tests/function` is always safe
to invoke.

## e2e/ — holistic app-flow tests

`tests/harness/` drives the real `lqh.agent.Agent` loop with an
LLM-simulated human answering `ask_user` calls. These tests cover the whole
product (spec → datagen → train → export); only the UI itself is out of
scope. Every test here is gated on auth **and** `LQH_E2E=1`, and can run
for hours.

- `test_auto_mode_e2e.py` — auto mode: spec in, trained model out
- `test_observability_gguf.py` — full flow from scratch: observability
  action model on LFM2.5-350M, exported to GGUF Q4_K (asserts the artifact
  registers in the backend store)

Each test also runs standalone with model/timeout args, e.g.
`LQH_E2E=1 python -m tests.e2e.test_observability_gguf orchestration:12 --timeout=7200`.
Run reports (markdown + JSON) land in `tests/harness/reports/` (gitignored;
`git add -f` to keep one as history).

## benchmarks/ — component comparisons

Not pytest — standalone sweep runners that produce score reports.

- **orchestration/** — compares orchestration models across scenario
  categories (spec capture, datagen pipeline, error recovery, …):
  `python -m tests.benchmarks.orchestration.runner --models orchestration:1,orchestration:12`
  Results in `tests/benchmarks/orchestration/results/`. Needs auth; the
  `auto_mode` category is excluded by default (expensive).
- **base_vs_instruct/** — which LFM2.5 variant fine-tunes best (local GPU
  training, judge scoring via API): `python -m tests.benchmarks.base_vs_instruct.run`
  — see its README for flags.

## Shared infrastructure

- `harness/` — `E2EHarness` (agent loop + simulated human), `Scenario`
  definitions, LLM judge, report generation. Imported by function tests,
  e2e tests, and every benchmark category.
- `experiments/` — one-off research drivers (`experiment_*.py`) and their
  task projects (`projects/`), e.g.
  `python -m tests.experiments.experiment_e2e_pipeline --task translation --remote-host=<host>`.
  Validation studies, not regression tests.
- `conftest.py` — shared fixtures (auth probes, project dirs, ChatML
  parquet builders, OpenAI doubles) and the marker auto-skip hook.
- `fixtures/` — shared data fixtures (e.g. debug images for VLM tests).

## Coverage notes (as of 2026-07)

Modules with no direct unit tests worth backfilling: `lqh/watcher.py`,
`lqh/project_log.py`, `lqh/project_meta.py`, `lqh/context_stats.py`; thin
coverage on `lqh/golden.py`, `lqh/artifacts.py`, `lqh/env_secrets.py`,
`lqh/models.py`, `lqh/cli.py`. `lqh/agent.py`'s loop is exercised via the
harness suites rather than unit tests.
