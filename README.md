
<p align="center">
  <a href="https://youtu.be/suC4VlY8z6Y">
    <img src="misc/image.png" alt="Liquid Harness — watch the demo" />
  </a>
</p>

# lqh — Liquid Harness

**From zero to a fine-tuned model in under an hour.**

Liquid Harness is a terminal agent that turns a plain-English description of your task into a small, fast, task-specific [Liquid Foundation Model](https://www.liquid.ai/). You describe the problem; the agent interviews you, writes the specification, generates and curates training data, fine-tunes, evaluates, and iterates until the model beats baseline. No ML experience required.

> [!IMPORTANT]
> ⚠️ **Closed beta** — visit [lqh.ai](https://lqh.ai) to request access.

```bash
uv tool install lqh   # or: pip install lqh
lqh
```

---

## 🤔 Why would I want this?

Large general-purpose models are expensive, slow, and overkill for most production tasks. A 350M–1.2B model fine-tuned on *your* task is cheaper, faster, runs on-device, and often more accurate — but getting there normally requires an ML team: data pipelines, judges, training loops, eval harnesses.

Liquid Harness collapses all of that into a conversation:

1. **💬 Describe your task** — the agent asks clarifying questions and writes a spec.
2. **☕ Let it work** — it generates synthetic data, scores every sample against a rubric, filters, runs baselines, fine-tunes, and evaluates.
3. **📦 Get a checkpoint** — a model that measurably beats the baseline on your task, ready to deploy.

## 💡 What can you build?

A few things people fine-tune with lqh:

| Use case | You tell the agent… |
|---|---|
| 🎧 **Support-reply rewriter** | *"Rewrite draft replies into our brand voice: warm, concise, never over-promising."* |
| 🧾 **Structured extraction** | *"Turn free-form purchase emails into strict JSON with vendor, amount, and date."* |
| 🚦 **Classifier / router** | *"Label incoming tickets as billing, bug, or feature request — with high recall on billing."* |
| 🖼️ **Vision Q&A** | *"Here's a folder of product photos — I need a model that answers questions about defects."* |
| 📱 **On-device assistant** | *"A tiny model that summarizes meeting notes offline on a phone."* |

Point it at a folder of raw, unlabelled images and it can build a vision fine-tune too — the data pipeline synthesizes the question-answer pairs for you.

## 🚀 Quickstart

```bash
# One-time: install uv (skip if you already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv tool install lqh
mkdir my-task && cd my-task
lqh
```

`uv tool install` puts the `lqh` command on your PATH in its own isolated environment — no virtualenv to create or activate, and uv downloads a compatible Python automatically if your system one is too old. Upgrade later with `uv tool upgrade lqh`. On Windows, install uv with `winget install astral-sh.uv`.

If you already manage your own Python environments, `pipx install lqh` or `pip install lqh` work just as well.

Inside the TUI:

```
> /login          # authenticate with lqh.ai (one-time)
> I want a model that rewrites support replies into our brand voice.
```

That's it. The agent takes it from there — it interviews you about requirements, writes `SPEC.md`, and offers to run the pipeline stage by stage. You can interject, inspect data samples, or change direction at any point.

> [!TIP]
> Run `/hf_login` (or set `HF_TOKEN`) to enable private HuggingFace dataset access and publishing.

## 🔁 The pipeline

One command runs all nine stages. Each is a real artifact you can inspect, stop at, or hand off.

```
spec → rubric → data gen → filter → baseline → SFT → DPO → eval → checkpoint
```

```
$ lqh --auto ./my-task
[stage: rubric]            writing scorer from spec
[stage: data_gen_draft]    5 samples generated, all valid
[stage: filter_validation] 1,427 / 2,000 kept
[stage: sft_initial]       score 6.8/10  (baseline 4.1)
[stage: dpo]               iter 3/5, score 7.4/10
[final: success]           DPO checkpoint beats baseline by +3.3
```

## ✨ Features

### 💬 Fully agentic
You chat, the agent works. It captures requirements through dialogue, manages the specs, and drives every downstream stage end-to-end.

### 📝 Specify in plain English
No DSL, no boilerplate, no ML jargon. Just describe what you want — the agent turns it into structured specs you can refine over time.

### 🧪 Synthetic data, scored & filtered
The agent authors a per-task data generation pipeline, generates samples concurrently on LQH Cloud, and scores each one with an LLM judge against your rubric. The dataset that hits training is already curated.

### 🖼️ Vision fine-tuning
Bring a folder of raw images — no labels needed. The agent synthesizes an image-question-answer dataset and fine-tunes the LFM2.5-VL vision models on it.

### 🏋️ Train anywhere
Eval and data generation run on LQH Cloud. Training runs locally on your own GPUs, or hands off to a remote machine over SSH — including SLURM clusters — with dataset sync handled for you.

### 🤖 Hands-off `--auto` mode
Point lqh at a directory with a spec and walk away. It either delivers a checkpoint that beats baseline or returns an explicit failure with the reason — never a hang, never a prompt.

### 🤗 HuggingFace integration
Push and pull datasets from the Hub, publish checkpoints, and convert to GGUF for deployment.

### 🖥️ Interactive TUI
Guide the agent, visualize progress, and inspect dataset samples — all from one terminal session with a slash-command palette and a live status bar.

### 📦 Project-as-directory
Any directory is a project — fully git-compatible, so you can version, branch, and collaborate on specs, datasets, and runs like any other code. `cd` to switch projects.

## 🧬 Models

Fine-tunes the **LFM2.5** family — pick a size for your task, from tiny on-device to MoE:

| Size | Best for |
|---|---|
| 230M / 350M | Very simple tasks, extreme on-device constraints |
| **1.2B** ⭐ | Recommended starting point for most tasks |
| 2.6B / 8B-A1B (MoE) | More complex tasks |
| 24B-A2B | Only when the task clearly calls for it |
| LFM2.5-VL 450M / 1.6B | Vision (image + text) tasks |

Base, instruct, and thinking variants are available; the agent recommends the right starting size and variant for your task and steps up if fine-tuning struggles.

## ⌨️ Slash commands

| Command | What it does |
|---|---|
| `/login` | Log in to lqh.ai |
| `/hf_login` | Store a HuggingFace token for cloud jobs |
| `/spec` | Start specification capture |
| `/datagen` | Start data generation |
| `/validate` | Start data validation |
| `/train` | Start training (requires `torch`) |
| `/eval` | Start evaluation |
| `/prompt` | Start prompt optimization |
| `/resume` | Resume a previous conversation |
| `/clear` | Start a fresh conversation |
| `/reconnect` | Retry a failed network/API operation |
| `/feedback` | Send feedback (with the current conversation) to the lqh team |
| `/help` · `/quit` | Show commands · exit |

## 🤖 Auto mode

For CI, batch jobs, or when you just want a result:

```bash
lqh --auto ./my-task                # runs the full pipeline against ./my-task/SPEC.md
lqh --spec "use the smallest base model"   # sticky run-time directive (works in both modes)
```

Auto mode requires an existing `SPEC.md` (write one interactively first, or by hand). It runs rubric → data gen → filter → baseline → SFT → DPO → report without ever prompting, and always terminates with an explicit success or failure.

## 📁 Your project is just a directory

`cd` into any directory and run `lqh` — the agent reads what's there to understand current state. No init command, no project marker file.

```
my-task/
├── SPEC.md              # the heart of the project: what you want the model to do
├── other_specs/         # additional specs for edge cases or sub-requirements
├── data_gen/            # generated pipeline scripts
├── evals/               # eval definitions and results, versioned (v1/, v2/, ...)
├── datasets/            # generated and curated datasets as parquet (v1/, v2/, ...)
├── runs/                # training runs with checkpoints, logs, and configs
└── .lqh/                # conversation logs and permissions (add to .gitignore)
```

Everything is plain files — specs are markdown, pipelines are Python, datasets are parquet. Edit `SPEC.md` directly any time; the agent picks up your changes on the next turn.

## 🔧 Requirements

- Python 3.11+ (`uv tool install` fetches one automatically if needed)
- A Liquid Harness account ([request access](https://lqh.ai))
- Optional: `torch` + `transformers` for local fine-tuning
- Optional: `HF_TOKEN` for HuggingFace dataset sync

## 🗺️ Roadmap

Things we're actively building. Open an issue if you want to weigh in.

- **QAT with evals** — quantization-aware training (train against quantization noise so the deployed quantized model matches its full-precision score), paired with local evaluation on the quantized artifact (llama.cpp) so we measure exactly what ships. GGUF export already works.
- **Async runner** — run `lqh` asynchronously on LQH Cloud and connect from the web app or mobile, so long jobs keep going after you close the terminal.
- **Audio models (LFM2-Audio)** — fine-tuning and data support for our LFM2-Audio series of audio language models.

## 🤝 Contributing

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) before opening a pull request. **Random PRs will be rejected** — open an issue first and agree on the approach with a maintainer; security hot fixes are the one exception. See the contribution policy for details.

---

Made with care by [Liquid AI](https://www.liquid.ai/).
