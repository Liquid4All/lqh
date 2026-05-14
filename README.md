> **Closed beta** — visit [lqh.ai](https://lqh.ai) to request access.

<p align="center">
  <img src="misc/image.png" alt="Liquid Harness" />
</p>

# lqh — Liquid Harness

From zero to a fine-tuned LFM in under an hour.

Liquid Harness is a terminal agent that turns a plain-English task description into a deployable model. It writes the data pipeline, scores and filters samples, runs baselines, fine-tunes, and iterates. No ML experience required. Built and maintained by Liquid AI as the official way to customize [Liquid Foundation Models](https://www.liquid.ai/), powered by **LQH Cloud**.

```bash
pip install lqh
lqh
```

## ✨ Features

### 💬 Fully agentic
You chat, the agent works. It interviews you about your task, writes and manages the specifications, then drives every downstream stage end-to-end.

### 📝 Specify in plain English
No DSL, no boilerplate, no ML jargon. Just describe what you want the model to do — the agent captures requirements through dialogue and turns them into structured specs you can refine over time.

### 🧪 Synthetic data, scored & filtered
The agent authors a per-task data generation pipeline, generates samples concurrently on LQH Cloud, and scores each one with an LLM judge against your rubric. The dataset that hits training is already curated.

### 🏋️ Fine-tune locally or in the cloud
Eval and data generation run on LQH Cloud. Training can run locally on your own GPUs, or hand off to a beefier machine — just sync the dataset and continue.

### 🤗 HuggingFace integration
Push and pull datasets from the Hub. Set `HF_TOKEN` to enable private dataset access and dataset publishing.

### 🤖 Hands-off `--auto` mode
Point lqh at a directory and walk away. It either delivers a checkpoint that beats baseline or returns an explicit failure with the reason — never a hang, never a prompt.

### 🖥️ Interactive TUI
Provide input, guide the agent, visualize progress, and inspect dataset samples — all from a single terminal session with a slash-command palette and a live status bar.

### 📦 Project-as-directory
Any directory is a project — fully git-compatible, so you can version, branch, and collaborate on specs, datasets, and runs like any other code. `cd` to switch projects.

## 🚀 The pipeline

One command runs all nine stages. Each is a real component you can inspect, stop at, or hand off.

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

## 🔧 Requirements

- Python 3.10+
- A Liquid Harness account ([request access](https://lqh.ai))
- Optional: `torch` + `transformers` for local fine-tuning
- Optional: `HF_TOKEN` for HuggingFace dataset sync

## 🔐 Authentication

```
lqh
> /login
```

The CLI stores your token in `~/.lqh/config.json` and authenticates all requests to LQH Cloud.

## 🧬 Base model

Default: **LFM2-1.2B-Instruct** — small, capable, runs anywhere.

---

Made with care by [Liquid AI](https://www.liquid.ai/).
