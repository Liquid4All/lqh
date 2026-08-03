"""Category 5: Next Steps benchmark scenarios.

Tests whether the agent chooses the correct next action given different
project states.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tests.harness.scenarios import Scenario


_TRANSLATION_SPEC = """\
# Specification: Multi-Language Translation

## Overview
Translate input text into 5 languages: German, French, Spanish, English, and Chinese.
Output as a JSON object with keys: de, fr, es, en, zh.

## Input Format
- **Type**: Plain text, 1-5 sentences
- **Language**: Any language (auto-detected)

## Output Format
- **Type**: JSON object
- **Keys**: de, fr, es, en, zh

## Requirements
1. All 5 target languages must be present in every response
2. Translations must be accurate and natural
3. Handle informal text, slang, and idioms gracefully
"""

_PIPELINE_CODE = '''\
from lqh.pipeline import Pipeline, ChatMLMessage, Conversation, GenerationError, step
import json
import liquidrandom

class TranslationPipeline(Pipeline):
    """Generate translation training samples."""

    async def generate(self, client, input=None) -> Conversation:
        persona = liquidrandom.persona()

        resp = await client.chat.completions.create(
            model="random:small",
            messages=[{
                "role": "user",
                "content": f"Write a short sentence that a {persona.brief()} would write. Output ONLY the text.",
            }],
        )
        source_text = resp.choices[0].message.content.strip()

        resp = await client.chat.completions.create(
            model="random:medium",
            messages=[
                {"role": "system", "content": "Translate into German, French, Spanish, English, and Chinese. Return ONLY JSON with keys: de, fr, es, en, zh."},
                {"role": "user", "content": source_text},
            ],
            response_format={"type": "json_object"},
        )
        translations = resp.choices[0].message.content.strip()

        return [
            ChatMLMessage("user", source_text),
            ChatMLMessage("assistant", translations),
        ]
'''

_SCORER = """\
# Scorer: Translation Quality

## Task
Score the translation quality (de, fr, es, en, zh as JSON).

## Scoring Scale
- **9-10**: All 5 translations present, accurate, valid JSON
- **7-8**: All present with minor issues
- **5-6**: Valid JSON but some inaccurate
- **3-4**: Missing keys or multiple wrong
- **1-2**: Not valid JSON or mostly missing
"""

_PROMPT_V0 = (
    "Translate the following text into German, French, Spanish, English, "
    "and Chinese. Output ONLY a JSON object with keys: de, fr, es, en, zh."
)

_PROMPT_V1 = (
    "You are an expert multilingual translator. Translate the following text "
    "into German, French, Spanish, English, and Chinese. Preserve the original "
    "tone and formality level. Output ONLY a JSON object with keys: de, fr, es, en, zh. "
    "Ensure all translations are natural and idiomatic."
)


def _write_stub_parquet(path: Path, num_rows: int) -> None:
    """Write a minimal valid parquet file with stub translation data."""
    path.mkdir(parents=True, exist_ok=True)
    messages_list = []
    for i in range(num_rows):
        msgs = json.dumps([
            {"role": "user", "content": f"Sample text number {i + 1}."},
            {"role": "assistant", "content": json.dumps(
                {"de": f"Text {i+1}", "fr": f"Texte {i+1}", "es": f"Texto {i+1}",
                 "en": f"Sample text number {i+1}.", "zh": f"示例文本{i+1}"}
            )},
        ])
        messages_list.append(msgs)

    table = pa.table(
        {"messages": messages_list, "audio": [None] * num_rows, "tools": [None] * num_rows},
        schema=pa.schema([
            pa.field("messages", pa.string()),
            pa.field("audio", pa.string()),
            pa.field("tools", pa.string()),
        ]),
    )
    pq.write_table(table, path / "data.parquet")


def _seed_after_spec(project_dir: Path) -> None:
    """State: only SPEC.md exists."""
    (project_dir / "SPEC.md").write_text(_TRANSLATION_SPEC, encoding="utf-8")


def _seed_after_draft(project_dir: Path) -> None:
    """State: SPEC + pipeline + draft dataset (no scorer yet)."""
    (project_dir / "SPEC.md").write_text(_TRANSLATION_SPEC, encoding="utf-8")
    dg = project_dir / "data_gen"
    dg.mkdir(parents=True, exist_ok=True)
    (dg / "translation_v1.py").write_text(_PIPELINE_CODE, encoding="utf-8")
    _write_stub_parquet(project_dir / "datasets" / "translation_v1_draft", 20)


def _seed_after_eval(project_dir: Path) -> None:
    """State: SPEC + pipeline + eval dataset + scorer."""
    _seed_after_draft(project_dir)
    scorers = project_dir / "evals" / "scorers"
    scorers.mkdir(parents=True, exist_ok=True)
    (scorers / "translation_v1.md").write_text(_SCORER, encoding="utf-8")
    _write_stub_parquet(project_dir / "datasets" / "translation_v1_eval", 50)


def _seed_after_validation_scored(project_dir: Path) -> None:
    """State: SPEC + pipeline + scorer + a scored-but-UNFILTERED validation set.

    The validation set was generated and data-quality-scored (scores.parquet
    present) but never passed through ``run_data_filter``. Per the v0.3.1
    filter-before-eval/train gate, the correct next step here is to *filter* the
    generated set (run_data_filter with the scorer) before any baseline eval or
    training — not to evaluate or train on the raw generated data.
    """
    _seed_after_eval(project_dir)
    # A training-scale generated set that has been scored but not filtered.
    val_dir = project_dir / "datasets" / "translation_v1"
    _write_stub_parquet(val_dir, 200)
    # A sibling scores.parquet marks the set as already data-quality-scored,
    # so the only remaining gate before eval/train is the filter.
    _write_stub_parquet(project_dir / "datasets" / "_scores_tmp", 200)
    (project_dir / "datasets" / "_scores_tmp" / "data.parquet").replace(
        val_dir / "scores.parquet"
    )
    (project_dir / "datasets" / "_scores_tmp").rmdir()


def _seed_after_baseline(project_dir: Path) -> None:
    """State: SPEC + eval + baseline eval run + prompt."""
    _seed_after_eval(project_dir)
    prompts = project_dir / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "translation_v0.md").write_text(_PROMPT_V0, encoding="utf-8")

    # Baseline eval run
    run_dir = project_dir / "evals" / "runs" / "baseline_small"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({
        "mean_score": 5.8, "median_score": 6.0, "num_samples": 50,
        "model": "lfm2.5-1.2b-instruct", "system_prompt": "prompts/translation_v0.md",
    }), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({
        "dataset": "datasets/translation_v1_eval",
        "scorer": "evals/scorers/translation_v1.md",
        "model": "lfm2.5-1.2b-instruct",
    }), encoding="utf-8")


def _seed_after_prompt_opt(project_dir: Path) -> None:
    """State: SPEC + optimized prompt + improved eval."""
    _seed_after_baseline(project_dir)
    prompts = project_dir / "prompts"
    (prompts / "translation_v1.md").write_text(_PROMPT_V1, encoding="utf-8")

    # Improved eval run
    run_dir = project_dir / "evals" / "runs" / "prompt_v1_small"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({
        "mean_score": 7.4, "median_score": 8.0, "num_samples": 50,
        "model": "lfm2.5-1.2b-instruct", "system_prompt": "prompts/translation_v1.md",
    }), encoding="utf-8")

    # Training data at scale
    _write_stub_parquet(project_dir / "datasets" / "translation_v1", 200)


def _attach_scores(dataset_dir: Path, num_rows: int) -> None:
    """Write a sibling scores.parquet into ``dataset_dir`` marking it as already
    data-quality-scored (high scores), without running a real scorer."""
    tmp = dataset_dir.parent / "_scores_tmp"
    _write_stub_parquet(tmp, num_rows)
    (tmp / "data.parquet").replace(dataset_dir / "scores.parquet")
    tmp.rmdir()


def _seed_before_train_unfiltered(project_dir: Path) -> None:
    """State: prompt optimized, good eval, training-scale set generated AND
    data-quality-scored — but never filtered. The filter-before-train gate means
    the correct next step is run_data_filter, not start_training."""
    _seed_after_prompt_opt(project_dir)
    # _seed_after_prompt_opt writes a 200-row datasets/translation_v1 training set.
    _attach_scores(project_dir / "datasets" / "translation_v1", 200)


def _seed_ready_to_generate(project_dir: Path) -> None:
    """State: SPEC + pipeline + a validated scorer (scored on the draft) but NO
    full dataset yet. The next step is to generate the real dataset — and the
    careful move is to generate a SMALL batch first before scaling up."""
    _seed_after_draft(project_dir)  # SPEC + pipeline + draft(20)
    scorers = project_dir / "evals" / "scorers"
    scorers.mkdir(parents=True, exist_ok=True)
    (scorers / "translation_v1.md").write_text(_SCORER, encoding="utf-8")
    _attach_scores(project_dir / "datasets" / "translation_v1_draft", 20)


def _seed_unaudited_generated(project_dir: Path) -> None:
    """State: SPEC + pipeline + a freshly generated dataset with NO scores and NO
    scorer. Before using it for eval/train the agent should audit the data —
    create/run a data-quality scorer and inspect the samples."""
    (project_dir / "SPEC.md").write_text(_TRANSLATION_SPEC, encoding="utf-8")
    dg = project_dir / "data_gen"
    dg.mkdir(parents=True, exist_ok=True)
    (dg / "translation_v1.py").write_text(_PIPELINE_CODE, encoding="utf-8")
    _write_stub_parquet(project_dir / "datasets" / "translation_v1", 100)


def _seed_small_success(project_dir: Path) -> None:
    """State: SPEC + pipeline + scorer + a SMALL generated run (30 samples) that
    was data-quality-scored and looks good. The next step is to SCALE UP the
    dataset to a training-appropriate size."""
    (project_dir / "SPEC.md").write_text(_TRANSLATION_SPEC, encoding="utf-8")
    dg = project_dir / "data_gen"
    dg.mkdir(parents=True, exist_ok=True)
    (dg / "translation_v1.py").write_text(_PIPELINE_CODE, encoding="utf-8")
    scorers = project_dir / "evals" / "scorers"
    scorers.mkdir(parents=True, exist_ok=True)
    (scorers / "translation_v1.md").write_text(_SCORER, encoding="utf-8")
    smoke = project_dir / "datasets" / "translation_v1_smoke"
    _write_stub_parquet(smoke, 30)
    _attach_scores(smoke, 30)


def _seed_completed_training_run(
    project_dir: Path, dataset: str = "datasets/translation_v1"
) -> Path:
    """Write a realistic *completed* training run under ``runs/sft_v1``.

    The ``summary`` tool only scans ``runs/`` for training runs (it does NOT
    look at ``checkpoints/`` or ``training/``), and a finished checkpoint lives
    at ``runs/<name>/model-lora/`` with a ``lineage.json`` — see
    ``lqh/train/sft.py:_write_checkpoint_lineage``. Mirroring that layout is what
    makes the agent recognise that training already happened. Returns the run dir.
    """
    run = project_dir / "runs" / "sft_v1"
    run.mkdir(parents=True, exist_ok=True)
    config = {
        "run_name": "sft_v1",
        "base_model": "lfm2.5-1.2b",
        "dataset": dataset,
        "training": {"learning_rate": 2e-5, "num_epochs": 3, "max_seq_length": 2048},
        "lora": {"enabled": True, "r": 16, "alpha": 32, "dropout": 0.02},
    }
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run / "progress.jsonl").write_text(
        "".join(
            json.dumps({"step": s, "loss": round(1.2 - s * 0.004, 3)}) + "\n"
            for s in (1, 100, 200)
        ),
        encoding="utf-8",
    )
    model = run / "model-lora"
    model.mkdir(parents=True, exist_ok=True)
    (model / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "lfm2.5-1.2b", "r": 16, "lora_alpha": 32}),
        encoding="utf-8",
    )
    (model / "lineage.json").write_text(json.dumps({
        "artifact_kind": "checkpoint",
        "training_method": "lora",
        "base_model": "lfm2.5-1.2b",
        "hyperparams": {
            "learning_rate": 2e-5, "num_epochs": 3, "max_seq_length": 2048,
            "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.02, "lora_base": "lfm2.5-1.2b",
        },
        "parent_ids": [],
    }, indent=2), encoding="utf-8")
    return run


def _seed_finetune_failed(project_dir: Path) -> None:
    """State: a fine-tune COMPLETED and was evaluated, but its model-eval scores
    are POOR (mean ~4/10). The correct next step is to INSPECT the fine-tuned
    model's failing responses and their per-sample scores (e.g. get_eval_failures
    or by reading the eval results) to diagnose before retraining — NOT to
    blindly launch another training run."""
    _seed_after_eval(project_dir)  # SPEC + pipeline + scorer + eval set
    _seed_completed_training_run(project_dir)
    run_dir = project_dir / "evals" / "runs" / "finetuned_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    # summary.json uses the scores.mean shape the summary tool surfaces, so the
    # poor result is visible as "mean 4.1/10" in project state.
    (run_dir / "summary.json").write_text(json.dumps({
        "scores": {"mean": 4.1, "median": 4.0},
        "num_samples": 50,
        "model": "runs/sft_v1/model-lora", "mode": "model_eval",
        "system_prompt": "prompts/translation_v1.md",
    }), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({
        "eval_dataset": "datasets/translation_v1_eval",
        "scorer": "evals/scorers/translation_v1.md",
        "model": "runs/sft_v1/model-lora", "mode": "model_eval",
    }), encoding="utf-8")
    # Per-sample eval results so the failing responses + scores can be inspected.
    _write_stub_parquet(run_dir / "results", 50)


def _seed_post_sft_outcome(
    project_dir: Path,
    *,
    baseline_mean: float,
    post_sft_mean: float,
    train_rows: int,
    base_model: str,
) -> None:
    """State: full pre-training pipeline done, a fine-tune COMPLETED and was
    evaluated. The baseline eval, the post-SFT eval, the base-model size, and
    the training-set row count are all visible — the failure_analysis routing
    inputs. Scenarios vary the four knobs to plant different correct answers."""
    _seed_after_eval(project_dir)  # SPEC + pipeline + scorer + eval set
    prompts = project_dir / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "translation_v0.md").write_text(_PROMPT_V0, encoding="utf-8")

    # Training set at the story's scale, scored + "filtered" (scores attached).
    train_ds = project_dir / "datasets" / "translation_v1"
    _write_stub_parquet(train_ds, train_rows)
    _attach_scores(train_ds, train_rows)

    run = _seed_completed_training_run(project_dir)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    config["base_model"] = base_model
    config["dataset_rows"] = {
        "train": train_rows, "train_effective": train_rows, "eval": 50,
    }
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Post-SFT eval of the checkpoint (run-root eval_result.json, the shape
    # training_status renders as "Final eval").
    (run / "eval_result.json").write_text(json.dumps({
        "num_scored": 50, "num_failed": 0,
        "scores": {"mean": post_sft_mean, "median": post_sft_mean},
        "scores_weighted_mean": post_sft_mean,
        "per_source": {"all": {"num_scored": 50, "scores": {
            "mean": post_sft_mean, "median": post_sft_mean,
            "std": 1.4, "min": max(1.0, post_sft_mean - 3),
            "max": min(10.0, post_sft_mean + 2),
        }}},
    }), encoding="utf-8")

    # Zero-shot baseline of the same base model on the same eval set.
    baseline = project_dir / "evals" / "runs" / "baseline_zero_shot"
    baseline.mkdir(parents=True, exist_ok=True)
    (baseline / "summary.json").write_text(json.dumps({
        "scores": {"mean": baseline_mean, "median": baseline_mean},
        "num_samples": 50, "model": base_model, "mode": "model_eval",
        "system_prompt": "prompts/translation_v0.md",
    }), encoding="utf-8")
    (baseline / "config.json").write_text(json.dumps({
        "eval_dataset": "datasets/translation_v1_eval",
        "scorer": "evals/scorers/translation_v1.md",
        "model": base_model, "mode": "model_eval",
    }), encoding="utf-8")


def _seed_sft_flat_small_dataset(project_dir: Path) -> None:
    """SFT did not improve (5.5 → 5.6) and the training set is only 1.5k rows.
    Per the failure_analysis routing, the FIRST check is dataset size: below
    ~2k, grow the data before blaming the model."""
    _seed_post_sft_outcome(
        project_dir,
        baseline_mean=5.5, post_sft_mean=5.6,
        train_rows=1500, base_model="lfm2.5-350m",
    )


def _seed_sft_flat_small_model(project_dir: Path) -> None:
    """SFT did not improve (5.5 → 5.6) with an ALREADY-sufficient dataset
    (4k rows) on a tiny model (350M). The routing says: dataset size checks
    out, so step UP the model size (zero-shot a larger model and/or train it
    on the same data) — not more data at 350M."""
    _seed_post_sft_outcome(
        project_dir,
        baseline_mean=5.5, post_sft_mean=5.6,
        train_rows=4000, base_model="lfm2.5-350m",
    )


def _seed_sft_improved_scale_data(project_dir: Path) -> None:
    """SFT clearly improved (4.2 → 6.0) from a modest 2k-row set on a 1.2B
    model. The routing's first lever when training works but isn't good
    enough: SCALE THE DATA (10-20k) before touching model size or DPO."""
    _seed_post_sft_outcome(
        project_dir,
        baseline_mean=4.2, post_sft_mean=6.0,
        train_rows=2000, base_model="lfm2.5-1.2b-instruct",
    )


def _seed_sft_good_offer_deploy(project_dir: Path) -> None:
    """Post-SFT eval is strong (8.4/10, up from 5.0). Per the outcome bands,
    the agent should OFFER the user deployment (push_to_production /
    create_inference_key) or GGUF export — not silently keep training."""
    _seed_post_sft_outcome(
        project_dir,
        baseline_mean=5.0, post_sft_mean=8.4,
        train_rows=10000, base_model="lfm2.5-1.2b-instruct",
    )


def _seed_probe_scored_for_inspection(project_dir: Path) -> None:
    """State: the quantitative levers are exhausted — a 1.2B model trained on
    20k rows improved 4.0 → 6.2 but is stuck — and a PROBE-set eval has just
    completed with per-sample results showing extreme low outliers (most
    samples 7-9, a cluster at 1-3). The next step per the failure_analysis
    skill is the qualitative loop: read the distribution shape, inspect ~20
    low scorers (get_eval_failures), and write a findings report under
    reports/ that names the failure patterns and plans targeted data."""
    _seed_post_sft_outcome(
        project_dir,
        baseline_mean=4.0, post_sft_mean=6.2,
        train_rows=20000, base_model="lfm2.5-1.2b-instruct",
    )
    # Probe dataset (separate from train/validation) + its completed eval.
    _write_stub_parquet(project_dir / "datasets" / "translation_probe_v1", 200)
    probe_run = project_dir / "runs" / "probe_eval_v1"
    probe_run.mkdir(parents=True, exist_ok=True)

    scores: list[float] = []
    messages: list[str] = []
    reasons: list[str] = []
    for i in range(200):
        if i < 20:
            # The planted pattern: idiomatic/slang inputs collapse to a
            # literal word-by-word translation.
            score = float(1 + i % 3)
            reasons.append(
                "Slang/idiomatic source translated word-by-word; meaning lost "
                "in all five languages."
            )
            user = f"Idiomatic sample {i}: 'spill the beans already, mate!'"
        else:
            score = float(7 + i % 3)
            reasons.append("All five translations present and natural.")
            user = f"Plain sample {i}: 'The meeting starts at nine.'"
        scores.append(score)
        messages.append(json.dumps([
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(
                {"de": "...", "fr": "...", "es": "...", "en": "...", "zh": "..."}
            )},
        ]))
    pq.write_table(
        pa.table({
            "sample_index": list(range(200)),
            "messages": messages,
            "score": scores,
            "reasoning": reasons,
        }),
        probe_run / "results.parquet",
    )
    (probe_run / "eval_result.json").write_text(json.dumps({
        "num_scored": 200, "num_failed": 0,
        "scores": {"mean": 7.2, "median": 8.0},
        "scores_weighted_mean": 7.2,
        "per_source": {"all": {"num_scored": 200, "scores": {
            "mean": 7.2, "median": 8.0, "std": 2.4, "min": 1.0, "max": 9.0,
        }}},
        "score_distribution": {
            "n": 200,
            "percentiles": {"p10": 3.0, "p25": 7.0, "p50": 8.0,
                            "p75": 8.0, "p90": 9.0},
            "histogram": {"0": 0, "1": 7, "2": 7, "3": 6, "4": 0, "5": 0,
                          "6": 0, "7": 60, "8": 60, "9": 60, "10": 0},
        },
    }), encoding="utf-8")
    (probe_run / "config.json").write_text(json.dumps({
        "type": "eval_hf", "base_model": "lfm2.5-1.2b-instruct",
        "eval_dataset": "datasets/translation_probe_v1",
        "scorer": "evals/scorers/translation_v1.md",
    }), encoding="utf-8")


def _seed_after_train_success(project_dir: Path) -> None:
    """State: data generated, prompt optimized, and a fine-tune has completed
    successfully (a run is present under runs/ with a final model-lora
    checkpoint) — but the checkpoint has NOT been evaluated yet (no model_eval
    run against it). The next step is to evaluate the fine-tuned checkpoint to
    confirm the improvement before doing anything else."""
    _seed_after_prompt_opt(project_dir)
    _seed_completed_training_run(project_dir)


_PASSIVE_USER = (
    "You are a passive user who follows the agent's suggestions. "
    "You want to continue making progress on the project.\n\n"
    "Behavior rules:\n"
    "- When the agent suggests a next step, agree and say 'sounds good, go ahead'\n"
    "- When asked to choose, pick whatever the agent recommends\n"
    "- After the agent takes the next action, say 'I'm done for now'\n"
    "- Do NOT suggest specific actions yourself"
)


NEXT_AFTER_SPEC = Scenario(
    name="bench_next_after_spec",
    description=_PASSIVE_USER,
    initial_message="What should we do next with this project?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    judge_criteria="data_generation",  # Expected next step (used by scorer)
    max_turns=20,
    stage_limits={"data_generation": 15},
    seed_fn=_seed_after_spec,
)

NEXT_AFTER_DRAFT = Scenario(
    name="bench_next_after_draft",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    judge_criteria="data_generation",  # Should create scorer + full eval set
    max_turns=20,
    stage_limits={"data_generation": 15},
    seed_fn=_seed_after_draft,
)

NEXT_AFTER_VALIDATION = Scenario(
    name="bench_next_after_validation",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # v0.3.1 filter-before gate: a scored-but-unfiltered generated set must be
    # filtered (run_data_filter) before eval/train.
    judge_criteria="data_filtering",
    max_turns=20,
    stage_limits={"data_filtering": 15},
    seed_fn=_seed_after_validation_scored,
)

NEXT_AFTER_EVAL = Scenario(
    name="bench_next_after_eval",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # The seeded eval set is pipeline-generated and unfiltered. Either running
    # the model eval OR first filtering the eval set (filter-before-eval) is a
    # correct next move per the current evaluation skill.
    judge_criteria="evaluation,data_filtering",
    max_turns=20,
    stage_limits={"evaluation": 15, "data_filtering": 15},
    seed_fn=_seed_after_eval,
)

NEXT_AFTER_BASELINE = Scenario(
    name="bench_next_after_baseline",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # evaluation/SKILL.md lists BOTH "optimize the system prompt" and "generate
    # training data and fine-tune" as valid next steps after a baseline eval.
    judge_criteria="prompt_optimization,train",
    max_turns=20,
    stage_limits={"prompt_optimization": 15, "train": 15},
    seed_fn=_seed_after_baseline,
)

NEXT_AFTER_PROMPT_OPT = Scenario(
    name="bench_next_after_prompt_opt",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    judge_criteria="train",  # Should suggest training
    max_turns=20,
    stage_limits={"train": 15},
    seed_fn=_seed_after_prompt_opt,
)


NEXT_FILTER_BEFORE_TRAIN = Scenario(
    name="bench_next_filter_before_train",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # Training-scale set is generated and scored but NOT filtered. The
    # filter-before-train gate means the correct next step is run_data_filter,
    # not jumping straight into start_training.
    judge_criteria="data_filtering",
    max_turns=20,
    stage_limits={"data_filtering": 15},
    seed_fn=_seed_before_train_unfiltered,
)

NEXT_ZEROSHOT_SYSTEM_PROMPT = Scenario(
    name="bench_next_zeroshot_system_prompt",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # SPEC + eval set + scorer, no baseline yet -> run a zero-shot baseline eval.
    judge_criteria="evaluation",
    next_step_quality_criteria=(
        "The project has an eval dataset and a scorer but no baseline run yet, so "
        "the next step is a zero-shot baseline evaluation of the base model. Rate "
        "(1-10) whether the agent supplied a real system prompt / task instructions "
        "(derived from SPEC.md) for the model being evaluated, rather than running "
        "the eval with an empty or missing system prompt. "
        "10 = clearly authored or passed a task system prompt for the baseline eval; "
        "6 = ran the eval but it is unclear whether any instructions were given; "
        "1 = ran with no system prompt at all, or never set up the eval."
    ),
    max_turns=20,
    stage_limits={"evaluation": 15},
    seed_fn=_seed_after_eval,
)

NEXT_SMALL_BEFORE_SCALE = Scenario(
    name="bench_next_small_before_scale",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # Validated scorer, no full dataset yet -> generate data.
    judge_criteria="data_generation",
    next_step_quality_criteria=(
        "The pipeline and a validated scorer are ready but no full dataset exists "
        "yet. The careful next step is to generate a SMALL batch first (on the "
        "order of tens of samples) to sanity-check before committing to a large, "
        "expensive run. Rate (1-10) whether the agent generated/requested a small "
        "initial batch rather than immediately generating thousands of samples. "
        "10 = explicitly generated a small batch first (<= ~100) and said it would "
        "scale up after checking; 5 = generated a moderate amount without a "
        "small-first rationale; 1 = jumped straight to a large run or generated "
        "nothing."
    ),
    max_turns=20,
    stage_limits={"data_generation": 15},
    seed_fn=_seed_ready_to_generate,
)

NEXT_AUDIT_GENERATED_DATA = Scenario(
    name="bench_next_audit_generated_data",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # A freshly generated, unscored dataset exists -> audit it (create/run a
    # data-quality scorer, inspect samples) before eval/train.
    judge_criteria="data_generation",
    next_step_quality_criteria=(
        "A dataset was just generated but has not been quality-checked (no scorer, "
        "no scores). The correct next step is to AUDIT the data before using it: "
        "create/run a data-quality scorer and/or inspect the generated samples. "
        "Rate (1-10) whether the agent moved to audit/inspect the data quality "
        "rather than blindly proceeding to training or evaluation. "
        "10 = clearly set up a data-quality check or inspected samples; 5 = "
        "mentioned quality but did not act; 1 = skipped straight to train/eval or "
        "did something unrelated."
    ),
    max_turns=20,
    stage_limits={"data_generation": 15},
    seed_fn=_seed_unaudited_generated,
)

NEXT_SCALE_AFTER_SUCCESS = Scenario(
    name="bench_next_scale_after_success",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # A small generation run succeeded and scored well -> scale up the dataset.
    judge_criteria="data_generation",
    next_step_quality_criteria=(
        "A small generation run (about 30 samples) succeeded and scored well on "
        "the data-quality scorer. The next step is to SCALE UP generation to a "
        "training-appropriate size. Rate (1-10) whether the agent scaled up the "
        "dataset (requested a substantially larger generation, e.g. hundreds or "
        "more) rather than re-running another tiny batch or jumping to training on "
        "just 30 samples. "
        "10 = explicitly scaled generation up to a training-sized run; 5 = "
        "continued generating but without clearly scaling up; 1 = trained on the "
        "tiny set or did nothing useful."
    ),
    max_turns=20,
    stage_limits={"data_generation": 15},
    seed_fn=_seed_small_success,
)

NEXT_INSPECT_ON_FAILURE = Scenario(
    name="bench_next_inspect_on_failure",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # A fine-tune was evaluated and scored poorly. The correct next step is to
    # inspect the failing responses + scores to diagnose, NOT to blindly retrain.
    # Inspecting failures is surfaced via get_eval_failures (prompt_optimization
    # bucket); re-running the eval is also acceptable diagnosis.
    judge_criteria="prompt_optimization,evaluation",
    next_step_quality_criteria=(
        "A fine-tuned checkpoint was evaluated and scored poorly (mean ~4/10). The "
        "correct next step is to DIAGNOSE before acting: inspect the fine-tuned "
        "model's actual generated responses and their per-sample scores (e.g. via "
        "get_eval_failures or by reading the eval results) to understand WHY it "
        "underperformed. Rate (1-10) whether the agent inspected the failing "
        "responses/scores to diagnose, rather than immediately launching another "
        "training run or regenerating data without looking. "
        "10 = clearly inspected the failing responses and their scores to diagnose; "
        "5 = acknowledged the poor result but did not actually look at failures; "
        "1 = blindly retrained / regenerated, or ignored the poor result."
    ),
    max_turns=20,
    stage_limits={"prompt_optimization": 15, "evaluation": 15},
    seed_fn=_seed_finetune_failed,
)

NEXT_AFTER_TRAIN_SUCCESS = Scenario(
    name="bench_next_after_train_success",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # Fine-tune completed but the checkpoint has not been evaluated yet ->
    # evaluate the fine-tuned model to confirm the improvement.
    judge_criteria="evaluation",
    max_turns=20,
    stage_limits={"evaluation": 15},
    seed_fn=_seed_after_train_success,
)


NEXT_SFT_FLAT_GROW_DATA = Scenario(
    name="bench_next_sft_flat_grow_data",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # SFT flat (5.5→5.6) with only 1.5k training rows → grow the dataset
    # to ≥2k first (failure_analysis routing, "no improvement" branch (a)).
    judge_criteria="data_generation,failure_analysis",
    next_step_quality_criteria=(
        "A fine-tune on a 1,500-row training set did NOT improve over the "
        "zero-shot baseline (5.5 → 5.6). The correct first move is to check the "
        "training-set size and GROW THE DATASET to at least ~2k samples before "
        "blaming the model — 2-5k good samples should show measurable movement. "
        "Rate (1-10) whether the agent identified the small dataset as the "
        "first suspect and moved to scale data generation. "
        "10 = explicitly reasoned about the 1.5k size and scaled generation to "
        "2k+; 5 = scaled data without connecting it to the size check, or only "
        "talked about it; 1 = retrained the same data, jumped to a bigger model "
        "or DPO, or ignored the flat result."
    ),
    max_turns=20,
    stage_limits={"data_generation": 15, "failure_analysis": 15},
    seed_fn=_seed_sft_flat_small_dataset,
)

NEXT_SFT_FLAT_BIGGER_MODEL = Scenario(
    name="bench_next_sft_flat_bigger_model",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # SFT flat (5.5→5.6) but the dataset is already 4k rows on a 350M model →
    # step up the model size (zero-shot the larger model and/or train it).
    judge_criteria="evaluation,train,failure_analysis",
    next_step_quality_criteria=(
        "A fine-tune of the 350M model on an already-sufficient 4,000-row "
        "training set did NOT improve over baseline (5.5 → 5.6). Dataset size "
        "checks out, so the correct next move is to STEP UP THE MODEL SIZE "
        "(e.g. 1.2B or 2.6B): zero-shot the larger model to gauge headroom "
        "and/or train it on the same dataset. Rate (1-10) whether the agent "
        "moved to a larger model rather than generating more data for the 350M. "
        "10 = explicitly picked a larger model (eval or train) with the "
        "capacity rationale; 5 = mentioned model size but kept working at 350M; "
        "1 = generated more data, retrained 350M, or ignored the flat result."
    ),
    max_turns=20,
    stage_limits={"evaluation": 15, "train": 15, "failure_analysis": 15},
    seed_fn=_seed_sft_flat_small_model,
)

NEXT_SFT_IMPROVED_SCALE_DATA = Scenario(
    name="bench_next_sft_improved_scale_data",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # SFT improved (4.2→6.0) from 2k rows → the first lever is scaling the
    # data (10-20k), not model size or DPO.
    judge_criteria="data_generation,failure_analysis",
    next_step_quality_criteria=(
        "A fine-tune improved clearly over baseline (4.2 → 6.0) from a 2,000-row "
        "training set, but 6.0 is not good enough yet. Training works, so the "
        "first lever is to SCALE THE TRAINING DATA (e.g. to 10-20k samples with "
        "the same pipeline) and retrain — before changing model size or trying "
        "DPO. Rate (1-10) whether the agent scaled data generation as the next "
        "step. 10 = explicitly scaled the dataset to ~10k+ citing the working "
        "trend; 5 = generated somewhat more data or hedged between levers; "
        "1 = jumped to a bigger model or DPO first, or stopped."
    ),
    max_turns=20,
    stage_limits={"data_generation": 15, "failure_analysis": 15},
    seed_fn=_seed_sft_improved_scale_data,
)

NEXT_SFT_GOOD_OFFER_DEPLOY = Scenario(
    name="bench_next_sft_good_offer_deploy",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # Post-SFT 8.4/10 → offer the user deployment (API or GGUF export) or
    # continued improvement — the ≥8 outcome band.
    judge_criteria="deployment,failure_analysis",
    next_step_quality_criteria=(
        "The fine-tuned checkpoint scores 8.4/10 (baseline was 5.0) — a strong "
        "result. Per the outcome bands the agent should tell the user the model "
        "is good and OFFER the choice: deploy it as an API (push_to_production "
        "+ create_inference_key), export it for on-device use (gguf_convert), "
        "or keep improving. Rate (1-10) whether the agent surfaced the "
        "deployment/export offer. 10 = clearly presented deploy/export options "
        "(e.g. via ask_user) or started deployment on the user's yes; 5 = noted "
        "the good score but offered no concrete next step; 1 = kept training/"
        "generating as if the result were poor, or ignored the score."
    ),
    max_turns=20,
    stage_limits={"failure_analysis": 15},
    seed_fn=_seed_sft_good_offer_deploy,
)


NEXT_PROBE_QUALITATIVE_LOOP = Scenario(
    name="bench_next_probe_qualitative_loop",
    description=_PASSIVE_USER,
    initial_message="What should we do next?",
    expected_tools=["summary"],
    expected_files=["SPEC.md"],
    # Quantitative levers exhausted; a probe eval with per-sample results and
    # an outlier-heavy distribution is ready → run the qualitative loop:
    # inspect ~20 low scorers, write a findings report, plan targeted data.
    judge_criteria="failure_analysis,prompt_optimization",
    next_step_quality_criteria=(
        "The model (1.2B, trained on 20k rows, improved 4.0 → 6.2) is stuck "
        "after the quantitative levers, and a probe-set eval just completed at "
        "runs/probe_eval_v1 with per-sample results whose distribution shows "
        "extreme low outliers (a 1-3 cluster under a 7-9 majority). The correct "
        "next step is the QUALITATIVE failure-analysis loop: read the "
        "distribution shape, inspect on the order of 20 low-scoring samples "
        "with their judge reasoning (get_eval_failures on runs/probe_eval_v1), "
        "identify the common failure pattern (idiomatic/slang inputs "
        "translated literally), and write a findings report (e.g. "
        "reports/failure_analysis_v1.md) that plans targeted supplemental "
        "training data for that pattern. Rate (1-10): "
        "10 = inspected a meaningful batch of low scorers, named the "
        "idiom/slang pattern, and wrote a report planning targeted data; "
        "6 = inspected failures but produced no report/plan, or wrote a plan "
        "without actually reading the samples; "
        "1 = ignored the probe results and jumped to more training/data "
        "generation, or did nothing relevant."
    ),
    max_turns=25,
    stage_limits={"failure_analysis": 20},
    seed_fn=_seed_probe_scored_for_inspection,
)


SCENARIOS = [
    NEXT_AFTER_SPEC,
    NEXT_AFTER_DRAFT,
    NEXT_AFTER_VALIDATION,
    NEXT_AFTER_EVAL,
    NEXT_AFTER_BASELINE,
    NEXT_AFTER_PROMPT_OPT,
    NEXT_FILTER_BEFORE_TRAIN,
    NEXT_ZEROSHOT_SYSTEM_PROMPT,
    NEXT_SMALL_BEFORE_SCALE,
    NEXT_AUDIT_GENERATED_DATA,
    NEXT_SCALE_AFTER_SUCCESS,
    NEXT_INSPECT_ON_FAILURE,
    NEXT_AFTER_TRAIN_SUCCESS,
    NEXT_SFT_FLAT_GROW_DATA,
    NEXT_SFT_FLAT_BIGGER_MODEL,
    NEXT_SFT_IMPROVED_SCALE_DATA,
    NEXT_SFT_GOOD_OFFER_DEPLOY,
    NEXT_PROBE_QUALITATIVE_LOOP,
]
