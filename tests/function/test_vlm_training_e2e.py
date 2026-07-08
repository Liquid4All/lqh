"""End-to-end VLM training smoke test: LoRA SFT of LFM2.5-VL-450M on the
PIL-generated debug shapes dataset (VLM.md workflow test).

Requires:
    - CUDA GPU (~24 GB is plenty; the 450M model is small)
    - ``pip install lqh[train]``
    - Internet access (downloads the model from HuggingFace)

Run via pytest::

    pytest tests/function/test_vlm_training_e2e.py -v -s

Gated on ``@pytest.mark.gpu``; ``conftest.py`` auto-skips without CUDA.

No API access is needed: the image-QA pairs are synthesized directly from
the debug images' ground-truth labels (counting / color / text questions),
which also lets the test sanity-check the fine-tuned model's answers.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.fixtures.debug_images import generate_debug_images

EVAL_SIZE = 4
TIMEOUT_S = 1800  # 30 min cap


def _data_url(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _qa_for(record: dict) -> tuple[str, str]:
    """Ground-truth question/answer for one debug image record."""
    if record["text"]:
        return "What text is written in this image?", record["text"]
    if record["count"] > 1:
        return (
            "How many shapes are in this image? Answer with a number.",
            str(record["count"]),
        )
    return (
        "What color is the shape in this image? Answer with one word.",
        record["color"],
    )


def _write_vlm_parquet(records: list[dict], images_dir: Path, out: Path) -> None:
    rows = []
    for rec in records:
        question, answer = _qa_for(rec)
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _data_url(images_dir / rec["file"])}},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": answer},
        ]
        rows.append(json.dumps(conv))
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"messages": rows}, schema=pa.schema([pa.field("messages", pa.string())])),
        out,
    )


@pytest.fixture(scope="module")
def vlm_workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    project = tmp_path_factory.mktemp("lqh_vlm_e2e")
    print(f"\nVLM e2e workspace: {project}")

    images_dir = project / "images"
    records = generate_debug_images(images_dir, count=20, seed=7)

    eval_recs, train_recs = records[:EVAL_SIZE], records[EVAL_SIZE:]
    train_path = project / "datasets" / "shapes" / "data.parquet"
    eval_path = project / "datasets" / "shapes_eval" / "data.parquet"
    _write_vlm_parquet(train_recs, images_dir, train_path)
    _write_vlm_parquet(eval_recs, images_dir, eval_path)
    print(f"Train: {len(train_recs)}  Eval: {len(eval_recs)}")

    return {
        "project": project,
        "images_dir": images_dir,
        "train_path": train_path,
        "eval_path": eval_path,
        "eval_records": eval_recs,
    }


@pytest.mark.gpu
class TestVLMTrainingE2E:
    def test_vlm_sft_smoke(self, vlm_workspace: dict) -> None:
        """1-epoch LoRA SFT of LFM2.5-VL-450M on 16 debug images; then
        reload the adapter and generate on held-out images."""
        from lqh.subprocess_manager import SubprocessManager
        from lqh.train.progress import read_progress

        project = vlm_workspace["project"]
        run = project / "runs" / "vlm_sft_smoke"

        config = {
            "type": "sft",
            "modality": "vision",
            "base_model": "LiquidAI/LFM2.5-VL-450M",
            "dataset": str(vlm_workspace["train_path"]),
            "eval_dataset": str(vlm_workspace["eval_path"]),
            "eval_on_checkpoints": True,
            "lora": {
                "enabled": True,
                "r": 8,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": [
                    "q_proj", "v_proj", "fc1", "fc2", "linear",
                    "gate_proj", "up_proj", "down_proj",
                ],
            },
            "training": {
                "num_epochs": 1,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "learning_rate": 5e-4,
                "warmup_ratio": 0.1,
                "logging_steps": 2,
                "gradient_checkpointing": True,
                "bf16": True,
                "max_seq_length": 2048,
                "max_image_tokens": 256,
                "eval_split_ratio": 0,
                "auto_batch": False,
                "dataloader_num_workers": 0,
            },
        }

        manager = SubprocessManager()
        pid = manager.start(run, config, project_dir=project)
        print(f"\nVLM training started (PID {pid})")

        deadline = time.monotonic() + TIMEOUT_S
        last_step = -1
        status = None
        while time.monotonic() < deadline:
            status = manager.get_status(run)
            if status.step is not None and status.step != last_step:
                last_step = status.step
                loss_str = f"loss={status.loss:.4f}" if status.loss else ""
                print(f"  Step {status.step} {loss_str}")
            if status.state in ("completed", "failed"):
                break
            time.sleep(5)

        assert status is not None and status.state == "completed", (
            f"training did not complete: state={getattr(status, 'state', None)} "
            f"error={getattr(status, 'error', None)}\n"
            f"progress tail: {read_progress(run, last_n=5)}"
        )

        # Adapter saved with processor config (image preprocessing must travel).
        adapter_dir = run / "model-lora"
        assert (adapter_dir / "adapter_config.json").is_file()
        assert (adapter_dir / "preprocessor_config.json").is_file(), (
            "processor (image preprocessor) config missing from the saved adapter"
        )

        # Final checkpoint eval produced predictions.
        final_preds = run / "checkpoints" / "final" / "predictions.parquet"
        assert final_preds.is_file(), "final checkpoint eval predictions missing"
        preds = pq.read_table(str(final_preds))
        assert preds.num_rows == EVAL_SIZE
        for raw in preds.column("messages").to_pylist():
            conv = json.loads(raw)
            assert conv[-1]["role"] == "assistant"
            assert not conv[-1]["content"].startswith("[generation error"), conv[-1]

        # Reload the adapter through the standard loader and generate on
        # held-out images (exercises load_for_inference vision dispatch +
        # vlm_generate outside the training subprocess).
        import torch

        from lqh.train.load_model import load_for_inference
        from lqh.train.vlm_data import vlm_generate

        model, processor = load_for_inference(
            str(adapter_dir), dtype=torch.bfloat16, device_map="auto",
        )
        images_dir = vlm_workspace["images_dir"]
        for rec in vlm_workspace["eval_records"][:2]:
            question, expected = _qa_for(rec)
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _data_url(images_dir / rec["file"])}},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            answer = vlm_generate(model, processor, prompt, max_new_tokens=32)
            print(f"  {rec['file']}: {question!r} -> {answer!r} (expected {expected!r})")
            assert answer.strip(), "empty generation from fine-tuned VLM"
