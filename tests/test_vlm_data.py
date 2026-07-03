"""Unit tests for the VLM training bridge (:mod:`lqh.train.vlm_data`) and
the vision branches of ``handle_start_training``.

CPU-only. The collator/generation tests use MagicMock processors with real
torch tensors; no model downloads.
"""

from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lqh.train.vlm_data import (
    VLMCollator,
    chatml_to_vlm_dataset,
    conversation_has_images,
    decode_image,
    split_image_parts,
    vlm_generate,
)


def _png_bytes(color=(220, 50, 47), size=(8, 8)) -> bytes:
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _data_url(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _vision_conv(*urls: str, question: str = "how many circles?") -> list[dict]:
    parts = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    parts.append({"type": "text", "text": question})
    return [
        {"role": "user", "content": parts},
        {"role": "assistant", "content": "two"},
    ]


# ---------------------------------------------------------------------------
# split_image_parts / chatml_to_vlm_dataset
# ---------------------------------------------------------------------------


class TestSplitImageParts:
    def test_round_trip_bytes(self) -> None:
        raw = _png_bytes()
        conv = _vision_conv(_data_url(raw))

        normalized, images = split_image_parts(conv)

        assert images == [raw]
        user_parts = normalized[0]["content"]
        assert user_parts[0] == {"type": "image"}
        assert user_parts[1] == {"type": "text", "text": "how many circles?"}
        assert normalized[1] == {"role": "assistant", "content": "two"}

    def test_multi_image_document_order(self) -> None:
        raw_a = _png_bytes((10, 20, 30))
        raw_b = _png_bytes((200, 100, 0))
        conv = _vision_conv(_data_url(raw_a), _data_url(raw_b))

        _, images = split_image_parts(conv)
        assert images == [raw_a, raw_b]

    def test_text_only_passthrough(self) -> None:
        conv = [
            {"role": "user", "content": "plain"},
            {"role": "assistant", "content": "reply"},
        ]
        normalized, images = split_image_parts(conv)
        assert normalized == conv
        assert images == []

    def test_input_not_mutated(self) -> None:
        conv = _vision_conv(_data_url(_png_bytes()))
        before = json.dumps(conv)
        split_image_parts(conv)
        assert json.dumps(conv) == before

    def test_malformed_base64_raises(self) -> None:
        conv = _vision_conv("data:image/png;base64,!!!not-base64!!!")
        with pytest.raises(ValueError, match="malformed image data-URL"):
            split_image_parts(conv)

    def test_remote_url_rejected(self) -> None:
        conv = _vision_conv("https://example.com/cat.png")
        with pytest.raises(ValueError, match="remote image URLs are not supported"):
            split_image_parts(conv)

    def test_conversation_has_images(self) -> None:
        assert conversation_has_images(_vision_conv(_data_url(_png_bytes())))
        assert not conversation_has_images(
            [{"role": "user", "content": "text only"}]
        )
        assert not conversation_has_images(
            [{"role": "user", "content": [{"type": "text", "text": "parts"}]}]
        )


class TestChatmlToVlmDataset:
    def test_rows_shape(self) -> None:
        raw = _png_bytes()
        rows = chatml_to_vlm_dataset(
            [
                _vision_conv(_data_url(raw)),
                [{"role": "user", "content": "text"}, {"role": "assistant", "content": "t"}],
            ]
        )
        assert len(rows) == 2
        assert rows[0]["images"] == [raw]
        assert isinstance(rows[0]["messages"], str)  # JSON-encoded for Arrow
        decoded = json.loads(rows[0]["messages"])
        assert decoded[0]["content"][0] == {"type": "image"}
        assert rows[1]["images"] == []

    def test_rows_load_into_hf_dataset(self) -> None:
        from datasets import Dataset

        rows = chatml_to_vlm_dataset([_vision_conv(_data_url(_png_bytes()))])
        ds = Dataset.from_list(rows)
        assert ds[0]["images"][0] == rows[0]["images"][0]


# ---------------------------------------------------------------------------
# VLMCollator
# ---------------------------------------------------------------------------


def _mock_processor(seq_len: int = 6, pad_id: int = 0):
    """Processor double: returns a fixed-size tokenized batch and records
    the conversations passed to apply_chat_template."""
    import torch

    processor = MagicMock()
    processor.tokenizer.pad_token_id = pad_id

    def _apply(convs, **kwargs):
        n = len(convs)
        input_ids = torch.ones((n, seq_len), dtype=torch.long)
        input_ids[:, -1] = pad_id  # one pad position per row
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones((n, seq_len), dtype=torch.long),
            "pixel_values": torch.zeros((n, 3, 4, 4)),
        }

    processor.apply_chat_template = MagicMock(side_effect=_apply)
    return processor


class TestVLMCollator:
    def test_reinserts_pil_images_and_masks_labels(self) -> None:
        from PIL import Image

        raw = _png_bytes()
        rows = chatml_to_vlm_dataset([_vision_conv(_data_url(raw))])
        processor = _mock_processor(seq_len=6, pad_id=0)

        batch = VLMCollator(processor)(rows)

        convs = processor.apply_chat_template.call_args.args[0]
        image_part = convs[0][0]["content"][0]
        assert image_part["type"] == "image"
        assert isinstance(image_part["image"], Image.Image)
        assert image_part["image"].mode == "RGB"

        # labels = input_ids with pad masked to -100
        assert batch["labels"].shape == batch["input_ids"].shape
        assert (batch["labels"][:, -1] == -100).all()
        assert (batch["labels"][:, :-1] == batch["input_ids"][:, :-1]).all()

    def test_tokenize_kwargs(self) -> None:
        rows = chatml_to_vlm_dataset([_vision_conv(_data_url(_png_bytes()))])
        processor = _mock_processor()
        VLMCollator(processor)(rows)
        kwargs = processor.apply_chat_template.call_args.kwargs
        assert kwargs["tokenize"] is True
        assert kwargs["return_dict"] is True
        assert kwargs["return_tensors"] == "pt"
        assert kwargs["padding"] is True

    def test_drops_overlong_samples_instead_of_truncating(self) -> None:
        import torch

        raw = _png_bytes()
        rows = chatml_to_vlm_dataset(
            [_vision_conv(_data_url(raw)), _vision_conv(_data_url(raw), question="q2")]
        )

        processor = MagicMock()
        processor.tokenizer.pad_token_id = 0

        def _apply(convs, **kwargs):
            # The batch (2 samples) and the second sample render over-long
            # (length 100); the first sample alone fits (length 4).
            long = any(
                p.get("text") == "q2"
                for conv in convs
                for m in conv
                if isinstance(m.get("content"), list)
                for p in m["content"]
                if isinstance(p, dict)
            )
            seq = 100 if (len(convs) > 1 or long) else 4
            ids = torch.ones((len(convs), seq), dtype=torch.long)
            return {"input_ids": ids}

        processor.apply_chat_template = MagicMock(side_effect=_apply)

        batch = VLMCollator(processor, max_length=10)(rows)
        # Only the short sample survives; nothing was truncated.
        assert batch["input_ids"].shape == (1, 4)

    def test_raises_when_all_samples_overlong(self) -> None:
        import torch

        rows = chatml_to_vlm_dataset([_vision_conv(_data_url(_png_bytes()))])
        processor = MagicMock()
        processor.tokenizer.pad_token_id = 0
        processor.apply_chat_template = MagicMock(
            side_effect=lambda convs, **kw: {
                "input_ids": torch.ones((len(convs), 999), dtype=torch.long)
            }
        )
        with pytest.raises(ValueError, match="max_length"):
            VLMCollator(processor, max_length=10)(rows)


# ---------------------------------------------------------------------------
# decode_image / vlm_generate
# ---------------------------------------------------------------------------


def test_decode_image_returns_rgb() -> None:
    from PIL import Image

    buf = BytesIO()
    Image.new("RGBA", (5, 5), (1, 2, 3, 4)).save(buf, format="PNG")
    img = decode_image(buf.getvalue())
    assert img.mode == "RGB"
    assert img.size == (5, 5)


def test_vlm_generate_moves_inputs_and_decodes() -> None:
    import torch

    processor = MagicMock()
    prompt_ids = torch.tensor([[1, 2, 3]])

    inputs_dict = {
        "input_ids": prompt_ids,
        "pixel_values": torch.zeros((1, 3, 4, 4)),
    }
    processor.apply_chat_template = MagicMock(return_value=inputs_dict)
    processor.tokenizer.decode.return_value = "a red square"

    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = torch.tensor([[1, 2, 3, 9, 9]])

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_url(_png_bytes())}},
                {"type": "text", "text": "what is this?"},
            ],
        }
    ]
    out = vlm_generate(model, processor, prompt, max_new_tokens=16)

    assert out == "a red square"
    # The conversation handed to the template has a PIL image re-inserted.
    conv = processor.apply_chat_template.call_args.args[0]
    assert conv[0]["content"][0]["type"] == "image"
    assert "image" in conv[0]["content"][0]
    # generate received the full inputs dict (pixel_values included).
    gen_kwargs = model.generate.call_args.kwargs
    assert "pixel_values" in gen_kwargs
    assert gen_kwargs["max_new_tokens"] == 16
    assert gen_kwargs["do_sample"] is False
    # Decode starts after the prompt length.
    decode_args = processor.tokenizer.decode.call_args.args
    assert decode_args[0].tolist() == [9, 9]


# ---------------------------------------------------------------------------
# handle_start_training — vision config
# ---------------------------------------------------------------------------


def _make_dataset(project: Path, name: str) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds_dir = project / "datasets" / name
    ds_dir.mkdir(parents=True)
    table = pa.table({"messages": [json.dumps([{"role": "user", "content": "hi"}])]})
    pq.write_table(table, ds_dir / "data.parquet")
    return f"datasets/{name}"


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point HOME + the global remotes/compute paths at a tmp dir so tests
    can't leak state into the developer's real ~/.lqh (same shape as
    tests/test_compute_resolve.py)."""
    import lqh.remote.config as remote_config

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(remote_config, "GLOBAL_CONFIG_DIR", home / ".lqh")
    yield home


@pytest.fixture
def training_project(tmp_path, isolated_home, monkeypatch):
    """Project wired so handle_start_training reaches config-build and the
    cloud submit is captured instead of executed."""
    import lqh.tools.handlers as handlers
    from lqh.tools.permissions import grant_training_permission

    project = tmp_path / "proj"
    project.mkdir()
    grant_training_permission(project, project_wide=True)
    monkeypatch.setattr(handlers, "_local_gpu_available", lambda: False)

    recorded: dict = {}

    async def fake_remote(project_dir, run_dir, config, run_name, remote_name, api_key, **kw):
        from lqh.tools.handlers import ToolResult

        recorded["config"] = config
        return ToolResult(content="stub: cloud")

    monkeypatch.setattr(handlers, "_execute_start_training_remote", fake_remote)
    return project, recorded


def test_vlm_base_gets_vision_config(training_project) -> None:
    import lqh.tools.handlers as handlers

    project, recorded = training_project
    asyncio.run(
        handlers.handle_start_training(
            project,
            type="sft",
            base_model="LiquidAI/LFM2.5-VL-450M",
            dataset=_make_dataset(project, "ds"),
            eval_dataset=_make_dataset(project, "ds_eval"),
            disable_scoring=True,
        )
    )

    config = recorded["config"]["base_config"]
    assert config["modality"] == "vision"
    assert config["training"]["max_image_tokens"] == 256
    assert config["training"]["learning_rate"] == 5e-4
    assert config["training"]["per_device_batch_size"] == 2
    assert config["training"]["effective_batch_size"] == 16
    assert config["training"]["auto_batch"] is True  # OOM self-heal stays on
    assert config["lora"]["r"] == 8
    assert config["lora"]["alpha"] == 16
    assert config["lora"]["target_modules"] == [
        "q_proj", "v_proj", "fc1", "fc2", "linear",
        "gate_proj", "up_proj", "down_proj",
    ]


def test_text_base_config_unchanged(training_project) -> None:
    """Golden regression: a text base must produce the exact pre-VLM config."""
    import lqh.tools.handlers as handlers

    project, recorded = training_project
    asyncio.run(
        handlers.handle_start_training(
            project,
            type="sft",
            base_model="LiquidAI/LFM2.5-1.2B-Instruct",
            dataset=_make_dataset(project, "ds"),
            eval_dataset=_make_dataset(project, "ds_eval"),
            disable_scoring=True,
        )
    )

    config = recorded["config"]["base_config"]
    assert "modality" not in config
    assert "max_image_tokens" not in config["training"]
    assert config["training"]["learning_rate"] == 2e-5
    assert config["training"]["per_device_batch_size"] == 256
    assert config["lora"]["r"] == 32
    assert config["lora"]["alpha"] == 64
    assert config["lora"]["target_modules"] == [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj", "out_proj", "w1", "w2", "w3",
    ]


def test_vlm_base_rejects_dpo(training_project) -> None:
    import lqh.tools.handlers as handlers

    project, recorded = training_project
    scorer = project / "evals" / "scorers" / "scorer.md"
    scorer.parent.mkdir(parents=True)
    scorer.write_text("# scorer")
    res = asyncio.run(
        handlers.handle_start_training(
            project,
            type="on_policy_dpo",
            base_model="lfm2.5-vl-1.6b",
            dataset=_make_dataset(project, "ds"),
            eval_dataset=_make_dataset(project, "ds_eval"),
            scorer="evals/scorers/scorer.md",
        )
    )
    assert "not supported for vision-language models" in res.content
    assert "config" not in recorded


# ---------------------------------------------------------------------------
# merge_lora — model-class selection for VL bases
# ---------------------------------------------------------------------------


def test_merge_lora_vision_base_uses_vlm_class_and_processor(tmp_path, monkeypatch):
    from unittest.mock import patch

    import lqh.train.load_model as lm
    from lqh.remote import merge_lora

    vision_cls = MagicMock(name="AutoModelForImageTextToText")
    monkeypatch.setattr(lm, "detect_modality", lambda *a, **k: "vision")
    monkeypatch.setattr(lm, "_model_cls", lambda modality: vision_cls)

    with patch("peft.PeftModel") as peft_model, patch(
        "transformers.AutoProcessor"
    ) as auto_proc, patch("transformers.AutoTokenizer") as auto_tok:
        merged = peft_model.from_pretrained.return_value.merge_and_unload.return_value
        merge_lora._merge("fake/vl-base", tmp_path / "adapter", tmp_path / "out")

        vision_cls.from_pretrained.assert_called_once()
        assert vision_cls.from_pretrained.call_args.args[0] == "fake/vl-base"
        merged.save_pretrained.assert_called_once()
        # The FULL processor travels with the merged checkpoint (image
        # preprocessor config + tokenizer + chat template), not just the
        # tokenizer.
        auto_proc.from_pretrained.assert_called_once_with("fake/vl-base")
        auto_proc.from_pretrained.return_value.save_pretrained.assert_called_once()
        auto_tok.from_pretrained.assert_not_called()


def test_merge_lora_text_base_unchanged(tmp_path, monkeypatch):
    from unittest.mock import patch

    import lqh.train.load_model as lm
    from lqh.remote import merge_lora

    text_cls = MagicMock(name="AutoModelForCausalLM")
    monkeypatch.setattr(lm, "detect_modality", lambda *a, **k: "text")
    monkeypatch.setattr(lm, "_model_cls", lambda modality: text_cls)

    with patch("peft.PeftModel"), patch(
        "transformers.AutoProcessor"
    ) as auto_proc, patch("transformers.AutoTokenizer") as auto_tok:
        merge_lora._merge("fake/text-base", tmp_path / "adapter", tmp_path / "out")

        text_cls.from_pretrained.assert_called_once()
        auto_tok.from_pretrained.assert_called_once_with("fake/text-base")
        auto_proc.from_pretrained.assert_not_called()


def test_vlm_generate_drops_rejected_model_kwargs() -> None:
    """generate() may reject processor-emitted keys its
    prepare_inputs_for_generation doesn't route (LFM2-VL:
    pixel_attention_mask / spatial_shapes) — vlm_generate must drop the
    named keys and retry once."""
    import torch

    processor = MagicMock()
    inputs_dict = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "pixel_values": torch.zeros((1, 3, 4, 4)),
        "pixel_attention_mask": torch.ones((1, 1)),
        "spatial_shapes": torch.tensor([[4, 4]]),
    }
    processor.apply_chat_template = MagicMock(return_value=inputs_dict)
    processor.tokenizer.decode.return_value = "two circles"

    model = MagicMock()
    model.device = "cpu"
    calls = []

    def _generate(**kwargs):
        calls.append(set(kwargs))
        if "pixel_attention_mask" in kwargs:
            raise ValueError(
                "The following `model_kwargs` are not used by the model: "
                "['pixel_attention_mask', 'spatial_shapes']"
            )
        return torch.tensor([[1, 2, 3, 9]])

    model.generate = MagicMock(side_effect=_generate)

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_url(_png_bytes())}},
                {"type": "text", "text": "how many?"},
            ],
        }
    ]
    out = vlm_generate(model, processor, prompt, max_new_tokens=8)

    assert out == "two circles"
    assert len(calls) == 2
    assert "pixel_attention_mask" not in calls[1]
    assert "spatial_shapes" not in calls[1]
    assert "pixel_values" in calls[1]  # only the named keys are dropped
    assert "input_ids" in calls[1]


def test_vlm_generate_reraises_unrelated_value_error() -> None:
    import torch

    processor = MagicMock()
    processor.apply_chat_template = MagicMock(
        return_value={"input_ids": torch.tensor([[1]])}
    )
    model = MagicMock()
    model.device = "cpu"
    model.generate = MagicMock(side_effect=ValueError("something else broke"))

    with pytest.raises(ValueError, match="something else broke"):
        vlm_generate(model, processor, [{"role": "user", "content": "hi"}])
