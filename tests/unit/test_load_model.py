"""Unit tests for :mod:`lqh.train.load_model`.

CPU-only, mock-based — no real model downloads. All ``from_pretrained``
calls are patched at module level so the tests run in well under a
second on a machine without GPUs, transformers, or peft installed.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lqh.train.load_model import detect_kind, resolve_base_model


# ---------------------------------------------------------------------------
# detect_kind
# ---------------------------------------------------------------------------


def test_detect_kind_hub_for_hub_id():
    assert detect_kind("LiquidAI/LFM2-1.2B") == "hub"


def test_detect_kind_hub_for_nonexistent_path(tmp_path: Path):
    assert detect_kind(str(tmp_path / "does-not-exist")) == "hub"


def test_detect_kind_merged_for_config_json(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    assert detect_kind(str(tmp_path)) == "merged"


def test_detect_kind_adapter_for_adapter_config_json(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text("{}")
    # Adapter dirs ALSO often contain a config.json from the base — adapter
    # wins because the presence of adapter_config.json is the
    # discriminating signal.
    (tmp_path / "config.json").write_text("{}")
    assert detect_kind(str(tmp_path)) == "adapter"


def test_detect_kind_empty_dir_falls_back_to_merged(tmp_path: Path):
    # No config.json, no adapter_config.json — we fall back to "merged"
    # so downstream AutoModel raises its own clear error.
    assert detect_kind(str(tmp_path)) == "merged"


# ---------------------------------------------------------------------------
# resolve_base_model
# ---------------------------------------------------------------------------


def test_resolve_base_model_from_config(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "fake/base"})
    )
    assert resolve_base_model(str(tmp_path)) == "fake/base"


def test_resolve_base_model_override_wins(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "from/json"})
    )
    assert resolve_base_model(str(tmp_path), override="from/override") == "from/override"


def test_resolve_base_model_missing_field_raises(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="base_model_name_or_path"):
        resolve_base_model(str(tmp_path))


def test_resolve_base_model_not_an_adapter_dir_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="not an adapter dir"):
        resolve_base_model(str(tmp_path))


def test_resolve_base_model_invalid_json_raises(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text("{not json")
    with pytest.raises(ValueError, match="invalid JSON"):
        resolve_base_model(str(tmp_path))


# ---------------------------------------------------------------------------
# load_for_inference / load_for_training
#
# These run only when torch + transformers + peft are importable. The
# point isn't to exercise real model loads (that's the e2e test's job)
# but to verify the *dispatch* logic — which `from_pretrained` chain
# gets called for each model kind.
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_torch_transformers_peft(monkeypatch: pytest.MonkeyPatch):
    """Inject minimal stubs for torch / transformers / peft.

    The load_model module imports them lazily inside the functions, so
    stubbing via sys.modules is enough — we don't need the real
    packages installed.
    """
    # Real torch is fine if installed; otherwise stub bfloat16.
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.bfloat16 = "bf16-sentinel"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", torch_stub)

    # transformers.AutoModelForCausalLM + AutoTokenizer (+ the vision twins
    # and AutoConfig, which modality auto-detection reads). The default
    # AutoConfig result is a text model; tests flip it to a VL config via
    # `s.set_model_type(...)`.
    transformers_stub = types.ModuleType("transformers")
    auto_model = MagicMock(name="AutoModelForCausalLM")
    auto_tokenizer = MagicMock(name="AutoTokenizer")
    auto_vlm = MagicMock(name="AutoModelForImageTextToText")
    auto_processor = MagicMock(name="AutoProcessor")
    auto_config = MagicMock(name="AutoConfig")

    config_obj = types.SimpleNamespace(model_type="lfm2")
    auto_config.from_pretrained.return_value = config_obj

    def set_model_type(model_type: str, *, vision_config=None):
        cfg = types.SimpleNamespace(model_type=model_type)
        if vision_config is not None:
            cfg.vision_config = vision_config
        auto_config.from_pretrained.return_value = cfg

    transformers_stub.AutoModelForCausalLM = auto_model  # type: ignore[attr-defined]
    transformers_stub.AutoTokenizer = auto_tokenizer  # type: ignore[attr-defined]
    transformers_stub.AutoModelForImageTextToText = auto_vlm  # type: ignore[attr-defined]
    transformers_stub.AutoProcessor = auto_processor  # type: ignore[attr-defined]
    transformers_stub.AutoConfig = auto_config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)

    # peft.PeftModel
    peft_stub = types.ModuleType("peft")
    peft_model = MagicMock(name="PeftModel")
    peft_stub.PeftModel = peft_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peft", peft_stub)

    return types.SimpleNamespace(
        AutoModelForCausalLM=auto_model,
        AutoTokenizer=auto_tokenizer,
        AutoModelForImageTextToText=auto_vlm,
        AutoProcessor=auto_processor,
        AutoConfig=auto_config,
        PeftModel=peft_model,
        set_model_type=set_model_type,
    )


def test_load_for_inference_dispatches_hub(stub_torch_transformers_peft):
    from lqh.train.load_model import load_for_inference

    s = stub_torch_transformers_peft
    model, tok = load_for_inference("fake/hub-id")

    s.AutoModelForCausalLM.from_pretrained.assert_called_once()
    args, kwargs = s.AutoModelForCausalLM.from_pretrained.call_args
    assert args[0] == "fake/hub-id"
    s.PeftModel.from_pretrained.assert_not_called()
    # tokenizer always loads
    s.AutoTokenizer.from_pretrained.assert_called_once_with("fake/hub-id")


def test_load_for_inference_dispatches_merged(stub_torch_transformers_peft, tmp_path: Path):
    from lqh.train.load_model import load_for_inference

    (tmp_path / "config.json").write_text("{}")
    s = stub_torch_transformers_peft
    load_for_inference(str(tmp_path))

    s.AutoModelForCausalLM.from_pretrained.assert_called_once()
    s.PeftModel.from_pretrained.assert_not_called()


def test_load_for_inference_dispatches_adapter(stub_torch_transformers_peft, tmp_path: Path):
    from lqh.train.load_model import load_for_inference

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "fake/base"})
    )
    s = stub_torch_transformers_peft
    base_obj = MagicMock(name="base_model_instance")
    s.AutoModelForCausalLM.from_pretrained.return_value = base_obj
    wrapped = MagicMock(name="peft_wrapped")
    s.PeftModel.from_pretrained.return_value = wrapped
    merged = MagicMock(name="merged_model")
    wrapped.merge_and_unload.return_value = merged

    model, tok = load_for_inference(str(tmp_path))

    # base loads first
    base_call = s.AutoModelForCausalLM.from_pretrained.call_args_list[0]
    assert base_call.args[0] == "fake/base"
    # adapter wraps base
    s.PeftModel.from_pretrained.assert_called_once_with(base_obj, str(tmp_path))
    # transient merge applied
    wrapped.merge_and_unload.assert_called_once()
    assert model is merged


def test_load_for_inference_adapter_base_override(stub_torch_transformers_peft, tmp_path: Path):
    """``base_override`` should beat adapter_config.json's base_model_name_or_path."""
    from lqh.train.load_model import load_for_inference

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "from/json"})
    )
    s = stub_torch_transformers_peft
    load_for_inference(str(tmp_path), base_override="from/override")

    # First positional arg of the first AutoModel call is the override.
    first_call = s.AutoModelForCausalLM.from_pretrained.call_args_list[0]
    assert first_call.args[0] == "from/override"


def test_load_for_training_returns_effective_base_for_hub(stub_torch_transformers_peft):
    from lqh.train.load_model import load_for_training

    model, tok, effective_base = load_for_training("fake/hub-id")
    assert effective_base == "fake/hub-id"


def test_load_for_training_adapter_merges_by_default(stub_torch_transformers_peft, tmp_path: Path):
    from lqh.train.load_model import load_for_training

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "fake/base"})
    )
    s = stub_torch_transformers_peft
    base_obj = MagicMock(name="base_model_instance")
    s.AutoModelForCausalLM.from_pretrained.return_value = base_obj
    wrapped = MagicMock(name="peft_wrapped")
    s.PeftModel.from_pretrained.return_value = wrapped
    merged = MagicMock(name="merged_model")
    wrapped.merge_and_unload.return_value = merged

    model, tok, effective_base = load_for_training(str(tmp_path))

    assert effective_base == "fake/base"
    wrapped.merge_and_unload.assert_called_once()
    assert model is merged


def test_load_for_training_adapter_no_merge_returns_peft(stub_torch_transformers_peft, tmp_path: Path):
    """``merge_before_attach=False`` returns the live PeftModel wrapper."""
    from lqh.train.load_model import load_for_training

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "fake/base"})
    )
    s = stub_torch_transformers_peft
    base_obj = MagicMock(name="base_model_instance")
    s.AutoModelForCausalLM.from_pretrained.return_value = base_obj
    wrapped = MagicMock(name="peft_wrapped")
    s.PeftModel.from_pretrained.return_value = wrapped

    model, tok, effective_base = load_for_training(
        str(tmp_path), merge_before_attach=False,
    )

    wrapped.merge_and_unload.assert_not_called()
    assert model is wrapped
    assert effective_base == "fake/base"


# ---------------------------------------------------------------------------
# detect_modality + vision dispatch
# ---------------------------------------------------------------------------


def test_detect_modality_text(stub_torch_transformers_peft):
    from lqh.train.load_model import detect_modality

    stub_torch_transformers_peft.set_model_type("lfm2")
    assert detect_modality("LiquidAI/LFM2.5-1.2B-Instruct") == "text"


def test_detect_modality_vision_by_model_type(stub_torch_transformers_peft):
    from lqh.train.load_model import detect_modality

    stub_torch_transformers_peft.set_model_type("lfm2_vl")
    assert detect_modality("LiquidAI/LFM2.5-VL-450M") == "vision"


def test_detect_modality_vision_by_vision_config(stub_torch_transformers_peft):
    from lqh.train.load_model import detect_modality

    stub_torch_transformers_peft.set_model_type("some_multimodal", vision_config={"hidden": 1})
    assert detect_modality("whatever/model") == "vision"


def test_detect_modality_adapter_resolves_base(stub_torch_transformers_peft, tmp_path: Path):
    from lqh.train.load_model import detect_modality

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "fake/vl-base"})
    )
    s = stub_torch_transformers_peft
    s.set_model_type("lfm2_vl")

    assert detect_modality(str(tmp_path)) == "vision"
    # AutoConfig must be read from the resolved BASE, not the adapter dir.
    assert s.AutoConfig.from_pretrained.call_args.args[0] == "fake/vl-base"


def test_load_for_inference_vision_uses_processor(stub_torch_transformers_peft):
    from lqh.train.load_model import load_for_inference

    s = stub_torch_transformers_peft
    s.set_model_type("lfm2_vl")

    model, tok = load_for_inference("fake/vl-id", max_image_tokens=128)

    s.AutoModelForImageTextToText.from_pretrained.assert_called_once()
    s.AutoModelForCausalLM.from_pretrained.assert_not_called()
    s.AutoProcessor.from_pretrained.assert_called_once_with(
        "fake/vl-id", max_image_tokens=128,
    )
    s.AutoTokenizer.from_pretrained.assert_not_called()


def test_load_for_inference_text_path_untouched_by_vision_support(stub_torch_transformers_peft):
    """Regression: text models must not touch the vision classes."""
    from lqh.train.load_model import load_for_inference

    s = stub_torch_transformers_peft
    s.set_model_type("lfm2")

    load_for_inference("fake/hub-id")

    s.AutoModelForCausalLM.from_pretrained.assert_called_once()
    s.AutoModelForImageTextToText.from_pretrained.assert_not_called()
    s.AutoProcessor.from_pretrained.assert_not_called()


def test_load_for_training_vision_adapter(stub_torch_transformers_peft, tmp_path: Path):
    from lqh.train.load_model import load_for_training

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "fake/vl-base"})
    )
    s = stub_torch_transformers_peft
    s.set_model_type("lfm2_vl")
    base_obj = MagicMock(name="vl_base_instance")
    s.AutoModelForImageTextToText.from_pretrained.return_value = base_obj
    wrapped = MagicMock(name="peft_wrapped")
    s.PeftModel.from_pretrained.return_value = wrapped

    model, tok, effective_base = load_for_training(str(tmp_path))

    assert effective_base == "fake/vl-base"
    assert s.AutoModelForImageTextToText.from_pretrained.call_args.args[0] == "fake/vl-base"
    s.PeftModel.from_pretrained.assert_called_once_with(base_obj, str(tmp_path))
    wrapped.merge_and_unload.assert_called_once()


def test_explicit_modality_skips_detection(stub_torch_transformers_peft):
    from lqh.train.load_model import load_for_inference

    s = stub_torch_transformers_peft
    load_for_inference("fake/hub-id", modality="text")

    s.AutoConfig.from_pretrained.assert_not_called()
    s.AutoModelForCausalLM.from_pretrained.assert_called_once()
