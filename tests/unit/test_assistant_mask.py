"""Assistant-turn-only loss is forced for text SFT (lqh/train/assistant_mask.py).

Runs without transformers: the launch-side check reads template text, and the
trainer-side helpers are exercised through a duck-typed tokenizer that returns
the ``assistant_masks`` shape transformers produces.
"""

from __future__ import annotations

import json

import pytest

from lqh.train import assistant_mask as am
from lqh.train.assistant_mask import (
    NO_GENERATION_BLOCK,
    assistant_mask_unsupported,
    drop_rows_without_assistant_labels,
    read_chat_template,
    require_assistant_mask_support,
    template_lacks_generation_block,
    unsupported_message,
)

MARKED = (
    "{%- for m in messages -%}<|im_start|>{{ m.role }}\n"
    "{%- if m.role == 'assistant' -%}{%- generation -%}{{ m.content }}<|im_end|>\n"
    "{%- endgeneration -%}{%- else -%}{{ m.content }}<|im_end|>\n{%- endif -%}"
    "{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}"
)
UNMARKED = MARKED.replace("{%- generation -%}", "").replace("{%- endgeneration -%}", "")


# ---------------------------------------------------------------------------
# Template text (client side)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag", ["{% generation %}", "{%- generation -%}", "{%generation%}", "{%- generation %}"]
)
def test_generation_tag_spellings_are_recognized(tag):
    assert not template_lacks_generation_block(f"a {tag} b {{% endgeneration %}}")


def test_add_generation_prompt_is_not_a_generation_block():
    # LFM2.5-*-Base templates have add_generation_prompt and nothing else.
    assert template_lacks_generation_block(UNMARKED)
    assert not template_lacks_generation_block(MARKED)


def test_local_dir_prefers_chat_template_jinja(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "chat_template.jinja").write_text(MARKED)
    (ckpt / "tokenizer_config.json").write_text(json.dumps({"chat_template": UNMARKED}))
    assert read_chat_template("ckpt", tmp_path) == MARKED
    assert read_chat_template(str(ckpt), None) == MARKED  # absolute path


def test_local_dir_falls_back_to_tokenizer_config(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "tokenizer_config.json").write_text(json.dumps({"chat_template": UNMARKED}))
    assert read_chat_template("ckpt", tmp_path) == UNMARKED

    # the older list-of-variants form: every variant's text counts
    (ckpt / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": [
            {"name": "default", "template": UNMARKED},
            {"name": "train", "template": MARKED},
        ]})
    )
    assert not template_lacks_generation_block(read_chat_template("ckpt", tmp_path))


def test_local_dir_without_tokenizer_files_is_undeterminable(tmp_path):
    (tmp_path / "adapter").mkdir()
    assert read_chat_template("adapter", tmp_path) is None
    (tmp_path / "adapter" / "tokenizer_config.json").write_text("{not json")
    assert read_chat_template("adapter", tmp_path) is None


def test_hub_template_is_read_from_chat_template_jinja_first(tmp_path, monkeypatch):
    files = {"chat_template.jinja": MARKED}
    calls: list[str] = []

    def fake_hub_file(repo_id, filename, project_dir):
        calls.append(filename)
        if filename not in files:
            raise FileNotFoundError(filename)
        p = tmp_path / filename
        p.write_text(files[filename])
        return str(p)

    monkeypatch.setattr(am, "_hub_file", fake_hub_file)
    assert read_chat_template("LiquidAI/LFM2.5-1.2B-Instruct", tmp_path) == MARKED
    assert calls == ["chat_template.jinja"]

    files.clear()
    files["tokenizer_config.json"] = json.dumps({"chat_template": UNMARKED})
    assert read_chat_template("LiquidAI/LFM2.5-1.2B-Base", tmp_path) == UNMARKED


def test_hub_unreachable_is_undeterminable(monkeypatch):
    def offline(repo_id, filename, project_dir):
        raise OSError("offline")

    monkeypatch.setattr(am, "_hub_file", offline)
    assert read_chat_template("LiquidAI/LFM2.5-1.2B-Instruct", None) is None


# ---------------------------------------------------------------------------
# Tokenizer probe + truncation guard (trainer side), duck-typed tokenizer
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """One token per character of each message's content, in message order;
    assistant content is masked 1 when ``marks`` is set. Mirrors what
    transformers returns for ``return_assistant_tokens_mask=True``."""

    def __init__(self, marks: bool = True, accepts_tools: bool = True):
        self.marks = marks
        self.accepts_tools = accepts_tools
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize=True, return_dict=False, **kw):
        if not self.accepts_tools and "tools" in kw:
            raise TypeError("apply_chat_template() got an unexpected keyword 'tools'")
        self.calls.append(kw)
        ids: list[int] = []
        mask: list[int] = []
        for m in messages:
            n = len(m["content"])
            ids.extend([ord(c) for c in m["content"]])
            mask.extend([1 if (self.marks and m["role"] == "assistant") else 0] * n)
        if not return_dict:
            return ids
        out = {"input_ids": ids}
        if kw.get("return_assistant_tokens_mask"):
            out["assistant_masks"] = mask
        return out


class Failing:
    def apply_chat_template(self, *a, **kw):
        raise ValueError("boom")


def test_probe_accepts_a_marking_template():
    assert assistant_mask_unsupported(FakeTokenizer()) is None
    require_assistant_mask_support(FakeTokenizer(), "m")  # does not raise


def test_probe_rejects_a_template_that_marks_nothing():
    assert assistant_mask_unsupported(FakeTokenizer(marks=False)) == NO_GENERATION_BLOCK
    with pytest.raises(ValueError) as exc:
        require_assistant_mask_support(FakeTokenizer(marks=False), "LiquidAI/LFM2.5-1.2B-Base")
    msg = str(exc.value)
    assert msg == unsupported_message("LiquidAI/LFM2.5-1.2B-Base", NO_GENERATION_BLOCK)
    assert "{% generation %}" in msg and "upstream" in msg


def test_probe_reports_a_template_that_fails_to_render():
    reason = assistant_mask_unsupported(Failing())
    assert reason.startswith("rendering its chat template failed: boom")


def test_rows_truncated_past_their_assistant_turn_are_dropped():
    tok = FakeTokenizer()
    rows = [
        {"messages": [{"role": "user", "content": "1234567890"},
                      {"role": "assistant", "content": "ok"}]},
        {"messages": [{"role": "user", "content": "12"},
                      {"role": "assistant", "content": "fine"}]},
    ]
    # first row: assistant tokens are positions 10..11 -> gone at max_length 10
    assert drop_rows_without_assistant_labels(rows, tok, 10) == ([rows[1]], 1)
    assert drop_rows_without_assistant_labels(rows, tok, 11) == (rows, 0)
    assert drop_rows_without_assistant_labels(rows, tok, None) == (rows, 0)


def test_tools_column_is_decoded_before_rendering():
    tok = FakeTokenizer()
    tools = [{"type": "function", "function": {"name": "f"}}]
    rows = [{"messages": [{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "a"}],
             "tools": json.dumps(tools)}]
    assert drop_rows_without_assistant_labels(rows, tok, 100) == (rows, 0)
    assert tok.calls[-1]["tools"] == tools  # a list, not the JSON string

    # an old tokenizer without a tools kwarg is retried without it (no tools to lose)
    rows[0]["tools"] = None
    assert drop_rows_without_assistant_labels(rows, FakeTokenizer(accepts_tools=False), 100) == (rows, 0)
    rows[0]["tools"] = json.dumps(tools)
    with pytest.raises(TypeError):
        drop_rows_without_assistant_labels(rows, FakeTokenizer(accepts_tools=False), 100)


# ---------------------------------------------------------------------------
# Launch-side wrapper in the tool handler
# ---------------------------------------------------------------------------


def test_launch_check_resolves_an_adapter_dir_to_its_base(tmp_path, monkeypatch):
    from lqh.tools.handlers import _assistant_mask_unsupported

    base = tmp_path / "runs" / "sft_1" / "merged"
    base.mkdir(parents=True)
    (base / "chat_template.jinja").write_text(UNMARKED)
    adapter = tmp_path / "runs" / "sft_2" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base)})
    )
    # the adapter dir carries no tokenizer files; lineage leads to the base
    assert _assistant_mask_unsupported(tmp_path, "runs/sft_2/adapter") == NO_GENERATION_BLOCK

    (base / "chat_template.jinja").write_text(MARKED)
    assert _assistant_mask_unsupported(tmp_path, "runs/sft_2/adapter") is None

    monkeypatch.setattr(am, "_hub_file", lambda *a: (_ for _ in ()).throw(OSError("offline")))
    assert _assistant_mask_unsupported(tmp_path, "LiquidAI/LFM2.5-1.2B-Instruct") is None
