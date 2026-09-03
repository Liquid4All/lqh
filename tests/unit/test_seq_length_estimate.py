"""Submit-time longest-row estimate (lqh/train/seq_length.py).

Runs without transformers: the tokenizer path uses a tiny ``tokenizers``
WordLevel model written to disk, the fallback path counts characters.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lqh.train import seq_length
from lqh.train.seq_length import (
    CHARS_PER_TOKEN,
    PER_CONVERSATION_OVERHEAD,
    PER_MESSAGE_OVERHEAD,
    estimate_longest_row_tokens,
)


def _write_dataset(path: Path, conversations: list[list[dict]], tools=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {"messages": [json.dumps(c) for c in conversations]}
    if tools is not None:
        cols["tools"] = [json.dumps(t) if t else None for t in tools]
    pq.write_table(pa.table(cols), path)
    return path


def _word_tokenizer_dir(root: Path) -> Path:
    """A base-model dir whose tokenizer.json splits on whitespace: one token
    per word, so expected counts are exact."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["hello", "world", "a", "b", "c", "x", "long", "row", "ping", "pong"]
    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    d = root / "local-base"
    d.mkdir(parents=True)
    tok.save(str(d / "tokenizer.json"))
    return d


def _convo(n_words: int, turns: int = 2) -> list[dict]:
    text = " ".join(["x"] * n_words)
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": text}
        for i in range(turns)
    ]


def test_tokenizer_path_counts_words_plus_template_overhead(tmp_path):
    base = _word_tokenizer_dir(tmp_path)
    ds = _write_dataset(tmp_path / "ds" / "data.parquet", [_convo(3), _convo(10, turns=4)])

    est = estimate_longest_row_tokens([ds], base_model=str(base), project_dir=tmp_path)

    assert est.source == "tokenizer"
    assert est.rows == 2
    assert est.longest_tokens == 4 * 10 + PER_CONVERSATION_OVERHEAD + 4 * PER_MESSAGE_OVERHEAD
    assert est.rows_over_ceiling == 0


def test_local_base_model_is_resolved_relative_to_the_project(tmp_path):
    base = _word_tokenizer_dir(tmp_path)
    ds = _write_dataset(tmp_path / "ds" / "data.parquet", [_convo(5)])
    est = estimate_longest_row_tokens(
        [ds], base_model=base.name, project_dir=tmp_path
    )
    assert est.source == "tokenizer"


def test_rows_over_the_ceiling_are_counted_not_clipped(tmp_path):
    base = _word_tokenizer_dir(tmp_path)
    ds = _write_dataset(tmp_path / "ds" / "data.parquet", [_convo(50), _convo(120), _convo(200)])
    est = estimate_longest_row_tokens(
        [ds], base_model=str(base), project_dir=tmp_path, ceiling=150
    )
    assert est.rows_over_ceiling == 2
    assert est.longest_tokens > 200  # the true longest, not the ceiling


def test_tools_column_and_tool_calls_are_part_of_the_row(tmp_path):
    base = _word_tokenizer_dir(tmp_path)
    convo = [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "pong"}]},
    ]
    with_tools = _write_dataset(
        tmp_path / "t" / "data.parquet", [convo], tools=[[{"name": "pong", "x": "a b c"}]]
    )
    without = _write_dataset(tmp_path / "n" / "data.parquet", [convo])
    a = estimate_longest_row_tokens([with_tools], base_model=str(base), project_dir=tmp_path)
    b = estimate_longest_row_tokens([without], base_model=str(base), project_dir=tmp_path)
    assert a.longest_tokens > b.longest_tokens


def test_hub_models_fall_back_to_characters_when_unreachable(tmp_path):
    """The unit conftest stubs the Hub fetch; a repo id therefore takes the
    character path. Never raises, never touches the network."""
    convo = _convo(4)
    ds = _write_dataset(tmp_path / "ds" / "data.parquet", [convo])
    est = estimate_longest_row_tokens(
        [ds], base_model="LiquidAI/LFM2.5-1.2B-Instruct", project_dir=tmp_path
    )
    chars = sum(len(m["content"]) for m in convo)
    assert est.source == "chars"
    assert est.longest_tokens == (
        chars // CHARS_PER_TOKEN + PER_CONVERSATION_OVERHEAD + 2 * PER_MESSAGE_OVERHEAD
    )


def test_hub_fetch_is_used_when_available(tmp_path, monkeypatch):
    base = _word_tokenizer_dir(tmp_path)
    monkeypatch.setattr(
        seq_length, "_hub_tokenizer_file",
        lambda repo_id, project_dir: str(base / "tokenizer.json"),
    )
    ds = _write_dataset(tmp_path / "ds" / "data.parquet", [_convo(7)])
    est = estimate_longest_row_tokens(
        [ds], base_model="LiquidAI/LFM2.5-1.2B-Instruct", project_dir=tmp_path
    )
    assert est.source == "tokenizer"
    assert est.longest_tokens == 2 * 7 + PER_CONVERSATION_OVERHEAD + 2 * PER_MESSAGE_OVERHEAD


def test_unreadable_files_and_multipart_content_do_not_fail_the_estimate(tmp_path):
    base = _word_tokenizer_dir(tmp_path)
    multipart = [[
        {"role": "user", "content": [
            {"type": "text", "text": "a b c"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]},
        {"role": "assistant", "content": "x"},
    ]]
    ds = _write_dataset(tmp_path / "ds" / "data.parquet", multipart)
    est = estimate_longest_row_tokens(
        [tmp_path / "missing.parquet", ds], base_model=str(base), project_dir=tmp_path
    )
    assert est.rows == 1
    assert est.longest_tokens == 4 + PER_CONVERSATION_OVERHEAD + 2 * PER_MESSAGE_OVERHEAD


def test_empty_input_is_a_zero_estimate(tmp_path):
    est = estimate_longest_row_tokens([], base_model="", project_dir=tmp_path)
    assert (est.longest_tokens, est.rows, est.rows_over_ceiling) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Sandbox side: exact lengths through the chat template (data_utils), driven
# here with a duck-typed tokenizer since transformers is a train extra.
# ---------------------------------------------------------------------------


class _FakeChatTokenizer:
    """apply_chat_template that returns one id per message plus one per tool,
    in whichever return shape the transformers version at hand uses."""

    def __init__(self, as_dict: bool = False, accepts_tools: bool = True):
        self.as_dict = as_dict
        self.accepts_tools = accepts_tools
        self.calls: list[tuple] = []

    def apply_chat_template(self, messages, tokenize=True, **kw):
        if not self.accepts_tools and "tools" in kw:
            raise TypeError("unexpected keyword argument 'tools'")
        tools = kw.get("tools") or []
        self.calls.append((len(messages), len(tools)))
        ids = list(range(len(messages) + len(tools)))
        return {"input_ids": ids} if self.as_dict else ids


def test_tokenized_row_lengths_decodes_the_tools_column():
    from lqh.train.data_utils import tokenized_row_lengths

    rows = [
        {"messages": [{"role": "user", "content": "a"}], "tools": None},
        {"messages": [{"role": "user", "content": "a"}] * 3,
         "tools": json.dumps([{"name": "t1"}, {"name": "t2"}])},
    ]
    tok = _FakeChatTokenizer()
    assert tokenized_row_lengths(rows, tok) == [1, 5]
    assert tok.calls == [(1, 0), (3, 2)]


def test_tokenized_row_lengths_handles_dict_returns_and_no_tools_kwarg():
    from lqh.train.data_utils import tokenized_row_lengths

    rows = [{"messages": [{"role": "user", "content": "a"}] * 2}]
    assert tokenized_row_lengths(rows, _FakeChatTokenizer(as_dict=True)) == [2]
    assert tokenized_row_lengths(rows, _FakeChatTokenizer(accepts_tools=False)) == [2]
    assert tokenized_row_lengths([], _FakeChatTokenizer()) == []


def test_chars_fallback_counts_non_ascii_one_to_one(tmp_path):
    """A CJK row must not be estimated at a third of its size."""
    from lqh.train.seq_length import _chars_estimate

    assert _chars_estimate(["abcdef"]) == 2
    assert _chars_estimate(["日本語のテキスト"]) == 8
    assert _chars_estimate(["abc", "日本"]) == 1 + 2


def test_tokenized_row_lengths_only_retries_for_a_missing_tools_kwarg():
    """A TypeError from inside the template is a real failure, not a signal
    to silently drop the tools from the measurement."""
    from lqh.train.data_utils import tokenized_row_lengths

    class Broken:
        def apply_chat_template(self, messages, tokenize=True, **kw):
            raise TypeError("unsupported operand type(s) for +: 'NoneType' and 'str'")

    rows = [{"messages": [{"role": "user", "content": "a"}],
             "tools": json.dumps([{"name": "t"}])}]
    with pytest.raises(TypeError):
        tokenized_row_lengths(rows, Broken())


# ---------------------------------------------------------------------------
# resolve_text_seq_length — the trainer's exact pass
# ---------------------------------------------------------------------------


class _LenTokenizer:
    """One token per character of content, no template overhead."""

    def apply_chat_template(self, messages, tokenize=True, **kw):
        return list(range(sum(len(m["content"]) for m in messages)))


def _rows(*lengths: int) -> list[dict]:
    return [{"messages": [{"role": "user", "content": "x" * n}]} for n in lengths]


def test_resolve_lowers_the_estimate_to_the_exact_bucket():
    from lqh.train.seq_length import resolve_text_seq_length

    cfg = {"max_seq_length": 8192, "auto_seq_length": True}
    logs: list[str] = []
    res = resolve_text_seq_length(cfg, _LenTokenizer(), _rows(100, 3000), _rows(50), log=logs.append)
    assert res.changed is True
    assert cfg["max_seq_length"] == res.max_seq_length == 3072
    assert cfg["longest_row_tokens"] == 3000
    assert (res.dropped_train, res.dropped_eval) == (0, 0)
    assert len(res.train_rows) == 2 and len(res.eval_rows) == 1
    assert any("sequence length: 3072" in line for line in logs)


def test_resolve_raises_above_the_estimate_when_the_data_is_longer():
    """The estimate was low (e.g. the character fallback); the exact value
    wins and the calibration probe is what guards the GPU choice."""
    from lqh.train.seq_length import resolve_text_seq_length

    cfg = {"max_seq_length": 1024, "auto_seq_length": True}
    res = resolve_text_seq_length(cfg, _LenTokenizer(), _rows(5000), [], log=lambda s: None)
    assert res.max_seq_length == 5120


def test_resolve_drops_rows_over_the_ceiling_and_reports_counts():
    from lqh.train import defaults
    from lqh.train.seq_length import resolve_text_seq_length

    ceiling = defaults.MAX_SEQ_LENGTH_CEILING
    cfg = {"max_seq_length": ceiling, "auto_seq_length": True}
    logs: list[str] = []
    res = resolve_text_seq_length(
        cfg, _LenTokenizer(), _rows(10, ceiling + 1, ceiling + 5), _rows(ceiling + 2, 7),
        log=logs.append,
    )
    assert res.max_seq_length == ceiling
    assert (res.dropped_train, res.dropped_eval) == (2, 1)
    assert len(res.train_rows) == 1 and len(res.eval_rows) == 1
    assert any("skipped 2 train / 1 eval" in line for line in logs)


def test_resolve_warns_when_every_eval_row_is_dropped():
    from lqh.train import defaults
    from lqh.train.seq_length import resolve_text_seq_length

    ceiling = defaults.MAX_SEQ_LENGTH_CEILING
    cfg = {"max_seq_length": ceiling, "auto_seq_length": True}
    logs: list[str] = []
    res = resolve_text_seq_length(
        cfg, _LenTokenizer(), _rows(10), _rows(ceiling + 1), log=logs.append
    )
    assert res.eval_rows == []
    assert any("NO in-training eval" in line for line in logs)


def test_resolve_fails_clearly_when_no_train_row_fits():
    from lqh.train import defaults
    from lqh.train.seq_length import resolve_text_seq_length

    ceiling = defaults.MAX_SEQ_LENGTH_CEILING
    cfg = {"max_seq_length": ceiling, "auto_seq_length": True}
    with pytest.raises(ValueError, match="Every training conversation"):
        resolve_text_seq_length(cfg, _LenTokenizer(), _rows(ceiling + 1), [], log=lambda s: None)


def test_resolve_caps_at_the_model_context_window():
    """An arbitrary Hub model with a 2k window must never be fed 4k positions."""
    from lqh.train.seq_length import resolve_text_seq_length

    cfg = {"max_seq_length": 8192, "auto_seq_length": True}
    logs: list[str] = []
    res = resolve_text_seq_length(
        cfg, _LenTokenizer(), _rows(100, 4000), [], model_max_positions=2048, log=logs.append
    )
    assert res.max_seq_length == cfg["max_seq_length"] == 2048
    assert res.dropped_train == 1
    assert any("context window" in line for line in logs)


def test_resolve_leaves_pinned_configs_untouched():
    from lqh.train.seq_length import resolve_text_seq_length

    for cfg in ({"max_seq_length": 2048}, {"max_seq_length": 16384, "auto_seq_length": False}):
        before = dict(cfg)
        rows = _rows(100_000)
        res = resolve_text_seq_length(cfg, _LenTokenizer(), rows, [], log=lambda s: None)
        assert res.changed is False
        assert cfg == before
        assert res.train_rows is rows  # nothing dropped: trl truncates as before


def test_resolve_keeps_the_estimate_when_measurement_fails():
    from lqh.train.seq_length import resolve_text_seq_length

    class Exploding:
        def apply_chat_template(self, *a, **k):
            raise RuntimeError("template blew up")

    cfg = {"max_seq_length": 4096, "auto_seq_length": True}
    logs: list[str] = []
    res = resolve_text_seq_length(cfg, Exploding(), _rows(10), [], log=logs.append)
    assert res.changed is False
    assert cfg["max_seq_length"] == 4096
    assert any("truncated" in line for line in logs)
