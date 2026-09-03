"""assistant_only_loss labels assistant turns and nothing else."""

from __future__ import annotations

import pytest

MESSAGES = [
    {
        "role": "user",
        "content": (
            "project 01 | tempo 70\nStereo Out | out | vol 6.0 | clip\n"
            "Why is the master clipping?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "findings:\n- [critical] Stereo Out | clip: clipping latched at 6 dB\n"
            '<|tool_call_start|>[set_volume(track="Stereo Out", db=0.0)]<|tool_call_end|>'
        ),
    },
    {"role": "user", "content": "And the bass?"},
    {"role": "assistant", "content": "Trap Bass sits at -3 dB and does not clip."},
]

# The shape of LFM2.5's template: the assistant header is outside the block, so
# only the reply body and its terminator carry a live label.
TEMPLATE = (
    "{%- for m in messages -%}"
    "{{- '<|im_start|>' + m.role + '\n' -}}"
    "{%- if m.role == 'assistant' -%}{%- generation -%}"
    "{{- m.content + '<|im_end|>\n' -}}"
    "{%- endgeneration -%}"
    "{%- else -%}{{- m.content + '<|im_end|>\n' -}}{%- endif -%}"
    "{%- endfor -%}"
)


def make_tokenizer(template: str = TEMPLATE):
    """A character-level tokenizer built in memory: no hub, no cache, exact offsets."""
    pytest.importorskip("transformers")
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    chars = [chr(c) for c in range(32, 127)] + ["\n"]
    inner = Tokenizer(
        models.WordLevel(
            vocab={t: i for i, t in enumerate(["<|pad|>", "<|unk|>", *chars])},
            unk_token="<|unk|>",
        )
    )
    inner.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    inner.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=inner,
        pad_token="<|pad|>",
        eos_token="\n",
        chat_template=template,
    )


@pytest.fixture(scope="module")
def tokenizer():
    return make_tokenizer()


def encode(tokenizer, messages=MESSAGES):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )


def test_template_marks_only_the_assistant_replies(tokenizer):
    encoded = encode(tokenizer)
    ids, mask = encoded["input_ids"], encoded["assistant_masks"]
    assert len(ids) == len(mask)
    assert 0 < sum(mask) < len(ids)

    live = tokenizer.decode([i for i, m in zip(ids, mask) if m])
    assert live == "".join(
        f"{m['content']}<|im_end|>\n" for m in MESSAGES if m["role"] == "assistant"
    )
    masked = tokenizer.decode([i for i, m in zip(ids, mask) if not m])
    for m in MESSAGES:
        if m["role"] == "user":
            assert m["content"] in masked
    assert "<|im_start|>assistant" in masked


def test_collator_masks_every_non_assistant_token(tokenizer):
    pytest.importorskip("torch")
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    encoded = encode(tokenizer)
    ids, mask = encoded["input_ids"], encoded["assistant_masks"]
    batch = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)(
        [{"input_ids": ids, "assistant_masks": mask}]
    )
    labels = batch["labels"][0]
    assert (labels == -100).tolist() == [m == 0 for m in mask]
