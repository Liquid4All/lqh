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


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "LiquidAI/LFM2.5-350M", local_files_only=True
        )
    except OSError:
        pytest.skip("LiquidAI/LFM2.5-350M is not in the local HF cache")


def test_collator_masks_every_non_assistant_token(tokenizer):
    pytest.importorskip("torch")
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    encoded = tokenizer.apply_chat_template(
        MESSAGES, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    ids, mask = encoded["input_ids"], encoded["assistant_masks"]
    assert len(ids) == len(mask)
    assert 0 < sum(mask) < len(ids)

    batch = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)(
        [{"input_ids": ids, "assistant_masks": mask}]
    )
    labels = batch["labels"][0]
    assert (labels == -100).tolist() == [m == 0 for m in mask]

    live = tokenizer.decode(labels[labels != -100])
    assert live == "".join(
        f"{m['content']}<|im_end|>\n" for m in MESSAGES if m["role"] == "assistant"
    )
    masked = tokenizer.decode(batch["input_ids"][0][labels == -100])
    for m in MESSAGES:
        if m["role"] == "user":
            assert m["content"] in masked
    assert "<|im_start|>assistant" in masked
