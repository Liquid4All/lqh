"""Category: Data Filtering (bring-your-own-data-for-scoring).

Tests the v0.3.1 `run_data_filter` flow: the user brings their own parquet of
ChatML samples of mixed quality and wants only the good ones kept. The agent
should turn the user's intent into a scorer, show it, and run `run_data_filter`
to produce a filtered dataset (data.parquet + scores.parquet + summary.json).

This is the bring-your-data path the `data_filtering` skill governs — distinct
from filtering data we just generated this session (that's `run_scoring`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tests.e2e.scenarios import Scenario


def _write_mixed_quality_parquet(path: Path, num_good: int, num_bad: int) -> None:
    """Write a ChatML parquet with a mix of good and clearly-bad samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []

    for i in range(num_good):
        rows.append(json.dumps([
            {"role": "user", "content": f"My order #{1000 + i} hasn't arrived yet, can you help?"},
            {"role": "assistant", "content": (
                f"I'm sorry for the delay on order #{1000 + i}. I've checked the "
                "tracking and it's currently in transit, expected to arrive within "
                "2 business days. I've also added a note to expedite it. Is there "
                "anything else I can help with?"
            )},
        ]))

    # Clearly low-quality samples: refusals, wrong language, empty, off-topic.
    bad_templates = [
        [{"role": "user", "content": "I was charged twice this month."},
         {"role": "assistant", "content": "I can't help with that."}],
        [{"role": "user", "content": "How do I reset my password?"},
         {"role": "assistant", "content": "Lo siento, no puedo ayudarte con eso ahora."}],
        [{"role": "user", "content": "My app keeps crashing on startup."},
         {"role": "assistant", "content": ""}],
        [{"role": "user", "content": "Can I get a refund for my subscription?"},
         {"role": "assistant", "content": "The weather is lovely today, thanks for asking!"}],
    ]
    for i in range(num_bad):
        rows.append(json.dumps(bad_templates[i % len(bad_templates)]))

    n = len(rows)
    table = pa.table(
        {"messages": rows, "audio": [None] * n, "tools": [None] * n},
        schema=pa.schema([
            pa.field("messages", pa.string()),
            pa.field("audio", pa.string()),
            pa.field("tools", pa.string()),
        ]),
    )
    pq.write_table(table, path)


def _seed_byo_dataset(project_dir: Path) -> None:
    """Place a user-brought, unscored, mixed-quality dataset to be filtered."""
    _write_mixed_quality_parquet(
        project_dir / "imported" / "support_replies.parquet",
        num_good=24,
        num_bad=16,
    )


_FILTER_USER = (
    "You are a user who brought your own dataset of 40 customer-support "
    "conversations (ChatML `messages`) saved at `imported/support_replies.parquet`. "
    "About a third are low quality — refusals, wrong-language replies, empty or "
    "off-topic answers — and you want to KEEP ONLY the good ones before training. "
    "A good reply is helpful, on-topic, in English, and actually resolves the "
    "customer's issue.\n\n"
    "Behavior rules:\n"
    "- When the agent asks what makes a sample good, give the criteria above\n"
    "- When the agent proposes a scorer or threshold, say 'that sounds good'\n"
    "- When shown the scorer file, approve it ('looks good')\n"
    "- When shown a dry-run / sample of kept vs dropped rows, say 'looks right, "
    "run it on the full set'\n"
    "- Do NOT suggest specific tools or thresholds yourself\n"
    "- After the full filtered dataset is produced and counts are reported, say "
    "'I'm done for now'"
)


DATA_FILTER_SUPPORT = Scenario(
    name="bench_filter_support_replies",
    description=_FILTER_USER,
    initial_message=(
        "I have a dataset of customer-support replies at "
        "imported/support_replies.parquet. A bunch of them are bad — filter it "
        "down to only the good ones so I can train on the rest."
    ),
    expected_tools=["create_file", "run_data_filter"],
    expected_files=[],  # BYO-data filtering produces no SPEC.md
    judge_criteria=(
        "Score the filter scorer file the agent wrote (e.g. evals/filter_*.md or "
        "evals/scorers/*.md) for how well it captures the user's quality bar for "
        "customer-support replies.\n"
        "Check for:\n"
        "- A numeric scoring scale and an explicit keep/drop rule\n"
        "- Criteria covering helpfulness/relevance, correct language (English), "
        "non-empty/on-topic answers\n"
        "- Drop conditions for refusals, wrong-language, and empty replies\n\n"
        "10 = precise, actionable filter rubric, 5 = vague, 1 = unusable/irrelevant"
    ),
    max_turns=30,
    stage_limits={"data_filtering": 25},
    seed_fn=_seed_byo_dataset,
)


SCENARIOS = [
    DATA_FILTER_SUPPORT,
]
