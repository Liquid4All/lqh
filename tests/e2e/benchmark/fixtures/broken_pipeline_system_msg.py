"""Broken pipeline: system message in output Conversation (forbidden)."""

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
                "content": (
                    f"Write a short sentence that a {persona.brief()} would write. "
                    f"Output ONLY the text."
                ),
            }],
        )
        source_text = resp.choices[0].message.content.strip()

        resp = await client.chat.completions.create(
            model="random:medium",
            messages=[
                {"role": "system", "content": "Translate into 5 languages. Return JSON with keys: de, fr, es, en, zh."},
                {"role": "user", "content": source_text},
            ],
            response_format={"type": "json_object"},
        )
        translations = resp.choices[0].message.content.strip()

        # BUG: Including a system message in the output Conversation.
        # System messages should NOT be in the training data output -
        # system prompts are managed separately via prompts/ directory.
        return [
            ChatMLMessage("system", "You are a professional translator. Translate text into German, French, Spanish, English, and Chinese as JSON."),
            ChatMLMessage("user", source_text),
            ChatMLMessage("assistant", translations),
        ]
