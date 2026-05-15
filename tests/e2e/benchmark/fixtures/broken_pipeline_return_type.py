"""Broken pipeline: Conversation() constructor misuse (TypeError)."""

from lqh.pipeline import Pipeline, ChatMLMessage, Conversation
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
                {"role": "system", "content": "Translate into German, French, Spanish, English, and Chinese. Return ONLY JSON with keys: de, fr, es, en, zh."},
                {"role": "user", "content": source_text},
            ],
            response_format={"type": "json_object"},
        )
        translations = resp.choices[0].message.content.strip()

        # BUG: Conversation is a type alias for list[ChatMLMessage], not a class
        return Conversation(messages=[
            ChatMLMessage("user", source_text),
            ChatMLMessage("assistant", translations),
        ])
