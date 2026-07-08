"""Broken pipeline: wrong import path (ModuleNotFoundError)."""

from data_gen.base import Pipeline, ChatMLMessage, Conversation
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
                    f"Write a short sentence (1-2 sentences) that a {persona.brief()} "
                    f"would write. Output ONLY the text."
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

        return [
            ChatMLMessage("user", source_text),
            ChatMLMessage("assistant", translations),
        ]
