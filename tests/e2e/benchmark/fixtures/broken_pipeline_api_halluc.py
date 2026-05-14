"""Broken pipeline: calls non-existent client method (AttributeError)."""

from lqh.pipeline import Pipeline, ChatMLMessage, Conversation, GenerationError, step
import json
import liquidrandom


class TranslationPipeline(Pipeline):
    """Generate translation training samples."""

    async def generate(self, client, input=None) -> Conversation:
        # BUG: client.models.list() doesn't exist on the AsyncOpenAI client
        # used in lqh pipelines. This is a hallucination of the OpenAI API.
        available_models = await client.models.list()
        model_id = available_models.data[0].id

        persona = liquidrandom.persona()

        resp = await client.chat.completions.create(
            model=model_id,
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

        return [
            ChatMLMessage("user", source_text),
            ChatMLMessage("assistant", translations),
        ]
