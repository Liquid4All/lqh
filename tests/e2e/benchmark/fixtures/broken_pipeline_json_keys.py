"""Broken pipeline: wrong JSON key names in validation (GenerationError every time)."""

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
                {"role": "system", "content": "Translate into German, French, Spanish, English, and Chinese. Return ONLY JSON with keys: de, fr, es, en, zh."},
                {"role": "user", "content": source_text},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)

        # BUG: Checking for full language names instead of ISO codes
        # The LLM returns keys like "de", "fr", etc. but we check for "german", "french"
        required_keys = {"german", "french", "spanish", "english", "chinese"}
        if not required_keys.issubset(data.keys()):
            raise GenerationError(f"Missing keys: {required_keys - set(data.keys())}")

        translations = json.dumps(data, ensure_ascii=False)

        return [
            ChatMLMessage("user", source_text),
            ChatMLMessage("assistant", translations),
        ]
