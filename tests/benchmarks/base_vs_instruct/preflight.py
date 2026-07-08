"""Transformers v4→v5 compatibility preflight for benchmark models.

The HF transformers v4→v5 transition changed the library API (e.g.
``from_pretrained(torch_dtype=...)`` → ``dtype=...``) *and* bumped the minimum
version some model architectures need. This benchmark deliberately mixes
generations — the older ``LiquidAI/LFM2-*`` alongside the newer
``LiquidAI/LFM2.5-*`` — precisely to compare their fine-tuneability. The risk:
a model whose ``model_type`` is not registered in the *installed* transformers,
or whose chat template doesn't render, fails deep inside a GPU subprocess —
**after** datagen and possibly hours of sweep time have already been spent.

This module front-loads that check. For each model id it loads only the
**config and tokenizer** (no weights — a couple of small file downloads) under
the installed transformers and verifies:
  1. the config loads and its ``model_type`` maps to a registered
     causal-LM class (the architecture is known to this transformers), and
  2. the chat template renders a tiny system+user conversation
     (the documented ``-Base`` concern — those repos ship a separate
     ``chat_template.jinja`` that only transformers ≥4.43 / 5.x picks up).

It also surfaces the ``transformers_version`` each repo was *saved* with, so a
v4-saved vs v5-saved mix is visible at a glance. The decision logic
(:func:`verdict`) is a pure function so it can be unit-tested offline; the IO
wrapper (:func:`check_model_compat`) hits the hub.

If transformers isn't importable in the orchestrator process, the preflight is
skipped with a warning (the GPU subprocess will surface the real error) rather
than blocking the run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("bvi.preflight")


@dataclass
class ModelCompat:
    key: str
    hf_id: str
    ok: bool
    model_type: str | None = None
    saved_with: str | None = None  # config.transformers_version (the repo's v4/v5 origin)
    causal_lm_ok: bool = False
    chat_template_ok: bool = False
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def verdict(
    *,
    config_loaded: bool,
    model_type: str | None,
    causal_lm_ok: bool,
    chat_template_ok: bool,
    load_error: str | None,
) -> tuple[bool, list[str]]:
    """Pure compatibility decision from the probed facts. Returns (ok, notes).

    A model is compatible iff its config loaded, its architecture maps to a
    causal-LM class in the installed transformers, and its chat template
    renders. Each failure mode gets a specific, actionable note.
    """
    notes: list[str] = []
    if not config_loaded:
        notes.append(
            "config failed to load under the installed transformers — likely an "
            f"architecture this version doesn't know{f' ({load_error})' if load_error else ''}. "
            "Upgrade transformers (this benchmark targets >=5,<6)."
        )
        return False, notes

    ok = True
    if not causal_lm_ok:
        ok = False
        notes.append(
            f"model_type {model_type!r} has no causal-LM mapping in this "
            "transformers — the architecture is unknown/removed at this version. "
            "Upgrade transformers."
        )
    if not chat_template_ok:
        ok = False
        notes.append(
            "tokenizer has no working chat template — apply_chat_template failed. "
            "For -Base repos this means the separate chat_template.jinja wasn't "
            "loaded (needs transformers >=4.43 / 5.x), or no template ships at all."
        )
    return ok, notes


def installed_transformers_version() -> str | None:
    try:
        import transformers
    except Exception:  # pragma: no cover - exercised only without the train extra
        return None
    return getattr(transformers, "__version__", None)


def check_model_compat(key: str, hf_id: str) -> ModelCompat:
    """Probe a single model's config + tokenizer (no weights) for compatibility."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        return ModelCompat(key=key, hf_id=hf_id, ok=True, notes=[
            f"transformers not importable here ({exc}); preflight skipped "
            "(the GPU subprocess will load the model for real)."
        ])

    model_type: str | None = None
    saved_with: str | None = None
    causal_lm_ok = False
    chat_template_ok = False
    config_loaded = False
    load_error: str | None = None

    try:
        cfg = AutoConfig.from_pretrained(hf_id)
        config_loaded = True
        model_type = getattr(cfg, "model_type", None)
        saved_with = getattr(cfg, "transformers_version", None)
        try:
            causal_lm_ok = type(cfg) in AutoModelForCausalLM._model_mapping
        except Exception:
            # Be lenient if the private mapping shape changes across versions:
            # a config that loaded at all is very likely usable.
            causal_lm_ok = True
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"

    if config_loaded:
        try:
            tok = AutoTokenizer.from_pretrained(hf_id)
            rendered = tok.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            chat_template_ok = bool(rendered)
        except Exception as exc:
            load_error = load_error or f"{type(exc).__name__}: {exc}"

    ok, notes = verdict(
        config_loaded=config_loaded,
        model_type=model_type,
        causal_lm_ok=causal_lm_ok,
        chat_template_ok=chat_template_ok,
        load_error=load_error,
    )
    return ModelCompat(
        key=key, hf_id=hf_id, ok=ok, model_type=model_type, saved_with=saved_with,
        causal_lm_ok=causal_lm_ok, chat_template_ok=chat_template_ok,
        error=load_error, notes=notes,
    )


def run_preflight(models: list[tuple[str, str]]) -> list[ModelCompat]:
    """Probe every model, log a per-model report, and return the results.

    The caller decides whether to abort on an incompatibility (see ``run.py``'s
    ``--skip-preflight``). Logs the installed transformers version and, per
    model, the version it was saved with so a v4/v5 mix is obvious.
    """
    ver = installed_transformers_version()
    if ver is None:
        logger.warning(
            "preflight: transformers not importable in the orchestrator; "
            "skipping model compatibility checks (the GPU subprocess will load "
            "models for real and surface any error there).",
        )
        return [ModelCompat(key=k, hf_id=h, ok=True, notes=["preflight skipped"]) for k, h in models]

    logger.info("preflight: installed transformers %s — probing %d model(s)", ver, len(models))
    results: list[ModelCompat] = []
    for key, hf_id in models:
        r = check_model_compat(key, hf_id)
        results.append(r)
        status = "OK" if r.ok else "INCOMPATIBLE"
        logger.info(
            "preflight: %-16s %-30s [%s] model_type=%s saved_with=%s chat_template=%s",
            key, r.hf_id, status, r.model_type, r.saved_with,
            "ok" if r.chat_template_ok else "FAIL",
        )
        for note in r.notes:
            (logger.error if not r.ok else logger.info)("  - %s", note)
    return results
