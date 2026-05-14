"""Skill loader for lqh agent modes."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def list_available_skills() -> list[dict[str, str]]:
    """List all available skills with name and description."""
    skills = [
        {
            "name": "auto",
            "description": "Run the full customization pipeline autonomously (auto mode)",
            "command": "/auto",
        },
        {
            "name": "spec_capture",
            "description": "Interview the user to understand requirements and create SPEC.md",
            "command": "/spec",
        },
        {
            "name": "data_generation",
            "description": "Create data generation pipelines from specifications",
            "command": "/datagen",
        },
        {
            "name": "data_validation",
            "description": "Create validation criteria for generated datasets",
            "command": "/validate",
        },
        {
            "name": "data_filtering",
            "description": "Score a user-brought dataset and emit a filtered subset",
            "command": "/filter",
        },
        {
            "name": "train",
            "description": "Fine-tune a model on generated data (requires torch)",
            "command": "/train",
        },
        {
            "name": "evaluation",
            "description": "Create evaluation suites for trained models",
            "command": "/eval",
        },
        {
            "name": "prompt_optimization",
            "description": "Create and optimize system prompts via iterative eval refinement",
            "command": "/prompt",
        },
    ]
    return skills


# Aliases for common variations the orchestration model may use.
_SKILL_ALIASES: dict[str, str] = {
    "eval": "evaluation",
    "/eval": "evaluation",
    "eval_setup": "evaluation",
    "scoring": "evaluation",
    "spec": "spec_capture",
    "/spec": "spec_capture",
    "datagen": "data_generation",
    "/datagen": "data_generation",
    "data_gen": "data_generation",
    "validate": "data_validation",
    "/validate": "data_validation",
    "filter": "data_filtering",
    "/filter": "data_filtering",
    "data_filter": "data_filtering",
    "prompt": "prompt_optimization",
    "/prompt": "prompt_optimization",
    "prompt_opt": "prompt_optimization",
    "/train": "train",
}


def load_skill_content(skill_name: str) -> str:
    """Load a skill's SKILL.md content.

    Accepts the canonical skill name or common aliases (e.g., ``"eval"``
    resolves to ``"evaluation"``).

    Returns the markdown content that should be injected as a system message.
    Raises FileNotFoundError if the skill doesn't exist.
    """
    resolved = _SKILL_ALIASES.get(skill_name, skill_name)
    skill_dir = Path(__file__).parent / resolved
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        available = [s["name"] for s in list_available_skills()]
        raise FileNotFoundError(
            f"Skill '{skill_name}' not found. Available skills: {', '.join(available)}"
        )
    return skill_file.read_text()
