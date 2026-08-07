"""The shipped package must not name the infrastructure it runs on.

Feedback #45: the agent told a user which cloud provider was "probably"
behind LQH Cloud, having inferred it from a phrase in a skill doc. The
whole `lqh/` tree is user-reachable — skill docs and tool descriptions go
straight into the model's context, and the source itself ships to PyPI —
so the provider name must not appear anywhere in it.

This is a wording guard, not a behavior test: the word keeps drifting in
from the backend repo, where it is legitimately everywhere.
"""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "lqh"

# The provider name, spelled so the guard can't match its own source.
_PROVIDER = "mo" + "dal"

# "modality", "multimodal", "cross-modal", "bimodal" are legitimate ML
# terms that merely contain it — require a standalone word, hyphen included.
_HIT = re.compile(rf"(?<![a-z-]){_PROVIDER}(?![a-z])", re.IGNORECASE)

_TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml"}


def _scannable_files() -> list[Path]:
    files = [
        p for p in sorted(PACKAGE_ROOT.rglob("*"))
        if p.is_file()
        and p.suffix in _TEXT_SUFFIXES
        and "__pycache__" not in p.parts
    ]
    # pyproject only-includes ["lqh", "README.md"] — the readme ships too.
    readme = PACKAGE_ROOT.parent / "README.md"
    if readme.is_file():
        files.append(readme)
    return files


def test_package_never_names_the_compute_provider():
    files = _scannable_files()
    assert files, f"nothing scanned under {PACKAGE_ROOT} — the guard is broken"

    hits: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if _HIT.search(line):
                rel = path.relative_to(PACKAGE_ROOT.parent)
                hits.append(f"{rel}:{lineno}: {line.strip()}")

    assert not hits, (
        "the compute provider is named in the shipped package "
        "(agent-visible and published to PyPI) — use neutral wording "
        "such as 'cloud sandbox' / 'the cloud image':\n" + "\n".join(hits)
    )


def test_guard_would_catch_a_reintroduced_mention(tmp_path: Path):
    """Negative control: the regex must match a real sentence and skip the
    ML vocabulary that merely contains the same letters."""
    assert _HIT.search(f"receives HF_TOKEN as a {_PROVIDER} secret")
    assert _HIT.search(f"the durable {_PROVIDER.capitalize()} volume")
    assert not _HIT.search("multimodal collate_fn for the VLM path")
    assert not _HIT.search("a cross-modal retrieval eval")
    assert not _HIT.search("Modality: normally set by handle_start_training")
    assert not _HIT.search("the distribution is bimodal")
