"""Location URI grammar shared by the pull/push tools.

One grammar, three explicit schemes — the scheme is mandatory and never
inferred, so there is no "auto-detect by probing":

    local path           e.g.  runs/run_001/model
    hf:owner/repo[@rev]  e.g.  hf:meta-llama/Llama-3.2-1B@main
    lqh:<artifact_id>    e.g.  lqh:6f3a1c...   (an R2 artifact)
"""

from __future__ import annotations

from dataclasses import dataclass

_KNOWN_SCHEMES = ("hf", "lqh")


@dataclass(frozen=True)
class Location:
    scheme: str  # "hf" | "lqh" | "local"
    value: str   # hf -> "owner/repo"; lqh -> artifact_id; local -> path
    revision: str | None = None  # hf only (the part after '@')


class LocationError(ValueError):
    """Raised for a malformed or unknown-scheme location string."""


def parse_location(s: str) -> Location:
    """Parse a location string into a Location. Raises LocationError on a
    malformed scheme. A bare path with no recognized scheme is treated as
    a local path."""
    raw = (s or "").strip()
    if not raw:
        raise LocationError("empty location")

    for scheme in _KNOWN_SCHEMES:
        prefix = scheme + ":"
        if raw.startswith(prefix):
            rest = raw[len(prefix):]
            if not rest:
                raise LocationError(f"{scheme}: location is missing its value")
            if scheme == "hf":
                repo, _, rev = rest.partition("@")
                if "/" not in repo:
                    raise LocationError(
                        f"hf location must be 'hf:owner/repo[@rev]', got {raw!r}"
                    )
                return Location("hf", repo, rev or None)
            return Location("lqh", rest)

    # No recognized scheme. Reject an unknown 'scheme:...' form (a
    # URI-scheme-like token before a colon, no slash) to avoid silently
    # treating a typo'd scheme as a path. Single-char heads (Windows
    # drive letters) are allowed through as local paths.
    head, sep, _ = raw.partition(":")
    if sep and "/" not in head and len(head) > 1 and _looks_like_scheme(head):
        raise LocationError(
            f"unknown location scheme {head!r}: in {raw!r}; "
            "use hf:owner/repo, lqh:<artifact_id>, or a local path"
        )
    return Location("local", raw)


def _looks_like_scheme(head: str) -> bool:
    """True if head matches the URI scheme charset: ALPHA *( ALPHA / DIGIT
    / '+' / '-' / '.' ). Catches s3, http, gs, ... while letting odd
    local filenames through only when they don't look scheme-like."""
    if not head or not head[0].isalpha():
        return False
    return all(c.isalnum() or c in "+-." for c in head)
