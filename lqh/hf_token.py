"""Locate the user's Hugging Face token in the local environment.

Two audiences, one resolver:

* **Local HF tools** (``hf_pull`` / ``hf_push`` / repo info) call
  :func:`local_hf_token` — the token stays on this machine.
* **Cloud submits** call :func:`donatable_hf_token`, which attaches the token
  to a single job as ``meta.hf_token``. The backend uses it for that job and
  never writes it to its database (see ``CLOUD_FT.md``/``cloud_jobs.go``).

Search order, first hit wins::

    HF_TOKEN                       (environment)
    HUGGING_FACE_HUB_TOKEN         (environment)
    $LQH_ENV_FILE                  (escape hatch, absolute or project-relative)
    <project>/.env.local
    <project>/.env
    huggingface_hub.get_token()    (~/.cache/huggingface/token, i.e. `huggingface-cli login`)

Deliberate non-features:

* **We never mutate ``os.environ``.** ``lqh.subprocess_manager`` inherits the
  parent environment into every local training/data-gen subprocess, and
  ``lqh.remote.ssh_direct`` copies ``os.environ["HF_TOKEN"]`` onto a *third-party
  host's disk*. A global ``load_dotenv()`` would silently feed both, unconsented.
* **We only look inside the project directory** — no cwd, no parent walk, no
  ``~/.env``. A user-level env file donated to every project is exactly the
  consent failure this module exists to avoid; ``/hf_login`` already covers the
  "use this everywhere" case by storing the token on the backend.
* **We do not parse ``.envrc``.** That is a direnv *shell script*
  (``export FOO=$(op read ...)``, ``source_up``, conditionals); a dotenv parser
  gives wrong answers on its common forms. direnv users already have the
  variable exported, so the environment lookup catches them first.

## Keeping the token out of the conversation

The plaintext must never reach the LLM — not as a tool argument, not in a tool
result, not in an exception message, and therefore not in the session JSONL or
the backend's payload capture.

That is enforced structurally rather than by redaction: :func:`donatable_hf_token`
and :func:`local_hf_token` are the *only* functions here that return plaintext,
and they are called exclusively from the network clients that consume them
(``lqh.remote.cloud.submit_run``, ``lqh.remote.transfer``,
``lqh.remote.gguf_convert``) and from the local ``huggingface_hub`` wrappers.
Everything user-facing — prompts, tool results, the status bar — calls
:func:`hf_token_origin` or :func:`hf_disclosure_line`, which return *provenance*
(a closed enum plus a file path). Those go through ``_resolve(..., want_value=
False)``, which drops the plaintext before returning, so the value is not in
their frames at all — not merely unused by them. It exists in exactly one
function, :func:`_resolve_full`, plus the two public accessors below.

When adding a call site: if a ``ToolResult`` is constructed anywhere in the same
function, you want :func:`hf_token_origin`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "HFTokenOrigin",
    "donatable_hf_token",
    "hf_disclosure_line",
    "hf_token_origin",
    "local_hf_token",
    "redact",
]

# Environment variables, in precedence order. HF_TOKEN is ours and
# huggingface_hub's; HUGGING_FACE_HUB_TOKEN is the older hub-only spelling.
_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

# Project-relative env files, in precedence order. ".local" overriding the
# base file is the Vite/Next convention and the one users expect.
_ENV_FILENAMES = (".env.local", ".env")

# An env file larger than this is not an env file. Guards against a config key
# pointing at a data blob.
_MAX_ENV_FILE_BYTES = 1024 * 1024

# Source kinds. Closed set — these strings reach tool results and the status
# bar, so they must stay free of user data.
KIND_ENV = "env"
KIND_ENVFILE = "envfile"
KIND_HF_CLI = "hf_cli"


@dataclass(frozen=True)
class HFTokenOrigin:
    """Where a token was found. Provenance only — never the value.

    Safe to render in a prompt, a tool result, or the status bar. ``path`` is a
    ``~``-relativized path to the *file*, never its contents.
    """

    kind: str  # KIND_ENV | KIND_ENVFILE | KIND_HF_CLI
    label: str  # human-readable, e.g. "your project .env"
    path: str | None = None
    donation_enabled: bool = True  # False when LQH_HF_DONATE=0

    @property
    def is_hub_cache(self) -> bool:
        """True when the token came from ``huggingface-cli login``.

        Worth distinguishing in consent copy: the user put that token there for
        the Hub CLI, not for us.
        """
        return self.kind == KIND_HF_CLI


# ---------------------------------------------------------------------------
# env-file parsing
# ---------------------------------------------------------------------------


def _unquote(value: str) -> str:
    """Strip one layer of quotes, unescaping only inside double quotes.

    Finds the CLOSING quote rather than requiring it to be the last
    character, so a trailing inline comment doesn't defeat the match:

        HF_TOKEN="hf_abc..."  # private read token

    Anchoring on ``value[-1]`` failed that line, fell through to the
    unquoted branch, and returned the token *with its quotes still
    attached* — which authenticates nowhere. Quoting a secret and
    annotating what it's for are both ordinary things to do, and doing
    both broke the token silently.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] in ("'", '"'):
        quote = value[0]
        close = _closing_quote(value, quote)
        if close != -1:
            inner = value[1:close]
            if quote == '"':
                inner = inner.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
            return inner
    # Unquoted values may carry a trailing inline comment.
    head = value.split(" #", 1)[0]
    return head.strip()


def _closing_quote(value: str, quote: str) -> int:
    """Index of the quote closing the one at position 0, or -1.

    Backslash escapes only apply inside double quotes — POSIX single
    quotes have no escape, so the first `'` closes the string.
    """
    i = 1
    while i < len(value):
        char = value[i]
        if char == "\\" and quote == '"':
            i += 2
            continue
        if char == quote:
            return i
        i += 1
    return -1


# Sentinel for "this level assigned the key, and assigned it nothing".
# Distinct from None (never mentioned here) because the two must behave
# differently: an explicit blank is a revocation and stops the search,
# while silence hands off to the next source. Collapsing them is how a
# cleared token gets donated anyway.
CLEARED = "\x00cleared"


def parse_env_file_value(text: str, key: str) -> str | None:
    """Extract ``key`` from dotenv-style ``text``.

    Returns the value, :data:`CLEARED` if the last assignment is empty, or
    None if the key never appears.

    Grammar deliberately mirrors the *writer* side in :mod:`lqh.env_secrets`:
    blank and ``#`` lines are skipped, a leading ``export`` is tolerated, the
    line splits on its first ``=``, and matched surrounding quotes are stripped.
    No ``${VAR}`` interpolation — we resolve one key, not a shell.

    **Last assignment wins, including an empty one.** ``append_env_secret``
    appends duplicates rather than rewriting (and warns about "the stale
    one"), so first-wins would make the CLI warn about the very entry it
    is using. And because the file is append-only, ``HF_TOKEN=`` at the
    end is how someone *clears* a stale token.
    """
    found: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export ") or line.startswith("export\t"):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        if name.strip() != key:
            continue
        found = _unquote(value.strip()) or CLEARED
    return found


def _read_env_file(path: Path, keys: tuple[str, ...]) -> str | None:
    """First of ``keys`` assigned in ``path``: a value, CLEARED, or None.

    Never raises. An unreadable or oversized file reads as "says nothing"
    rather than "cleared" — we only honour a revocation we actually saw.
    """
    try:
        if not path.is_file():
            return None
        if path.stat().st_size > _MAX_ENV_FILE_BYTES:
            logger.debug("ignoring oversized env file: %s", path)
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.debug("could not read env file %s: %s", path, e)
        return None
    for key in keys:
        value = parse_env_file_value(text, key)
        if value is not None:
            return value
    return None


def _candidate_env_files(project_dir: Path | None) -> list[Path]:
    """Env files to search, in precedence order.

    ``LQH_ENV_FILE`` comes first so a user with a differently-named file
    (``.env.dev``, ``secrets.env``) has a supported way in without us growing a
    hardcoded list. Relative values resolve against the project directory.
    """
    candidates: list[Path] = []
    override = (os.environ.get("LQH_ENV_FILE") or "").strip()
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute() and project_dir is not None:
            p = project_dir / p
        candidates.append(p)
    if project_dir is not None:
        candidates.extend(project_dir / name for name in _ENV_FILENAMES)
    return candidates


def _display_path(path: Path) -> str:
    """``~``-relativized path for display. The path only — never contents."""
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def _hub_cached_token() -> str | None:
    """Token written by ``huggingface-cli login``, via the hub's own resolver.

    ``huggingface_hub`` is a base dependency, and ``get_token()`` already honors
    ``HF_HOME``/``HF_TOKEN_PATH`` and strips whitespace — reimplementing that
    would drift. Deliberately broad except: a broken hub install must degrade to
    "no token", never break a submit.
    """
    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception as e:  # noqa: BLE001 - see docstring
        logger.debug("huggingface_hub.get_token() unavailable: %s", e)
        return None
    return token.strip() if token else None


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def _resolve(
    project_dir: Path | None, *, want_value: bool = True,
) -> tuple[str | None, HFTokenOrigin | None]:
    """Resolve a token, optionally discarding the plaintext.

    Private on purpose — public callers pick a side (value or provenance) so
    that a function handling a ``ToolResult`` never has the value in scope.

    ``want_value=False`` returns ``(None, origin)``. The plaintext is still
    read — there is no way to know a token exists without reading it — but it
    is dropped HERE and never crosses the return boundary, so provenance
    callers cannot hold it even by accident. Before this flag,
    :func:`hf_token_origin` received the plaintext and merely declined to
    use it, which is a weaker guarantee than the surrounding comments
    claimed. The value now lives in exactly one frame: :func:`_resolve_full`.
    """
    token, origin = _resolve_full(project_dir)
    if not want_value:
        # Rebind before returning so the plaintext is unreachable from the
        # caller's frame rather than merely unused by it.
        token = None
    return token, origin


def _resolve_full(project_dir: Path | None) -> tuple[str | None, HFTokenOrigin | None]:
    """The actual search. The only function here that pairs plaintext with origin."""
    # An explicit assignment wins at its own precedence level, even when
    # it is empty. That is what makes revocation work: clearing HF_TOKEN
    # in .env.local has to beat a stale value still sitting in .env, and
    # `export HF_TOKEN=` has to beat both. Falling through on empty would
    # donate exactly the token the user just tried to take away.
    for var in _ENV_VARS:
        if var not in os.environ:
            continue
        value = (os.environ[var] or "").strip()
        if value:
            return value, HFTokenOrigin(kind=KIND_ENV, label=f"your {var} environment variable")
        logger.debug("%s is set but empty — treating as revoked", var)
        return None, None

    for path in _candidate_env_files(project_dir):
        value = _read_env_file(path, _ENV_VARS)
        if value is CLEARED:
            logger.debug("token explicitly cleared in %s", path)
            return None, None
        if value:
            shown = _display_path(path)
            return value, HFTokenOrigin(
                kind=KIND_ENVFILE, label=f"your project {path.name}", path=shown
            )

    value = _hub_cached_token()
    if value:
        return value, HFTokenOrigin(
            kind=KIND_HF_CLI, label="your huggingface-cli login token"
        )

    return None, None


def _donation_disabled() -> bool:
    """True when the user opted out with ``LQH_HF_DONATE``."""
    raw = (os.environ.get("LQH_HF_DONATE") or "").strip().lower()
    return raw in {"0", "false", "no", "off"}


def hf_token_origin(project_dir: Path | None) -> HFTokenOrigin | None:
    """Where a local HF token would come from, without reading its value.

    This is what prompts, tool results and the status bar call. Returns None
    when no token is available anywhere. When donation is disabled via
    ``LQH_HF_DONATE=0`` the origin is still reported, with
    ``donation_enabled=False``, so the UI can say "found, but disabled" rather
    than "not found".
    """
    _unused, origin = _resolve(project_dir, want_value=False)
    assert _unused is None  # provenance callers never receive plaintext
    if origin is None:
        return None
    if _donation_disabled():
        return replace(origin, donation_enabled=False)
    return origin


def donatable_hf_token(
    project_dir: Path | None,
) -> tuple[str | None, HFTokenOrigin | None]:
    """Token to attach to a single cloud job, plus where it came from.

    Returns ``(None, None)`` when no token is available or the user set
    ``LQH_HF_DONATE=0``.

    **Call this only from the network client that puts the value on the wire.**
    See the module docstring.
    """
    if _donation_disabled():
        return None, None
    return _resolve(project_dir)


def local_hf_token(project_dir: Path | None) -> str | None:
    """Token for HF calls made *from this machine* (hf_pull, hf_push, HfApi).

    Not gated on ``LQH_HF_DONATE`` — that opt-out is about sending the token to
    our backend, not about using it locally.
    """
    token, _origin = _resolve(project_dir)
    return token


def hf_disclosure_line(project_dir: Path | None, *, indent: str = "  ") -> str:
    """One consent line naming the token's source, or "" when there is none.

    Used by the cloud consent prompts so all of them describe the donation the
    same way. Contains the source label and file path — never the value, and
    never a prefix, length or fingerprint of it (a stable credential identifier
    in a transcript is still a credential identifier).
    """
    origin = hf_token_origin(project_dir)
    if origin is None or not origin.donation_enabled:
        return ""
    # Two claims this must not overstate. It says "you'll be asked",
    # not "is sent", because the donation is consented separately and may
    # be declined. And it says "not added to your account" rather than
    # "not stored", because the backend does hold it — encrypted, keyed
    # to this job, deleted when the job ends — so that a worker replacing
    # a preempted one still has it.
    tail = (
        "held only for this job, then deleted — it is not added to your LQH "
        "account (LQH_HF_DONATE=0 stops us offering)"
    )
    if origin.is_hub_cache:
        # Distinct wording: this token was created for the Hub CLI, not for us.
        return (
            f"{indent}HF:      you'll be asked whether to send the token from your "
            f"`huggingface-cli login` with this job — {tail}\n"
        )
    where = origin.label
    if origin.path:
        where += f" ({origin.path})"
    return (
        f"{indent}HF:      you'll be asked whether to send a Hugging Face token "
        f"from {where} with this job — {tail}\n"
    )


def redact(text: str, secret: str | None) -> str:
    """Blank out ``secret`` wherever it appears in ``text``.

    Server error bodies are the one path by which a donated token can
    come back to us and end up in a tool result (and from there in the
    session log and the backend's payload capture): a validation error
    that echoes the request body would carry it. The backend is not
    supposed to echo request bodies, but "the other side promised" is not
    a control we own, so every client scrubs what it puts into an
    exception message.

    Short secrets get the whole body dropped instead of substring
    replacement. Blanking a 3-character "token" out of prose would hit
    every incidental occurrence and leave an unreadable message — but
    silently returning the text unchanged, which is what this used to do
    below the 8-character floor, breaks the invariant the function
    exists for. The submit API accepts any non-empty value up to 512
    bytes, so "too short to be a real HF token" is not something this
    layer gets to assume. Real tokens are long and take the redaction
    path; the drop is a defense-in-depth backstop.
    """
    if not text or not secret:
        return text
    if len(secret) < _MIN_REDACTABLE_SECRET:
        if secret in text:
            return (
                "«server error withheld: it contained the donated HF token "
                "and the token is too short to redact safely»"
            )
        return text
    return text.replace(secret, "«redacted-hf-token»")


# Below this length, substring redaction does more damage than good, so
# redact() drops the surrounding text instead.
_MIN_REDACTABLE_SECRET = 8
