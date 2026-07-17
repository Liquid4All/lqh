"""Tested helpers for loading user-brought data in pipeline ``source()`` methods.

These helpers exist to narrow the agent's write surface: rather than hand-
rolling glob/parquet/HF/JSONL boilerplate (which the agent gets wrong and
then fix-loops on), pipelines import one of these helpers and get a typed
iterable ready to feed into ``generate(client, input)``.

Design rules:

* Every helper validates that resolved paths stay inside ``Path.cwd()``
  (the project directory — lqh always runs with cwd set to the project).
* Missing files raise ``FileNotFoundError`` *at call time* (before the
  engine schedules work) so the agent gets a crisp, non-retryable error.
* Items are typed dataclasses so ``generate(self, client, input: ImageItem)``
  is self-documenting.
* Image bytes and HF rows are loaded lazily to keep memory flat on large
  inputs.
"""

from __future__ import annotations

import base64
import csv
import json
import mimetypes
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

__all__ = [
    "ImageItem",
    "PromptItem",
    "image_folder",
    "prompts",
    "parquet",
    "jsonl",
    "hf_dataset",
    "seed_data",
    "record_source_paths",
    "hf_dataset_was_used",
]


# ---------------------------------------------------------------------------
# Typed items
# ---------------------------------------------------------------------------


@dataclass
class ImageItem:
    """An image file from an ``image_folder`` source.

    Attributes
    ----------
    path:
        Absolute path to the image on disk.
    subfolder:
        Name of the immediate parent folder relative to the source root,
        or ``""`` if the image lies directly in the source root.  Useful
        as a coarse label (e.g. ``"dog"``, ``"cat"``).
    metadata:
        Free-form dict for user-supplied metadata (reserved for future use).
    """

    path: Path
    subfolder: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()

    def mime_type(self) -> str:
        guess, _ = mimetypes.guess_type(self.path.name)
        return guess or "image/jpeg"

    def as_data_url(
        self,
        *,
        max_dim: int | None = 1024,
        jpeg_quality: int = 90,
    ) -> str:
        """Return a ``data:image/...;base64,...`` URL suitable for OpenAI vision.

        By default the image is preprocessed to a size that VLM APIs handle
        well: decoded with PIL, converted to RGB, downscaled so the long edge
        is at most *max_dim* pixels (never upscaled), and re-encoded as JPEG
        at *jpeg_quality*. Images with an alpha channel are re-encoded as PNG
        instead so transparency survives. This bounds both the base64 payload
        (the API caps request bodies) and the per-image token cost.

        Pass ``max_dim=None`` to skip preprocessing and embed the raw file
        bytes unchanged.
        """
        if max_dim is None:
            b64 = base64.b64encode(self.read_bytes()).decode("ascii")
            return f"data:{self.mime_type()};base64,{b64}"

        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(self.read_bytes())) as img:
            img.load()
            has_alpha = img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            )
            if img.width > max_dim or img.height > max_dim:
                img.thumbnail((max_dim, max_dim), Image.LANCZOS)
            buf = BytesIO()
            if has_alpha:
                img.convert("RGBA").save(buf, format="PNG")
                mime = "image/png"
            else:
                img.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
                mime = "image/jpeg"

        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:{mime};base64,{b64}"


@dataclass
class PromptItem:
    """A user-brought prompt awaiting completion.

    ``prompt`` is the raw text.  ``metadata`` carries any extra columns
    from the source file (e.g. an ``id`` or ``category``).
    """

    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Path validation + source-path recording
# ---------------------------------------------------------------------------

# Active recording set for record_source_paths(). ContextVar (not a
# module global) so concurrent pipeline runs in one process don't see
# each other's paths; asyncio tasks created inside the context inherit
# it automatically.
_RECORDED_PATHS: ContextVar[set[Path] | None] = ContextVar(
    "_RECORDED_PATHS", default=None
)

# Parallel flag set when hf_dataset() runs inside record_source_paths():
# observed (not guessed) HF usage drives whether a cloud data_gen
# sandbox receives the user's stored HF token. One-element list so the
# same object is mutated across asyncio tasks sharing the context.
_RECORDED_HF: ContextVar[list[bool] | None] = ContextVar(
    "_RECORDED_HF", default=None
)


@contextmanager
def record_source_paths() -> Iterator[set[Path]]:
    """Collect every project file lqh.sources helpers actually consumed.

    The engine wraps a whole pipeline run in this so a successful local
    run yields the exact set of input files the pipeline read — the
    trusted manifest for shipping those inputs to a cloud data_gen job
    (CLOUD_OFFLOAD_PLAN.md §2). Recording is per-file, not per-folder:
    ``image_folder`` records only the images that matched its extension
    filter and ``seed_data`` records only the chosen seed file, so the
    bundle never ships unrelated files that merely share a directory.
    hf_dataset touches no local path and records nothing.
    """
    recorded: set[Path] = set()
    token = _RECORDED_PATHS.set(recorded)
    hf_token = _RECORDED_HF.set([False])
    try:
        yield recorded
    finally:
        _RECORDED_PATHS.reset(token)
        _RECORDED_HF.reset(hf_token)


def _record_path(p: Path) -> None:
    """Add one consumed file to the active record_source_paths() set."""
    recorded = _RECORDED_PATHS.get()
    if recorded is not None:
        recorded.add(p)


def hf_dataset_was_used() -> bool:
    """Whether hf_dataset() ran inside the current record context.

    Must be called while the ``record_source_paths()`` context is still
    active (the flag resets on exit).
    """
    flag = _RECORDED_HF.get()
    return bool(flag and flag[0])


def _resolve_inside_project(path: Path | str) -> Path:
    """Resolve *path* and ensure it is inside the project directory.

    The project directory is ``Path.cwd()`` (lqh sets cwd to the project
    root before importing pipelines).  Rejects ``..`` escapes and
    symlinks pointing outside the project.
    """
    project_dir = Path.cwd().resolve()
    p = Path(path)
    if not p.is_absolute():
        p = project_dir / p
    p = p.resolve()
    try:
        p.relative_to(project_dir)
    except ValueError as exc:
        raise ValueError(
            f"Path {path!r} resolves to {p}, which is outside the project directory {project_dir}. "
            "Paths passed to lqh.sources helpers must stay inside the project."
        ) from exc
    return p


# ---------------------------------------------------------------------------
# image_folder
# ---------------------------------------------------------------------------


_DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".gif")


def image_folder(
    path: Path | str,
    *,
    recursive: bool = True,
    extensions: Sequence[str] = _DEFAULT_IMAGE_EXTENSIONS,
    include_subfolder_label: bool = False,
) -> list[ImageItem]:
    """List images in *path* as ``ImageItem``s.

    Parameters
    ----------
    path:
        Folder containing the images.  May be relative to the project dir.
    recursive:
        If ``True`` (default), walks subfolders.  If ``False``, only
        files directly in *path* are returned.
    extensions:
        File extensions (case-insensitive, leading dot) to include.
    include_subfolder_label:
        If ``True``, each item's ``subfolder`` is set to the immediate
        parent folder name relative to *path* (useful for categorical
        labels).  If ``False``, ``subfolder`` is always ``""``.

    Returns
    -------
    list[ImageItem]
        Sorted deterministically by path so repeat runs are reproducible.
    """
    root = _resolve_inside_project(path)
    if not root.exists():
        raise FileNotFoundError(f"image_folder: {root} does not exist")
    if not root.is_dir():
        raise NotADirectoryError(f"image_folder: {root} is not a directory")

    ext_set = {e.lower() for e in extensions}
    iterator = root.rglob("*") if recursive else root.iterdir()
    items: list[ImageItem] = []
    for p in iterator:
        if not p.is_file():
            continue
        if p.suffix.lower() not in ext_set:
            continue
        subfolder = ""
        if include_subfolder_label:
            try:
                rel_parent = p.parent.relative_to(root)
                subfolder = rel_parent.parts[0] if rel_parent.parts else ""
            except ValueError:
                subfolder = ""
        items.append(ImageItem(path=p, subfolder=subfolder))

    items.sort(key=lambda it: str(it.path))
    if not items:
        raise FileNotFoundError(
            f"image_folder: no images matching {sorted(ext_set)} found under {root}"
        )
    # Record only the matched images — not the folder root, which may
    # hold unrelated files that must never ride a cloud bundle.
    for it in items:
        _record_path(it.path)
    return items


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def prompts(
    path: Path | str,
    *,
    column: str = "prompt",
) -> list[PromptItem]:
    """Load prompts from a ``.jsonl``, ``.txt``, ``.csv``, or ``.parquet`` file.

    * ``.txt`` — one prompt per non-empty line.  ``metadata`` is empty.
    * ``.jsonl`` — each line is a JSON object; the value at *column* is
      the prompt, remaining keys go into ``metadata``.  If the line is a
      bare string, it is used directly.
    * ``.csv`` — the *column* column is the prompt; other columns go into
      ``metadata``.
    * ``.parquet`` — same semantics as CSV.
    """
    p = _resolve_inside_project(path)
    if not p.exists():
        raise FileNotFoundError(f"prompts: {p} does not exist")
    if not p.is_file():
        raise IsADirectoryError(f"prompts: {p} is not a file")
    _record_path(p)

    suffix = p.suffix.lower()
    if suffix == ".txt":
        return [
            PromptItem(prompt=line.strip())
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".jsonl":
        items: list[PromptItem] = []
        with p.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"prompts: {p}:{lineno}: invalid JSON: {exc}") from exc
                if isinstance(obj, str):
                    items.append(PromptItem(prompt=obj))
                elif isinstance(obj, dict):
                    if column not in obj:
                        raise KeyError(
                            f"prompts: {p}:{lineno}: missing required column {column!r}. "
                            f"Available keys: {sorted(obj.keys())}"
                        )
                    prompt_val = obj[column]
                    if not isinstance(prompt_val, str):
                        raise TypeError(
                            f"prompts: {p}:{lineno}: column {column!r} is {type(prompt_val).__name__}, expected str"
                        )
                    metadata = {k: v for k, v in obj.items() if k != column}
                    items.append(PromptItem(prompt=prompt_val, metadata=metadata))
                else:
                    raise TypeError(
                        f"prompts: {p}:{lineno}: expected str or object, got {type(obj).__name__}"
                    )
        return items
    if suffix == ".csv":
        items = []
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise KeyError(
                    f"prompts: {p}: CSV has no column {column!r}. "
                    f"Available columns: {reader.fieldnames}"
                )
            for row in reader:
                prompt_val = row.get(column, "")
                if not prompt_val:
                    continue
                metadata = {k: v for k, v in row.items() if k != column}
                items.append(PromptItem(prompt=prompt_val, metadata=metadata))
        return items
    if suffix == ".parquet":
        rows = list(parquet(p))
        if not rows:
            return []
        if column not in rows[0]:
            raise KeyError(
                f"prompts: {p}: parquet has no column {column!r}. "
                f"Available columns: {sorted(rows[0].keys())}"
            )
        return [
            PromptItem(
                prompt=str(r[column]),
                metadata={k: v for k, v in r.items() if k != column},
            )
            for r in rows
            if r.get(column)
        ]
    raise ValueError(
        f"prompts: unsupported file type {suffix!r} for {p}. "
        "Supported: .txt, .jsonl, .csv, .parquet"
    )


# ---------------------------------------------------------------------------
# parquet / jsonl
# ---------------------------------------------------------------------------


def parquet(
    path: Path | str,
    *,
    columns: Sequence[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream rows from a parquet file as dicts.

    If *columns* is given, only those columns are read (saves memory on
    wide tables).  Rows are yielded in file order.
    """
    import pyarrow.parquet as pq

    p = _resolve_inside_project(path)
    if not p.exists():
        raise FileNotFoundError(f"parquet: {p} does not exist")
    _record_path(p)
    table = pq.read_table(p, columns=list(columns) if columns else None)
    names = table.column_names
    for i in range(len(table)):
        yield {name: table.column(name)[i].as_py() for name in names}


def jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream JSON objects from a ``.jsonl`` file.

    Blank lines are skipped.  Raises on malformed JSON with line number.
    """
    p = _resolve_inside_project(path)
    if not p.exists():
        raise FileNotFoundError(f"jsonl: {p} does not exist")
    _record_path(p)
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"jsonl: {p}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise TypeError(
                    f"jsonl: {p}:{lineno}: expected object, got {type(obj).__name__}"
                )
            yield obj


# ---------------------------------------------------------------------------
# hf_dataset
# ---------------------------------------------------------------------------


def hf_dataset(
    repo: str,
    *,
    split: str = "train",
    streaming: bool = True,
    columns: Sequence[str] | None = None,
    revision: str | None = None,
) -> Iterable[dict[str, Any]]:
    """Load a Hugging Face dataset as an iterable of row dicts.

    Uses ``datasets.load_dataset`` under the hood.  Respects ``HF_TOKEN``
    from the environment for private repos.

    Parameters
    ----------
    repo:
        Hub repo ID like ``"squad"`` or ``"user/my_dataset"``.
    split:
        Dataset split (e.g. ``"train"``, ``"validation"``).
    streaming:
        If ``True`` (default), stream rows without downloading the full
        dataset — right for 1M-row datasets where the pipeline will cap
        with ``num_samples``.  If ``False``, downloads to the HF cache.
    columns:
        If given, only these columns are yielded per row.
    revision:
        Optional Hub revision or commit SHA. Pin a commit for reproducible
        long-running cloud jobs and restart continuations.
    """
    flag = _RECORDED_HF.get()
    if flag is not None:
        flag[0] = True

    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        repo,
        split=split,
        streaming=streaming,
        revision=revision,
        # Explicit rather than relying on implicit library discovery: cloud
        # jobs receive HF_TOKEN as a Modal secret under the trusted-pipeline
        # contract, including direct streaming of private datasets.
        token=os.environ.get("HF_TOKEN") or None,
    )
    col_set = set(columns) if columns else None
    for row in ds:
        if col_set is not None:
            yield {k: v for k, v in row.items() if k in col_set}
        else:
            yield dict(row)


# ---------------------------------------------------------------------------
# seed_data
# ---------------------------------------------------------------------------


_SEED_DATA_EXTS: tuple[str, ...] = (".jsonl", ".csv", ".txt")


def seed_data(name: str) -> list[Any]:
    """Load user-brought seed data from ``seed_data/<name>.{jsonl,csv,txt}``.

    Convention: users drop lightweight seed files (e.g. a list of flower
    names) into a ``seed_data/`` folder at the project root.  Pipelines
    combine these with ``liquidrandom`` for diversity:

        flowers = lqh.sources.seed_data("flowers")
        seed = random.choice(flowers)
        style = liquidrandom.writing_style().brief()

    Returns
    -------
    list
        * ``.txt`` — list[str], one per non-empty line.
        * ``.jsonl`` — list of whatever each line deserialises to (str or dict).
        * ``.csv`` — list[dict] (one per row, keyed by header).
    """
    base = _resolve_inside_project("seed_data")
    if not base.exists():
        raise FileNotFoundError(
            f"seed_data: no seed_data/ directory at {base}. "
            "Create it and drop your seed file there."
        )
    for ext in _SEED_DATA_EXTS:
        candidate = base / f"{name}{ext}"
        if candidate.exists():
            break
    else:
        raise FileNotFoundError(
            f"seed_data: no {name}.{{jsonl,csv,txt}} in {base}"
        )
    _record_path(candidate)

    suffix = candidate.suffix.lower()
    if suffix == ".txt":
        return [
            line.strip()
            for line in candidate.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".jsonl":
        items: list[Any] = []
        with candidate.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"seed_data: {candidate}:{lineno}: invalid JSON: {exc}"
                    ) from exc
        return items
    if suffix == ".csv":
        with candidate.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    raise ValueError(f"seed_data: unsupported extension {suffix!r}")
