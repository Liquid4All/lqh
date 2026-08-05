"""The study's design matrix.

A **cell** is one (task × train_size × model) combination — a context in which
a set of hyperparameters either works or doesn't. The study trains the whole HP
grid inside every cell, so that "which default is best?" can be answered as
"which config loses the least, averaged over contexts?" rather than "which
config won the one experiment we ran".

The three axes are exactly the dimensions the defaults might need to vary along:

- **task** — does the best learning rate depend on what the model is learning?
- **train_size** — does it depend on how much data there is?
- **model** — does it depend on size, or on base-vs-instruct?

Model and size are deliberately *separate* readings of the same axis: a
recommendation like "base models want a higher LR" is only trustworthy if it
holds at both 350M and 1.2B.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    """One context in which the hyperparameter grid gets evaluated."""

    task: str
    train_size: int
    model_key: str
    hf_id: str
    model_kind: str      # "base" | "instruct"
    param_count: float   # in billions: 0.35, 1.2

    @property
    def id(self) -> str:
        return f"{self.task}__{self.model_key}__n{self.train_size}"

    def dimensions(self) -> dict[str, object]:
        """The levels this cell occupies, keyed by dimension name.

        The analysis asks "does any dimension deserve its own default?" by
        grouping cells on exactly these keys.
        """
        return {
            "task": self.task,
            "train_size": self.train_size,
            "model_kind": self.model_kind,
            "param_count": self.param_count,
        }


# Friendly key -> (HuggingFace id, kind, params in billions). The 350M instruct
# variant has no -Instruct suffix upstream.
MODELS: dict[str, tuple[str, str, float]] = {
    "350M-Instruct": ("LiquidAI/LFM2.5-350M", "instruct", 0.35),
    "350M-Base": ("LiquidAI/LFM2.5-350M-Base", "base", 0.35),
    "1.2B-Instruct": ("LiquidAI/LFM2.5-1.2B-Instruct", "instruct", 1.2),
    "1.2B-Base": ("LiquidAI/LFM2.5-1.2B-Base", "base", 1.2),
}

# Four of the six base_vs_instruct pipelines, chosen to span output shapes:
# free text with format discipline, structured JSON, a short label, and one
# genuinely hard judgement task. voice_satisfaction is the non-saturating one —
# the others reach >8/10 after SFT, which compresses the differences between
# hyperparameter configs and would make every config look equally good.
TASKS: tuple[str, ...] = (
    "translation",
    "extraction",
    "classification",
    "voice_satisfaction",
)

# Spanning the range where the default actually has to work: a few hundred rows
# (where a pilot lands) up to the low tens of thousands (a scaled run).
TRAIN_SIZES: tuple[int, ...] = (500, 2000, 8000)


def build_cells(
    *,
    tasks: tuple[str, ...] = TASKS,
    sizes: tuple[int, ...] = TRAIN_SIZES,
    models: tuple[str, ...] = tuple(MODELS),
) -> list[Cell]:
    """The full factorial over the three axes."""
    cells: list[Cell] = []
    for task in tasks:
        for size in sizes:
            for key in models:
                hf_id, kind, params = MODELS[key]
                cells.append(Cell(
                    task=task, train_size=size, model_key=key,
                    hf_id=hf_id, model_kind=kind, param_count=params,
                ))
    return cells


def anchor_cells(cells: list[Cell], *, per_dimension: int = 3) -> list[Cell]:
    """A balanced subset covering every level of every dimension.

    Stage A screens the full HP grid, which is the expensive part; running it on
    all 48 cells is wasteful when its only job is to narrow 15 configs down to a
    handful of finalists. This picks a small subset in which **no level of any
    dimension is missing**, so the screen cannot be biased by, say, only seeing
    instruct models or only seeing large datasets.

    Greedy by design: repeatedly take the cell that adds the most
    not-yet-covered levels, until every level is covered *per_dimension* times.
    That is not a minimal cover, but it is deterministic, easy to read in the
    report, and the cost difference is a cell or two.
    """
    if not cells:
        return []

    wanted: dict[tuple[str, object], int] = {}
    for cell in cells:
        for dim, level in cell.dimensions().items():
            wanted[(dim, level)] = per_dimension

    chosen: list[Cell] = []
    remaining = list(cells)
    while remaining and any(v > 0 for v in wanted.values()):
        def coverage(cell: Cell) -> int:
            return sum(
                1 for dim, level in cell.dimensions().items()
                if wanted.get((dim, level), 0) > 0
            )

        # Tie-break on cell id so the anchor set is reproducible across runs.
        best = max(remaining, key=lambda c: (coverage(c), c.id))
        if coverage(best) == 0:
            break
        chosen.append(best)
        remaining.remove(best)
        for dim, level in best.dimensions().items():
            key = (dim, level)
            if wanted.get(key, 0) > 0:
                wanted[key] -= 1

    return sorted(chosen, key=lambda c: c.id)


def resolve_cells(
    *,
    tasks: str = "",
    sizes: str = "",
    models: str = "",
    anchors_only: bool = False,
    anchor_coverage: int = 3,
) -> list[Cell]:
    """Build the cell list from comma-separated CLI selections."""
    task_sel = _split(tasks) or list(TASKS)
    model_sel = _split(models) or list(MODELS)
    size_sel = [int(s) for s in _split(sizes)] or list(TRAIN_SIZES)

    unknown_tasks = [t for t in task_sel if t not in TASKS]
    if unknown_tasks:
        raise SystemExit(
            f"unknown task(s) {', '.join(unknown_tasks)}; "
            f"choose from {', '.join(TASKS)}"
        )
    unknown_models = [m for m in model_sel if m not in MODELS]
    if unknown_models:
        raise SystemExit(
            f"unknown model(s) {', '.join(unknown_models)}; "
            f"choose from {', '.join(MODELS)}"
        )

    cells = build_cells(
        tasks=tuple(task_sel), sizes=tuple(size_sel), models=tuple(model_sel),
    )
    if anchors_only:
        return anchor_cells(cells, per_dimension=anchor_coverage)
    return cells


def _split(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]
