# 🧪 Base vs Instruct — fine-tuning benchmark

`train_size=10000` · `eval_size=400` · `grid=small` · `judge=medium` · `dpo=on` · `compute=local-gpu`

---

## ⚡ TL;DR

**What we did.** For each of 6 models × 6 tasks we ran a hyperparameter sweep of supervised
fine-tuning (6 configs: lr ∈ {2e-5, 5e-5, 1e-4} × epochs ∈ {2, 3}) on 10k training
examples, then a DPO sweep (6 configs: lr ∈ {3e-7, 1e-6, 3e-6} × β ∈ {0.05, 0.1}) on top of
the winning SFT checkpoint. Every model — baseline, best SFT, best DPO — was then scored by
a judge model on a held-out 400-example eval set, from **local** inference of the weights
(not the API). Scores are mean judge ratings on a 0–10 scale; higher is better.

**What we found.**

| | Finding |
|---|---|
| 🎯 | **Base vs Instruct doesn't matter after fine-tuning.** Baselines differ hugely (LFM2.5-350M-Base averages 3.21 vs 5.46 for Instruct), but post-SFT they land within 0.05 of each other at both sizes. That's noise. Instruct tuning buys a better *starting* point and nothing at the finish line. |
| 📏 | **Size matters, base/instruct doesn't.** 1.2B lands at ~8.9, 350M at ~8.45 — a gap ~10× larger than the base-vs-instruct gap. Spend your choice on size. |
| 🎨 | **One real exception: `style_rewrite`.** LFM2.5-1.2B-Base wins 8.56 vs 8.45 while starting from 0.06 vs 4.98. When the target style fights instruct-tuned habits, starting from Base helps. |
| 🔁 | **DPO isn't worth it once SFT is already near-perfect.** It only produced a usable model in 15 of 36 cells; everywhere else the SFT policy was too close to gold to generate preference pairs. Where it ran, median gain is **+0.02** and 4 of 15 cells regressed — for roughly double the compute. |
| 📉 | **LFM2 is superseded at both sizes.** ~0.3 behind at 1.2B, ~0.27 behind at 350M, and the gap is widest on the hard tasks. |
| ⚠️ | **This benchmark is saturating.** 4 of 6 tasks hit ~9.9–10.0 after SFT and no longer discriminate between options. |

**Recommendation:** fine-tune **LFM2.5-1.2B**, SFT only, and take whichever of
Base/Instruct is more convenient to license and serve.

---

## 📊 Per-task results

Δ columns are improvement over each model's own baseline. `—` means DPO produced no usable
model for that cell (see [DPO coverage](#-dpo-coverage-and-why-it-failed)).

### Task: `translation`

| Model | Baseline | Best SFT | Best DPO | ΔSFT | ΔDPO | Best | SFT sweep | DPO sweep |
|---|---|---|---|---|---|---|---|---|
| LFM2-350M | 5.78 | 7.57 | 7.65 | +1.79 | +1.87 | 7.65 | 2h 45m | 6h 2m |
| LFM2.5-350M-Base | 3.65 | 8.08 | 8.09 | +4.43 | +4.44 | 8.09 | 2h 49m | 4h 27m |
| LFM2.5-350M-Instruct | 6.12 | 7.87 | 7.91 | +1.75 | +1.79 | 7.91 | 3h 40m | 4h 50m |
| LFM2-1.2B | 8.97 | 9.24 | 9.32 | +0.27 | +0.35 | 9.32 | 4h 32m | 4h 30m |
| LFM2.5-1.2B-Base | 8.05 | 9.31 | — | +1.26 | — | 9.31 | 3h 34m | 4h 25m |
| LFM2.5-1.2B-Instruct | 9.14 | 9.39 | — | +0.25 | — | **9.39** | 3h 43m | 4h 33m |

- 🏆 **Highest final score:** `LFM2.5-1.2B-Instruct` at 9.39.
- 📈 **Largest lift:** `LFM2.5-350M-Base` (+4.44).

### Task: `extraction`

| Model | Baseline | Best SFT | Best DPO | ΔSFT | ΔDPO | Best | SFT sweep | DPO sweep |
|---|---|---|---|---|---|---|---|---|
| LFM2-350M | 6.92 | 10.00 | — | +3.08 | — | **10.00** | 3h 2m | 44m |
| LFM2.5-350M-Base | 8.37 | 10.00 | — | +1.63 | — | **10.00** | 2h 58m | 43m |
| LFM2.5-350M-Instruct | 9.39 | 10.00 | — | +0.61 | — | **10.00** | 2h 58m | 42m |
| LFM2-1.2B | 9.50 | 9.99 | — | +0.49 | — | 9.99 | 3h 11m | 43m |
| LFM2.5-1.2B-Base | 8.79 | 10.00 | — | +1.21 | — | **10.00** | 3h 14m | 44m |
| LFM2.5-1.2B-Instruct | 9.34 | 10.00 | — | +0.66 | — | **10.00** | 3h 19m | 43m |

- 🏆 **Highest final score:** 5-way tie at 10.00.
- 📈 **Largest lift:** `LFM2-350M` (+3.08).
- ⚠️ **Saturated** — all six land within 0.01. This task no longer discriminates.

### Task: `classification`

| Model | Baseline | Best SFT | Best DPO | ΔSFT | ΔDPO | Best | SFT sweep | DPO sweep |
|---|---|---|---|---|---|---|---|---|
| LFM2-350M | 9.77 | 9.97 | — | +0.20 | — | 9.97 | 1h 13m | 26m |
| LFM2.5-350M-Base | 3.02 | 10.00 | — | +6.98 | — | **10.00** | 58m | 12m |
| LFM2.5-350M-Instruct | 6.13 | 9.97 | — | +3.84 | — | 9.97 | 1h 3m | 26m |
| LFM2-1.2B | 8.60 | 9.98 | — | +1.38 | — | 9.98 | 1h 53m | 15m |
| LFM2.5-1.2B-Base | 9.06 | 9.99 | — | +0.93 | — | 9.99 | 1h 10m | 47m |
| LFM2.5-1.2B-Instruct | 8.81 | 9.99 | — | +1.18 | — | 9.99 | 1h 16m | 24m |

- 🏆 **Highest final score:** `LFM2.5-350M-Base` at 10.00.
- 📈 **Largest lift:** `LFM2.5-350M-Base` (+6.98).
- ⚠️ **Saturated** — all six land within a 0.03 band.

### Task: `messy_extraction`

| Model | Baseline | Best SFT | Best DPO | ΔSFT | ΔDPO | Best | SFT sweep | DPO sweep |
|---|---|---|---|---|---|---|---|---|
| LFM2-350M | 7.38 | 9.90 | — | +2.52 | — | 9.90 | 5h 8m | 9h 9m |
| LFM2.5-350M-Base | 4.05 | 9.95 | — | +5.90 | — | **9.95** | 4h 19m | 7h 39m |
| LFM2.5-350M-Instruct | 8.42 | 9.95 | — | +1.53 | — | **9.95** | 4h 26m | 7h 44m |
| LFM2-1.2B | 8.93 | 9.89 | — | +0.96 | — | 9.89 | 4h 41m | 8h 49m |
| LFM2.5-1.2B-Base | 9.08 | 9.91 | — | +0.83 | — | 9.91 | 4h 27m | 6h 40m |
| LFM2.5-1.2B-Instruct | 9.44 | 9.95 | — | +0.51 | — | **9.95** | 4h 30m | 7h 50m |

- 🏆 **Highest final score:** 3-way tie at 9.95.
- 📈 **Largest lift:** `LFM2.5-350M-Base` (+5.90).
- ⚠️ **Approaching saturation** — spread is 0.06.

### Task: `style_rewrite`

| Model | Baseline | Best SFT | Best DPO | ΔSFT | ΔDPO | Best | SFT sweep | DPO sweep |
|---|---|---|---|---|---|---|---|---|
| LFM2-350M | 2.55 | 8.07 | 7.99 | +5.52 | +5.44 | 8.07 | 4h 52m | 9h 41m |
| LFM2.5-350M-Base | 0.00 | 8.21 | 8.21 | +8.21 | +8.21 | 8.21 | 5h 25m | 9h 50m |
| LFM2.5-350M-Instruct | 1.84 | 8.39 | 8.41 | +6.55 | +6.57 | 8.41 | 5h 28m | 10h 26m |
| LFM2-1.2B | 3.46 | 7.89 | 7.92 | +4.43 | +4.46 | 7.92 | 5h 30m | 10h 9m |
| LFM2.5-1.2B-Base | 0.06 | 8.54 | 8.56 | +8.48 | +8.50 | **8.56** | 4h 55m | 9h 49m |
| LFM2.5-1.2B-Instruct | 4.98 | 8.45 | 8.03 | +3.47 | +3.05 | 8.45 | 5h 50m | 9h 11m |

- 🏆 **Highest final score:** `LFM2.5-1.2B-Base` at 8.56.
- 📈 **Largest lift:** `LFM2.5-1.2B-Base` (+8.50).
- 🎨 **The one task where Base genuinely beats Instruct** (8.56 vs 8.45), despite starting
  from 0.06 vs 4.98. The target style appears to conflict with instruct-tuned habits, so
  the instruct prior is a liability rather than a head start.

### Task: `voice_satisfaction`

| Model | Baseline | Best SFT | Best DPO | ΔSFT | ΔDPO | Best | SFT sweep | DPO sweep |
|---|---|---|---|---|---|---|---|---|
| LFM2-350M | 0.45 | 3.72 | — | +3.27 | — | 3.72 | 14h 6m | 12h 19m |
| LFM2.5-350M-Base | 0.20 | 4.58 | 4.72 | +4.38 | +4.52 | 4.72 | 14h 15m | 16h 7m |
| LFM2.5-350M-Instruct | 0.85 | 4.42 | 4.25 | +3.57 | +3.40 | 4.42 | 12h 41m | 15h 40m |
| LFM2-1.2B | 1.15 | 4.64 | 4.85 | +3.49 | +3.70 | 4.85 | 15h 41m | 20h 55m |
| LFM2.5-1.2B-Base | 1.89 | 5.95 | 6.01 | +4.06 | +4.12 | **6.01** | 14h 33m | 21h 36m |
| LFM2.5-1.2B-Instruct | 2.40 | 5.67 | 5.58 | +3.27 | +3.18 | 5.67 | 12h 45m | 20h 25m |

- 🏆 **Highest final score:** `LFM2.5-1.2B-Base` at 6.01.
- 📈 **Largest lift:** `LFM2.5-350M-Base` (+4.52).
- ✅ **The only task with real headroom left** — nothing is near 10, the spread across
  models is 2.3 points, and the size gap is clearly visible. This is the task that still
  discriminates.

---

## 🧮 Cross-task aggregates

### Overall — best achieved

Mean across all 6 tasks of each model's *best achieved* score (max of baseline / SFT / DPO
per task) and its lift over baseline.

| Model | Mean baseline | Mean best | Mean lift |
|---|---|---|---|
| LFM2-350M | 5.48 | 8.22 | +2.74 |
| LFM2.5-350M-Base | 3.21 | 8.49 | +5.28 |
| LFM2.5-350M-Instruct | 5.46 | 8.44 | +2.98 |
| LFM2-1.2B | 6.77 | 8.66 | +1.89 |
| LFM2.5-1.2B-Base | 6.16 | **8.96** | +2.81 |
| LFM2.5-1.2B-Instruct | 7.35 | 8.91 | +1.56 |

> ⚠️ **Don't read "mean lift" as "learns better."** Base's much larger lift is entirely an
> artifact of its much lower baseline. Both variants converge to the same ceiling — the
> lift column measures how far away the starting point was, not how good the destination
> is.

## 🧾 Conclusions

### 1. Base vs Instruct: no difference at the ceiling

Once you fine-tune on 10k examples, the two converge. Base ends up 0.04–0.05 ahead at both
sizes, which on a 0–10 judge scale across six tasks is noise — the honest reading is "no
difference," not "Base wins."

Instruct tuning buys a much better starting point (mean baseline 5.46 vs 3.21 at 350M; 7.35
vs 6.16 at 1.2B) and nothing at the finish line. All of the apparent "Base learns better"
signal (+5.28 mean lift vs +2.98) is the low baseline, not a higher ceiling.

**So choose on practical grounds** — licensing, serving convenience, what's already in the
registry.

**Exception:** `style_rewrite`, where LFM2.5-1.2B-Base ends at 8.56 vs 8.45 for Instruct
while starting from 0.06 vs 4.98. If a task's target style conflicts with instruct-tuned
habits, starting from Base actually helps. Worth checking task-by-task when the output
format is unusual.

### 2. Size dominates the base/instruct choice

1.2B lands at ~8.9 and 350M at ~8.45 regardless of variant. The size gap (~0.48) is an
order of magnitude larger than the base-vs-instruct gap (~0.04). Pick the bigger model;
don't agonize over the variant.

### 3. LFM2 is superseded at both sizes

| Size | LFM2 | Best LFM2.5 | Gap |
|---|---|---|---|
| 350M | 8.22 | 8.49 | −0.27 |
| 1.2B | 8.66 | 8.96 | −0.30 |

The gap is widest exactly where it matters — on the hard tasks. `voice_satisfaction` has
LFM2-1.2B at 4.85 vs 6.01 for LFM2.5-1.2B-Base, and LFM2-350M at 3.72 is the **worst cell in
the entire run**. On the saturated tasks everything hits ~10 and LFM2 looks fine, which is
precisely why those tasks don't tell you anything.

LFM2-350M does still have the single best *untuned* baseline on `classification` (9.77), so
it remains decent zero-shot at some things — irrelevant if you're fine-tuning anyway.

**Retire LFM2 for customization work.**
