# Auto Audio Depth Estimation

**Project slug:** `auto-audio-depth-estimation`

Autonomous research for **depth estimation from binaural echoes**. An AI research agent iteratively
forms a hypothesis, edits a single training file, trains a ray-conditioned depth model under a fixed
1-hour budget, evaluates it on a fixed metric, concludes **PASS/FAIL** with a scientific rationale,
updates a multi-lineage archive, and continues indefinitely.

The model (**RayDPT**) maps a binaural-audio spectral representation to an equirectangular (ERP)
radial-depth map using per-ray spherical queries that cross-attend audio tokens — depth is decoded
*per ray direction*, not regressed from a global bottleneck. Data is SoundSpaces (256×512 ERP).

## Repository layout

| File | Role |
|---|---|
| `program.md` | The agent's operating manual: protocol, the hypothesis-driven workflow, decision & noise policy. |
| `train.py` | **The only file experiments edit.** Model, composite loss, training/eval loop. |
| `prepare.py` | **Fixed / read-only.** Data split, target depth, audio-feature construction, and the ground-truth metric (`compute_errors`). |
| `results.tsv` | Authoritative per-run log (one row per training run). Append-only history. |
| `EXPERIMENTS.md` | Human-readable per-experiment findings and running commentary. |
| `hypotheses.tsv` | **Study-level** scientific conclusions (general + detailed hypothesis, type, PASS/FAIL). |
| `archive.json` | Global + per-lineage champions, informative failures, specialists. |
| `studies.json` | Active study state, adaptive-HPO progression, next experiment id. |
| `research.py` | Lightweight helper: `python research.py status` / `composite ...` / `next-id`. |

## The metric

Selection uses an **honest composite** (lower is better):

```
composite = rmse/1.6 + (1 - d1)/0.46 + 0.3 · abs_rel/0.4
```

RMSE and d1 (δ<1.25) are the trustworthy signals; ABS_REL is discounted because the relative loss
term can game it. Never crown a config that wins one metric while badly regressing the others.

## Current status

- **Global champion:** `E127` — eval-time L/R-flip test-time augmentation — commit `494b5e2`,
  composite ≈ **2.079** (abs_rel 0.340 / rmse 1.464 / d1 0.582), confirmed over 3 draws.
- Baseline → champion so far: **ABS_REL −23%, RMSE −8.0%, d1 +5.9 pts** across ~130 experiments.
- Research **continues from the current state** (see `studies.json` for the next experiment id) — it
  does not restart from scratch.

Run `python research.py status` for a live summary.

## How the workflow works

```
general hypothesis  →  detailed hypothesis  →  experiment note
  →  structural screen  →  adaptive HPO (3 → 5 → 7 → 10, justified)  →  confirm if near noise
  →  PASS / FAIL conclusion  →  archive update  →  choose next action
        (explore / refine / tune / combine / confirm)  →  continue indefinitely
```

Each experiment is exactly one of: **new** (novel mechanism), **refine** (better implementation of an
existing mechanism), **tune** (hyperparameters), **combine** (merge two lineages, with a stated reason),
or **confirm** (validate a near-noise or champion-candidate result). See `program.md` for the full
policy, including the noise floor (σ ≈ 0.008–0.019) and the rule to never crown a sub-0.015 candidate
on fewer than 3 confirming draws.

## Running an experiment (single GPU)

```bash
conda activate ss
python train.py --mode train > run.log 2>&1
grep "ABS_REL\|RMSE\|Best" run.log
```

`prepare.py` (data + metric) is fixed for evaluation fairness and reproducibility; all model/optimizer/
loss/augmentation changes go in `train.py`.
