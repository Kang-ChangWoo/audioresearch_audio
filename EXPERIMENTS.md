# RayDPT Autoresearch — Experiment Findings

Audio → ERP radial depth (SoundSpaces, 256×512). Fixed 1-hour training budget per run.
Metric: `compute_errors` in `prepare.py` (ABS_REL, RMSE, d1=δ<1.25). Goal: low ABS_REL **and** RMSE.

## How to read the metrics (they mean different things — judge together)
- **ABS_REL** = mean(|D−gt|/gt): relative error, weighted toward **near** pixels. ⚠️ Now **directly optimized** by the relative loss → partly *gamed*, least trustworthy on its own.
- **RMSE** = sqrt(mean((D−gt)²)): absolute error, dominated by **far / large-depth** pixels. Not directly optimized → **honest** signal.
- **d1** (δ<1.25) = % pixels roughly correct: **overall accuracy**, not directly optimized → **honest, most holistic**.

Rule: trust **RMSE + d1** as the real quality signal; don't crown a config that only wins ABS_REL while RMSE/d1 regress.

## Results (1 hr each; best epoch by composite ABS_REL+RMSE)

| run | change | ABS_REL ↓ | RMSE ↓ | d1 ↑ | verdict |
|---|---|---|---|---|---|
| baseline | RayDPT 5ch+flip, fp32 bs16 lr3e-4 (~5 ep) | 0.4434 | 1.5907 | 0.5236 | keep |
| **E0b** | **bf16 AMP + bs32 + lr6e-4 + anneal (~7 ep)** | 0.4151 | 1.5887 | 0.5398 | **keep** (clean gain) |
| **E0c** | E0b, **lr 4e-4** | 0.4259 | **1.5199** | **0.5471** | **keep** (best RMSE+d1) |
| E_d | + shared ray_proj | 0.4513 | 1.5280 | 0.5330 | discard |
| E_e | + full_decode (lr6e-4) | 0.4569 | 1.5273 | 0.5391 | discard (under-annealed) |
| E_f | + full_decode + time-anneal | 0.4594 | **1.5011** | 0.5447 | discard (ABS_REL froze) |
| E1 | + relative loss **w_rel=0.25** | **0.3340** | 1.7181 | 0.5173 | discard (RMSE broken) |
| **E2** | + relative loss **w_rel=0.1** | 0.3746 | 1.5540 | 0.5395 | **keep — best balanced** |
| E3 | rel0.25 + full_decode | 0.3443 | 1.7311 | 0.5182 | discard |
| E4 | + SILog w_silog=0.5 | 0.3989 | 1.5468 | 0.5192 | keep |
| E5 | rel w_rel=0.13 | 0.3587 | 1.6377 | 0.5297 | discard (RMSE>baseline) |
| E6 | lr4e-4 + rel0.1 | 0.3570 | 1.5837 | 0.5337 | keep (best ABS_REL, but worst RMSE/d1 of group — gamed) |
| E7 | lr4e-4 + rel0.1 + SILog0.3 | running | | | — |

(E0 fp16 AMP crashed: NaN at epoch 2 → fixed with bf16.)

## What helped
1. **bf16 AMP + batch 32 + LR cosine anneal (E0b)** — the foundation. fp16 → NaN, **bf16** fixed it. More epochs/hour + real LR decay → ABS_REL 0.4434→0.4151 with RMSE flat (a clean, both-safe gain).
2. **lr 4e-4 (E0c)** — genuinely lifts the *honest* metrics: RMSE 1.589→1.520, d1 0.5398→0.5471.
3. **Light relative loss `w_rel=0.1` (E2)** — `|D−gt|/gt` ≈ ABS_REL itself. At light weight + annealing, lowers ABS_REL to 0.3746 while keeping RMSE 1.554 & d1 0.5395 → best balanced config.
4. **full_decode (learned upsample, E_f)** — best RMSE alone (1.501) and good d1 (0.5447); the un-gamed quality levers live here.

## What did NOT help
- **fp16 AMP** → NaN (attention/softmax overflow). Use bf16.
- **Heavy relative `w_rel≥0.13` (E1, E5)** → great ABS_REL but RMSE breaks (1.64–1.72 every epoch). Over-weighting near pixels sacrifices far pixels.
- **full_decode + relative loss (E3)** → don't stack: once the rel loss is on, RMSE is loss-limited, so the decoder's RMSE benefit vanishes.
- **shared ray_proj (E_d)** → worse.
- **time-based LR anneal (E_f)** → froze LR early; great RMSE but ABS_REL stalls.

## Key principles discovered
- **ABS_REL ↔ RMSE anti-correlate** (across configs *and* epoch-to-epoch within a run). High LR / heavy rel → ABS_REL↓ RMSE↑; hard anneal → RMSE↓ ABS_REL↑. Loss/schedule changes mostly **slide along a frontier**, not push it in.
- The frontier is governed by the **loss balance**, not the architecture (full_decode can't rescue RMSE once rel loss is on).
- Because ABS_REL is now directly optimized, **RMSE and d1 are the trustworthy quality signals.**
- **Model-selection must be multi-metric**: ABS_REL-only checkpoint selection systematically saved the RMSE-spike epoch → switched to a composite (`abs_rel/0.4 + rmse/1.6`).

## Current best & strategy
- **Best balanced: E2** (lr6e-4 + rel0.1) — 0.3746 / 1.554 / 0.5395.
- **Best honest metrics: E0c** (lr4e-4) — 0.4259 / 1.520 / 0.5471.
- **Next:** stop chasing ABS_REL; lift **d1 + RMSE** genuinely via **lr4e-4 + full_decode (+ light rel)** and model-capacity levers.

_Live results table: `results.tsv`. This file summarizes; see git log for per-experiment commits._
