# Auto Audio Depth Estimation

Autonomous research — binaural echoes → ERP radial depth (SoundSpaces).

**Reference model** = BatVision U-Net (`base/`, plain pix2pix encoder→decoder, trained by
`run_base.py`). **My model** = the ray-conditioned RayDPT (`train.py`), iterated to beat the
reference under the same fixed split / target / metric / selection composite.

**Input representation** — named binaural cues, each on/off, plus a `use_log` switch
(`prepare.build_channel_names`): `logL/L, logR/R, ILD, cosIPD, sinIPD`. Default = all five,
`use_log=True` → the 5ch `[logL,logR,ILD,cosIPD,sinIPD]` stack.

## Visual results

Held-out val scenes — `RGB | GT depth | batvision | best1 | best2` (best1/best2 fill in as
improved "my model" checkpoints are found; RGB is unavailable in the simplified dataset).

![qualitative depth comparison](out/display/qualitative.png)

Performance vs experiment (honest composite `rmse/1.6 + (1-d1)/0.46 + 0.35·abs_rel`, lower = better;
running best highlighted):

![performance progress](out/display/score_progress.png)

*Regenerate: `conda activate ss && python utils/report.py all`.*

## Results

<!-- RESULTS:START -->
| # | commit | ABS_REL | RMSE | d1 | composite | status | description |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | *(no experiments logged yet)* |
<!-- RESULTS:END -->

## Progression (composite, lower = better)

| phase | best | note |
|---|---|---|
| 2026-June (archived) | ~2.030 | multi-res STFT + interaural coherence + TTA |
| 2026-July (this) | — | BatVision reference + named-cue inputs + fixed coarse/low loss target |

## Network flowchart

Shared audio front-end, then two decoder heads — the BatVision reference (plain U-Net) and
my model RayDPT (ray-conditioned):

```mermaid
flowchart TD
    A["Binaural echo waveform (2ch)"] --> B["STFT"]
    B --> C["Named cue stack (in_ch)<br/>logL/L · logR/R · ILD · cosIPD · sinIPD"]
    C --> D["UNet8 encoder<br/>256x512 → 1x2 · skips e2/e3/e4"]
    D --> E{"decoder"}
    E -->|reference| F["BatVision U-Net<br/>ConvTranspose decoder + skips"]
    E -->|my model| G["RayBank ray queries ×<br/>audio cross-attention (scales 16/32/64)"]
    G --> H["DPT fusion +<br/>local spherical window attention"]
    F --> I["Sigmoid head"]
    H --> I
    I --> J["ERP radial depth<br/>256x512, [0,1] × max_depth"]
```
