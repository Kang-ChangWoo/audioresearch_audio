#!/usr/bin/env bash
# Stage H0 (I25): log-depth output to cure near-field median-pull compression.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 221>out/.h0.lock
flock -n 221 || { echo "[h0] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# DIAGNOSIS (I24, four zero-GPU probes): the champion's real deficit is near-field median-pull.
# At 1-2m (52.5% of pixels) RayDPT ties batvision on boundaries but trails +0.0196 on flat-wall
# INTERIORS. Its bias and variance are both SMALLER than batvision's, yet d1 is worse -- because
# d1 is a +-25% RATIO threshold and RayDPT piles pixels at pred/gt 0.9-1.0 (just under) while
# batvision sits at 1.0-1.11 (centre). masked-MAE on normalised depth optimises the ARITHMETIC
# median, which for a right-skewed depth posterior sits BELOW the geometric median (ratio 1).
#
# I25: regress LOG-depth. The model's free variable becomes log d, so the same masked-MAE now
# drives to the conditional GEOMETRIC median -- pred/gt ratio 1 -- which is exactly what d1 rewards.
# It is a re-parameterisation: same params, same loss code, only the output map changes.
# This is NOT I13 (which re-weighted the loss and failed): the diagnosis then was range-shaped and
# wrong; now it is metric-shaped (median-pull) and the fix matches it.
#
# CONTROL is E23 (1.8962), the champion; E25 differs from it in one thing (--depth-out log).
# PRE-REGISTERED: judge on 1-2m INTERIOR d1 (the diagnosed locus) and overall d1. If the near-field
# interior does not improve, drop I25 whatever the composite does. ABS_REL is not evidence.
run raydpt_e32_logout --epochs 22 --depth-out log

# I26: log-out (near-field cure) + EchoDelayVolume (far-field cure). The two target DIFFERENT,
# non-overlapping deficits -- median-pull near vs range compression far -- so unlike E30 they carry
# different information and should compose. This is the combine the diagnoses jointly predict.
run raydpt_e33_logout_ede --epochs 22 --depth-out log --depth-volume True

echo "H0 DONE $(date -Is)"
