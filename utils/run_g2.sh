#!/usr/bin/env bash
# Stage G1b: E25 was budget-truncated. Re-run with the schedule matched to the budget.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 218>out/.g2.lock
flock -n 218 || { echo "[g2] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# E25 (champion + EchoDelayVolume) scored 1.9083 against E23's 1.8962 -- the OPPOSITE sign from
# E24, which beat its own control by 0.0112 on the fast parent. Before reading that as
# "the mechanism depends on its parent", look at the schedule:
#
#   run              --epochs   ran   anneal   best epoch
#   E23 control         24       22     92%      16/22   converged
#   E24 fast + EDE      28       24     86%      23/24   just barely
#   E25 champ + EDE     26       21     81%      21/21   NOT converged
#
# EchoDelayVolume costs 12 s/epoch, so on the champion architecture (157.1 vs 144.6 s/epoch) it
# fits one epoch fewer AND its cosine is cut off at 81% of schedule. Three disadvantages, all in
# the same direction, none of them the mechanism. This is the E10 trap again: reading d1 off a
# model that was still learning.
#
# Correction (idea I3): set --epochs to what actually fits, so cosine anneals to zero inside the
# budget. Two draws, because program.md forbids crowning on one -- and because E23 just showed what
# happens when I stop a 2x2 early.
run raydpt_e27_ede_champ_ep21 --epochs 21 --depth-volume True
run raydpt_e28_ede_champ_confirm --epochs 21 --depth-volume True

echo "G2 DONE $(date -Is)"
