#!/usr/bin/env bash
# Stage 8 (S5 attribution): fill the missing cell of the 2x2 -- L1 x kv=e4.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 207>out/.queue8.lock
flock -n 207 || { echo "[queue8] another queue8 runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# E11 changed TWO things at once versus E9 (a second cross layer, and a coarse KV set), and
# its d1 recovered +0.0079. The credit cannot be assigned. The 2x2, judged on d1:
#            kv=e3                     kv=e4
#   L1   E9  0.5631  21ep converged    E12  ?      30ep   <- this run
#   L2   E10 0.5665  15ep NOT conv.    E11  0.5710 23ep converged
#
# If E12 lands near 0.5710, the recovery belongs to the coarse KV (and the epochs it buys).
# If it lands near 0.5631, it belongs to the second cross layer.
# --epochs 30 because that is what the measured 119.4 s/epoch fits; cosine must anneal inside
# the budget (I3). Unequal epochs across cells are inherent to a wall-clock benchmark -- report
# epochs_ran and best_epoch with every number (D2/D5).
run raydpt_e12_d32L1_kve4 --amp bf16 --decode-scale 32 --ray-cross-layers 1 \
    --cross-kv32 e4 --batch-size 64 --lr 1.2e-3 --epochs 30

echo "QUEUE8 DONE $(date -Is)"
