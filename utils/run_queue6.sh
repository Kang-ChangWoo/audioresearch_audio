#!/usr/bin/env bash
# Stage 6 (S4): ablate E9's compound change. Restore cross-attention depth at decode 32.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 205>out/.queue6.lock
flock -n 205 || { echo "[queue6] another queue6 runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# D9: with both models converged, RayDPT trails batvision by 0.0741, of which 0.0691 is d1
# alone -- the ANGULAR metric. The audio<->ray cross-attention is where per-ray direction
# information enters the model, and E9 halved it (2 layers -> 1). That is the amputation most
# likely to have caused the angular deficit, and it is ours, not the mechanism's.
#
# E10 restores ray_cross_layers=2 at decode 32. MEASURED 216.5 s/epoch = 16.6 epochs/h before
# validation; expect ~14 epochs after. --epochs 16 so cosine anneals inside the budget.
#
# Judge on d1 FIRST, not the composite. Fewer epochs fit at 2 layers, which is itself a
# confound (D2/D5) -- report epochs_ran and best_epoch.
run raydpt_e10_d32L2_b64 --amp bf16 --decode-scale 32 --ray-cross-layers 2 \
    --batch-size 64 --lr 1.2e-3 --epochs 16

echo "QUEUE6 DONE $(date -Is)"
