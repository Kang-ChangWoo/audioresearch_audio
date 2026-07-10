#!/usr/bin/env bash
# Stage 9 (S6 / I13): far-field range compression is where RayDPT loses d1.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 208>out/.queue9.lock
flock -n 208 || { echo "[queue9] another queue9 runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# MEASURED (utils/diag_d1.py, full val): RayDPT's d1 deficit is NOT azimuthal. Across azimuth
# sectors it is flat (std 0.0056 on a mean of 0.0245); across elevation the floor shows ZERO
# deficit; across GT depth it explodes -- at 8-9 m batvision scores d1 0.4023 and RayDPT 0.1751.
# Both models under-predict far depth badly (GT 8.5 m -> batvision 5.74 m, RayDPT 4.97 m).
#
# d1 is a +-25% RELATIVE threshold, so far-field compression fails it outright while RMSE barely
# notices (far pixels are rare). Masked MAE on normalised depth drives each pixel to the
# CONDITIONAL MEDIAN, and far depths are a minority of the mass, so the median sits short.
#
# PRE-REGISTERED PREDICTION: a relative dense term should recover the far deciles and raise d1,
# while RMSE gets slightly WORSE (it rewards hedging toward the mean). Judged on d1 and RMSE.
# ABS_REL will improve almost by construction and is therefore NOT evidence -- program.md.
#
# FALSIFICATION: if the far deciles (7-10 m) do not improve, compression was not caused by the
# loss and I13 is dropped, whatever the composite does.
#
# Parent = E11 (the RayDPT champion). Both arms keep its architecture exactly.
run raydpt_e13_relmae --amp bf16 --decode-scale 32 --ray-cross-layers 2 --cross-kv32 e4 \
    --batch-size 64 --lr 1.2e-3 --epochs 24 --main-loss rel_mae

run raydpt_e14_logmae --amp bf16 --decode-scale 32 --ray-cross-layers 2 --cross-kv32 e4 \
    --batch-size 64 --lr 1.2e-3 --epochs 24 --main-loss log_mae

echo "QUEUE9 DONE $(date -Is)"
