#!/usr/bin/env bash
# Staged experiment queue. Runs AFTER the batvision grid finishes, one experiment at a time.
#
# Serialisation is enforced twice, deliberately:
#   1. this script waits for the grid driver to print ALL DONE, and
#   2. every scored run takes the eval_lock (utils/evallock.py) anyway.
# TIME_BUDGET is wall-clock, so overlapping runs fit fewer epochs and stop being comparable.
#
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."

# Single-instance guard: two queue runners would launch the same experiment twice and
# fight over the GPU. flock is released by the kernel if this process dies.
exec 200>out/.queue.lock
flock -n 200 || { echo "[queue] another queue runner holds out/.queue.lock; exiting."; exit 0; }
echo "[queue] single-instance lock acquired (pid $$)"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs

# --- wait for the grid to finish (it holds the GPU for its remaining cells) ---
echo "[queue] waiting for the batvision grid to finish... $(date -Is)"
while ! grep -q "ALL DONE" out/grid_driver.log 2>/dev/null; do sleep 60; done
echo "[queue] grid finished. starting queue. $(date -Is)"

run() {  # name  script  extra-args...
    local name="$1"; local script="$2"; shift 2
    echo "=== $name START $(date -Is) ==="
    python "$script" --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# E4 -- raydpt-baseline lineage: re-anchor MY MODEL under the planar target.
# Prerequisite for every RayDPT improvement; nothing before commit 87b3047 is comparable.
run raydpt_e4_planar train.py

# I1 probe -- acoustic-representation / temporal resolution, on the CHEAPEST parent
# (batvision reference), NOT on the champion. Anti-anchoring: if finer time-of-flight
# resolution is a real mechanism it must show up on the simplest model.
#
# DESIGN CORRECTION (before any GPU was spent): the STFT's temporal RESOLUTION is set by
# the analysis WINDOW, not by the hop. The window's support smears an echo over
# c*win/(2*sr) of one-way depth; the hop only controls how densely that smeared function
# is sampled. At win=400 the smear is 1.417 m -- the same size as the achieved RMSE 1.3207 m.
# Shrinking the hop alone leaves the smear at 1.417 m and merely oversamples.
#
# So the two arms DISSOCIATE sampling density from resolution. Control = E2 (win 400, hop 160).
#   A: win 400, hop 40  -> T=71,  smear 1.417 m (UNCHANGED)  = density only
#   B: win  64, hop 16  -> T=177, smear 0.227 m              = true resolution (costs freq. resolution)
#
# PRE-REGISTERED PREDICTION: if the bottleneck is time-of-flight resolution, B improves RMSE
# and A does not. If A and B both improve, the gain is extra sampling/capacity, not resolution.
# If neither improves, I1 is DROPped -- temporal resolution is not the binding constraint.
# Per D3, watch WHICH metric moves: I1 should buy RMSE (range), not d1 (angle).
BV5="--use-log False"
run batvision_5ch_win400_hop40 run_base.py $BV5 --stft-win 400 --stft-hop 40
run batvision_5ch_win64_hop16  run_base.py $BV5 --stft-win 64  --stft-hop 16

echo "QUEUE DONE $(date -Is)"
