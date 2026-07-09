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
# The control already exists: E2 = batvision_5ch_nolog at hop=160 (0.567 m depth quantum).
# Falsification: RMSE must fall MONOTONICALLY across hop 160 -> 80 -> 40.
#   hop 80 -> T=36 frames, 0.283 m quantum
#   hop 40 -> T=71 frames, 0.142 m quantum
BV5="--use-log False"
run batvision_5ch_nolog_hop80 run_base.py $BV5 --stft-hop 80
run batvision_5ch_nolog_hop40 run_base.py $BV5 --stft-hop 40

echo "QUEUE DONE $(date -Is)"
