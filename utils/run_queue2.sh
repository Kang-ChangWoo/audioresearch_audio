#!/usr/bin/env bash
# Second-stage queue. Waits for run_queue.sh to print QUEUE DONE, then runs the
# I6-vs-I7 discriminating ablation.
#
# A separate file on purpose: bash reads a running script incrementally from a byte
# offset, so editing run_queue.sh while it executes can corrupt it. Never edit a live
# queue script -- add a new one.
#
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."

exec 201>out/.queue2.lock
flock -n 201 || { echo "[queue2] another queue2 runner holds the lock; exiting."; exit 0; }
echo "[queue2] single-instance lock acquired (pid $$)"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs

echo "[queue2] waiting for run_queue.sh to finish... $(date -Is)"
while ! grep -q "QUEUE DONE" out/queue_driver.log 2>/dev/null; do sleep 60; done
echo "[queue2] stage-1 queue finished. starting. $(date -Is)"

run() { local name="$1"; local script="$2"; shift 2
    echo "=== $name START $(date -Is) ==="
    python "$script" --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# I6 vs I7 -- ONE run decides TWO competing explanations of the same observation.
#
# Observed (zero GPU): the model is a LOW-PASS predictor. 97.4% of its azimuthal power sits
# in k<=6 vs GT's 75.8%, and only ~5% of GT's power survives at k>=17.
#   I6: the objective is to blame. The two low-frequency auxiliaries (coarse-layout 16x32,
#       low-pass sigma=3) carry 58.2% of the loss at convergence, so the model is trained blurry.
#   I7: the sensor is to blame. Two microphones give a broad directional response, so fine
#       azimuthal structure is close to unobservable.
#
# Discriminating run: same parent as E3 (batvision 5ch log), auxiliaries ZEROED.
#   If azimuthal power above k=6 rises  -> I6 supported, I7 dropped.
#   If the spectrum still collapses     -> I6 refuted, I7 survives, and the lever moves to
#                                          the representation rather than the loss.
# Guard against a confound: report d1 too. If d1 REGRESSES the auxiliaries were load-bearing
# regularisers, and "blurry" was the price of stability, not a bias.
run batvision_5ch_log_noaux run_base.py --use-log True --w-coarse-layout 0 --w-low 0

echo "QUEUE2 DONE $(date -Is)"
