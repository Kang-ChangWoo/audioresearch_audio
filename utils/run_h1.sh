#!/usr/bin/env bash
# Stage H1 (I27): log_mae LOSS on the champion -- the correct geometric-median objective.
cd "$(dirname "$0")/.."
exec 222>out/.h1.lock
flock -n 222 || { echo "[h1] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}
# I25 refuted: re-parameterising the OUTPUT to log-depth left masked-MAE on LINEAR depth, so the
# optimum stayed the arithmetic median and the 1-2m ratio histogram did not budge. The correct form
# is the LOSS in log space: |log D - log gt|, whose optimum IS the geometric median (ratio 1).
#
# main_loss=log_mae already exists (from I13). But I13 ran it on the FAST parent and judged FAR
# deciles on a range mis-diagnosis -- it failed THERE. Here it runs on the CHAMPION and is judged on
# the 1-2m INTERIOR, the correct median-pull locus. Different parent, different locus, different test.
#
# PRE-REGISTERED: the 1-2m interior ratio histogram must shift the 0.9-1.0 pile toward 1.0-1.11, and
# 1-2m interior d1 must rise over E23. ABS_REL is not evidence (log_mae improves it by construction).
run raydpt_e34_logmae_champ --epochs 22 --main-loss log_mae
echo "H1 DONE $(date -Is)"
