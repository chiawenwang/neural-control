#!/usr/bin/env bash
# Full batch of the shape task S for one target (paper Sec. VII), in order:
#
#   1. targets        build the target, check the design criteria
#   2. check          state-based / frozen-tangent gradient vs finite differences (H1)
#   3. sweep          only with SWEEP=1: Adam rate on seed 0 (frozen tangent), then the fixed
#                     rule -> OUT/adam_lr.json; by default the rate of the config (train.adam.lr)
#   4. jobs           {state, proxy} x {adam, lbfgs} x seeds, and the baselines (direct rates,
#                     no obstacles, end pose tuned / aimed over 20 and 60 steps), in parallel
#   5. greedy         after 4 (its step cap is the largest step of the state/L-BFGS solutions)
#   6. figure         re-render every run's figures next to its data (3 snapshots, no title)
#   7. verify, report every run complete and consistent; the summary table and curves
#
# Usage:
#   bash scripts/run_shape_batch.sh TARGET OUT NPROC [SEEDS] [EXTRA]
#   bash scripts/run_shape_batch.sh P1 runs/shape_p1 8                      # the real batch
#   bash scripts/run_shape_batch.sh P1 runs/smoke 2 "0" "--epochs 2"        # a smoke test
#   SWEEP=1 bash scripts/run_shape_batch.sh P1 runs/shape_p1 8              # with the Adam-rate sweep
#
# EXTRA is appended to every train / greedy command (e.g. "--epochs 2").
# RESUME=1 skips every run whose OUT/<name>.json exists (to continue after an interruption).
# Everything goes to OUT/logs; the script stops at the first failing stage.
set -euo pipefail
TARGET=${1:?target, e.g. P1}
OUT=${2:?output directory}
NPROC=${3:-4}
SEEDS=${4:-"0 1 2 3 4"}
EXTRA=${5:-}
C=${CONFIG:-configs/task_shape_pegs_3d.yaml}
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
mkdir -p "$OUT/logs"
run_parallel() {   # read commands on stdin, append EXTRA before the redirection, run NPROC at a time
    # RESUME=1: skip a run whose result OUT/<name>.json already exists (after a disconnect)
    grep '^python' | while IFS= read -r line; do
        name=$(printf '%s' "$line" | sed -n 's|.*/logs/\([^ ]*\)\.log.*|\1|p')
        if [ "${RESUME:-0}" = "1" ] && [ -n "$name" ] && [ -f "$OUT/$name.json" ]; then
            echo "skip $name (done)" >&2
            continue
        fi
        printf '%s\n' "$line"
    done | sed "s| > | $EXTRA > |" | xargs -P "$NPROC" -I{} sh -c "{}"
}
echo "== 1. targets"; $PY scripts/train_shape.py targets --config "$C" --out "$OUT" --targets "$TARGET" | tee "$OUT/logs/targets.log"
grep -q "criteria: all OK" "$OUT/logs/targets.log" || { echo "design criteria not met"; exit 1; }
echo "== 2. gradient check"; $PY scripts/train_shape.py check --config "$C" --out "$OUT" --target "$TARGET" --param net --no-pegs-too | tee "$OUT/logs/check.log"
if [ "${SWEEP:-0}" = "1" ]; then
    echo "== 3. Adam-rate sweep (seed 0)"
    $PY scripts/train_shape.py sweep --config "$C" --out "$OUT" --targets "$TARGET" > "$OUT/sweep_jobs.sh"
    bash -c "$(head -n 1 "$OUT/sweep_jobs.sh")"          # its setup line: OUT/sweep with the target
    run_parallel < "$OUT/sweep_jobs.sh"
    $PY scripts/train_shape.py sweep --config "$C" --out "$OUT" --targets "$TARGET" --pick | tee "$OUT/logs/sweep_pick.log"
else
    echo "== 3. Adam rate: train.adam.lr of $C (no sweep; SWEEP=1 runs one)"
    rm -f "$OUT/adam_lr.json"
fi
echo "== 4. main runs and baselines"
# shellcheck disable=SC2086
$PY scripts/train_shape.py jobs --config "$C" --out "$OUT" --targets "$TARGET" --seeds $SEEDS --baselines > "$OUT/jobs.sh"
grep -v ' greedy ' "$OUT/jobs.sh" | run_parallel
echo "== 5. greedy"; grep ' greedy ' "$OUT/jobs.sh" | run_parallel
echo "== 6. figures"
for f in "$OUT"/"$TARGET"_*.npz; do
    $PY scripts/train_shape.py figure --config "$C" --npz "$f" --snapshots 3 --no-title > "$OUT/logs/figure_$(basename "$f" .npz).log" 2>&1
done
echo "== 7. verify and report"
$PY scripts/train_shape.py verify --config "$C" --out "$OUT" | tee "$OUT/logs/verify.log"
$PY scripts/train_shape.py report --config "$C" --out "$OUT" > "$OUT/logs/report.log"
echo "done: $OUT/report.md"
