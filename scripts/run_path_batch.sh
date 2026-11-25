#!/usr/bin/env bash
# Full batch of the path-tracking task P for one target (paper Sec. VII), in order:
#
#   1. targets        build the reference path, check the design criteria
#   2. inits          the common initialization of every seed (snap-free, cached in OUT/inits)
#      check          state-based / frozen-tangent gradient vs finite differences (H1),
#                     trajectory objective
#   3. jobs           {state, proxy} x {adam, lbfgs} x seeds (network), and the baselines
#                     (direct rates; end pose only, tuned, over 20 and 60 steps), in parallel
#   4. greedy         after 3 (its step cap is the largest step of the state/L-BFGS solutions)
#   5. figure         re-render every run's figures next to its data (3 snapshots, no title)
#   6. verify, report every run complete and consistent; the summary table and curves
#
# The Adam rate is train.adam.lr of the config (3e-4, that of S; no sweep).
#
# Usage:
#   bash scripts/run_path_batch.sh TARGET OUT NPROC [SEEDS] [EXTRA]
#   bash scripts/run_path_batch.sh P1 runs/path_p1 6                                 # the real batch
#   bash scripts/run_path_batch.sh P1 runs/path_smoke 2 "0" "--epochs 2 --no-gif"    # a smoke test
#
# EXTRA is appended to every train / greedy command. RESUME=1 skips every run whose
# OUT/<name>.json exists. Everything goes to OUT/logs; the script stops at the first
# failing stage.
set -euo pipefail
TARGET=${1:?target, e.g. P1}
OUT=${2:?output directory}
NPROC=${3:-4}
SEEDS=${4:-"0 1 2 3 4"}
EXTRA=${5:-}
C=${CONFIG:-configs/task_path_peg_3d.yaml}
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
mkdir -p "$OUT/logs"
run_parallel() {
    grep '^python' | while IFS= read -r line; do
        name=$(printf '%s' "$line" | sed -n 's|.*/logs/\([^ ]*\)\.log.*|\1|p')
        if [ "${RESUME:-0}" = "1" ] && [ -n "$name" ] && [ -f "$OUT/$name.json" ]; then
            echo "skip $name (done)" >&2
            continue
        fi
        printf '%s\n' "$line"
    done | sed "s| > | $EXTRA > |" | xargs -P "$NPROC" -I{} sh -c "{}"
}
echo "== 1. targets"; $PY scripts/train_path.py targets --config "$C" --out "$OUT" --targets "$TARGET" | tee "$OUT/logs/targets.log"
grep -q "criteria: all OK" "$OUT/logs/targets.log" || { echo "design criteria not met"; exit 1; }
echo "== 2a. common initializations (one per seed, in parallel)"
for s in $SEEDS; do
    echo "python scripts/train_path.py inits --config $C --out $OUT --target $TARGET --seeds $s > $OUT/logs/init_s$s.log 2>&1"
done | xargs -P "$NPROC" -I{} sh -c "{}"
n_ok=$(cat "$OUT"/logs/init_s*.log | grep -c '^\[init' || true)
[ "$n_ok" -eq "$(echo $SEEDS | wc -w)" ] || { echo "an initialization failed (see $OUT/logs/init_s*.log)"; exit 1; }
grep -h '^\[init' "$OUT"/logs/init_s*.log
echo "== 2b. gradient check"; $PY scripts/train_path.py check --config "$C" --out "$OUT" --target "$TARGET" --param net | tee "$OUT/logs/check.log"
echo "== 3. main runs and baselines"
# shellcheck disable=SC2086
$PY scripts/train_path.py jobs --config "$C" --out "$OUT" --targets "$TARGET" --seeds $SEEDS --baselines > "$OUT/jobs.sh"
grep -v ' greedy ' "$OUT/jobs.sh" | run_parallel
echo "== 4. greedy"; grep ' greedy ' "$OUT/jobs.sh" | run_parallel
echo "== 5. figures"
for f in "$OUT"/"$TARGET"_*.npz; do
    $PY scripts/train_path.py figure --config "$C" --npz "$f" --snapshots 3 --no-title > "$OUT/logs/figure_$(basename "$f" .npz).log" 2>&1
done
echo "== 6. verify and report"
$PY scripts/train_path.py verify --config "$C" --out "$OUT" | tee "$OUT/logs/verify.log"
$PY scripts/train_path.py report --config "$C" --out "$OUT" > "$OUT/logs/report.log"
echo "done: $OUT/report.md"
