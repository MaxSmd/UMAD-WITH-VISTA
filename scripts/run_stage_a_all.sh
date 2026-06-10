#!/bin/bash
# Stage A driver: run Vista rollout inference over every viable AnoVox scenario
# in Final_Output_2026_05_26-21_18, sequentially on GPU 1.
#
# Resumable: scenarios whose <OUT_BASE>/<tag>/manifest.json already exists are
# skipped, so a re-run after interruption continues where it left off.
#
# Per-scenario stdout/stderr -> outputs/anovox-inference/_logs/<tag>.log
# Index/timing summary       -> outputs/anovox-inference/_logs/run.log

set -u

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

# Absolute paths required: run_inference_anovox.py chdirs into external/Vista
# early, breaking any relative --scenario / --out arguments.
SCENS_DIR=$REPO/data/anovox/Outputs/Final_Output_2026_05_26-21_18
OUT_BASE=$REPO/outputs/anovox-inference
LOG_DIR=$OUT_BASE/_logs
mkdir -p "$LOG_DIR"

RUN_LOG=$LOG_DIR/run.log

ts() { date -Iseconds; }

echo "[stage-A all] started $(ts) on host $(hostname)" | tee -a "$RUN_LOG"

t_all=$(date +%s)
for s in "$SCENS_DIR"/Scenario_*-*; do
    [ -d "$s" ] || continue
    n=$(ls "$s"/RGB-CAM*/*.png 2>/dev/null | wc -l)
    if [ "$n" -lt 200 ]; then
        echo "[skip] $(basename "$s"): only $n RGB frames" | tee -a "$RUN_LOG"
        continue
    fi
    tag=$(basename "$s" | cut -d- -f1)
    out=$OUT_BASE/$tag
    if [ -f "$out/manifest.json" ]; then
        echo "[skip] $tag: manifest.json already exists" | tee -a "$RUN_LOG"
        continue
    fi
    echo "[run]  $tag ($n frames) start $(ts)" | tee -a "$RUN_LOG"
    t0=$(date +%s)
    external/Vista/.venv/bin/python scripts/run_inference_anovox.py \
        --gpu 1 --low-vram --height 512 --width 832 \
        --n-conds 3 --rollout-stride 12 \
        --scenario "$s" --out "$OUT_BASE" \
        > "$LOG_DIR/$tag.log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    echo "[done] $tag rc=$rc in ${dt}s $(ts)" | tee -a "$RUN_LOG"
done

dt_all=$(( $(date +%s) - t_all ))
echo "[stage-A all] finished $(ts) total=${dt_all}s" | tee -a "$RUN_LOG"
