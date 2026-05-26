#!/bin/bash
# compare_models.sh
# Auto-discovers all .tar checkpoints in ./checkpoints/ and runs
# vanilla softmax GradCAM pipeline on each, with its own output folder.
#
# Usage:
#   bash compare_models.sh                  # 5 images, all checkpoints
#   bash compare_models.sh --n 10           # 10 images
#   bash compare_models.sh --skip           # skip already processed
#   bash compare_models.sh --ckpt_dir path  # custom checkpoint dir

set -e

# -----------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------
SCRIPT="test_04_gradcam_text_softmax.py"
PROMPTS="data/train_prompts_all.json"
TRAIN_CSV="data/siim_train.csv"
MASK_CSV="data/siim_with_masks.csv"
N_IMAGES=5
LOG="compare_models.log"
CKPT_DIR="./checkpoints"
SKIP_FLAG=""

# -----------------------------------------------------------------------
# Parse args
# -----------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --n)        N_IMAGES=$2;  shift 2 ;;
        --n=*)      N_IMAGES="${1#*=}"; shift ;;
        --skip)     SKIP_FLAG="--skip_existing"; shift ;;
        --ckpt_dir) CKPT_DIR=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; shift ;;
    esac
done

# -----------------------------------------------------------------------
# Auto-discover checkpoints
# -----------------------------------------------------------------------
CHECKPOINTS=()
for tar in "$CKPT_DIR"/*.tar; do
    [ -f "$tar" ] || continue
    CHECKPOINTS+=("$tar")
done

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "No .tar files found in $CKPT_DIR"
    echo "Run: python download_artifacts.py --entity YOUR_ENTITY --project YOUR_PROJECT"
    exit 1
fi

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
log() { echo "$1" | tee -a $LOG; }
sep() { log ""; log "$(printf '=%.0s' {1..60})"; log "  $1"; log "$(printf '=%.0s' {1..60})"; }

# -----------------------------------------------------------------------
# Build deduplicated CSV with N positives
# -----------------------------------------------------------------------
DEDUP_CSV="data/siim_compare_n${N_IMAGES}.csv"
python3 -c "
import pandas as pd
df = pd.read_csv('$TRAIN_CSV', index_col=0)
df = df.drop_duplicates(subset='image', keep='first')
pos = df[df['label']==1].head($N_IMAGES)
neg = df[df['label']==0].head(50)
out = pd.concat([pos, neg]).reset_index(drop=True)
out.to_csv('$DEDUP_CSV')
print(f'CSV: {len(pos)} positives + {len(neg)} negatives → $DEDUP_CSV')
" 2>&1 | tee -a $LOG

# -----------------------------------------------------------------------
# Summary header
# -----------------------------------------------------------------------
sep "Model Comparison Pipeline — $(date)"
log "  Script     : $SCRIPT"
log "  N images   : $N_IMAGES"
log "  Ckpt dir   : $CKPT_DIR"
log "  Log        : $LOG"
log ""
log "  Discovered checkpoints:"
for ckpt in "${CHECKPOINTS[@]}"; do
    log "    $(basename $ckpt)"
done

# -----------------------------------------------------------------------
# Per-model loop
# -----------------------------------------------------------------------
RESULTS=()
FAILED=()

for CKPT in "${CHECKPOINTS[@]}"; do
    # Nickname = filename without extension
    NICKNAME=$(basename "$CKPT" .tar)
    OUT_DIR="./cam_compare/${NICKNAME}"
    VIZ_DIR="./cam_compare/${NICKNAME}_viz"
    mkdir -p "$OUT_DIR" "$VIZ_DIR"

    sep "Model: $NICKNAME"
    log "  Checkpoint : $CKPT"
    log "  Output     : $OUT_DIR"
    log "  Started    : $(date)"

    # Generate CAMs
    if python "$SCRIPT" \
        --checkpoint "$CKPT" \
        --prompts_json "$PROMPTS" \
        --train_csv "$DEDUP_CSV" \
        --positive_only \
        --out_dir "$OUT_DIR" \
        $SKIP_FLAG \
        2>&1 | tee -a $LOG; then

        # CUDA cleanup
        python3 -c "
import torch, gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f'  GPU cleared. Allocated: {torch.cuda.memory_allocated()/1e6:.1f}MB')
" 2>&1 | tee -a $LOG

        # Evaluate
        log ""
        log "  --- IoU: $NICKNAME ---"
        python eval_cam_iou.py \
            --mask_csv "$MASK_CSV" \
            --train_csv "$DEDUP_CSV" \
            --cam_dir "$OUT_DIR" \
            --cam_type raw \
            2>&1 | tee -a $LOG

        # Visualise
        log ""
        log "  --- Viz: $NICKNAME ---"
        python visualise_masks.py \
            --mask_csv "$MASK_CSV" \
            --train_csv "$DEDUP_CSV" \
            --positive_only \
            --cam_dir "$OUT_DIR" \
            --out_dir "$VIZ_DIR" \
            2>&1 | tee -a $LOG

        RESULTS+=("$NICKNAME")
        log "  Finished : $(date)"
    else
        log "  [FAILED] $NICKNAME — check log for errors"
        FAILED+=("$NICKNAME")
    fi
done

# -----------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------
sep "FINAL SUMMARY — $(date)"
log ""
log "  Completed (${#RESULTS[@]}):"
for r in "${RESULTS[@]}"; do
    log "    ✓ $r"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    log ""
    log "  Failed (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do
        log "    ✗ $f"
    done
fi

log ""
log "  IoU comparison (copy this):"
log "  $(printf '%-40s %-12s %-12s %-12s' 'Model' 'MeanBestIoU' 'IoU@0.3' 'IoU@0.5')"
grep -A4 "Best IoU (oracle" $LOG | grep -E "mean=|^--" | \
    paste - - | awk '{print $0}' 2>/dev/null || \
    log "  Run: grep -A4 'Best IoU' $LOG"

log ""
log "  Quick IoU extract:"
log "    grep -A4 'Best IoU (oracle' $LOG"