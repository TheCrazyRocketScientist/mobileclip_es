#!/bin/bash
# run_full_pipeline.sh
# Full WSSS evaluation pipeline on deduplicated SIIM dataset
# Logs all output to pipeline.log

set -e
LOG="pipeline.log"
CKPT="checkpoints/fastvit_sa12.apple_in1k_distil-biobert.tar"
PROMPTS="data/train_prompts_all.json"
MASK_CSV="data/siim_with_masks.csv"

echo "================================================" | tee $LOG
echo "  WSSS Pipeline — $(date)" | tee -a $LOG
echo "================================================" | tee -a $LOG

# -----------------------------------------------------------------------
# Step 1: Deduplicate CSV
# -----------------------------------------------------------------------
echo "" | tee -a $LOG
echo "=== Step 1: Deduplicate CSV ===" | tee -a $LOG
python3 -c "
import pandas as pd
df = pd.read_csv('data/siim_train.csv', index_col=0)
print(f'Before dedup: {len(df)} rows, {df[\"label\"].sum()} positives')
df_dedup = df.drop_duplicates(subset='image', keep='first')
print(f'After dedup:  {len(df_dedup)} rows, {df_dedup[\"label\"].sum()} positives')
df_dedup.to_csv('data/siim_train_dedup.csv')
print('Saved: data/siim_train_dedup.csv')
" 2>&1 | tee -a $LOG

# -----------------------------------------------------------------------
# Step 2: Generate CAMs — GradCAM (best method, no logit_scale)
# We need to temporarily disable logit_scale for best results
# -----------------------------------------------------------------------
echo "" | tee -a $LOG
echo "=== Step 2: Generate CAMs (GradCAM, all positives) ===" | tee -a $LOG
python test_04_gradcam_text.py \
  --checkpoint $CKPT \
  --prompts_json $PROMPTS \
  --train_csv data/siim_train_dedup.csv \
  --positive_only \
  --cam_method gradcam \
  --out_dir ./cams_full_gradcam \
  --skip_existing \
  2>&1 | tee -a $LOG

# -----------------------------------------------------------------------
# Step 3: Generate CAMs — GradCAM + negative calibration
# -----------------------------------------------------------------------
echo "" | tee -a $LOG
echo "=== Step 3: Generate CAMs (GradCAM + neg calibration) ===" | tee -a $LOG
python test_04_gradcam_text.py \
  --checkpoint $CKPT \
  --prompts_json $PROMPTS \
  --train_csv data/siim_train_dedup.csv \
  --positive_only \
  --cam_method gradcam \
  --neg_calibration \
  --n_neg 100 \
  --out_dir ./cams_full_negcal \
  --skip_existing \
  2>&1 | tee -a $LOG

# -----------------------------------------------------------------------
# Step 4: Generate CAMs — Hybrid alpha=0.3
# -----------------------------------------------------------------------
echo "" | tee -a $LOG
echo "=== Step 4: Generate CAMs (Hybrid alpha=0.3) ===" | tee -a $LOG
python test_04_gradcam_text.py \
  --checkpoint $CKPT \
  --prompts_json $PROMPTS \
  --train_csv data/siim_train_dedup.csv \
  --positive_only \
  --cam_method hybrid \
  --eigen_alpha 0.3 \
  --out_dir ./cams_full_hybrid_03 \
  --skip_existing \
  2>&1 | tee -a $LOG

# -----------------------------------------------------------------------
# Step 5: Evaluate all three methods
# -----------------------------------------------------------------------
echo "" | tee -a $LOG
echo "=== Step 5: IoU Evaluation ===" | tee -a $LOG

echo "" | tee -a $LOG
echo "--- GradCAM ---" | tee -a $LOG
python eval_cam_iou.py \
  --mask_csv $MASK_CSV \
  --train_csv data/siim_train_dedup.csv \
  --cam_dir ./cams_full_gradcam \
  --cam_type raw \
  2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "--- GradCAM + Neg Calibration ---" | tee -a $LOG
python eval_cam_iou.py \
  --mask_csv $MASK_CSV \
  --train_csv data/siim_train_dedup.csv \
  --cam_dir ./cams_full_negcal \
  --cam_type raw \
  2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "--- Hybrid alpha=0.3 ---" | tee -a $LOG
python eval_cam_iou.py \
  --mask_csv $MASK_CSV \
  --train_csv data/siim_train_dedup.csv \
  --cam_dir ./cams_full_hybrid_03 \
  --cam_type raw \
  2>&1 | tee -a $LOG

# -----------------------------------------------------------------------
# Step 6: Visualise sample of 20 positives for each method
# -----------------------------------------------------------------------
echo "" | tee -a $LOG
echo "=== Step 6: Visualise samples ===" | tee -a $LOG

python visualise_masks.py \
  --mask_csv $MASK_CSV \
  --train_csv data/siim_train_dedup.csv \
  --positive_only \
  --n 20 \
  --cam_dir ./cams_full_gradcam \
  --out_dir ./viz_full_gradcam \
  2>&1 | tee -a $LOG

python visualise_masks.py \
  --mask_csv $MASK_CSV \
  --train_csv data/siim_train_dedup.csv \
  --positive_only \
  --n 20 \
  --cam_dir ./cams_full_negcal \
  --out_dir ./viz_full_negcal \
  2>&1 | tee -a $LOG

python visualise_masks.py \
  --mask_csv $MASK_CSV \
  --train_csv data/siim_train_dedup.csv \
  --positive_only \
  --n 20 \
  --cam_dir ./cams_full_hybrid_03 \
  --out_dir ./viz_full_hybrid_03 \
  2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "================================================" | tee -a $LOG
echo "  Pipeline complete — $(date)" | tee -a $LOG
echo "  Log saved to: $LOG" | tee -a $LOG
echo "================================================" | tee -a $LOG