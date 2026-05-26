# -*- coding: utf-8 -*-
"""
eval_cam_iou.py
---------------
Compute IoU of raw/refined CAMs against GT RLE masks across SIIM positives.
Thresholds the CAM at multiple values and reports best IoU + optimal threshold.

Usage:
    python eval_cam_iou.py \
        --mask_csv data/siim_with_masks.csv \
        --train_csv data/siim_train.csv \
        --cam_dir ./test_04_output \
        --cam_type raw      # or refined
"""

import os
import argparse
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# RLE decode — SIIM native 1024x1024
# ---------------------------------------------------------------------------

SIIM_H, SIIM_W = 1024, 1024

def rle_decode(rle_string):
    if str(rle_string).strip() in ('-1', '', 'nan'):
        return np.zeros((SIIM_H, SIIM_W), dtype=np.uint8)
    array = np.asarray([int(x) for x in str(rle_string).split()])
    starts, lengths = array[0::2], array[1::2]
    mask = np.zeros(SIIM_W * SIIM_H, dtype=np.uint8)
    current_position = 0
    for start, length in zip(starts, lengths):
        current_position += start
        mask[current_position: current_position + length] = 1
        current_position += length
    return mask.reshape(SIIM_W, SIIM_H).T


def merge_masks(rle_list, img_h, img_w):
    combined = np.zeros((SIIM_H, SIIM_W), dtype=np.uint8)
    for rle in rle_list:
        combined = np.maximum(combined, rle_decode(rle))
    if (img_h, img_w) != (SIIM_H, SIIM_W):
        combined = cv2.resize(combined, (img_w, img_h),
                              interpolation=cv2.INTER_NEAREST)
    return combined


def build_mask_lookup(mask_csv):
    df = pd.read_csv(mask_csv, index_col=0)
    lookup = {}
    for _, row in df.iterrows():
        img_id = str(row['ImageId']).strip()
        rle    = str(row['EncodedPixels']).strip()
        lookup.setdefault(img_id, []).append(rle)
    return lookup


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def compute_iou(pred_binary, gt_binary):
    intersection = (pred_binary & gt_binary).sum()
    union        = (pred_binary | gt_binary).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def best_iou_over_thresholds(cam, gt_mask, thresholds):
    """Try multiple thresholds, return best IoU and the threshold that gave it."""
    best_iou   = 0.0
    best_thresh = thresholds[0]
    for t in thresholds:
        pred = (cam >= t).astype(np.uint8)
        iou  = compute_iou(pred, gt_mask)
        if iou > best_iou:
            best_iou    = iou
            best_thresh = t
    return best_iou, best_thresh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_csv',  required=True)
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--cam_dir',   required=True)
    parser.add_argument('--cam_type',  default='raw',
                        choices=['raw', 'refined'],
                        help='Which CAM to evaluate')
    parser.add_argument('--img_root',  default='')
    parser.add_argument('--positive_only', action='store_true', default=True)
    args = parser.parse_args()

    mask_lookup = build_mask_lookup(args.mask_csv)

    df = pd.read_csv(args.train_csv, index_col=0)
    if args.positive_only:
        df = df[df['label'] == 1]
    print(f"Evaluating {len(df)} images ({args.cam_type} CAM)\n")

    thresholds = np.arange(0.1, 0.95, 0.05).tolist()

    results = []
    missing_cam  = 0
    missing_mask = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        rel   = str(row['image'])
        path  = os.path.join(args.img_root, rel) if args.img_root else rel
        stem  = os.path.splitext(os.path.basename(rel))[0]

        # Load CAM
        # Support both test_04 naming ({stem}_{type}_cam.npy)
        # and CRF prob naming ({stem}.npy)
        cam_path = os.path.join(args.cam_dir, f'{stem}_{args.cam_type}_cam.npy')
        if not os.path.exists(cam_path):
            cam_path = os.path.join(args.cam_dir, f'{stem}.npy')
        if not os.path.exists(cam_path):
            missing_cam += 1
            continue
        cam = np.load(cam_path).astype(np.float32)

        # Normalise CAM to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            missing_cam += 1
            continue
        cam = (cam - cam_min) / (cam_max - cam_min)

        # Load GT mask
        rle_list = mask_lookup.get(stem, ['-1'])
        if all(r.strip() in ('-1', 'nan', '') for r in rle_list):
            missing_mask += 1
            continue

        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_mask = merge_masks(rle_list, h, w)
        cam_r   = cv2.resize(cam, (w, h))

        best_iou, best_t = best_iou_over_thresholds(cam_r, gt_mask, thresholds)

        # Also compute IoU at fixed thresholds for comparison
        iou_30 = compute_iou((cam_r >= 0.3).astype(np.uint8), gt_mask)
        iou_50 = compute_iou((cam_r >= 0.5).astype(np.uint8), gt_mask)

        results.append({
            'image_id':  stem,
            'best_iou':  best_iou,
            'best_thresh': best_t,
            'iou_t0.3':  iou_30,
            'iou_t0.5':  iou_50,
        })

    if not results:
        print("No results — check cam_dir and that CAM .npy files exist")
        return

    # Summary
    best_ious = [r['best_iou']  for r in results]
    iou_30    = [r['iou_t0.3']  for r in results]
    iou_50    = [r['iou_t0.5']  for r in results]

    print(f"\n{'='*55}")
    print(f"  Results ({args.cam_type} CAM)  —  {len(results)} images evaluated")
    print(f"{'='*55}")
    print(f"  Best IoU (oracle threshold):  "
          f"mean={np.mean(best_ious):.3f}  "
          f"median={np.median(best_ious):.3f}  "
          f"min={np.min(best_ious):.3f}  "
          f"max={np.max(best_ious):.3f}")
    print(f"  IoU @ threshold=0.3:          "
          f"mean={np.mean(iou_30):.3f}  "
          f"median={np.median(iou_30):.3f}")
    print(f"  IoU @ threshold=0.5:          "
          f"mean={np.mean(iou_50):.3f}  "
          f"median={np.median(iou_50):.3f}")

    # Most common best threshold
    best_thresholds = [r['best_thresh'] for r in results]
    thresh_counts = {}
    for t in best_thresholds:
        t_r = round(t, 2)
        thresh_counts[t_r] = thresh_counts.get(t_r, 0) + 1
    most_common_t = max(thresh_counts, key=thresh_counts.get)
    print(f"\n  Most common optimal threshold: {most_common_t:.2f}")

    print(f"\n  Missing CAM files:  {missing_cam}")
    print(f"  Missing GT masks:   {missing_mask}")

    # Per-image breakdown
    print(f"\n{'='*55}")
    print(f"  Per-image breakdown")
    print(f"{'='*55}")
    print(f"  {'ImageId (last 20)':22s}  {'BestIoU':>8}  {'@t':>5}  "
          f"{'IoU@0.3':>8}  {'IoU@0.5':>8}")
    for r in sorted(results, key=lambda x: -x['best_iou']):
        print(f"  ...{r['image_id'][-20:]}  "
              f"{r['best_iou']:>8.3f}  "
              f"{r['best_thresh']:>5.2f}  "
              f"{r['iou_t0.3']:>8.3f}  "
              f"{r['iou_t0.5']:>8.3f}")


if __name__ == '__main__':
    main()