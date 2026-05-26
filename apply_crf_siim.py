# -*- coding: utf-8 -*-
"""
apply_crf_siim.py
-----------------
DenseCRF post-processing on raw CAM .npy files, adapted directly from
CLIP-ES eval_cam_with_crf.py for binary SIIM pneumothorax segmentation.

Exact CLIP-ES CRF parameters preserved:
  iter_max=10, pos_xy_std=1, pos_w=3, bi_xy_std=67, bi_rgb_std=3, bi_w=4

Binary adaptation:
  - 2 classes: background (0) and pneumothorax (1)
  - bg_score = (1 - cam_fg) ^ 1   (same power as original)
  - prob = [bg_score, fg_score]
  - confidence threshold 0.95 → pixels below set to 255 (ignore)

Install pydensecrf first:
  pip install git+https://github.com/lucasb-eyer/pydensecrf.git

Usage:
    python apply_crf_siim.py \
        --cam_dir ./test_04_output \
        --image_root final_512/siim/images \
        --train_csv data/siim_train.csv \
        --out_dir ./crf_output \
        --cam_type raw
"""

import os
import argparse
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import joblib
import multiprocessing

try:
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as utils
except ImportError:
    raise ImportError(
        "pydensecrf not installed. Run:\n"
        "  pip install git+https://github.com/lucasb-eyer/pydensecrf.git"
    )


# ---------------------------------------------------------------------------
# DenseCRF class — copied verbatim from CLIP-ES eval_cam_with_crf.py
# ---------------------------------------------------------------------------

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max   = iter_max
        self.pos_w      = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w       = bi_w
        self.bi_xy_std  = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)
        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))
        return Q


# ---------------------------------------------------------------------------
# CLIP-ES CRF parameters (exact values from original repo)
# ---------------------------------------------------------------------------

# CRF parameters adapted for CXR greyscale images
# Original CLIP-ES values (VOC natural images):
#   iter_max=10, pos_xy_std=1, pos_w=3, bi_xy_std=67, bi_rgb_std=3, bi_w=4
#   confidence_threshold=0.95
#
# CXR adaptations:
#   bi_rgb_std: increased 3→10 — CXR is greyscale/low-colour-contrast,
#               strict colour similarity (3) kills the bilateral filter
#   bi_xy_std:  reduced 67→50 — encourage more local spatial coherence
#   pos_w:      increased 3→4 — stronger spatial smoothness prior
#   confidence_threshold: reduced 0.95→0.50 — CXR softmax scores are
#               inherently lower than natural image scores; 0.95 eliminates
#               almost all pixels on CXR data
CRF_PARAMS = dict(
    iter_max   = 10,
    pos_xy_std = 1,
    pos_w      = 4,
    bi_xy_std  = 50,
    bi_rgb_std = 10,
    bi_w       = 4,
)

# Mean BGR subtraction — same as CLIP-ES (ImageNet BGR mean)
MEAN_BGR = np.array([104.008, 116.669, 122.675], dtype=np.float32)

# Confidence threshold — lowered from 0.95 (VOC) to 0.50 (CXR)
CONFIDENCE_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Per-image processing function
# ---------------------------------------------------------------------------

def process_one(args_tuple):
    """
    Process a single image: load CAM .npy, run DenseCRF, save pseudo mask.
    Returns (image_id, success, message).
    """
    image_id, img_path, cam_path, out_dir, cam_type, save_vis = args_tuple

    postprocessor = DenseCRF(**CRF_PARAMS)

    try:
        # --- Load image ---
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            return image_id, False, f"Cannot read image: {img_path}"
        H, W = image.shape[:2]

        # Mean subtraction (same as CLIP-ES)
        image_float = image.astype(np.float32) - MEAN_BGR
        # HWC → CHW
        image_chw = image_float.transpose(2, 0, 1)

        # --- Load CAM ---
        if not os.path.exists(cam_path):
            return image_id, False, f"CAM not found: {cam_path}"

        cam_npy = np.load(cam_path, allow_pickle=True)
        # Handle both plain array and dict (from generate_cams_siim.py)
        if isinstance(cam_npy, np.ndarray) and cam_npy.dtype == object:
            cam_dict = cam_npy.item()
            if 'attn_highres' in cam_dict:
                cam_fg = cam_dict['attn_highres'].astype(np.float32)
            elif cam_type in cam_dict:
                cam_fg = cam_dict[cam_type].astype(np.float32)
            else:
                cam_fg = list(cam_dict.values())[0].astype(np.float32)
        else:
            cam_fg = cam_npy.astype(np.float32)

        # Resize CAM to image size if needed
        if cam_fg.shape != (H, W):
            cam_fg = cv2.resize(cam_fg, (W, H), interpolation=cv2.INTER_LINEAR)

        # Normalise to [0, 1]
        cam_min, cam_max = cam_fg.min(), cam_fg.max()
        if cam_max - cam_min < 1e-8:
            return image_id, False, "CAM is flat (zero range)"
        cam_fg = (cam_fg - cam_min) / (cam_max - cam_min)

        # --- Build 2-class probability map (same as CLIP-ES) ---
        # bg_score = (1 - max_fg)^1
        bg_score = np.power(1.0 - cam_fg[np.newaxis], 1)       # (1, H, W)
        fg_score = cam_fg[np.newaxis]                            # (1, H, W)
        prob = np.concatenate([bg_score, fg_score], axis=0)     # (2, H, W)

        # Clip to avoid log(0) in unary energy
        prob = np.clip(prob, 1e-5, 1.0)
        prob = prob / prob.sum(axis=0, keepdims=True)

        # --- Run CRF ---
        image_uint8 = image.astype(np.uint8)   # pydensecrf needs uint8 HWC
        Q = postprocessor(image_uint8, prob)    # (2, H, W)

        # --- Generate pseudo mask ---
        label = np.argmax(Q, axis=0).astype(np.uint8)  # 0=bg, 1=ptx

        # Confidence-guided: low-confidence pixels → 255 (ignore)
        confidence = np.max(Q, axis=0)
        label[confidence < CONFIDENCE_THRESHOLD] = 255

        # --- Save pseudo mask ---
        mask_path = os.path.join(out_dir, 'masks', f'{image_id}.png')
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(mask_path, label)

        # --- Save CRF probability map as .npy for IoU evaluation ---
        prob_path = os.path.join(out_dir, 'probs', f'{image_id}.npy')
        os.makedirs(os.path.dirname(prob_path), exist_ok=True)
        np.save(prob_path, Q[1].astype(np.float16))  # save fg probability

        # --- Save visualisation ---
        if save_vis:
            vis_dir = os.path.join(out_dir, 'vis')
            os.makedirs(vis_dir, exist_ok=True)

            # CRF fg probability heatmap
            crf_prob = Q[1]
            crf_norm = (crf_prob * 255).astype(np.uint8)
            heatmap  = cv2.applyColorMap(crf_norm, cv2.COLORMAP_JET)
            overlay  = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

            # Raw CAM heatmap for comparison
            raw_norm = (cam_fg * 255).astype(np.uint8)
            raw_heat = cv2.applyColorMap(raw_norm, cv2.COLORMAP_JET)
            raw_over = cv2.addWeighted(image, 0.5, raw_heat, 0.5, 0)

            # Binary pseudo mask overlay
            mask_vis = image.copy()
            mask_vis[label == 1]   = mask_vis[label == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
            mask_vis[label == 255] = mask_vis[label == 255] * 0.5 + np.array([128, 128, 128]) * 0.5

            panel = np.concatenate([raw_over, overlay, mask_vis], axis=1)

            # Add labels
            for i, txt in enumerate(['Raw CAM', 'CRF Prob', 'Pseudo Mask']):
                x = i * W + 10
                cv2.putText(panel, txt, (x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 0), 3)
                cv2.putText(panel, txt, (x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 1)

            cv2.imwrite(os.path.join(vis_dir, f'{image_id}_crf.jpg'), panel)

        return image_id, True, 'OK'

    except Exception as e:
        return image_id, False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_dir',    required=True,
                        help='Directory containing *_raw_cam.npy files')
    parser.add_argument('--image_root', required=True,
                        help='Root directory of SIIM images')
    parser.add_argument('--train_csv',  required=True,
                        help='Path to siim_train.csv')
    parser.add_argument('--out_dir',    default='./crf_output')
    parser.add_argument('--cam_type',   default='raw',
                        choices=['raw', 'refined'],
                        help='Which CAM file to use as input')
    parser.add_argument('--positive_only', action='store_true',
                        help='Only process label=1 images')
    parser.add_argument('--n_jobs',     type=int,
                        default=min(4, multiprocessing.cpu_count()),
                        help='Parallel workers (default: min(4, cpu_count))')
    parser.add_argument('--vis',        action='store_true',
                        help='Save 3-panel visualisation (raw|crf|mask)')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build image list
    df = pd.read_csv(args.train_csv, index_col=0)
    if args.positive_only:
        df = df[df['label'] == 1]
        print(f"Filtered to {len(df)} positive cases")
    print(f"Total images: {len(df)}")

    tasks = []
    for _, row in df.iterrows():
        rel      = str(row['image'])
        stem     = os.path.splitext(os.path.basename(rel))[0]
        img_path = os.path.join(args.image_root, os.path.basename(rel)) \
                   if args.image_root else rel
        # Also try the full relative path
        if not os.path.exists(img_path):
            img_path = rel

        cam_path = os.path.join(args.cam_dir, f'{stem}_{args.cam_type}_cam.npy')

        if args.skip_existing:
            prob_path = os.path.join(args.out_dir, 'probs', f'{stem}.npy')
            if os.path.exists(prob_path):
                continue

        tasks.append((stem, img_path, cam_path, args.out_dir,
                      args.cam_type, args.vis))

    print(f"Processing {len(tasks)} images with {args.n_jobs} workers...\n")

    if args.n_jobs == 1:
        results = [process_one(t) for t in tqdm(tasks)]
    else:
        results = joblib.Parallel(n_jobs=args.n_jobs, verbose=5)(
            joblib.delayed(process_one)(t) for t in tasks
        )

    # Summary
    ok      = sum(1 for _, s, _ in results if s)
    failed  = [(iid, msg) for iid, s, msg in results if not s]
    print(f"\nDone: {ok}/{len(results)} succeeded")
    if failed:
        print(f"Failed ({len(failed)}):")
        for iid, msg in failed[:10]:
            print(f"  {iid[-25:]}: {msg}")

    print(f"\nOutputs saved to:")
    print(f"  {args.out_dir}/masks/   — binary pseudo masks (.png)")
    print(f"  {args.out_dir}/probs/   — CRF fg probability (.npy)")
    if args.vis:
        print(f"  {args.out_dir}/vis/     — 3-panel visualisations")


if __name__ == '__main__':
    main()