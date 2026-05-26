# -*- coding: utf-8 -*-
"""
visualise_masks.py
------------------
Overlays SIIM RLE masks onto CXR images for visual inspection.
Useful for comparing CAM outputs against ground truth segmentation.

Usage:
    # Single image
    python visualise_masks.py \
        --mask_csv data/siim_mask.csv \
        --image final_512/siim/images/1.2.276.0.7230010.3.1.4.8323329.10026.1517875221.22915.jpg \
        --out_dir ./mask_viz

    # Whole split (uses train_csv to get image paths)
    python visualise_masks.py \
        --mask_csv data/siim_mask.csv \
        --train_csv data/siim_train.csv \
        --img_root final_512/siim/images \
        --out_dir ./mask_viz \
        --n 20
"""

import os
import argparse
import numpy as np
import cv2
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# RLE decode — SIIM column-major (Fortran) order
# ---------------------------------------------------------------------------

# SIIM masks are always encoded at 1024x1024 regardless of stored image size
SIIM_MASK_HEIGHT = 1024
SIIM_MASK_WIDTH  = 1024

def rle_decode(rle_string, height=SIIM_MASK_HEIGHT, width=SIIM_MASK_WIDTH):
    """
    Decode SIIM RLE to binary mask.
    Always decodes at 1024x1024 (native SIIM encoding resolution).
    Caller must resize to actual image dimensions after decoding.
    """
    if str(rle_string).strip() in ('-1', '', 'nan'):
        return np.zeros((height, width), dtype=np.uint8)
    array = np.asarray([int(x) for x in str(rle_string).split()])
    starts  = array[0::2]
    lengths = array[1::2]
    mask = np.zeros(width * height, dtype=np.uint8)
    current_position = 0
    for start, length in zip(starts, lengths):
        current_position += start
        mask[current_position: current_position + length] = 1
        current_position += length
    return mask.reshape(width, height).T  # (1024, 1024)


def build_mask_lookup(mask_csv_path):
    """
    ImageId (DICOM stem) -> list of RLE strings.
    siim_mask.csv row index is NOT a foreign key to siim_train.csv index.
    Join is on ImageId == filename stem (no extension).
    """
    df = pd.read_csv(mask_csv_path, index_col=0)
    lookup = {}
    for _, row in df.iterrows():
        img_id = str(row['ImageId']).strip()
        rle    = str(row['EncodedPixels']).strip()
        lookup.setdefault(img_id, []).append(rle)
    return lookup


def build_train_index(train_csv_path, img_root=''):
    """
    Returns list of dicts with image_path and image_id (DICOM stem).
    The 'index' column in siim_train.csv is just a row number — we
    derive image_id from the filename stem for joining with mask lookup.
    """
    df = pd.read_csv(train_csv_path, index_col=0)
    records = []
    for _, row in df.iterrows():
        rel  = str(row['image'])
        path = os.path.join(img_root, rel) if img_root else rel
        stem = os.path.splitext(os.path.basename(rel))[0]
        records.append({
            'image_path': path,
            'image_id':   stem,
            'label':      int(row['label']),
            'class_name': str(row['class']),
        })
    return records


def merge_masks(rle_list, img_height, img_width):
    """
    Union of all RLE masks, decoded at 1024x1024 then resized to
    actual image dimensions (img_height, img_width).
    """
    combined = np.zeros((SIIM_MASK_HEIGHT, SIIM_MASK_WIDTH), dtype=np.uint8)
    for rle in rle_list:
        combined = np.maximum(combined, rle_decode(rle))
    # Resize to actual image size
    if (img_height, img_width) != (SIIM_MASK_HEIGHT, SIIM_MASK_WIDTH):
        combined = cv2.resize(combined, (img_width, img_height),
                              interpolation=cv2.INTER_NEAREST)
    return combined


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def overlay_mask_on_image(img_bgr, mask, color=(0, 255, 0), alpha=0.4):
    """
    Draws a semi-transparent coloured overlay where mask==1,
    plus a solid contour outline.
    Returns a copy of img_bgr with the overlay applied.
    """
    overlay = img_bgr.copy()
    # Fill
    overlay[mask == 1] = (
        alpha * np.array(color) +
        (1 - alpha) * overlay[mask == 1]
    ).astype(np.uint8)
    # Contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def make_heatmap_panel(img, cam_array, mask_r, with_mask=False):
    """Return one panel: cam heatmap blended with image, optionally with mask contour."""
    h, w = img.shape[:2]
    cam_norm = (cam_array - cam_array.min()) / (cam_array.max() - cam_array.min() + 1e-8)
    cam_resized = cv2.resize(cam_norm.astype(np.float32), (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    panel = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    if with_mask:
        contours, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, (0, 255, 0), 2)
    return panel


def add_label(img, text, pos=(10, 30), color=(255, 255, 255)):
    """Add a text label to a panel copy."""
    out = img.copy()
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1)
    return out


def load_npy(path):
    """Load a .npy file as float32, return None if missing or not .npy."""
    if path and os.path.exists(path) and path.endswith('.npy'):
        return np.load(path).astype(np.float32)
    return None


def overlay_cam_and_mask(img_path, raw_cam_path, refined_cam_path, mask,
                         out_path, crf_prob_path=None, crf_mask_path=None):
    """
    Up to 7-panel side-by-side:
      1. GT Mask
      2. Raw CAM
      3. Raw CAM + Mask
      4. CRF Prob        (if crf_prob_path provided)
      5. CRF Prob + Mask (if crf_prob_path provided)
      6. CRF Pseudo Mask (if crf_mask_path provided)
      7. CRF Mask + GT   (if crf_mask_path provided)
    Falls back to 5-panel (no CRF) or 3-panel (raw only) gracefully.
    """
    img  = cv2.imread(img_path)
    h, w = img.shape[:2]

    mask_r = cv2.resize(mask.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST)              if mask.shape[:2] != (h, w) else mask.astype(np.uint8)

    panels = [add_label(overlay_mask_on_image(img.copy(), mask_r), 'GT Mask')]

    # --- Raw CAM panels ---
    raw_cam = load_npy(raw_cam_path)
    if raw_cam is not None:
        panels.append(add_label(
            make_heatmap_panel(img, raw_cam, mask_r), 'Raw CAM'))
        panels.append(add_label(
            make_heatmap_panel(img, raw_cam, mask_r, with_mask=True),
            'Raw + Mask'))

    # --- CRF probability panels ---
    crf_prob = load_npy(crf_prob_path)
    if crf_prob is not None:
        panels.append(add_label(
            make_heatmap_panel(img, crf_prob, mask_r), 'CRF Prob'))
        panels.append(add_label(
            make_heatmap_panel(img, crf_prob, mask_r, with_mask=True),
            'CRF + Mask'))

    # --- CRF pseudo mask panels ---
    if crf_mask_path and os.path.exists(crf_mask_path):
        pseudo = cv2.imread(crf_mask_path, cv2.IMREAD_GRAYSCALE)
        if pseudo is not None:
            pseudo = cv2.resize(pseudo, (w, h), interpolation=cv2.INTER_NEAREST)
            # Colour the pseudo mask: ptx=green, ignore=grey, bg=black
            pseudo_vis = np.zeros((h, w, 3), dtype=np.uint8)
            pseudo_vis[pseudo == 1]   = (0, 200, 0)
            pseudo_vis[pseudo == 255] = (128, 128, 128)
            pseudo_blend = cv2.addWeighted(img, 0.4, pseudo_vis, 0.6, 0)
            panels.append(add_label(pseudo_blend, 'CRF Pseudo'))
            # CRF pseudo + GT contour overlay
            pseudo_gt = pseudo_blend.copy()
            contours, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pseudo_gt, contours, -1, (0, 255, 255), 2)
            panels.append(add_label(pseudo_gt, 'CRF + GT'))

    panel = np.concatenate(panels, axis=1)
    cv2.imwrite(out_path, panel)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_single(img_path, mask_lookup, out_dir, cam_path=None,
                   image_id=None, crf_dir=None):
    if image_id is None:
        image_id = os.path.splitext(os.path.basename(img_path))[0]
    rle_list = mask_lookup.get(image_id, ['-1'])
    if rle_list == ['-1']:
        for key in mask_lookup:
            if key.endswith(image_id) or image_id.endswith(key):
                rle_list = mask_lookup[key]
                break

    img = cv2.imread(img_path)
    if img is None:
        print(f"  [WARN] Could not read: {img_path}")
        return
    h, w = img.shape[:2]

    is_positive = not all(r.strip() in ('-1', 'nan', '') for r in rle_list)
    mask = merge_masks(rle_list, h, w)

    out_name = f"{image_id}_{'pos' if is_positive else 'neg'}_mask.jpg"
    out_path = os.path.join(out_dir, out_name)

    # Resolve raw CAM path
    raw_path = None
    if cam_path:
        raw_path = cam_path.replace('_refined_cam', '_raw_cam')
        raw_path = raw_path if os.path.exists(raw_path) else                    (cam_path if os.path.exists(cam_path) else None)

    # Resolve CRF paths
    crf_prob_path = None
    crf_mask_path = None
    if crf_dir:
        p = os.path.join(crf_dir, 'probs', f'{image_id}.npy')
        m = os.path.join(crf_dir, 'masks', f'{image_id}.png')
        crf_prob_path = p if os.path.exists(p) else None
        crf_mask_path = m if os.path.exists(m) else None

    if raw_path or crf_prob_path:
        overlay_cam_and_mask(img_path, raw_path, None, mask, out_path,
                             crf_prob_path=crf_prob_path,
                             crf_mask_path=crf_mask_path)
        n = 1 + (2 if raw_path else 0) +             (2 if crf_prob_path else 0) +             (2 if crf_mask_path else 0)
        print(f"  [{'POS' if is_positive else 'NEG'}] {out_name}  ({n}-panel)")
    else:
        vis = overlay_mask_on_image(img.copy(), mask)
        cv2.imwrite(out_path, vis)
        print(f"  [{'POS' if is_positive else 'NEG'}] {out_name}  "
              f"mask_area={mask.sum()} px")
    return out_path


def main():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--mask_csv',  required=True,
                        help='Path to siim_mask.csv')

    # Input mode A: single image
    parser.add_argument('--image',     default=None,
                        help='Path to a single CXR jpg')

    # Input mode B: batch from train csv
    parser.add_argument('--train_csv', default=None,
                        help='Path to siim_train.csv for batch mode')
    parser.add_argument('--img_root',  default='',
                        help='Root directory prepended to image paths in CSV')
    parser.add_argument('--n',         type=int, default=20,
                        help='How many images to process in batch mode')
    parser.add_argument('--positive_only', action='store_true',
                        help='Only visualise positive (pneumothorax) cases')

    # Optional: pair with CAM output directory
    parser.add_argument('--cam_dir',   default=None,
                        help='Directory containing _raw_cam.npy files from test_04')
    parser.add_argument('--crf_dir',   default=None,
                        help='Directory containing crf_output/probs/ and crf_output/masks/ '
                             'from apply_crf_siim.py')

    parser.add_argument('--out_dir',   default='./mask_viz')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading mask lookup...")
    mask_lookup = build_mask_lookup(args.mask_csv)
    print(f"  {len(mask_lookup)} unique ImageIds in mask CSV")

    # -----------------------------------------------------------------------
    # Mode A: single image
    # -----------------------------------------------------------------------
    if args.image:
        image_id = os.path.splitext(os.path.basename(args.image))[0]
        cam_path = None
        if args.cam_dir:
            for ext in ('_refined_cam.npy', '_raw_cam.npy',
                        '_refined_cam.jpg', '_raw_cam.jpg'):
                candidate = os.path.join(args.cam_dir, image_id + ext)
                if os.path.exists(candidate):
                    cam_path = candidate
                    break
        process_single(args.image, mask_lookup, args.out_dir,
                       cam_path, image_id=image_id, crf_dir=args.crf_dir)
        return

    # -----------------------------------------------------------------------
    # Mode B: batch from train CSV
    # -----------------------------------------------------------------------
    if args.train_csv is None:
        print("Provide either --image or --train_csv")
        return

    records = build_train_index(args.train_csv, args.img_root)
    if args.positive_only:
        records = [r for r in records if r['label'] == 1]
        print(f"  Filtered to {len(records)} positive cases")

    records = records[:args.n]
    print(f"  Processing {len(records)} images...\n")

    for rec in records:
        image_id = rec['image_id']
        # Find refined cam first (preferred), fall back to raw
        cam_path = None
        if args.cam_dir:
            for ext in ('_refined_cam.npy', '_raw_cam.npy',
                        '_refined_cam.jpg', '_raw_cam.jpg'):
                candidate = os.path.join(args.cam_dir, image_id + ext)
                if os.path.exists(candidate):
                    cam_path = candidate
                    break

        process_single(rec['image_path'], mask_lookup, args.out_dir,
                       cam_path, image_id=image_id, crf_dir=args.crf_dir)

    print(f"\nDone. Saved to {args.out_dir}/")


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# Debug helper — run standalone to check ID matching
# ---------------------------------------------------------------------------
def debug_id_matching(train_csv, mask_csv, img_root='', n=10):
    mask_lookup = build_mask_lookup(mask_csv)
    records = build_train_index(train_csv, img_root)
    pos = [r for r in records if r['label'] == 1][:n]
    print(f"Checking {len(pos)} positive cases...\n")
    for r in pos:
        iid = r['image_id']
        found = iid in mask_lookup
        rles  = mask_lookup.get(iid, ['-1'])
        is_neg = all(x.strip() in ('-1','nan','') for x in rles)
        print(f"  {'OK ' if found and not is_neg else 'MISS'} | {iid[-20:]} | "
              f"rle_count={len(rles)} | negative={is_neg}")