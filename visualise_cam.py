import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def rle2mask(rle, width, height):
    if pd.isna(rle) or str(rle).strip() == "-1" or not rle:
        return np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros(width * height, dtype=np.uint8)
    array = np.asarray([int(x) for x in str(rle).split()])
    starts, lengths = array[0::2], array[1::2]
    pos = 0
    for start, length in zip(starts, lengths):
        pos += start
        mask[pos:pos + length] = 255
        pos += length
    return mask.reshape(width, height).T


def load_cam(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    cam  = data["attn_highres"].astype(np.float32)
    keys = data["keys"]

    print(f"\n{'='*60}")
    print(f"Raw CAM shape: {cam.shape}  dtype: {cam.dtype}")
    print(f"CAM min: {cam.min():.4f}  max: {cam.max():.4f}")
    print(f"CAM mean: {cam.mean():.4f}  std: {cam.std():.4f}")
    print(f"Label keys: {keys}")

    # Handle unexpected shapes
    if cam.ndim == 3:
        print(f"CAM has {cam.ndim} dims with shape {cam.shape} — collapsing to 2D")
        # Try each axis — print stats for each channel
        print("Channel stats:")
        axis = cam.shape[-1] if cam.shape[-1] < cam.shape[0] else 0
        if cam.shape[-1] <= 16:
            # Last dim is channels — mean across channels
            for c in range(cam.shape[-1]):
                ch = cam[:, :, c]
                print(f"  ch{c}: min={ch.min():.3f} max={ch.max():.3f} mean={ch.mean():.3f}")
            cam = cam.mean(axis=-1)
        else:
            cam = cam.mean(axis=0)
        print(f"Collapsed CAM shape: {cam.shape}")

    print(f"{'='*60}\n")
    return cam, keys


def overlay_cam_on_image(image_rgb, cam, alpha=0.5):
    # Normalise to [0, 1]
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    # Convert to uint8 for colormap
    cam_uint8 = (cam_norm * 255).astype(np.uint8)
    heatmap   = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap   = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # Resize heatmap to match image if needed
    if heatmap.shape[:2] != image_rgb.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    overlay = (alpha * heatmap + (1 - alpha) * image_rgb).astype(np.uint8)
    return overlay, cam_norm


def visualise_single(img_path, npy_path, gt_mask=None, save_path=None):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Could not read: {img_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W    = img_rgb.shape[:2]
    print(f"Image: {os.path.basename(img_path)}  ({W}x{H})")

    cam, keys = load_cam(npy_path)

    # Resize CAM to image dimensions
    if cam.shape != (H, W):
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

    overlay, cam_norm = overlay_cam_on_image(img_rgb, cam)

    has_gt = gt_mask is not None
    n_cols = 4 if has_gt else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    fig.suptitle(f"{os.path.basename(img_path)}  |  label={keys}", fontsize=10)

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(cam_norm, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("CAM (normalised)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("CAM overlay")
    axes[2].axis("off")

    if has_gt:
        gt_display = cv2.resize(gt_mask, (W, H)) if gt_mask.shape != (H, W) else gt_mask
        axes[3].imshow(img_rgb)
        axes[3].imshow(gt_display, alpha=0.4, cmap="Reds")
        axes[3].set_title("GT mask")
        axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def run(args):
    cam_dir  = Path(args.cam_dir)
    img_root = Path(args.img_root)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mask lookup
    mask_lookup = {}
    if args.mask_csv and os.path.exists(args.mask_csv):
        mask_df = pd.read_csv(args.mask_csv)
        print(f"Mask CSV: {len(mask_df)} rows")
        id_col  = mask_df.columns[0]
        rle_col = " EncodedPixels" if " EncodedPixels" in mask_df.columns else mask_df.columns[1]
        for _, row in mask_df.iterrows():
            img_id = str(row[id_col])
            mask_lookup.setdefault(img_id, []).append(str(row[rle_col]))

    manifest  = pd.read_csv(args.manifest_txt)
    npy_files = sorted(cam_dir.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files in {cam_dir}")
        sys.exit(1)
    print(f"Found {len(npy_files)} CAM file(s)\n")

    for npy_path in npy_files:
        stem    = npy_path.stem
        matches = manifest[manifest["image"].str.contains(stem, regex=False)]
        if matches.empty:
            print(f"No manifest entry for {stem}")
            continue

        img_path = str(img_root / matches.iloc[0]["image"])
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        gt_mask = None
        if mask_lookup:
            rle_list = mask_lookup.get(stem, ["-1"])
            img_bgr  = cv2.imread(img_path)
            H, W     = img_bgr.shape[:2]
            merged   = np.zeros((H, W), dtype=np.uint8)
            for rle in rle_list:
                merged = np.maximum(merged, rle2mask(rle, W, H))
            gt_mask = merged

        visualise_single(
            img_path, str(npy_path),
            gt_mask=gt_mask,
            save_path=str(out_dir / f"{stem}_vis.png")
        )

    print(f"\nDone. Visualisations in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_dir",      type=str, required=True)
    parser.add_argument("--img_root",     type=str, required=True)
    parser.add_argument("--manifest_txt", type=str, required=True)
    parser.add_argument("--mask_csv",     type=str, default=None)
    parser.add_argument("--out_dir",      type=str, default="./cam_visualisations")
    args = parser.parse_args()
    run(args)