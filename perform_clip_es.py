import os
import sys
import json
import argparse
import warnings
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from tqdm import tqdm
from torch import multiprocessing
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytorch_grad_cam import GradCAM

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# WRAPPER
# Vision model + image projection head.
# Outputs softmax similarity scores over [fg, bg].
# GradCAM hooks into the last conv block and backprops through:
#   conv features → GAP → img_proj → normalize → cosine sim → softmax
# ─────────────────────────────────────────────────────────────
class FastVITCLIPWrapper(torch.nn.Module):
    def __init__(self, vision_model, proj_weight, proj_bias, tau=0.07):
        super().__init__()
        self.vision_model = vision_model
        self.tau = tau
        self.text_features = None  # 2 x 512, set per image before cam()

        # Image projection: 1024 → 512
        out_dim, in_dim = proj_weight.shape
        self.img_proj = torch.nn.Linear(in_dim, out_dim, bias=True)
        self.img_proj.weight = torch.nn.Parameter(proj_weight.clone())
        self.img_proj.bias   = torch.nn.Parameter(proj_bias.clone())

    def forward(self, x):
        img_features = self.vision_model(x)             # B x 1024
        img_features = self.img_proj(img_features)      # B x 512
        img_features = F.normalize(img_features, dim=-1)
        text = F.normalize(self.text_features, dim=-1)  # 2 x 512
        logits = img_features @ text.T                  # B x 2
        # CLIP-ES softmax: fg and bg compete
        # suppresses activations in non-target regions
        return F.softmax(logits / self.tau, dim=-1)     # B x 2


# ─────────────────────────────────────────────────────────────
# TARGET
# Backprop through fg class probability (index 0)
# ─────────────────────────────────────────────────────────────
class ClipOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────
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


def split_dataset(dataset, n_splits):
    if n_splits == 1:
        return [dataset]
    part = len(dataset) // n_splits
    splits = [dataset[i * part:(i + 1) * part] for i in range(n_splits - 1)]
    splits.append(dataset[(n_splits - 1) * part:])
    return splits


def load_medical_prompts(json_path, target_class):
    """
    Maps manifest class strings to JSON prompt keys.
    'Pneumothorax'    → pos=fg, neg=bg
    'No Pneumothorax' → neg=fg, pos=bg  (looking for absence)
    unc prompts excluded — add noise to both directions.
    """
    with open(json_path) as f:
        data = json.load(f)

    if target_class.startswith("No "):
        base_class = target_class[3:]
        if base_class not in data:
            raise KeyError(f"'{base_class}' (from '{target_class}') not in {json_path}")
        pos_prompts = data[base_class]["neg"]
        neg_prompts = data[base_class]["pos"]
    else:
        if target_class not in data:
            raise KeyError(f"'{target_class}' not in {json_path}")
        pos_prompts = data[target_class]["pos"]
        neg_raw     = data[target_class].get("neg", [""])
        neg_prompts = neg_raw if neg_raw != [""] else data["No Finding"]["pos"]

    if not pos_prompts:
        raise ValueError(f"Empty pos prompts for '{target_class}'")
    if not neg_prompts:
        raise ValueError(f"Empty neg prompts for '{target_class}'")

    return pos_prompts, neg_prompts


def encode_prompt_batch(prompt_list, text_model, tokenizer, text_proj, device):
    """
    Tokenize → ClinicalBERT → mean pool → text_proj (768→512) → normalize.
    Returns single 512-dim embedding representing the prompt list.
    """
    with torch.no_grad():
        inputs = tokenizer(
            prompt_list,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        outputs  = text_model(**inputs)
        embs     = outputs.last_hidden_state.mean(dim=1)  # N x 768
        embs     = F.normalize(embs, dim=-1)
        embs     = text_proj(embs)                        # N x 512
        embs     = F.normalize(embs, dim=-1)
        mean_emb = embs.mean(dim=0)                       # 512
        mean_emb = F.normalize(mean_emb, dim=0)
    return mean_emb


def _transform_resize(h, w):
    return Compose([
        Resize((h, w), interpolation=Image.BICUBIC),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        ),
    ])


def get_target_layer(vision_model):
    target = vision_model.stages[-1].blocks[-1]
    print(f"Target layer: stages[-1].blocks[-1] → {type(target).__name__}")
    return [target]


def load_full_checkpoint(model_path, device):
    """
    Load the full exported checkpoint containing:
      backbone, proj_weight, proj_bias,
      text_proj_weight, text_proj_bias,
      logit_scale, model_variant
    """
    full = torch.load(model_path, map_location="cpu")

    required = ["backbone", "proj_weight", "proj_bias",
                "text_proj_weight", "text_proj_bias", "model_variant"]
    missing = [k for k in required if k not in full]
    if missing:
        raise KeyError(
            f"Checkpoint missing keys: {missing}. "
            f"Re-run the export script to generate a full checkpoint."
        )

    model_variant = full["model_variant"]
    print(f"Model variant from checkpoint: {model_variant}")

    # Vision backbone
    vision_model = timm.create_model(model_variant, pretrained=False, num_classes=0)
    vision_model.load_state_dict(full["backbone"], strict=True)
    vision_model.eval()

    # Image projection wrapper
    wrapped = FastVITCLIPWrapper(
        vision_model,
        full["proj_weight"],
        full["proj_bias"],
        tau=0.07  # overridden by args.tau in main
    )
    wrapped.eval()

    # Text projection: 768 → 512
    txt_w = full["text_proj_weight"]
    txt_b = full["text_proj_bias"]
    text_proj = torch.nn.Linear(txt_w.shape[1], txt_w.shape[0], bias=True)
    text_proj.weight = torch.nn.Parameter(txt_w)
    text_proj.bias   = torch.nn.Parameter(txt_b)
    text_proj.eval()

    return vision_model, wrapped, text_proj, model_variant


# ─────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────
def perform(process_id, dataset_list, args, wrapped_model,
            text_model, tokenizer, text_proj, target_layers):

    n_gpus = torch.cuda.device_count()
    device = f"cuda:{process_id % n_gpus}" if n_gpus > 0 else "cpu"
    databin = dataset_list[process_id]

    wrapped_model = wrapped_model.to(device)
    wrapped_model.tau = args.tau
    text_model    = text_model.to(device)
    text_proj     = text_proj.to(device)

    patch_size = args.patch_size

    # No reshape_transform — FastViT outputs B x C x H x W spatial maps
    cam = GradCAM(
        model=wrapped_model,
        target_layers=target_layers,
        reshape_transform=None
    )

    failed = 0
    for row in tqdm(databin, desc=f"Worker {process_id}"):
        img_path  = os.path.join(args.img_root, row["image"])
        class_str = row["class"]
        label     = row["label"]

        if not os.path.exists(img_path):
            print(f"Missing: {img_path}")
            failed += 1
            continue

        cv_img = cv2.imread(img_path)
        if cv_img is None:
            print(f"OpenCV failed: {img_path}")
            failed += 1
            continue
        ori_height, ori_width = cv_img.shape[:2]

        # Prompts
        try:
            fg_prompts, bg_prompts = load_medical_prompts(args.prompt_json, class_str)
        except (KeyError, ValueError) as e:
            print(f"Prompt error for '{class_str}': {e}")
            failed += 1
            continue

        # Text embeddings — projected to 512
        fg_emb = encode_prompt_batch(fg_prompts, text_model, tokenizer, text_proj, device)
        bg_emb = encode_prompt_batch(bg_prompts, text_model, tokenizer, text_proj, device)
        text_features = torch.stack([fg_emb, bg_emb], dim=0)  # 2 x 512

        # Set on wrapper before GradCAM forward pass
        wrapped_model.text_features = text_features.to(device)

        # Image preprocessing
        h = int(np.ceil(ori_height / patch_size) * patch_size)
        w = int(np.ceil(ori_width  / patch_size) * patch_size)
        preprocess   = _transform_resize(h, w)
        image_tensor = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

        # GradCAM — raw image tensor, target fg (index 0)
        targets = [ClipOutputTarget(0)]
        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=targets
        )
        grayscale_cam = grayscale_cam[0]  # H x W

        # Resize to original resolution
        cam_highres = cv2.resize(
            grayscale_cam,
            (ori_width, ori_height),
            interpolation=cv2.INTER_LINEAR
        )

        # Save
        out_name  = os.path.basename(row["image"]).replace(".jpg", ".npy")
        save_path = os.path.join(args.cam_out_dir, out_name)
        np.save(save_path, {
            "keys":         np.array([label]),
            "attn_highres": cam_highres.astype(np.float16)
        })

    print(f"Worker {process_id} done. Failed: {failed}/{len(databin)}")
    return 0


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastViT CXR CAM Extractor (CLIP-ES)")

    parser.add_argument("--img_root",     type=str, required=True)
    parser.add_argument("--manifest_txt", type=str, required=True,
                        help="CSV with columns: image, class, label")
    parser.add_argument("--prompt_json",  type=str, required=True)
    parser.add_argument("--cam_out_dir",  type=str, default="./cam_output")
    parser.add_argument("--model",        type=str, required=True,
                        help="Path to full exported .pt checkpoint")
    parser.add_argument("--mask_csv",     type=str, default=None)
    parser.add_argument("--text_encoder", type=str, default="medicalai/ClinicalBERT")
    parser.add_argument("--tau",          type=float, default=0.07)
    parser.add_argument("--patch_size",   type=int, default=16)
    parser.add_argument("--num_workers",  type=int, default=1)
    parser.add_argument("--dry_run",      type=int, default=0,
                        help="Process only N images for testing")

    args = parser.parse_args()
    os.makedirs(args.cam_out_dir, exist_ok=True)

    # Manifest
    df = pd.read_csv(args.manifest_txt)
    print(f"Manifest: {len(df)} rows")
    print(f"Class distribution:\n{df['class'].value_counts()}\n")

    df = df[df.apply(
        lambda r: os.path.exists(os.path.join(args.img_root, r["image"])), axis=1
    )]
    print(f"After path filter: {len(df)} images on disk")

    if len(df) == 0:
        print("No files found. Check --img_root and --manifest_txt.")
        sys.exit(1)

    if args.dry_run > 0:
        df = df.head(args.dry_run)
        print(f"Dry run: {len(df)} images")

    records = df.to_dict(orient="records")
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Text encoder
    print(f"Loading text encoder: {args.text_encoder}")
    tokenizer  = AutoTokenizer.from_pretrained(args.text_encoder)
    text_model = AutoModel.from_pretrained(args.text_encoder).eval().to(device)

    # Vision model + projections from full checkpoint
    print(f"Loading checkpoint: {args.model}")
    vision_model, wrapped, text_proj, model_variant = load_full_checkpoint(
        args.model, device
    )
    wrapped   = wrapped.to(device)
    wrapped.tau = args.tau
    text_proj = text_proj.to(device)

    # Target layer
    target_layers = get_target_layer(vision_model)

    # Sanity check
    print("\nRunning sanity check...")
    with torch.no_grad():
        dummy      = torch.randn(1, 3, 224, 224).to(device)
        dummy_text = F.normalize(torch.randn(2, 512), dim=-1).to(device)
        wrapped.text_features = dummy_text
        out = wrapped(dummy)
        assert out.shape == torch.Size([1, 2]), f"Bad output shape: {out.shape}"
        assert abs(out.sum().item() - 1.0) < 1e-4, f"Softmax sum != 1: {out.sum()}"
        print(f"Output shape: {out.shape}  sum: {out.sum().item():.4f}")
    print("Sanity check passed.\n")

    # Run
    splits = split_dataset(records, args.num_workers)
    if args.num_workers == 1:
        perform(0, splits, args, wrapped,
                text_model, tokenizer, text_proj, target_layers)
    else:
        multiprocessing.spawn(
            perform,
            nprocs=args.num_workers,
            args=(splits, args, wrapped, text_model, tokenizer,
                  text_proj, target_layers)
        )