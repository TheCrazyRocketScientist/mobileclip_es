# -*- coding: utf-8 -*-
"""
test_04_gradcam_text.py
-----------------------
CLIP-ES CAM generation for SIIM with:
  - Hybrid EigenCAM × GradCAM: spatial prior × class discriminator
  - Pneumothorax pos prompts as fg class
  - Neg prompts from ALL other conditions as bg classes
  - Stage 3 attention refinement (vanilla CLIP-ES)

Usage:
    # Single image
    python test_04_gradcam_text.py \
        --checkpoint checkpoints/fastvit_sa12.apple_in1k_distil-biobert.tar \
        --prompts_json data/train_prompts_all.json \
        --image final_512/siim/images/<id>.jpg \
        --out_dir ./test_04_output

    # Full CSV
    python test_04_gradcam_text.py \
        --checkpoint checkpoints/fastvit_sa12.apple_in1k_distil-biobert.tar \
        --prompts_json data/train_prompts_all.json \
        --train_csv data/siim_train.csv \
        --positive_only \
        --out_dir ./test_04_output
"""

import os
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import AutoTokenizer, AutoModel
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import scale_cam_image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def get_transform(size=512):
    return Compose([
        Resize(size, interpolation=BICUBIC),
        CenterCrop(size),
        lambda img: img.convert('RGB'),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


# ---------------------------------------------------------------------------
# Image encoder — spatial per-patch similarity, NO global pooling before score
# ---------------------------------------------------------------------------

class ImageEncoderForCAM(nn.Module):
    """
    Computes per-patch similarity against each class text feature.
    Returns softmax probabilities over classes — GradCAM differentiates
    through the softmax to get class-discriminative spatial gradients.

    text_features: (num_classes, proj_dim) set before each forward pass
    logit_scale:   scalar — learned temperature from checkpoint (exp(ls) ≈ 20.4)
                   Applied before softmax to match training distribution.
                   Without this, similarities are ~20x too small and softmax
                   is nearly uniform, killing spatial discrimination.
    """
    def __init__(self, timm_model, projection, logit_scale=1.0):
        super().__init__()
        self.timm_model    = timm_model
        self.projection    = projection
        self.text_features = None   # (num_classes, proj_dim) — set externally
        self.logit_scale   = logit_scale  # scalar float, exp(checkpoint logit_scale)

    def forward(self, x):
        feat = self.timm_model.forward_features(x)   # (B, C, H, W)
        B, C, H, W = feat.shape
        # Per-patch projection
        tokens    = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, C)
        projected = self.projection(tokens)                          # (B, N, proj_dim)
        projected = F.normalize(projected, dim=-1)
        # Per-patch similarity scaled by logit_scale → (B, N, num_classes)
        # This matches the training objective: logit_scale * cosine_sim
        text = self.text_features.to(projected.device)              # (num_classes, proj_dim)
        sim  = torch.einsum('bnc,kc->bnk', projected, text)        # (B, N, num_classes)
        sim  = sim * self.logit_scale                                # scale before softmax
        # Softmax over classes at each patch — CLIP-ES key contribution
        prob = sim.softmax(dim=-1)                                   # (B, N, num_classes)
        # Sum spatial locations → (B, num_classes) scalar score per class
        return prob.sum(dim=1)                                       # (B, num_classes)


# ---------------------------------------------------------------------------
# Softmax GradCAM target — gradient of target class probability
# ---------------------------------------------------------------------------

class SoftmaxClassTarget:
    """
    GradCAM target: probability of the target class after spatial softmax.
    This is the CLIP-ES approach — gradient flows back through the
    inter-class competition, suppressing regions that activate all classes
    equally (e.g. the PORTABLE text label).
    """
    def __init__(self, class_idx):
        self.class_idx = class_idx

    def __call__(self, model_output):
        # model_output: (B, num_classes)
        if model_output.dim() == 1:
            return model_output[self.class_idx]
        return model_output[:, self.class_idx].squeeze()


# ---------------------------------------------------------------------------
# Projection head — named child to match checkpoint keys
# ---------------------------------------------------------------------------

class LinearProjection(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.dropout    = nn.Dropout(dropout)
        self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.projection(self.dropout(x))


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_prompts(prompts, tokenizer, text_model, text_proj, device,
                   max_length=256, batch_size=64):
    """Encode list of raw strings → mean-pooled L2-normalised (proj_dim,)."""
    all_embs = []
    for i in range(0, len(prompts), batch_size):
        batch  = prompts[i: i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_length, return_tensors='pt')
        input_ids      = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        out = text_model(input_ids=input_ids, attention_mask=attention_mask)
        # EOS pooling
        seq_len = attention_mask.sum(dim=1) - 1
        emb = out.last_hidden_state[
            torch.arange(out.last_hidden_state.size(0)), seq_len
        ]
        emb = text_proj(emb)
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    stacked = torch.cat(all_embs, dim=0)
    return F.normalize(stacked.mean(dim=0), dim=-1)


# ---------------------------------------------------------------------------
# CAM refinement — vanilla CLIP-ES
# ---------------------------------------------------------------------------

def scoremap2bbox(scoremap, threshold=0.4, multi_contour_eval=True):
    binary = (scoremap >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = scoremap.shape
        return [(0, 0, w, h)], 1
    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        boxes.append((x, y, x + bw, y + bh))
    return boxes, len(boxes)


def refine_cam_with_attention(grayscale_cam, attn_weights_list, ori_w, ori_h):
    h_feat, w_feat = grayscale_cam.shape
    n_tokens = h_feat * w_feat

    # Average blocks + heads
    processed = []
    for aw in attn_weights_list:
        processed.append(aw[0].mean(0))
    attn_avg = torch.stack(processed, dim=0).mean(0).float()

    # Resize attention if token counts differ
    n_attn = attn_avg.shape[0]
    if n_attn != n_tokens:
        attn_avg = F.interpolate(
            attn_avg.unsqueeze(0).unsqueeze(0),
            size=(n_tokens, n_tokens), mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0)

    # Bbox mask
    boxes, cnt = scoremap2bbox(grayscale_cam, threshold=0.4,
                               multi_contour_eval=True)
    aff_mask = torch.zeros((h_feat, w_feat))
    for (x0, y0, x1, y1) in boxes:
        aff_mask[y0:y1, x0:x1] = 1
    aff_mask = aff_mask.view(1, n_tokens)

    # Vanilla CLIP-ES normalisation
    trans_mat = attn_avg
    trans_mat = trans_mat / trans_mat.sum(dim=0, keepdim=True).clamp(min=1e-6)
    trans_mat = trans_mat / trans_mat.sum(dim=1, keepdim=True).clamp(min=1e-6)
    for _ in range(2):
        trans_mat = trans_mat / trans_mat.sum(dim=0, keepdim=True).clamp(min=1e-6)
        trans_mat = trans_mat / trans_mat.sum(dim=1, keepdim=True).clamp(min=1e-6)
    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2
    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)
    trans_mat = trans_mat * aff_mask

    cam_flat    = torch.FloatTensor(grayscale_cam).view(-1, 1)
    cam_refined = torch.matmul(trans_mat, cam_flat)
    cam_refined = cam_refined.reshape(h_feat, w_feat).numpy()

    return scale_cam_image([cam_refined.astype(np.float32)], (ori_w, ori_h))[0]


# ---------------------------------------------------------------------------
# TTA / multi-scale helpers
# ---------------------------------------------------------------------------

def run_cam_single(img_tensor, cam_method, eigen_runner, grad_runner,
                   target, attn_captured):
    """Run CAM on a single preprocessed tensor. Returns (cam, attn_list)."""
    def norm01(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    attn_captured.clear()

    if cam_method == 'gradcam':
        out = grad_runner(input_tensor=img_tensor,
                          targets=[target], target_size=None)
        cam = out[0] if isinstance(out, tuple) else out
        cam = cam.squeeze()

    elif cam_method == 'eigencam':
        with torch.no_grad():
            out = eigen_runner(input_tensor=img_tensor,
                               targets=None, target_size=None)
        cam = out[0] if isinstance(out, tuple) else out
        cam = cam.squeeze()
        attn_captured.clear()

    else:  # hybrid
        with torch.no_grad():
            eigen_out = eigen_runner(input_tensor=img_tensor,
                                     targets=None, target_size=None)
        eigen_cam = eigen_out[0] if isinstance(eigen_out, tuple) else eigen_out
        eigen_cam = eigen_cam.squeeze()

        attn_captured.clear()
        grad_out = grad_runner(input_tensor=img_tensor,
                               targets=[target], target_size=None)
        grad_cam = grad_out[0] if isinstance(grad_out, tuple) else grad_out
        grad_cam = grad_cam.squeeze()

        alpha = 0.3  # set per args outside; passed via closure
        cam = norm01((norm01(eigen_cam) ** alpha) * (norm01(grad_cam) ** (1 - alpha)))

    attn_list = [attn_captured[f'block{bi}'] for bi in range(2)
                 if f'block{bi}' in attn_captured]
    return cam, attn_list


def attention_rollout(attn_list):
    """
    Attention rollout (Abnar & Zuidema 2020):
    Multiply attention matrices across blocks to get full
    information flow from input to output.
    For 2 blocks: rollout = attn_block0 @ attn_block1
    Returns (N, N) averaged over heads.
    """
    processed = []
    for aw in attn_list:
        # aw: (B, heads, N, N) → avg heads → (N, N)
        processed.append(aw[0].mean(0))

    if len(processed) == 0:
        return None
    elif len(processed) == 1:
        return processed[0].float()
    else:
        # Rollout: sequential matrix multiplication
        rollout = processed[0].float()
        for mat in processed[1:]:
            rollout = torch.matmul(rollout, mat.float())
        # Normalise rows
        rollout = rollout / rollout.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return rollout


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_cam_overlay(img_path, cam, out_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (cam.shape[1], cam.shape[0]))
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap  = cv2.applyColorMap((cam_norm * 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
    overlay  = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    cv2.imwrite(out_path, overlay)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',    required=True)
    parser.add_argument('--prompts_json',  required=True)
    parser.add_argument('--out_dir',       default='./test_04_output')
    parser.add_argument('--image',         default=None)
    parser.add_argument('--train_csv',     default=None)
    parser.add_argument('--img_root',      default='')
    parser.add_argument('--positive_only', action='store_true')
    parser.add_argument('--skip_existing', action='store_true')

    # CAM method arguments
    parser.add_argument('--cam_method', default='hybrid',
                        choices=['gradcam', 'eigencam', 'hybrid'],
                        help='CAM method: gradcam | eigencam | hybrid (default: hybrid)')
    parser.add_argument('--eigen_alpha', type=float, default=0.5,
                        help='Hybrid only: weight for EigenCAM in product '
                             'eigen^alpha * grad^(1-alpha). '
                             'alpha=0.5 = equal weight (default), '
                             'alpha=0.3 = more GradCAM, '
                             'alpha=0.7 = more EigenCAM')

    # TTA / multi-scale / rollout
    parser.add_argument('--tta', action='store_true',
                        help='Test-time augmentation: average CAM over '
                             'original + horizontal flip')
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0],
                        help='Multi-scale CAM: list of scale factors e.g. '
                             '--scales 1.0 1.5 (images resized to 512*scale)')
    parser.add_argument('--attn_rollout', action='store_true',
                        help='Use attention rollout (block0 @ block1) instead '
                             'of block averaging for the transition matrix')
    args = parser.parse_args()

    if args.image is None and args.train_csv is None:
        parser.error('Provide either --image or --train_csv')

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # -----------------------------------------------------------------------
    # 1. Load checkpoint
    # -----------------------------------------------------------------------
    print("=== 1. Loading checkpoint ===")
    checkpoint = torch.load(args.checkpoint, map_location='cpu',
                            weights_only=False)
    state_dict = checkpoint['model']
    cfg        = checkpoint['config']
    timm_name  = cfg.model.image_encoder.name.replace('timm/', '')
    tok_name   = cfg.tokenizer.pretrained_model_name_or_path
    proj_dim   = cfg.model.projection_head.proj_dim

    def extract_sd(prefix):
        return {k[len(prefix):]: v for k, v in state_dict.items()
                if k.startswith(prefix)}

    # -----------------------------------------------------------------------
    # 2. Build FastViT
    # -----------------------------------------------------------------------
    print(f"\n=== 2. Building FastViT: {timm_name} ===")
    timm_model = timm.create_model(timm_name, pretrained=False, num_classes=0)
    for bi in range(2):
        timm_model.stages[3].blocks[bi].token_mixer.fused_attn = False

    timm_sd = extract_sd('image_encoder.image_encoder.timm_model.')
    timm_model.load_state_dict(timm_sd, strict=True)
    print(f"  Loaded {len(timm_sd)} keys cleanly")

    image_proj = LinearProjection(timm_model.num_features, proj_dim)
    image_proj.load_state_dict(extract_sd('image_projection.'))

    # Extract trained logit_scale — model was trained with this temperature
    # exp(logit_scale) ≈ 20.4 for this checkpoint
    logit_scale = state_dict['logit_scale'].exp().item()
    print(f"  logit_scale: {state_dict['logit_scale'].item():.4f}  "
          f"→ temperature multiplier: {logit_scale:.2f}")

    image_encoder = ImageEncoderForCAM(timm_model, image_proj,
                                        logit_scale=logit_scale).to(device).eval()

    # -----------------------------------------------------------------------
    # 3. Register attention hooks (stage 3) + GradCAM target (stage 3)
    # -----------------------------------------------------------------------
    print("\n=== 3. Registering hooks ===")
    attn_captured = {}
    attn_hooks    = []
    for bi in range(2):
        def make_hook(name):
            def fn(module, input, output):
                attn_captured[name] = output.detach().cpu()
            return fn
        h = timm_model.stages[3].blocks[bi].token_mixer.attn_drop\
            .register_forward_hook(make_hook(f'block{bi}'))
        attn_hooks.append(h)
        print(f"  Attn hook  : stages[3].blocks[{bi}].token_mixer.attn_drop")

    target_layer = timm_model.stages[3].blocks[1].norm
    print(f"  EigenCAM   : stages[3].blocks[1].norm  (spatial prior)")
    print(f"  GradCAM    : stages[3].blocks[1].norm  (class discriminator)")

    # -----------------------------------------------------------------------
    # 4. Build text encoder
    # -----------------------------------------------------------------------
    print(f"\n=== 4. Building text encoder: {tok_name} ===")
    tokenizer  = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    text_model = AutoModel.from_pretrained(tok_name, trust_remote_code=True)
    text_model.load_state_dict(extract_sd('text_encoder.text_encoder.'))
    text_proj  = LinearProjection(text_model.config.hidden_size, proj_dim)
    text_proj.load_state_dict(extract_sd('text_projection.'))
    text_model = text_model.to(device).eval()
    text_proj  = text_proj.to(device).eval()

    # -----------------------------------------------------------------------
    # 5. Build class text features
    #    Class 0: Pneumothorax (positive prompts)
    #    Classes 1-N: negative prompts from every OTHER condition
    # -----------------------------------------------------------------------
    print("\n=== 5. Building text features ===")
    with open(args.prompts_json) as f:
        all_prompts = json.load(f)

    # -----------------------------------------------------------------------
    # Foreground: expanded pneumothorax prompt set
    # Original 7 prompts are purely label-level ("pneumothorax is present.")
    # We add anatomical synonyms and visual descriptors that match how
    # radiologists actually describe pneumothorax in reports — the same
    # text distribution the model was trained on.
    # -----------------------------------------------------------------------
    PTX_EXPANDED_POS = [
        # Original 7 from JSON
        "pneumothorax.",
        "there is pneumothorax.",
        "pneumothorax is present.",
        "pneumothorax is seen.",
        "pneumothorax is noted.",
        "the presence of pneumothorax is seen.",
        "the presence of pneumothorax is noted.",
        # Article variants (common in radiology reports)
        "there is a pneumothorax.",
        "a pneumothorax is present.",
        "a pneumothorax is seen.",
        "a pneumothorax is noted.",
        "pneumothorax is identified.",
        "pneumothorax is demonstrated.",
        "pneumothorax is evident.",
        "pneumothorax is apparent.",
        # Size qualifiers
        "small pneumothorax.",
        "moderate pneumothorax.",
        "large pneumothorax.",
        "there is a small pneumothorax.",
        "there is a moderate pneumothorax.",
        "there is a large pneumothorax.",
        "a small pneumothorax is present.",
        "a moderate pneumothorax is present.",
        "a large pneumothorax is present.",
        # Laterality — most pneumothoraces are unilateral
        "left pneumothorax.",
        "right pneumothorax.",
        "left-sided pneumothorax.",
        "right-sided pneumothorax.",
        "there is a left-sided pneumothorax.",
        "there is a right-sided pneumothorax.",
        "pneumothorax is seen on the left.",
        "pneumothorax is seen on the right.",
        "small left pneumothorax.",
        "small right pneumothorax.",
        "left apical pneumothorax.",
        "right apical pneumothorax.",
        # Clinical subtypes
        "tension pneumothorax.",
        "apical pneumothorax.",
        "there is a tension pneumothorax.",
        "there is an apical pneumothorax.",
        # Visual appearance — pleural air descriptors
        "pleural air is present.",
        "there is pleural air.",
        "air in the pleural space.",
        "air is seen in the pleural space.",
        "pleural air collection is present.",
        "there is a pleural air collection.",
        "free air in the pleural cavity.",
        "intrapleural air is present.",
        # Lung margin descriptors — what you actually see on CXR
        "visceral pleural line is visible.",
        "the visceral pleural line is seen.",
        "the visceral pleural line is identified.",
        "lung edge is visible.",
        "the lung edge is seen.",
        "the lung margin is visible.",
        # Lung collapse
        "collapsed lung with pneumothorax.",
        "lung is partially collapsed.",
        "partial lung collapse due to pneumothorax.",
        "partial collapse of the lung.",
        "there is partial collapse of the lung.",
        # Combination findings common in reports
        "pneumothorax with subcutaneous emphysema.",
        "pneumothorax without tension.",
        "stable pneumothorax.",
        "new pneumothorax.",
        "interval development of pneumothorax.",
        "pneumothorax has developed.",
    ]

    # Merge with original JSON prompts (deduplication)
    json_pos  = all_prompts['Pneumothorax']['pos']
    ptx_pos   = list(dict.fromkeys(PTX_EXPANDED_POS + json_pos))  # preserves order, deduplicates
    print(f"  Pneumothorax pos prompts : {len(ptx_pos)} "          f"({len(json_pos)} original + {len(ptx_pos)-len(json_pos)} expanded)")
    fg_feature = encode_prompts(ptx_pos, tokenizer, text_model, text_proj, device)
    print(f"  fg_feature: {fg_feature.shape}  norm={fg_feature.norm():.4f}")

    # Also add No Finding pos prompts as an explicit background class
    # (1448 prompts describing normal CXR — richest background signal)
    nf_pos_feature = encode_prompts(
        all_prompts['No Finding']['pos'], tokenizer, text_model, text_proj, device
    )

    # Background: neg prompts from all conditions EXCEPT Pneumothorax
    # Each condition gets its own class vector (richer competition in softmax)
    bg_features = [nf_pos_feature]  # No Finding first
    bg_names    = ['No Finding (pos)']
    for condition, prompts in all_prompts.items():
        if condition in ('Pneumothorax', 'No Finding'):
            continue
        neg_prompts = prompts.get('neg', [])
        if len(neg_prompts) == 0:
            continue
        feat = encode_prompts(neg_prompts, tokenizer, text_model, text_proj, device)
        bg_features.append(feat)
        bg_names.append(condition)
    print(f"  Background classes: {len(bg_features)} "          f"(No Finding pos + {len(bg_features)-1} condition neg sets)")

    # Stack all class features: [fg, bg_0, bg_1, ..., bg_N]
    all_features = torch.stack([fg_feature] + bg_features, dim=0).to(device)
    print(f"\n  Total classes for softmax: {all_features.shape[0]} "
          f"(1 fg + {len(bg_features)} bg)")
    print(f"  Pneumothorax class index: 0")

    # Set on encoder
    image_encoder.text_features = all_features

    # -----------------------------------------------------------------------
    # 6. CAM runner setup
    # -----------------------------------------------------------------------
    print(f"\n  CAM method : {args.cam_method}  "
          f"(eigen_alpha={args.eigen_alpha:.2f})" if args.cam_method == 'hybrid'
          else f"\n  CAM method : {args.cam_method}")
    eigen_runner = EigenCAM(model=image_encoder, target_layers=[target_layer]) \
                   if args.cam_method in ('eigencam', 'hybrid') else None
    grad_runner  = GradCAM(model=image_encoder,  target_layers=[target_layer]) \
                   if args.cam_method in ('gradcam',  'hybrid') else None
    target       = SoftmaxClassTarget(class_idx=0)
    tf           = get_transform(512)

    # -----------------------------------------------------------------------
    # 7. Build image list
    # -----------------------------------------------------------------------
    if args.image:
        image_list = [{'image_path': args.image,
                       'image_id': os.path.splitext(
                           os.path.basename(args.image))[0]}]
    else:
        df = pd.read_csv(args.train_csv, index_col=0)
        if args.positive_only:
            df = df[df['label'] == 1]
            print(f"\n  Filtered to {len(df)} positive cases")
        image_list = []
        for _, row in df.iterrows():
            rel  = str(row['image'])
            path = os.path.join(args.img_root, rel) if args.img_root else rel
            stem = os.path.splitext(os.path.basename(rel))[0]
            image_list.append({'image_path': path, 'image_id': stem})
    print(f"\n  Total images to process: {len(image_list)}")

    # -----------------------------------------------------------------------
    # 8. Process loop
    # -----------------------------------------------------------------------
    for rec in tqdm(image_list, desc='Generating CAMs'):
        img_path = rec['image_path']
        stem     = rec['image_id']

        if args.skip_existing:
            if os.path.exists(os.path.join(args.out_dir,
                                           f'{stem}_refined_cam.npy')):
                continue

        if not os.path.exists(img_path):
            tqdm.write(f'  [WARN] Not found: {img_path}')
            continue

        try:
            pil_img = Image.open(img_path).convert('RGB')
            ori_w, ori_h = pil_img.size

            # Monkey-patch alpha into run_cam_single closure
            import types
            _alpha = args.eigen_alpha
            _orig_src = run_cam_single.__code__
            # Simpler: just override the alpha=0.3 line via a wrapper
            def _run_cam(tensor):
                def _norm01(x):
                    mn, mx = x.min(), x.max()
                    return (x - mn) / (mx - mn + 1e-8)
                attn_captured.clear()
                if args.cam_method == 'gradcam':
                    out = grad_runner(input_tensor=tensor,
                                      targets=[target], target_size=None)
                    cam = (out[0] if isinstance(out, tuple) else out).squeeze()
                elif args.cam_method == 'eigencam':
                    with torch.no_grad():
                        out = eigen_runner(input_tensor=tensor,
                                           targets=None, target_size=None)
                    cam = (out[0] if isinstance(out, tuple) else out).squeeze()
                    attn_captured.clear()
                else:  # hybrid
                    with torch.no_grad():
                        eout = eigen_runner(input_tensor=tensor,
                                            targets=None, target_size=None)
                    ecam = (eout[0] if isinstance(eout, tuple) else eout).squeeze()
                    attn_captured.clear()
                    gout = grad_runner(input_tensor=tensor,
                                       targets=[target], target_size=None)
                    gcam = (gout[0] if isinstance(gout, tuple) else gout).squeeze()
                    a = args.eigen_alpha
                    cam = _norm01((_norm01(ecam)**a) * (_norm01(gcam)**(1-a)))
                attn = [attn_captured[f'block{bi}'] for bi in range(2)
                        if f'block{bi}' in attn_captured]
                return cam, attn

            # ---------------------------------------------------------------
            # Multi-scale + TTA loop
            # For each scale: run on original [+ flip if TTA]
            # All CAMs resized to base (16,16) feature grid then averaged
            # ---------------------------------------------------------------
            all_cams  = []
            all_attns = []
            base_size = 512

            for scale in args.scales:
                size = int(base_size * scale)
                # Round to multiple of 32 (FastViT downsampling factor)
                size = max(32, (size // 32) * 32)
                tf_s = get_transform(size)

                # Original orientation
                t_orig = tf_s(pil_img).unsqueeze(0).to(device)
                cam_orig, attn_orig = _run_cam(t_orig)
                all_cams.append(cam_orig)
                all_attns.extend(attn_orig)

                # Horizontal flip (TTA)
                if args.tta:
                    pil_flip = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                    t_flip = tf_s(pil_flip).unsqueeze(0).to(device)
                    cam_flip, attn_flip = _run_cam(t_flip)
                    # Flip CAM back before averaging
                    # cam_flip may be numpy or torch depending on cam_method
                    cam_flip_np = cam_flip.numpy()                                   if isinstance(cam_flip, torch.Tensor)                                   else cam_flip
                    cam_flip_back = np.fliplr(cam_flip_np).copy()
                    all_cams.append(cam_flip_back)
                    all_attns.extend(attn_flip)

            # Average all CAMs (resize each to match base grid 16x16 first)
            base_h = base_w = base_size // 32  # 16
            cams_resized = []
            for c in all_cams:
                c_np = c.numpy() if isinstance(c, torch.Tensor) else c
                if c_np.shape != (base_h, base_w):
                    c_np = cv2.resize(c_np.astype(np.float32),
                                      (base_w, base_h))
                cams_resized.append(c_np)
            grayscale_cam = np.mean(cams_resized, axis=0)

            # Normalise averaged CAM
            mn, mx = grayscale_cam.min(), grayscale_cam.max()
            grayscale_cam = (grayscale_cam - mn) / (mx - mn + 1e-8)

            raw_highres = cv2.resize(grayscale_cam.astype(np.float32),
                                     (ori_w, ori_h))

            # ---------------------------------------------------------------
            # Attention: rollout or average
            # ---------------------------------------------------------------
            if args.attn_rollout and len(all_attns) >= 2:
                last_two = all_attns[-2:]
                rollout  = attention_rollout(last_two)
                if rollout is not None:
                    # Wrap (N,N) → (1, 1, N, N) to match (B, heads, N, N) format
                    attn_for_refine = [rollout.unsqueeze(0).unsqueeze(0)
                                       .expand(1, 1, -1, -1)]
                else:
                    attn_for_refine = all_attns[-2:]
            else:
                attn_for_refine = all_attns[-2:] if len(all_attns) >= 2                                   else all_attns

            refined_highres = refine_cam_with_attention(
                grayscale_cam, attn_for_refine, ori_w, ori_h
            )

            # Save
            save_cam_overlay(img_path, raw_highres,
                             os.path.join(args.out_dir, f'{stem}_raw_cam.jpg'))
            save_cam_overlay(img_path, refined_highres,
                             os.path.join(args.out_dir, f'{stem}_refined_cam.jpg'))
            np.save(os.path.join(args.out_dir, f'{stem}_raw_cam.npy'),
                    raw_highres.astype(np.float16))
            np.save(os.path.join(args.out_dir, f'{stem}_refined_cam.npy'),
                    refined_highres.astype(np.float16))

        except Exception as e:
            tqdm.write(f'  [ERROR] {stem}: {e}')

    for h in attn_hooks:
        h.remove()

    print(f"\n[DONE] CAMs saved to {args.out_dir}/")


if __name__ == '__main__':
    main()