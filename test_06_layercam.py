# -*- coding: utf-8 -*-
"""
test_06_layercam.py
-------------------
LayerCAM on 5 positive SIIM images.
LayerCAM: element-wise product of relu(gradients) and activations
before spatial averaging — produces finer-grained maps than GradCAM.

Usage:
    python test_06_layercam.py \
        --checkpoint checkpoints/fastvit_sa12.apple_in1k_distil-biobert.tar \
        --prompts_json data/train_prompts_all.json \
        --train_csv data/siim_train.csv \
        --mask_csv data/siim_with_masks.csv \
        --out_dir ./test_06_layercam
"""

import os, json, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from transformers import AutoTokenizer, AutoModel

SIIM_H, SIIM_W = 1024, 1024

def get_transform(size=512):
    return Compose([
        Resize(size, interpolation=BICUBIC),
        CenterCrop(size),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

class LinearProjection(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.dropout    = nn.Dropout(dropout)
        self.projection = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.projection(self.dropout(x))

class SoftmaxImageEncoder(nn.Module):
    def __init__(self, timm_model, projection, logit_scale=1.0):
        super().__init__()
        self.timm_model    = timm_model
        self.projection    = projection
        self.logit_scale   = logit_scale
        self.text_features = None
    def forward(self, x):
        feat      = self.timm_model.forward_features(x)
        B,C,H,W   = feat.shape
        tokens    = feat.permute(0,2,3,1).reshape(B,H*W,C)
        projected = F.normalize(self.projection(tokens), dim=-1)
        text      = self.text_features.to(projected.device)
        sim       = torch.einsum("bnc,kc->bnk", projected, text) * self.logit_scale
        return sim.softmax(dim=-1).sum(dim=1)

@torch.no_grad()
def encode_prompts(prompts, tokenizer, text_model, text_proj, device, batch_size=64):
    all_embs = []
    for i in range(0, len(prompts), batch_size):
        batch  = prompts[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True,
                           max_length=256, return_tensors="pt")
        iids   = tokens["input_ids"].to(device)
        amask  = tokens["attention_mask"].to(device)
        out    = text_model(input_ids=iids, attention_mask=amask)
        seq_len = amask.sum(dim=1) - 1
        emb = out.last_hidden_state[torch.arange(out.last_hidden_state.size(0)), seq_len]
        emb = F.normalize(text_proj(emb), dim=-1)
        all_embs.append(emb.cpu())
    return F.normalize(torch.cat(all_embs).mean(0), dim=-1)

def layer_cam(image_encoder, img_tensor, target_class, target_layer):
    activations, gradients = {}, {}
    def save_act(m,i,o): activations["feat"] = o
    def save_grad(m,i,o):
        gradients["grad"] = o[0] if isinstance(o,tuple) else o
    h_act  = target_layer.register_forward_hook(save_act)
    h_grad = target_layer.register_full_backward_hook(save_grad)

    img_tensor = img_tensor.requires_grad_(True)
    out   = image_encoder(img_tensor)
    score = out[0, target_class]
    image_encoder.zero_grad()
    score.backward()

    h_act.remove(); h_grad.remove()

    act  = activations["feat"].detach()   # (1,C,H,W)
    grad = gradients["grad"].detach()     # (1,C,H,W)

    # LayerCAM: relu(grad) * act element-wise, then mean over channels
    cam = (torch.relu(grad) * act).mean(dim=1)[0]  # (H,W)
    cam = torch.relu(cam).cpu().numpy()
    mn,mx = cam.min(), cam.max()
    if mx > mn: cam = (cam-mn)/(mx-mn)
    return cam

def rle_decode(rle_string):
    if str(rle_string).strip() in ("-1","","nan"): return np.zeros((SIIM_H,SIIM_W),dtype=np.uint8)
    array = np.asarray([int(x) for x in str(rle_string).split()])
    starts,lengths = array[0::2], array[1::2]
    mask = np.zeros(SIIM_W*SIIM_H, dtype=np.uint8)
    pos = 0
    for s,l in zip(starts,lengths):
        pos+=s; mask[pos:pos+l]=1; pos+=l
    return mask.reshape(SIIM_W,SIIM_H).T

def build_mask_lookup(mask_csv):
    df = pd.read_csv(mask_csv, index_col=0)
    lookup = {}
    for _,row in df.iterrows():
        lookup.setdefault(str(row["ImageId"]).strip(),[]).append(str(row["EncodedPixels"]).strip())
    return lookup

def merge_masks(rle_list, h, w):
    combined = np.zeros((SIIM_H,SIIM_W),dtype=np.uint8)
    for rle in rle_list: combined = np.maximum(combined, rle_decode(rle))
    if (h,w)!=(SIIM_H,SIIM_W): combined=cv2.resize(combined,(w,h),interpolation=cv2.INTER_NEAREST)
    return combined

def compute_iou(pred, gt):
    inter=(pred&gt).sum(); union=(pred|gt).sum()
    return inter/union if union>0 else 0.0

def save_panel(img_path, cam, gt_mask, out_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    gt_r = cv2.resize(gt_mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
    p1 = img.copy(); p1[gt_r==1]=(p1[gt_r==1]*0.5+np.array([0,255,0])*0.5).astype(np.uint8)
    contours,_=cv2.findContours(gt_r,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p1,contours,-1,(0,255,0),2)
    cam_r=cv2.resize(cam.astype(np.float32),(w,h))
    heatmap=cv2.applyColorMap((cam_r*255).astype(np.uint8),cv2.COLORMAP_JET)
    p2=cv2.addWeighted(img,0.5,heatmap,0.5,0)
    p3=p2.copy(); cv2.drawContours(p3,contours,-1,(0,255,0),2)
    cv2.imwrite(out_path,np.concatenate([p1,p2,p3],axis=1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--prompts_json", required=True)
    parser.add_argument("--train_csv",    required=True)
    parser.add_argument("--mask_csv",     required=True)
    parser.add_argument("--out_dir",      default="./test_06_layercam")
    parser.add_argument("--n",            type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt       = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]; cfg = ckpt["config"]
    timm_name  = cfg.model.image_encoder.name.replace("timm/","")
    tok_name   = cfg.tokenizer.pretrained_model_name_or_path
    proj_dim   = cfg.model.projection_head.proj_dim

    def extract(prefix):
        return {k[len(prefix):]:v for k,v in state_dict.items() if k.startswith(prefix)}

    timm_model = timm.create_model(timm_name, pretrained=False, num_classes=0)
    for bi in range(2): timm_model.stages[3].blocks[bi].token_mixer.fused_attn = False
    timm_model.load_state_dict(extract("image_encoder.image_encoder.timm_model."), strict=True)
    image_proj = LinearProjection(timm_model.num_features, proj_dim)
    image_proj.load_state_dict(extract("image_projection."))
    logit_scale   = state_dict["logit_scale"].exp().item()
    image_encoder = SoftmaxImageEncoder(timm_model, image_proj, logit_scale).to(device).eval()
    target_layer  = timm_model.stages[3].blocks[1].norm

    tokenizer  = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    text_model = AutoModel.from_pretrained(tok_name, trust_remote_code=True)
    text_model.load_state_dict(extract("text_encoder.text_encoder."))
    text_proj  = LinearProjection(text_model.config.hidden_size, proj_dim)
    text_proj.load_state_dict(extract("text_projection."))
    text_model = text_model.to(device).eval(); text_proj = text_proj.to(device).eval()

    with open(args.prompts_json) as f: all_prompts = json.load(f)
    fg_feature  = encode_prompts(all_prompts["Pneumothorax"]["pos"], tokenizer, text_model, text_proj, device)
    bg_features = []
    for cond,prompts in all_prompts.items():
        if cond=="Pneumothorax": continue
        neg=prompts.get("neg",[])
        if not neg: continue
        bg_features.append(encode_prompts(neg, tokenizer, text_model, text_proj, device))
    all_features = torch.stack([fg_feature]+bg_features, dim=0).to(device)
    image_encoder.text_features = all_features
    print(f"Classes: 1 fg + {len(bg_features)} bg")

    df = pd.read_csv(args.train_csv, index_col=0)
    df = df[df["label"]==1].drop_duplicates(subset="image").head(args.n)
    mask_lookup = build_mask_lookup(args.mask_csv)
    tf = get_transform(512)

    results = []
    for _, row in df.iterrows():
        path = str(row["image"])
        stem = os.path.splitext(os.path.basename(path))[0]
        if not os.path.exists(path): continue

        pil = Image.open(path).convert("RGB")
        ori_w, ori_h = pil.size
        tensor = tf(pil).unsqueeze(0).to(device)

        print(f"\nLayerCAM: {stem[-25:]}")
        cam = layer_cam(image_encoder, tensor, target_class=0, target_layer=target_layer)

        rles    = mask_lookup.get(stem,["-1"])
        gt_mask = merge_masks(rles, ori_h, ori_w)
        cam_full= cv2.resize(cam.astype(np.float32),(ori_w,ori_h))

        best_iou,best_t = 0, 0.5
        for t in np.arange(0.1,0.95,0.05):
            iou = compute_iou((cam_full>=t).astype(np.uint8), gt_mask)
            if iou > best_iou: best_iou,best_t = iou,t
        print(f"  Best IoU: {best_iou:.3f} @ t={best_t:.2f}")
        results.append({"id":stem[-20:],"iou":best_iou,"t":best_t})

        np.save(os.path.join(args.out_dir,f"{stem}_raw_cam.npy"), cam_full.astype(np.float16))
        save_panel(path, cam_full, gt_mask, os.path.join(args.out_dir,f"{stem}_layercam.jpg"))

    print("\n=== LayerCAM Results ===")
    for r in results: print(f"  {r['id']:25s}  IoU={r['iou']:.3f}  @t={r['t']:.2f}")
    print(f"  Mean Best IoU: {np.mean([r['iou'] for r in results]):.3f}")

if __name__ == "__main__":
    main()