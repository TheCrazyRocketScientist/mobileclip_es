# -*- coding: utf-8 -*-
"""
test_03_real_weights.py
-----------------------
Step 3: Load the actual checkpoint weights into the FastViT model,
verify the state dict loads cleanly, run a real CXR image through,
and confirm:
  - GradCAM target layer (stages[3].blocks[1].norm) output shape
  - Attention weights still captured correctly with real weights
  - forward_features output shape and value range

Usage:
    python test_03_real_weights.py \
        --checkpoint checkpoints/fastvit_sa12.apple_in1k_distil-biobert.tar \
        --image final_512/siim/images/1.2.276.0.7230010.3.1.4.8323329.1000.1517875165.878027.jpg
"""

import argparse
import torch
import timm
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    import PIL
    BICUBIC = PIL.Image.BICUBIC


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def get_transform(size=224):
    return Compose([
        Resize(size, interpolation=BICUBIC),
        CenterCrop(size),
        lambda img: img.convert('RGB'),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


captured = {}

def make_hook(name):
    def fn(module, input, output):
        captured[name] = output.detach().cpu()
    return fn


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True,
                        help='Path to any CXR jpg to test with')
    args = parser.parse_args()

    # --- Load checkpoint ---
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu',
                            weights_only=False)
    state_dict = checkpoint['model']
    cfg        = checkpoint['config']
    timm_name  = cfg.model.image_encoder.name.replace('timm/', '')
    print(f"  arch     : {timm_name}")
    print(f"  epoch    : {checkpoint.get('epoch', '?')}")
    print(f"  sd keys  : {len(state_dict)}")

    # --- Build model ---
    print("\nBuilding model...")
    model = timm.create_model(timm_name, pretrained=False, num_classes=0)

    # Disable fused_attn BEFORE loading weights
    for bi in range(2):
        model.stages[3].blocks[bi].token_mixer.fused_attn = False
    print("  fused_attn disabled on both attention blocks")

    # --- Extract and load image encoder state dict ---
    # Prefix in checkpoint: 'image_encoder.image_encoder.timm_model.'
    prefix = 'image_encoder.image_encoder.timm_model.'
    timm_sd = {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }
    print(f"  timm state dict keys extracted: {len(timm_sd)}")

    missing, unexpected = model.load_state_dict(timm_sd, strict=True)
    if missing:
        print(f"  [WARN] missing  : {missing[:5]}")
    if unexpected:
        print(f"  [WARN] unexpected: {unexpected[:5]}")
    if not missing and not unexpected:
        print("  state dict loaded cleanly (strict=True, 0 missing, 0 unexpected)")

    model.eval()

    # --- Register hooks ---
    hooks = []

    # GradCAM target layer — last spatial layer before final attention block
    gradcam_layer = model.stages[3].blocks[1].norm
    hooks.append(
        gradcam_layer.register_forward_hook(make_hook('gradcam_target'))
    )
    print("\nHooks registered:")
    print("  GradCAM target : stages[3].blocks[1].norm  [BatchNorm2d]")

    # Attention hooks on both blocks
    for bi in range(2):
        h = model.stages[3].blocks[bi].token_mixer.attn_drop.register_forward_hook(
            make_hook(f'attn_block{bi}')
        )
        hooks.append(h)
        print(f"  Attn hook      : stages[3].blocks[{bi}].token_mixer.attn_drop")

    # --- Load and preprocess image ---
    print(f"\nLoading image: {args.image}")
    tf = get_transform(512)
    img_tensor = tf(Image.open(args.image)).unsqueeze(0)  # (1, 3, 512, 512)
    print(f"  input shape : {img_tensor.shape}")

    # --- Forward pass ---
    print("\nRunning forward pass...")
    with torch.no_grad():
        features = model.forward_features(img_tensor)   # (1, C, H, W)

    print(f"  forward_features output : {features.shape}")
    print(f"  feature value range     : [{features.min():.4f}, {features.max():.4f}]")

    # --- Check captured tensors ---
    print("\n--- Captured tensors ---")
    for name, tensor in captured.items():
        print(f"\n  [{name}]")
        print(f"    shape : {tensor.shape}")
        print(f"    dtype : {tensor.dtype}")
        print(f"    range : [{tensor.min():.4f}, {tensor.max():.4f}]")
        if 'attn' in name:
            row_sums = tensor[0, 0].sum(dim=-1)
            print(f"    softmax row sum (head 0): "
                  f"min={row_sums.min():.4f}  max={row_sums.max():.4f}  "
                  f"mean={row_sums.mean():.4f}  ← should be 1.0")

    # --- GradCAM target shape check ---
    gt = captured.get('gradcam_target')
    if gt is not None:
        B, C, H, W = gt.shape
        print(f"\n--- GradCAM target layer summary ---")
        print(f"  Output shape : (B={B}, C={C}, H={H}, W={W})")
        print(f"  Spatial grid : {H}×{W} = {H*W} positions")
        attn_n = captured.get('attn_block0')
        attn_n = attn_n.shape[-1] if attn_n is not None else '?'
        match = '✓ MATCH' if H*W == attn_n else f'✗ MISMATCH (attn N={attn_n})'
        print(f"  Consistent with attn N?  {H*W} vs {attn_n}  {match}")

    # --- Cleanup ---
    for h in hooks:
        h.remove()

    print("\n[DONE] If all shapes look correct, proceed to test_04 (GradCAM + real text).")


if __name__ == '__main__':
    main()