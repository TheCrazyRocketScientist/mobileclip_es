# -*- coding: utf-8 -*-
"""
test_02_attention_hooks.py
--------------------------
Step 2: Hook stages[3].blocks[0,1].token_mixer and verify we can
capture attention weights from a forward pass.

Key issue to solve: FastViT uses fused_attn (scaled_dot_product_attention)
by default which never materialises the attn matrix as a Python tensor.
We test two approaches:
  A) Disable fused_attn on the existing module (just flip the flag)
  B) Forward hook that intercepts qkv and computes attn manually

We want to confirm which approach works before touching the real checkpoint.

Usage:
    python test_02_attention_hooks.py \
        --checkpoint checkpoints/fastvit_sa12.apple_in1k_distil-biobert.tar
"""

import argparse
import torch
import timm

# -----------------------------------------------------------------------
# Approach A: just flip fused_attn=False and hook the output of attn_drop
# -----------------------------------------------------------------------

captured_attn = {}

def make_hook(name):
    def hook_fn(module, input, output):
        # attn_drop receives the softmax attn matrix (B, heads, N, N)
        # and returns it unchanged (dropout during eval is identity)
        captured_attn[name] = output.detach().cpu()
    return hook_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu',
                            weights_only=False)
    cfg = checkpoint['config']
    timm_name = cfg.model.image_encoder.name.replace('timm/', '')

    print(f"Building: {timm_name}")
    model = timm.create_model(timm_name, pretrained=False, num_classes=0)
    model.eval()

    # -----------------------------------------------------------------------
    # Step A: check fused_attn flag on both attention blocks
    # -----------------------------------------------------------------------
    print("\n--- Step A: fused_attn flag ---")
    for bi in range(2):
        attn_mod = model.stages[3].blocks[bi].token_mixer
        print(f"  stages[3].blocks[{bi}].token_mixer.fused_attn = "
              f"{attn_mod.fused_attn}")

    # -----------------------------------------------------------------------
    # Step B: disable fused_attn and register hook on attn_drop
    # -----------------------------------------------------------------------
    print("\n--- Step B: disable fused_attn, hook attn_drop ---")
    hooks = []
    for bi in range(2):
        attn_mod = model.stages[3].blocks[bi].token_mixer
        attn_mod.fused_attn = False
        h = attn_mod.attn_drop.register_forward_hook(
            make_hook(f"block{bi}_attn")
        )
        hooks.append(h)
        print(f"  Registered hook on stages[3].blocks[{bi}].token_mixer.attn_drop")

    # -----------------------------------------------------------------------
    # Step C: run a dummy forward pass
    # -----------------------------------------------------------------------
    print("\n--- Step C: dummy forward pass (1, 3, 224, 224) ---")
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model.forward_features(dummy)

    print(f"  forward_features output shape: {out.shape}")

    # -----------------------------------------------------------------------
    # Step D: inspect captured attention weights
    # -----------------------------------------------------------------------
    print("\n--- Step D: captured attention weights ---")
    if not captured_attn:
        print("  [FAIL] Nothing captured — hooks did not fire")
    else:
        for name, tensor in captured_attn.items():
            B, heads, N, N2 = tensor.shape
            h_feat = w_feat = int(N ** 0.5)
            print(f"  {name}:")
            print(f"    shape : {tensor.shape}  (B, heads, N, N)")
            print(f"    N={N} → spatial grid ≈ {h_feat}×{w_feat}")
            print(f"    dtype : {tensor.dtype}")
            print(f"    min={tensor.min():.4f}  max={tensor.max():.4f}  "
                  f"sum_over_N={tensor[0,0].sum(dim=-1).mean():.4f}  "
                  f"(should be ≈1.0 — softmax check)")

    # -----------------------------------------------------------------------
    # Step E: average the two blocks (what we'll do in the real pipeline)
    # -----------------------------------------------------------------------
    print("\n--- Step E: average both blocks → transition matrix ---")
    if len(captured_attn) == 2:
        weights = list(captured_attn.values())
        # Average over batch and heads → (N, N)
        avg = torch.stack(
            [w[0].mean(0) for w in weights], dim=0
        ).mean(0)
        print(f"  Averaged attn shape: {avg.shape}  (N, N)")
        print(f"  Row sum mean: {avg.sum(dim=-1).mean():.4f}  (should be ≈1.0)")
        print(f"  Col sum mean: {avg.sum(dim=0).mean():.4f}")
        print(f"  Symmetry check (|A - A^T| mean): "
              f"{(avg - avg.t()).abs().mean():.6f}")
    else:
        print(f"  Only {len(captured_attn)}/2 blocks captured — skipping")

    # Cleanup
    for h in hooks:
        h.remove()

    print("\n[DONE] If Step D shows shape (1, heads, N, N) with softmax≈1.0,")
    print("       we are ready to move to test_03 (load real weights + image).")


if __name__ == '__main__':
    main()