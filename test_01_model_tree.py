# -*- coding: utf-8 -*-
"""
test_01_model_tree.py
---------------------
Step 1: Load the checkpoint and print the full module tree.
Goal: confirm the exact layer names and class types before we hook anything.

Usage:
    python test_01_model_tree.py --checkpoint checkpoints/fastvit_sa12.apple_in1k_distil-biobert.tar
"""

import argparse
import torch
import torch.nn as nn


def print_tree(module, prefix='', max_depth=6, current_depth=0):
    if current_depth >= max_depth:
        return
    for name, child in module.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        cls = type(child).__name__
        print(f"{prefix}{name}  [{cls}]  ({n_params:,} params)")
        print_tree(child, prefix + '    ', max_depth, current_depth + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--max_depth', type=int, default=6)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu',
                            weights_only=False)

    # The checkpoint stores a state dict, not a live module.
    # We need to reconstruct the model — but for this test we just want
    # to see the tree, so let's use timm directly and confirm it matches
    # the state dict key prefixes we saw in the inspector.

    import timm
    cfg = checkpoint['config']
    timm_name = cfg.model.image_encoder.name.replace('timm/', '')
    print(f"Building: {timm_name}\n")

    model = timm.create_model(timm_name, pretrained=False, num_classes=0)

    print("=" * 70)
    print("  Full FastViT module tree")
    print("=" * 70)
    print_tree(model, max_depth=args.max_depth)

    print("\n" + "=" * 70)
    print("  All named modules (flat list with class names)")
    print("=" * 70)
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls in ('Attention', 'RepMixer', 'AttentionBlock',
                   'RepMixerBlock', 'FastVitBlock'):
            print(f"  {name:70s}  [{cls}]")

    print("\n" + "=" * 70)
    print("  Stage[3] block detail (the attention stage)")
    print("=" * 70)
    for bi, block in enumerate(model.stages[3].blocks):
        print(f"\n  stages[3].blocks[{bi}]  [{type(block).__name__}]")
        for name, child in block.named_children():
            print(f"    {name}  [{type(child).__name__}]")
            for n2, c2 in child.named_children():
                print(f"      {n2}  [{type(c2).__name__}]")


if __name__ == '__main__':
    main()