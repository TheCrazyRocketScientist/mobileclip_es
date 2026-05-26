# -*- coding: utf-8 -*-
"""
inspect_checkpoint.py
---------------------
Loads a .tar checkpoint and prints everything we need to know
to wire up the model loading in generate_cams_siim.py.

Usage:
    python inspect_checkpoint.py --checkpoint /path/to/model.tar
"""

import argparse
import torch
import torch.nn as nn


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def inspect_tensor_dict(d, prefix='', max_entries=20):
    """Print keys + shapes of a state dict."""
    items = list(d.items())
    for i, (k, v) in enumerate(items[:max_entries]):
        if isinstance(v, torch.Tensor):
            print(f"  {prefix}{k:60s} {str(v.shape):30s} {v.dtype}")
        else:
            print(f"  {prefix}{k:60s} {type(v).__name__}")
    if len(items) > max_entries:
        print(f"  ... and {len(items) - max_entries} more keys")


def inspect_module(module, max_depth=3, prefix=''):
    """Recursively print module tree up to max_depth."""
    for name, child in list(module.named_children()):
        cls = type(child).__name__
        n_params = sum(p.numel() for p in child.parameters())
        print(f"  {prefix}{name}: {cls}  ({n_params:,} params)")
        if max_depth > 1:
            inspect_module(child, max_depth - 1, prefix + '    ')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .tar checkpoint file')
    args = parser.parse_args()

    print(f"\nLoading: {args.checkpoint}")

    # PyTorch 2.6+ defaults weights_only=True which blocks OmegaConf internals
    # (DictConfig, ContainerMetadata, etc.). Since this is your own trusted
    # checkpoint, weights_only=False is the correct solution here.
    checkpoint = torch.load(args.checkpoint, map_location='cpu',
                            weights_only=False)
    print("  [loaded with weights_only=False]")

    # ------------------------------------------------------------------ #
    # 1. Top-level type and keys
    # ------------------------------------------------------------------ #
    print_section("1. Checkpoint type")
    print(f"  type(checkpoint): {type(checkpoint).__name__}")

    if isinstance(checkpoint, nn.Module):
        print("  >>> Checkpoint IS a full nn.Module (entire model saved directly)")
        print_section("Module structure (depth=3)")
        inspect_module(checkpoint)
        return

    if isinstance(checkpoint, dict):
        print(f"  Top-level keys: {list(checkpoint.keys())}")

    # ------------------------------------------------------------------ #
    # 2. Config / arch info
    # ------------------------------------------------------------------ #
    print_section("2. Config / architecture info")
    for key in ('config', 'cfg', 'arch', 'model_args', 'hparams',
                'hyper_parameters', 'model_name', 'args'):
        if key in checkpoint:
            val = checkpoint[key]
            print(f"  checkpoint['{key}'] -> {type(val).__name__}:")
            if isinstance(val, dict):
                for k, v in val.items():
                    print(f"      {k}: {v}")
            else:
                print(f"      {val}")

    # ------------------------------------------------------------------ #
    # 3. State dict
    # ------------------------------------------------------------------ #
    print_section("3. State dict")
    state_dict = None
    for key in ('model', 'state_dict', 'model_state', 'model_state_dict',
                'network', 'net'):
        if key in checkpoint:
            candidate = checkpoint[key]
            if isinstance(candidate, dict):
                print(f"  Found state dict under key='{key}' "
                      f"({len(candidate)} keys)")
                state_dict = candidate
                break
            elif isinstance(candidate, nn.Module):
                print(f"  checkpoint['{key}'] is a live nn.Module — "
                      f"extracting state dict")
                state_dict = candidate.state_dict()
                print_section("Module structure (depth=3)")
                inspect_module(candidate)
                break

    if state_dict is not None:
        print(f"\n  First {min(20, len(state_dict))} state dict entries:")
        inspect_tensor_dict(state_dict)

        # Detect module prefix (DataParallel wrapping)
        sample_keys = list(state_dict.keys())[:5]
        if any(k.startswith('module.') for k in sample_keys):
            print("\n  [!] Keys have 'module.' prefix (DataParallel) — "
                  "will need stripping")
        
        # Try to identify sub-model boundaries
        print("\n  Top-level parameter namespaces:")
        namespaces = {}
        for k in state_dict.keys():
            top = k.replace('module.', '').split('.')[0]
            namespaces[top] = namespaces.get(top, 0) + 1
        for ns, count in sorted(namespaces.items(), key=lambda x: -x[1]):
            print(f"      {ns:40s} {count:5d} tensors")
    else:
        print("  No state dict found under common keys.")
        print("  All top-level values:")
        if isinstance(checkpoint, dict):
            for k, v in checkpoint.items():
                print(f"    '{k}': {type(v).__name__}")

    # ------------------------------------------------------------------ #
    # 4. Optimizer / epoch / other metadata
    # ------------------------------------------------------------------ #
    print_section("4. Training metadata")
    for key in ('epoch', 'step', 'global_step', 'best_val', 'optimizer',
                'scheduler', 'scaler', 'loss'):
        if key in checkpoint:
            val = checkpoint[key]
            if isinstance(val, (int, float, str)):
                print(f"  {key}: {val}")
            else:
                print(f"  {key}: {type(val).__name__}")

    # ------------------------------------------------------------------ #
    # 5. If model is a live module, print its named_modules with attn scan
    # ------------------------------------------------------------------ #
    if state_dict is not None:
        print_section("5. Attention-related key scan")
        attn_keys = [k for k in state_dict.keys()
                     if any(x in k.lower() for x in
                            ('attn', 'attention', 'mixer', 'mhsa', 'qkv',
                             'query', 'key', 'value'))]
        if attn_keys:
            print(f"  Found {len(attn_keys)} attention-related keys. "
                  f"First 20:")
            for k in attn_keys[:20]:
                print(f"    {k}")
        else:
            print("  No obvious attention keys found.")

    print("\n" + "="*60)
    print("  Done. Paste this output back to continue.")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()