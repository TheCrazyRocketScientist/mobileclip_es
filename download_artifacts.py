#!/usr/bin/env python3
"""
download_artifacts.py
---------------------
Downloads CXR-CLIP model artifacts from W&B and renames them
to be compatible with compare_models.sh naming convention.

Artifact name format on W&B:  entity/project/run_name:version
Local checkpoint name format:  {vision_encoder}_{text_encoder}.tar

Usage:
    python download_artifacts.py --entity YOUR_ENTITY --project YOUR_PROJECT
    python download_artifacts.py --entity YOUR_ENTITY --project YOUR_PROJECT --version best
"""

import os
import shutil
import argparse
import wandb


# -----------------------------------------------------------------------
# Edit this mapping: W&B artifact name → local checkpoint filename
# The local name becomes the model nickname in compare_models.sh
# Format: 'artifact_name_on_wandb': 'local_filename_without_extension'
# -----------------------------------------------------------------------
# W&B entity and project (from URL)
ENTITY  = 'starmariner027-manipal-institute-of-technology-bangalore'
PROJECT = 'CXR-CLIP'

# Artifact name on W&B → local checkpoint filename (no extension)
# Full artifact names visible in sidebar — CLIP_{vision}_{text}
ARTIFACT_MAP = {
    # W&B artifact name                                               : local checkpoint name
    'CLIP_vit_small_patch16_224.augreg_in21k_ft_in1k_distil-clinicalbert': 'vit_small_distilclinical',
    'CLIP_vit_tiny_patch16_224.augreg_in21k_ft_in1k_distil-clinicalbert':  'vit_tiny_distilclinical',
    'CLIP_fastvit_sa24.apple_in1k_distil-clinicalbert':                     'fastvit_sa24_distilclinical',
    'CLIP_fastvit_s12.apple_dist_in1k_distil-clinicalbert':                 'fastvit_s12_distilclinical',
}

# File name inside each artifact (W&B saves as model-best.tar)
ARTIFACT_FILENAME = 'model-best.tar'

CHECKPOINTS_DIR = './checkpoints'


def download_artifact(entity, project, artifact_name, version, nickname, dry_run=False):
    """Download a single artifact and rename it."""
    full_name  = f'{entity}/{project}/{artifact_name}:{version}'
    local_name = f'{nickname}.tar'
    local_path = os.path.join(CHECKPOINTS_DIR, local_name)

    if os.path.exists(local_path):
        print(f'  [SKIP] Already exists: {local_name}')
        return local_path

    print(f'  Downloading: {full_name}')
    print(f'  → Saving as: {local_name}')

    if dry_run:
        print(f'  [DRY RUN] Would save to: {local_path}')
        return local_path

    try:
        api = wandb.Api()
        art = api.artifact(full_name, type='model')
        dl  = art.download()

        # Look for ARTIFACT_FILENAME first (model-best.tar), then any .tar/.pt
        tar_file = None
        preferred = os.path.join(dl, ARTIFACT_FILENAME)
        if os.path.exists(preferred):
            tar_file = preferred
        else:
            for fname in sorted(os.listdir(dl)):
                if fname.endswith('.tar') or fname.endswith('.pt') or fname.endswith('.pth'):
                    tar_file = os.path.join(dl, fname)
                    break

        if tar_file is None:
            files = os.listdir(dl)
            if files:
                tar_file = os.path.join(dl, files[0])
                print(f'  [WARN] Using fallback file: {files[0]}')
            else:
                print(f'  [ERROR] No files in artifact: {dl}')
                return None

        print(f'  Found: {os.path.basename(tar_file)} ({os.path.getsize(tar_file)/1e9:.2f}GB)')
        print(f'  Copying to: {local_path}')
        shutil.copy2(tar_file, local_path)
        print(f'  [OK] {local_path}')
        return local_path

    except Exception as e:
        print(f'  [ERROR] {full_name}: {e}')
        return None


def list_project_artifacts(entity, project, artifact_type='model'):
    """List all artifacts in a project to help identify names."""
    print(f'\nListing artifacts in {entity}/{project} (type={artifact_type}):\n')
    api = wandb.Api()
    try:
        project_obj = api.project(entity=entity, name=project)
        artifacts   = api.artifacts(type_name=artifact_type,
                                     project=f'{entity}/{project}')
        for art in artifacts:
            print(f'  {art.name:50s}  versions={art.version}')
    except Exception as e:
        print(f'  Error listing artifacts: {e}')
        print(f'  Try: wandb artifact ls {entity}/{project}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity',   default=ENTITY,
                        help=f'W&B entity (default: {ENTITY})')
    parser.add_argument('--project',  default=PROJECT,
                        help=f'W&B project (default: {PROJECT})')
    parser.add_argument('--version',  default='latest',
                        help='Artifact version (default: latest). '
                             'Use "best" or "v0", "v1" etc.')
    parser.add_argument('--list',     action='store_true',
                        help='List all artifacts in project and exit')
    parser.add_argument('--dry_run',  action='store_true',
                        help='Show what would be downloaded without downloading')
    parser.add_argument('--artifact', default=None,
                        help='Download a single artifact by name '
                             '(overrides ARTIFACT_MAP)')
    parser.add_argument('--nickname', default=None,
                        help='Nickname for single artifact download')
    args = parser.parse_args()

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # Login to W&B
    wandb.login()

    # Just list artifacts
    if args.list:
        list_project_artifacts(args.entity, args.project)
        return

    # Single artifact download
    if args.artifact:
        if not args.nickname:
            # Auto-generate nickname from artifact name
            args.nickname = args.artifact.replace('-', '_').replace(' ', '_').lower()
        print(f'\nDownloading single artifact: {args.artifact}')
        download_artifact(args.entity, args.project, args.artifact,
                          args.version, args.nickname, args.dry_run)
        return

    # Batch download from ARTIFACT_MAP
    print(f'\nDownloading {len(ARTIFACT_MAP)} artifacts from '
          f'{args.entity}/{args.project}\n')

    success, failed = [], []
    for artifact_name, nickname in ARTIFACT_MAP.items():
        print(f'\n--- {artifact_name} → {nickname}.tar ---')
        result = download_artifact(args.entity, args.project, artifact_name,
                                   args.version, nickname, args.dry_run)
        if result:
            success.append(nickname)
        else:
            failed.append(artifact_name)

    # Summary
    print(f'\n{"="*50}')
    print(f'Downloaded: {len(success)}/{len(ARTIFACT_MAP)}')
    for s in success:
        print(f'  ✓ checkpoints/{s}.tar')
    if failed:
        print(f'\nFailed ({len(failed)}):')
        for f in failed:
            print(f'  ✗ {f}')
        print('\nRun with --list to see available artifact names')
    print(f'{"="*50}')

    # Print what compare_models.sh will see
    print(f'\nCheckpoints available for compare_models.sh:')
    for f in sorted(os.listdir(CHECKPOINTS_DIR)):
        if f.endswith('.tar'):
            print(f'  checkpoints/{f}')


if __name__ == '__main__':
    main()