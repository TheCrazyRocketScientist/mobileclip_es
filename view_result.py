import os
import sys
import json
import argparse
import warnings
import cv2
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from tqdm import tqdm
from torch import multiprocessing
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Hugging Face clinical language infrastructure
from transformers import AutoTokenizer, AutoModel

# Inject local module paths explicitly to prioritize your copied repositories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import scale_cam_image

warnings.filterwarnings("ignore")

def rle2mask(rle, width, height):
    """Decodes a standard SIIM run-length string container into an alignment matrix mask."""
    if pd.isna(rle) or rle.strip() == "-1" or not rle:
        return np.zeros((height, width), dtype=np.uint8)
    
    mask = np.zeros(width * height, dtype=np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height).T

def split_dataset(dataset, n_splits):
    if n_splits == 1:
        return [dataset]
    part = len(dataset) // n_splits
    dataset_list = []
    for i in range(n_splits - 1):
        dataset_list.append(dataset[i*part:(i+1)*part])
    dataset_list.append(dataset[(i+1)*part:])
    return dataset_list

def load_medical_prompts(json_path, target_class):
    """Extracts explicit positive and background negation prompt strings directly from JSON."""
    with open(json_path, "r") as f:
        prompt_data = json.load(f)
    
    if target_class not in prompt_data:
        raise KeyError(f"Target class label '{target_class}' not discovered inside {json_path}")
        
    pos_prompts = prompt_data[target_class]["pos"]
    neg_prompts = prompt_data[target_class]["neg"] if prompt_data[target_class]["neg"] != [""] else prompt_data["No Finding"]["pos"]
    return pos_prompts, neg_prompts

def encode_prompt_batch(prompt_list, text_model, tokenizer, device_id):
    """Tokenizes and builds normalized embeddings via native ClinicalBERT vocabulary constraints."""
    inputs = tokenizer(
        prompt_list, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    ).to(device_id)
    
    outputs = text_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings /= embeddings.norm(dim=-1, keepdim=True)
    mean_embedding = embeddings.mean(dim=0)
    mean_embedding /= mean_embedding.norm()
    return mean_embedding

class ClipOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

def _transform_resize(h, w):
    return Compose([
        Resize((h, w), interpolation=Image.BICUBIC),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def find_target_and_attention_properties(model):
    """Dynamically scans the loaded timm architecture to figure out exact hook points."""
    total_sa_blocks = 0
    target_layer = None
    
    if hasattr(model, 'stages'):
        final_stage = model.stages[-1]
        last_block = final_stage.blocks[-1]
        
        if hasattr(last_block, 'mlp'):
            if hasattr(last_block.mlp, 'norm2'):
                target_layer = [last_block.mlp.norm2]
            elif hasattr(last_block.mlp, 'fc2'):
                target_layer = [last_block.mlp.fc2]
            else:
                target_layer = [last_block.mlp]
        else:
            target_layer = [last_block]
        
        for block in final_stage.blocks:
            if hasattr(block, 'mixer') and ('Attention' in type(block.mixer).__name__ or 'Attn' in type(block.mixer).__name__):
                total_sa_blocks += 1
    else:
        if hasattr(model, 'blocks'):
            target_layer = [model.blocks[-1].norm1]
            total_sa_blocks = len(model.blocks)
            
    return target_layer, total_sa_blocks

def perform(process_id, dataset_list, args, model, text_model, tokenizer, json_prompt_path, mask_df, target_layers, total_sa_blocks):
    n_gpus = torch.cuda.device_count()
    device_id = f"cuda:{process_id % n_gpus}" if n_gpus > 0 else "cpu"
    databin = dataset_list[process_id]
    
    model = model.to(device_id)
    text_model = text_model.to(device_id)
    patch_size = 16

    def custom_reshape(tensor):
        tokens_count = tensor.size(0) - 1 if tensor.size(0) % 2 != 0 else tensor.size(0)
        calc_side = int(np.sqrt(tokens_count))
        tensor = tensor.permute(1, 0, 2)
        if tensor.size(1) > tokens_count:
            res = tensor[:, 1:, :].reshape(tensor.size(0), calc_side, calc_side, tensor.size(2))
        else:
            res = tensor.reshape(tensor.size(0), calc_side, calc_side, tensor.size(2))
        return res.transpose(2, 3).transpose(1, 2)

    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=custom_reshape)

    print(f"\n[Worker {process_id}] Commencing loop over {len(databin)} localized samples...")

    for row_idx, row in enumerate(databin):
        img_relative_path = row['image']
        class_str = row['class']
        img_path = os.path.join(args.img_root, img_relative_path)

        print("\n" + "="*80)
        print(f"🔍 PROCESSING ROW {row_idx}: {img_relative_path}")
        print("="*80)

        if not os.path.exists(img_path):
            print(f"❌ CRITICAL error: Disk file does not exist at {img_path}")
            sys.exit(1)

        cv_img = cv2.imread(img_path)
        if cv_img is None:
            print(f"❌ CRITICAL error: OpenCV failed to read matrix data from {img_path}")
            sys.exit(1)
            
        ori_height, ori_width, _ = cv_img.shape
        print(f"-> Canvas dimensions detected: Width={ori_width}, Height={ori_height}")

        # Loading text conditions (No catch block to reveal structural errors instantly)
        fg_prompts, bg_prompts = load_medical_prompts(json_prompt_path, class_str)
        print(f"-> Prompts extracted for '{class_str}': {len(fg_prompts)} positive, {len(bg_prompts)} background.")

        # Text embedding synthesis
        fg_features = encode_prompt_batch(fg_prompts, text_model, tokenizer, device_id).unsqueeze(0)
        bg_features = encode_prompt_batch(bg_prompts, text_model, tokenizer, device_id).unsqueeze(0)
        text_features_tensor = torch.cat([fg_features, bg_features], dim=0)
        print(f"-> ClinicalBERT matrix shape synthesized: {list(text_features_tensor.shape)}")

        # Image preprocessing
        preprocess = _transform_resize(
            int(np.ceil(ori_height / patch_size) * patch_size), 
            int(np.ceil(ori_width / patch_size) * patch_size)
        )
        image_tensor = preprocess(Image.open(img_path)).unsqueeze(0).to(device_id)
        h, w = image_tensor.shape[-2], image_tensor.shape[-1]
        print(f"-> Preprocessed execution tensor dimensions: {list(image_tensor.shape)} (H={h}, W={w})")

        # Forward pass feature extraction
        image_features = model(image_tensor)
        print(f"-> Vision backbone output vector map shape: {list(image_features.shape)}")
        
        attn_weight_list = [] 
        input_tensor = [image_features, text_features_tensor, h, w]
        targets = [ClipOutputTarget(0)]
        
        # Computing backpropagation gradients
        print("-> Executing Grad-CAM backward propagation phase...")
        grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor, targets=targets, target_size=None)
        
        grayscale_cam = grayscale_cam[0, :]
        print(f"-> Raw low-res Grad-CAM map matrix extracted. Shape: {grayscale_cam.shape}")

        if total_sa_blocks > 0 and len(attn_weight_list) > 0:
            print(f"-> Processing transformer random walk layers across {total_sa_blocks} attention maps...")
            attn_weight_list.append(attn_weight_last)
            attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list if aw.ndim == 3]
            attn_weight = torch.stack(attn_weight, dim=0)[-total_sa_blocks:]
            attn_weight = torch.mean(attn_weight, dim=0)[0].cpu().detach().float()

            aff_mat = attn_weight
            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
            
            for _ in range(2):
                trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
                trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
            trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2
            
            for _ in range(1):
                trans_mat = torch.matmul(trans_mat, trans_mat)

            cam_to_refine = torch.FloatTensor(grayscale_cam).view(-1, 1)
            cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h // patch_size, w // patch_size)
            final_cam_matrix = cam_refined.cpu().numpy().astype(np.float32)
        else:
            print("-> Skipping structural self-attention refinement (backbone type is purely convolutional).")
            final_cam_matrix = grayscale_cam

        # Interpolating back to source matrix resolution
        cam_highres = scale_cam_image([final_cam_matrix], (ori_width, ori_height))[0]
        print(f"-> Interpolated high-res activation dimensions: {cam_highres.shape}")

        # Direct filename extraction to avoid path nesting bugs
        raw_filename = img_relative_path.split('/')[-1]
        output_filename = raw_filename.replace('.jpg', '.npy')
        save_path = os.path.join(args.cam_out_dir, output_filename)
        
        # Comprehensive activation metrics print statement
        print("\n" + "-"*40)
        print("📊 DIAGNOSTIC MATRIX PROFILE REPORT:")
        print(f"  ├── Target Output Path: {save_path}")
        print(f"  ├── Intensity Minimum:  {cam_highres.min():.6f}")
        print(f"  ├── Intensity Maximum:  {cam_highres.max():.6f}")
        print(f"  ├── Intensity Average:  {cam_highres.mean():.6f}")
        print(f"  └── Saved Array Shape:  {cam_highres.shape}")
        print("-"*40)

        np.save(save_path, {
            "keys": np.array([row['label']]),
            "attn_highres": cam_highres.astype(np.float16)
        })
        print(f"✨ File saved successfully to disk location!")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ClinicalBERT-Driven Segmenter')
    parser.add_argument('--img_root', type=str, required=True)
    parser.add_argument('--manifest_txt', type=str, required=True)
    parser.add_argument('--mask_csv', type=str, default=None)
    parser.add_argument('--prompt_json', type=str, required=True)
    parser.add_argument('--cam_out_dir', type=str, default='./testing/output_masks')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.cam_out_dir, exist_ok=True)

    df_manifest = pd.read_csv(args.manifest_txt, sep=",")
    
    # Accelerated validation filter block for local execution testing
    print(f"Original manifest rows loaded: {len(df_manifest)}")
    def file_exists_locally(row):
        target_path = os.path.join(args.img_root, row['image'])
        return os.path.exists(target_path)
    df_manifest = df_manifest[df_manifest.apply(file_exists_locally, axis=1)]
    print(f"Filtered manifest records matching local files: {len(df_manifest)}")
    
    if len(df_manifest) == 0:
        print("❌ CRITICAL: 0 matching files discovered on your machine. Terminating script.")
        sys.exit(1)

    manifest_records = df_manifest.to_dict(orient="records")
    mask_df = pd.read_csv(args.mask_csv) if args.mask_csv and os.path.exists(args.mask_csv) else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Initializing ClinicalBERT Language Transformer Context...")
    checkpoint_target = "medicalai/ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_target)
    text_model = AutoModel.from_pretrained(checkpoint_target).to(device)
    text_model.eval()

    print(f"Rebuilding Vision Skeleton Structure...")
    model_variant = "fastvit_sa12" if "sa12" in args.model.lower() else "fastvit_s12"
    model = timm.create_model(model_variant, pretrained=False, num_classes=0)
    
    clean_state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(clean_state_dict, strict=True)
    model = model.to(device)
    model.eval()

    target_layers, total_sa_blocks = find_target_and_attention_properties(model)
    print(f"✅ Setup verified. Model variant loaded: '{model_variant}' containing {total_sa_blocks} attention blocks.")

    dataset_splits = split_dataset(manifest_records, n_splits=args.num_workers)

    if args.num_workers == 1:
        perform(0, dataset_splits, args, model, text_model, tokenizer, args.prompt_json, mask_df, target_layers, total_sa_blocks)
    else:
        multiprocessing.spawn(perform, nprocs=args.num_workers,
                              args=(dataset_splits, args, model, text_model, tokenizer, args.prompt_json, mask_df, target_layers, total_sa_blocks))