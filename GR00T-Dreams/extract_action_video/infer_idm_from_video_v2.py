import os
import json
import cv2
import torch
import argparse
import numpy as np
import PIL.Image
from pathlib import Path
from tqdm import tqdm
import safetensors.torch

from gr00t.model.idm import IDM, IDMConfig
from gr00t.model.action_head.siglip import SiglipProcessor

# ================= DEFAULT CONFIG =================

DEFAULT_MODEL_CONFIG = "idm/m2_bottle300/checkpoint-9500/config.json"
DEFAULT_MODEL_CKPT = "idm/m2_bottle300/checkpoint-9500/model.safetensors"
DEFAULT_OUTPUT_DIR = "output_actions1"

# Stats file for denormalization
DEFAULT_STATS_FILE = "IDM_dump/data/m2_zed/m2_pick.data/meta/stats.json"
DEFAULT_MODALITY_FILE = "IDM_dump/data/m2_zed/m2_pick.data/meta/modality.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training pipeline: data is 256x256 -> VideoCrop(0.95) -> VideoResize(224x224) -> SigLIP processor
# At inference: resize to 256x256 -> center crop 95% -> SigLIP processor handles final resize/normalize
TARGET_SIZE = (256, 256)  # Initial resize (matching training data resolution)
CROP_SCALE = 0.95  # Must match VideoCrop(scale=0.95) in M2DataConfig
CROP_RESIZE = (224, 224)  # Must match VideoResize(height=224, width=224) in M2DataConfig

# SigLIP model path (must match GR00TIDMTransform.siglip_processor)
SIGLIP_MODEL_PATH = "/home/aitv/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c"

# Embodiment
DEFAULT_EMBODIMENT_TAG = "m2"

EMBODIMENT_TAG_MAPPING = {
    "gr1": 24,
    "franka": 17,
    "so100": 26,
    "new_embodiment": 31,
    "robocasa_panda_omron": 13,
    "g1": 21,
    "m2": 26
}

# ========================================
# ACTION CONCAT ORDER (from M2DataConfig in data_config_idm.py)
# This MUST match the training config!
# ========================================
ACTION_CONCAT_ORDER = [
    "action.left_arm",   # 7 dims (joints)
    "action.left_ee",    # 6 dims (gripper, range [0,1])
    "action.right_arm",  # 7 dims (joints)
    "action.right_ee",   # 6 dims (gripper, range [0,1])
]

# EE indices in the final 26-dim action (after concat)
# left_arm: 0-7, left_ee: 7-13, right_arm: 13-20, right_ee: 20-26
EE_INDICES = list(range(7, 13)) + list(range(20, 26))  # 12 values total

# ========================================


def load_siglip_processor(model_path=SIGLIP_MODEL_PATH):
    """Load SigLIP processor to match training preprocessing."""
    print(f"Loading SigLIP processor from {model_path}")
    processor = SiglipProcessor.from_pretrained(model_path, use_fast=False)
    return processor


def center_crop_frame(img, scale=CROP_SCALE):
    """
    Apply center crop to an image, matching VideoCrop(scale) at eval time.
    
    Training uses VideoCrop(scale=0.95) which at eval does CenterCrop.
    """
    h, w = img.shape[:2]
    new_h = int(h * scale)
    new_w = int(w * scale)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return img[top:top + new_h, left:left + new_w]


def load_idm_model(config_path, ckpt_path, device):
    """Load IDM model from config and checkpoint."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    model_config = IDMConfig(**config_dict)
    model = IDM(model_config)

    state_dict = safetensors.torch.load_file(ckpt_path, device=device)
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)

    return model


def load_stats(stats_path):
    """Load statistics for denormalization."""
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return stats


def load_modality(modality_path):
    """Load modality metadata for index mapping."""
    with open(modality_path, "r") as f:
        modality = json.load(f)
    return modality


class ActionDenormalizer:
    """
    Denormalize actions from [-1, 1] to original scale.
    
    CRITICAL: Must match the training pipeline exactly!
    
    Training pipeline (from data_config_idm.py):
    1. StateActionTransform normalizes each key separately using min_max:
       normalized = 2 * (x - min) / (max - min) - 1
    2. ConcatTransform concatenates in order: left_arm, left_ee, right_arm, right_ee
    
    So we need to:
    1. Get stats for each key from stats.json (sliced by indices from modality.json)
    2. Build the denorm arrays in the same concat order
    3. Denormalize: x = (normalized + 1) / 2 * (max - min) + min
    4. Clip EE values to [0, 1] range
    """
    
    def __init__(self, stats_path, modality_path):
        stats = load_stats(stats_path)
        modality = load_modality(modality_path)
        
        action_stats = stats["action"]  # 26-dim stats for full action
        action_modality = modality["action"]  # Slice indices for each key
        
        # Build min/max arrays following CONCAT ORDER (not stats order!)
        self.action_min = []
        self.action_max = []
        self.key_ranges = {}  # Track which indices belong to which key
        
        current_idx = 0
        for key in ACTION_CONCAT_ORDER:
            subkey = key.replace("action.", "")
            
            # Get indices from modality.json
            start_idx = action_modality[subkey]["start"]
            end_idx = action_modality[subkey]["end"]
            dim = end_idx - start_idx
            
            # Slice stats by these indices
            key_min = np.array(action_stats["min"])[start_idx:end_idx]
            key_max = np.array(action_stats["max"])[start_idx:end_idx]
            
            self.action_min.extend(key_min.tolist())
            self.action_max.extend(key_max.tolist())
            
            # Track range for this key in the concat output
            self.key_ranges[key] = (current_idx, current_idx + dim)
            current_idx += dim
            
            print(f"  {key}: stats indices [{start_idx}:{end_idx}] -> concat indices [{self.key_ranges[key][0]}:{self.key_ranges[key][1]}]")
            print(f"    min: {key_min[:3].round(4)}... max: {key_max[:3].round(4)}...")
        
        self.action_min = np.array(self.action_min)
        self.action_max = np.array(self.action_max)
        
        print(f"\nFinal denorm arrays shape: min={self.action_min.shape}, max={self.action_max.shape}")
        
        # Identify EE indices (for clipping to [0,1])
        self.ee_indices = []
        for key in ["action.left_ee", "action.right_ee"]:
            start, end = self.key_ranges[key]
            self.ee_indices.extend(range(start, end))
        print(f"EE indices (will be clipped to [0,1]): {self.ee_indices}")
    
    def denormalize(self, normalized_action, clip_ee=True):
        """
        Denormalize action from [-1, 1] to original scale.
        
        Args:
            normalized_action: numpy array of shape (26,) with values in [-1, 1]
            clip_ee: if True, clip EE values to [0, 1] range
        
        Returns:
            Denormalized action in original scale
        """
        normalized_action = np.array(normalized_action)
        
        # Formula: x = (normalized + 1) / 2 * (max - min) + min
        denormalized = (normalized_action + 1) / 2 * (self.action_max - self.action_min) + self.action_min
        
        # Clip EE values to [0, 1] range (they represent gripper positions)
        if clip_ee:
            for idx in self.ee_indices:
                denormalized[idx] = np.clip(denormalized[idx], 0.0, 1.0)
        
        return denormalized
    
    def normalize(self, raw_action):
        """
        Normalize action from original scale to [-1, 1].
        (For debugging/verification purposes)
        
        Args:
            raw_action: numpy array of shape (26,) in original scale
        
        Returns:
            Normalized action in [-1, 1] range
        """
        raw_action = np.array(raw_action)
        
        # Formula: normalized = 2 * (x - min) / (max - min) - 1
        # Handle division by zero for constant values
        range_val = self.action_max - self.action_min
        range_val = np.where(range_val == 0, 1.0, range_val)  # Avoid division by zero
        
        normalized = 2 * (raw_action - self.action_min) / range_val - 1
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized


# -------- FRAME PREPROCESS --------

def resize_with_padding(img, target_size=(256, 256)):
    """Resize image with padding to target size."""
    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_size[0] - new_w
    pad_h = target_size[1] - new_h

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    if padded.ndim == 2:
        padded = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
    elif padded.shape[2] == 4:
        padded = cv2.cvtColor(padded, cv2.COLOR_RGBA2RGB)

    return padded.astype(np.uint8)


# -------- EXTRACT FRAMES --------

def extract_frames(video_path, target_size=(256, 256)):
    """Extract all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_with_padding(frame, target_size)
        frames.append(frame)

    cap.release()
    return frames


# -------- INFER ACTIONS --------

def infer_actions_from_frames(model, frames, device, embodiment_tag, denormalizer=None, siglip_processor=None):
    """
    Infer actions from a sequence of frames using IDM model.
    
    IMPORTANT: Image preprocessing MUST match the training pipeline:
    Training: VideoToTensor -> VideoCrop(0.95, eval=CenterCrop) -> VideoResize(224,224) 
              -> VideoColorJitter(skipped at eval) -> VideoToNumpy -> SigLIP image_processor
    
    Inference: resize_with_padding(256) -> center_crop(0.95) -> SigLIP image_processor
    
    Args:
        model: IDM model
        frames: list of frames (numpy uint8, shape [H,W,3])
        device: torch device
        embodiment_tag: embodiment tag string
        denormalizer: ActionDenormalizer instance (optional, for denormalization)
        siglip_processor: SiglipProcessor instance for image preprocessing
    
    Returns:
        Tuple of (denormalized_actions, normalized_actions)
    """
    actions = []
    normalized_actions = []

    embodiment_id = EMBODIMENT_TAG_MAPPING.get(embodiment_tag, 0)
    embodiment_id_tensor = torch.tensor([embodiment_id], dtype=torch.long, device=device)

    action_dim = model.action_head.action_dim
    action_horizon = model.action_head.action_horizon
    num_inferences = (len(frames) - 1) // action_horizon
    print(f'This has: {len(frames)} frames, so will run {num_inferences} inferences. Each produces {action_horizon} actions.')
    
    # Pre-process all frames: center crop + resize + SigLIP processor
    # This matches training eval pipeline: CenterCrop(95%) -> Resize(224) -> SigLIP normalize
    print(f'Preprocessing frames: resize({TARGET_SIZE}) -> center_crop({CROP_SCALE}) -> resize({CROP_RESIZE}) -> SigLIP processor...')
    processed_frames = []
    for frame in tqdm(frames, desc="Preprocessing", leave=False):
        # 1. Center crop (matches VideoCrop(0.95) at eval time)
        cropped = center_crop_frame(frame, CROP_SCALE)
        # 2. Resize to 224x224 (matches VideoResize(height=224, width=224) in M2DataConfig)
        resized = cv2.resize(cropped, CROP_RESIZE, interpolation=cv2.INTER_LINEAR)
        # 3. SigLIP processor handles final normalization (matches GR00TIDMTransform._prepare_video)
        img_pil = PIL.Image.fromarray(resized)
        processed = siglip_processor.image_processor(images=[img_pil])["pixel_values"]
        processed_frames.append(processed[0])  # shape: [C, H, W]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(frames) - action_horizon, action_horizon), desc="Infer actions", leave=False):
            # Use pre-processed frames (already normalized by SigLIP)
            img1 = torch.from_numpy(processed_frames[i]).float()
            img2 = torch.from_numpy(processed_frames[i + action_horizon]).float()

            # [num_images, C, H, W] - matches training's (t v) flattening with 1 view
            images = torch.stack([img1, img2], dim=0).to(device)
            num_images = images.shape[0]  # 2

            B = 1
            T = 2  # 2 frames

            # view ids (train uses single cam, so view_id=0,1 for frame 0 and 1)
            view_ids = torch.arange(num_images, dtype=torch.long, device=device)

            # dummy actions (flow matching init)
            # horizon=action_horizon, so we need [B, action_horizon, action_dim]
            dummy_actions = torch.zeros(B, action_horizon, action_dim, device=device)

            # embodiment
            cat_ids = embodiment_id_tensor.expand(B)

            # ================= MATCH TRAIN BATCH =================
            # Build token IDs matching GR00TIDMTransform._build_token_ids
            num_visual_tokens_per_frame = 16  # matches GR00TIDMTransform default
            max_sequence_length = 112  # matches GR00TIDMTransform default
            
            # VL token IDs: [IMG_TOKEN * num_visual_tokens_per_frame] * num_images
            _IMG_TOKEN = 1
            _ACT_TOKEN = 4
            _PAD_TOKEN = 0
            
            vl_tokens = [_IMG_TOKEN] * (num_images * num_visual_tokens_per_frame)  # 2 * 16 = 32
            sa_tokens = [_ACT_TOKEN] * action_horizon  # action tokens
            
            # Pad VL tokens to max_sequence_length (left pad)
            vl_seq_len = len(vl_tokens)
            left_pad = max_sequence_length - vl_seq_len
            vl_token_ids = [_PAD_TOKEN] * left_pad + vl_tokens
            vl_attn_mask = [0] * left_pad + [1] * vl_seq_len
            
            sa_token_ids = torch.tensor([sa_tokens], dtype=torch.long, device=device)  # [B, 1]
            vl_token_ids = torch.tensor([vl_token_ids], dtype=torch.long, device=device)  # [B, max_seq_len]
            vl_attention_mask = torch.tensor([vl_attn_mask], dtype=torch.long, device=device)  # [B, max_seq_len]

            inputs = {
                "images": images,
                "view_ids": view_ids,
                "actions": dummy_actions,
                "embodiment_id": embodiment_id_tensor,
                "cat_ids": cat_ids,
                "sa_token_ids": sa_token_ids,
                "vl_token_ids": vl_token_ids,
                "vl_attn_mask": vl_attention_mask,
            }

            # =====================================================

            out = model.get_action(inputs)

            if "action" in out:
                action = out["action"]
            else:
                action = list(out.values())[0]

            # Get normalized action (model output is [B, Horizon, Dim] -> [1, action_horizon, 26])
            normalized_actions_seq = action[0].detach().cpu().numpy()
            
            for normalized_action in normalized_actions_seq:
                normalized_actions.append(normalized_action.tolist())
                
                # Denormalize if denormalizer provided
                if denormalizer is not None:
                    denormalized_action = denormalizer.denormalize(normalized_action, clip_ee=True)
                    actions.append(denormalized_action.tolist())
                else:
                    actions.append(normalized_action.tolist())

    return actions, normalized_actions


def get_video_files(input_path):
    """Get all video files from a folder or return single video path."""
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single video file
        if input_path.suffix.lower() in video_extensions:
            return [input_path]
        else:
            raise ValueError(f"File {input_path} is not a valid video file")
    
    elif input_path.is_dir():
        # Folder containing videos
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.glob(f"*{ext}"))
            video_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        video_files = sorted(set(video_files))
        
        if not video_files:
            raise ValueError(f"No video files found in {input_path}")
        
        return video_files
    
    else:
        raise ValueError(f"Input path {input_path} does not exist")


def process_single_video(model, video_path, output_dir, device, embodiment_tag, denormalizer=None, siglip_processor=None):
    """Process a single video and save actions to JSON."""
    video_path = Path(video_path)
    
    # Extract frames
    frames = extract_frames(str(video_path), TARGET_SIZE)
    
    action_horizon = model.action_head.action_horizon
    if len(frames) <= action_horizon:
        print(f"⚠️ Skipping {video_path.name}: less than {action_horizon + 1} frames")
        return None
    
    # Infer actions
    actions, normalized_actions = infer_actions_from_frames(
        model, frames, device, embodiment_tag, denormalizer, siglip_processor
    )
    
    # Reorder actions from model output order to trajectory.json order
    # Model output order: left_arm[0:7], left_ee[7:13], right_arm[13:20], right_ee[20:26]
    # Target order: left_arm[0:7], right_arm[7:14], left_ee[14:20], right_ee[20:26]
    reordered_actions = []
    for action in actions:
        reordered = (
            action[0:7] +      # left_arm
            action[13:20] +    # right_arm
            action[7:13] +     # left_ee
            action[20:26]      # right_ee
        )
        reordered_actions.append(reordered)
    
    # Prepare output matching the expected format
    output_data = {
        "video_name": video_path.name,
        "num_frames": len(frames),
        "num_actions": len(reordered_actions),
        "action_dim": 26,
        "actions": reordered_actions,  # 26-dim vector: [left_arm(7), right_arm(7), left_ee(6), right_ee(6)]
    }
    
    # Save to JSON
    output_path = Path(output_dir) / f"{video_path.stem}_actions.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    return output_path


# -------- MAIN --------

def main():
    parser = argparse.ArgumentParser(description="Infer actions from videos using trained IDM model")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input video file or folder containing videos"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for action JSON files (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=DEFAULT_MODEL_CONFIG,
        help=f"Path to model config.json (default: {DEFAULT_MODEL_CONFIG})"
    )
    parser.add_argument(
        "--checkpoint", "-ckpt",
        type=str,
        default=DEFAULT_MODEL_CKPT,
        help=f"Path to model.safetensors (default: {DEFAULT_MODEL_CKPT})"
    )
    parser.add_argument(
        "--stats", "-s",
        type=str,
        default=DEFAULT_STATS_FILE,
        help=f"Path to stats.json for denormalization (default: {DEFAULT_STATS_FILE})"
    )
    parser.add_argument(
        "--modality", "-m",
        type=str,
        default=DEFAULT_MODALITY_FILE,
        help=f"Path to modality.json for index mapping (default: {DEFAULT_MODALITY_FILE})"
    )
    parser.add_argument(
        "--embodiment", "-e",
        type=str,
        default=DEFAULT_EMBODIMENT_TAG,
        help=f"Embodiment tag (default: {DEFAULT_EMBODIMENT_TAG})"
    )
    parser.add_argument(
        "--no-denormalize",
        action="store_true",
        help="Skip denormalization (output raw model values in [-1, 1])"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video files
    print(f"🔍 Scanning input: {args.input}")
    video_files = get_video_files(args.input)
    print(f"📹 Found {len(video_files)} video(s)")
    
    # Load model
    print(f"🔧 Loading model from {args.checkpoint}")
    model = load_idm_model(args.config, args.checkpoint, DEVICE)
    print(f"✅ Model loaded (action_dim={model.action_head.action_dim})")
    print(f" Model loaded (action horizon=) {model.action_head.action_horizon}")
    # Load SigLIP processor (CRITICAL: must match training preprocessing)
    siglip_processor = load_siglip_processor()
    print(f"✅ SigLIP processor loaded")
    
    # Load denormalizer (if not skipped)
    denormalizer = None
    if not args.no_denormalize:
        print(f"\n📊 Loading denormalization config:")
        print(f"   Stats: {args.stats}")
        print(f"   Modality: {args.modality}")
        try:
            denormalizer = ActionDenormalizer(args.stats, args.modality)
            print(f"✅ Denormalizer loaded")
        except Exception as e:
            print(f"⚠️ Failed to load denormalizer: {e}")
            print(f"⚠️ Proceeding without denormalization")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Denormalization disabled")
    
    # Process each video
    results = []
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            output_path = process_single_video(
                model, video_path, output_dir, DEVICE, args.embodiment, denormalizer, siglip_processor
            )
            if output_path:
                results.append({
                    "video": str(video_path),
                    "output": str(output_path),
                    "status": "success"
                })
                print(f"✅ {video_path.name} → {output_path.name}")
        except Exception as e:
            results.append({
                "video": str(video_path),
                "output": None,
                "status": f"error: {str(e)}"
            })
            print(f"❌ {video_path.name}: {e}")
    
    # Save summary
    summary_path = output_dir / "summary.json"
    summary = {
        "total_videos": len(video_files),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] != "success"),
        "model_config": args.config,
        "model_checkpoint": args.checkpoint,
        "stats_file": args.stats if not args.no_denormalize else None,
        "modality_file": args.modality if not args.no_denormalize else None,
        "denormalized": not args.no_denormalize,
        "embodiment_tag": args.embodiment,
        "action_structure": {
            "order": ["left_arm", "right_arm", "left_ee", "right_ee"],
            "dims": {
                "left_arm": "indices 0-7 (7 joints)",
                "right_arm": "indices 7-14 (7 joints)", 
                "left_ee": "indices 14-20 (6 gripper values, clipped to [0,1])",
                "right_ee": "indices 20-26 (6 gripper values, clipped to [0,1])",
            },
            "total_dim": 26,
        },
        "results": results
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Processed {summary['successful']}/{summary['total_videos']} videos")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"📋 Summary: {summary_path}")
    if denormalizer:
        print(f"📊 Actions DENORMALIZED using:")
        print(f"   - Stats: {args.stats}")
        print(f"   - Modality: {args.modality}")
        print(f"   - EE values clipped to [0, 1]")
    else:
        print(f"⚠️ Actions NOT denormalized (raw model output in [-1, 1])")
    print(f"\n📐 Action structure (26 dims) - order: left_arm, right_arm, left_ee, right_ee")
    print(f"   [0:7]   left_arm  - 7 joint angles")
    print(f"   [7:14]  right_arm - 7 joint angles")
    print(f"   [14:20] left_ee   - 6 gripper values [0,1]")
    print(f"   [20:26] right_ee  - 6 gripper values [0,1]")


if __name__ == "__main__":
    main()


# ==================================================
# USAGE EXAMPLES
# ==================================================

# Infer with denormalization (recommended):
# python extract_action_video/infer_idm_from_video_v3.py \
#     --input /path/to/video_folder \
#     --output /path/to/output_actions \
#     --config idm/m2_bottle300/checkpoint-9500/config.json \
#     --checkpoint idm/m2_bottle300/checkpoint-9500/model.safetensors \
#     --stats IDM_dump/data/m2_zed/m2_pick.data/meta/stats.json \
#     --modality IDM_dump/data/m2_zed/m2_pick.data/meta/modality.json \
#     --embodiment m2

# Infer WITHOUT denormalization (raw model output):
# python extract_action_video/infer_idm_from_video_v3.py \
#     --input /path/to/video.mp4 \
#     --output /path/to/output_actions \
#     --no-denormalize
