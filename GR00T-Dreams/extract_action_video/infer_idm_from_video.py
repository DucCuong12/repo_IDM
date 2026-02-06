import os
import json
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import safetensors.torch

from gr00t.model.idm import IDM, IDMConfig

# ================= DEFAULT CONFIG =================

DEFAULT_MODEL_CONFIG = "idm/m2_new_update/checkpoint-9500/config.json"
DEFAULT_MODEL_CKPT = "idm/m2_new_update/checkpoint-9500/model.safetensors"
DEFAULT_OUTPUT_DIR = "output_actions"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (256, 256)

# Embodiment
DEFAULT_EMBODIMENT_TAG = "m2"

EMBODIMENT_TAG_MAPPING = {
    "gr1": 24,
    "franka": 17,
    "so100": 26,
    "robocasa_panda_omron": 13,
    "new_embodiment": 31,
    "g1": 21,
    "m2": 26
}

# ========================================


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

def infer_actions_from_frames(model, frames, device, embodiment_tag):
    """Infer actions from a sequence of frames using IDM model."""
    actions = []

    embodiment_id = EMBODIMENT_TAG_MAPPING.get(embodiment_tag, 0)
    embodiment_id_tensor = torch.tensor([embodiment_id], dtype=torch.long, device=device)

    action_dim = model.action_head.action_dim

    with torch.no_grad():
        for i in tqdm(range(len(frames) - 1), desc="Infer actions", leave=False):
            img1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float() / 255.0
            img2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).float() / 255.0

            # [T,C,H,W]
            images = torch.stack([img1, img2], dim=0)

            # [B,T,C,H,W]
            images = images.unsqueeze(0).to(device)
            B, T, C, H, W = images.shape

            # flatten time
            images = images.view(B * T, C, H, W)

            # view ids (train uses single cam)
            view_ids = torch.zeros(B * T, dtype=torch.long, device=device)

            # dummy actions (flow matching init)
            dummy_actions = torch.zeros(B, T, action_dim, device=device)

            # embodiment
            cat_ids = embodiment_id_tensor.expand(B)

            # ================= MATCH TRAIN BATCH =================

            max_txt_len = 32

            sa_token_ids = torch.zeros((B, max_txt_len), dtype=torch.long, device=device)
            vl_token_ids = torch.zeros((B, max_txt_len), dtype=torch.long, device=device)
            vl_attention_mask = torch.zeros((B, max_txt_len), dtype=torch.long, device=device)

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

            actions.append(action[0, -1].detach().cpu().numpy().tolist())

    return actions


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


def process_single_video(model, video_path, output_dir, device, embodiment_tag):
    """Process a single video and save actions to JSON."""
    video_path = Path(video_path)
    
    # Extract frames
    frames = extract_frames(str(video_path), TARGET_SIZE)
    
    if len(frames) < 2:
        print(f"⚠️ Skipping {video_path.name}: less than 2 frames")
        return None
    
    # Infer actions
    actions = infer_actions_from_frames(model, frames, device, embodiment_tag)
    
    # Prepare output
    output_data = {
        "video_name": video_path.name,
        "num_frames": len(frames),
        "num_actions": len(actions),
        "action_dim": len(actions[0]) if actions else 0,
        "actions": actions
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
        "--embodiment", "-e",
        type=str,
        default=DEFAULT_EMBODIMENT_TAG,
        help=f"Embodiment tag (default: {DEFAULT_EMBODIMENT_TAG})"
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
    
    # Process each video
    results = []
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            output_path = process_single_video(
                model, video_path, output_dir, DEVICE, args.embodiment
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
        "embodiment_tag": args.embodiment,
        "results": results
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"✅ Processed {summary['successful']}/{summary['total_videos']} videos")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"📋 Summary: {summary_path}")


if __name__ == "__main__":
    main()
