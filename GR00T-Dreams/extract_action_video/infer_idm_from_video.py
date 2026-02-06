import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
import safetensors.torch

from gr00t.model.idm import IDM, IDMConfig

# ================= CONFIG =================

VIDEO_PATH = "IDM_dump/data/m2_unified.data/videos/chunk-000/observation.images.cam_head/episode_000001.mp4"
# /media/aitv/32202CEA64082D5F/cuong/repo_IDM/GR00T-Dreams/idm/m2_checkkkk/checkpoint-9500
MODEL_CONFIG = "idm/m2_new_update/checkpoint-9500/config.json"
MODEL_CKPT = "idm/m2_new_update/checkpoint-9500/model.safetensors"
OUTPUT_JSON = "output_trajectory1.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (256, 256)

# Embodiment
EMBODIMENT_TAG = "m2"

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
def infer_actions_from_frames(model, frames, device):
    actions = []

    embodiment_id = EMBODIMENT_TAG_MAPPING.get(EMBODIMENT_TAG, 0)
    embodiment_id_tensor = torch.tensor([embodiment_id], dtype=torch.long, device=device)

    action_dim = model.action_head.action_dim

    with torch.no_grad():
        for i in tqdm(range(len(frames) - 1), desc="Infer actions"):
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



# -------- MAIN --------

def main():
    frames = extract_frames(VIDEO_PATH, TARGET_SIZE)
    print(f"Extracted {len(frames)} frames")

    model = load_idm_model(MODEL_CONFIG, MODEL_CKPT, DEVICE)

    actions = infer_actions_from_frames(model, frames, DEVICE)

    with open(OUTPUT_JSON, "w") as f:
        json.dump({"actions": actions}, f, indent=2)

    abs_path = os.path.abspath(OUTPUT_JSON)
    print(f"Saved trajectory to {abs_path}")


if __name__ == "__main__":
    main()
