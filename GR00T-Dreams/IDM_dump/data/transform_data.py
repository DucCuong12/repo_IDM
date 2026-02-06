import os
import re
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import imageio.v2 as imageio

def natural_sort(l):
    return sorted(l, key=lambda x: [
        int(t) if t.isdigit() else t
        for t in re.findall(r'\d+|\D+', x)
    ])
def resize_with_padding(img, target_size=(256, 256)):

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    h, w = img.shape[:2]

    scale = min(target_size[1] / w, target_size[0] / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_size[1] - new_w
    pad_h = target_size[0] - new_h

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
        value=(0, 0, 0),
    )

    assert padded.shape == (target_size[0], target_size[1], 3)

    return padded.astype(np.uint8)

def images_to_video(images_dir, output_video, fps=30, target_size=(256, 256)):

    image_files = natural_sort([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])

    Path(os.path.dirname(output_video)).mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        output_video,
        fps=fps,
        codec="libx264",
        format="FFMPEG"
    )

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)

        frame = imageio.imread(img_path)
        frame = resize_with_padding(frame, target_size)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        writer.append_data(frame)

    writer.close()
    print("✅ Video saved:", output_video)

def flatten_state_action(item):
    out = []
    for v in item.values():
        if isinstance(v, dict):
            out.extend(flatten_state_action(v))
        elif isinstance(v, list):
            out.extend(v)
        else:
            out.append(v)
    return out
src_root = os.getcwd()     # chứa episode_0000, episode_0001...
dst_folder = "resore.data"

episodes = sorted([
    f for f in os.listdir(src_root)
    if f.startswith("episode_") and os.path.isdir(f) and len(f.split('_')[-1])==4
])

os.makedirs(f"{dst_folder}/data/chunk-000", exist_ok=True)
os.makedirs(f"{dst_folder}/videos/chunk-000/observation.images.cam_head", exist_ok=True)
os.makedirs(f"{dst_folder}/meta", exist_ok=True)

all_states = []
all_actions = []
episodes_jsonl = []

metadata_ref = None

for epi_idx, ep in enumerate(tqdm(episodes, desc="Episodes")):

    src_folder = ep

    ################ VIDEO ################

    images_dir = Path(f"{src_folder}/images")
    video_path = f"{dst_folder}/videos/chunk-000/observation.images.cam_head/episode_{epi_idx:06d}.mp4"

    images_to_video(images_dir, video_path)

    ################ METADATA ################

    with open(f"{src_folder}/metadata.json") as f:
        metadata = json.load(f)

    if metadata_ref is None:
        metadata_ref = metadata

    ################ TRAJECTORY ################

    with open(f"{src_folder}/trajectory.json") as f:
        data = json.load(f)

    trajectory = []
    for frame in data["frames"]:
        if isinstance(frame,dict) and "states" in frame and "actions" in frame:
            trajectory.append(frame)

    if not trajectory:
        raise RuntimeError(f"{ep}: empty trajectory")

    states = [flatten_state_action(t["states"]) for t in trajectory]
    actions = [flatten_state_action(t["actions"]) for t in trajectory]

    all_states.extend(states)
    all_actions.extend(actions)

    num_frames = len(states)
    fps = metadata.get("fps",30)
    timestamps = np.arange(num_frames)/fps

    ################ PARQUET ################

    df = pd.DataFrame({
        "observation.state": states,
        "action": actions,
        "timestamp": timestamps.tolist(),
        "episode_index": [epi_idx]*num_frames,
        "index": list(range(num_frames)),
        "task_index": [0]*num_frames,
        "annotation.human.annotation.task": [metadata.get("task","unknown")]*num_frames,
    })

    pq = f"{dst_folder}/data/chunk-000/episode_{epi_idx:06d}.parquet"
    df.to_parquet(pq)

    episodes_jsonl.append({
        "episode_id": f"{epi_idx:06d}",
        "length": num_frames
    })

######################## META ########################

# modality.json (y chang code gốc)

modality = {
    "video": ["observation.images.cam_head"],
    "state": list(metadata_ref.get("state_keys", [])),
    "action": list(metadata_ref.get("action_keys", [])),
    "language": list(metadata_ref.get("language_keys", []))
}

with open(f"{dst_folder}/meta/modality.json","w") as f:
    json.dump(modality,f,indent=2)

# info.json

info = {
    "embodiment_tag":"m2",
    "num_episodes": len(episodes),
    "other_info": metadata_ref.get("other_info",{})
}

with open(f"{dst_folder}/meta/info.json","w") as f:
    json.dump(info,f,indent=2)

# episodes.jsonl

with open(f"{dst_folder}/meta/episodes.jsonl","w") as f:
    for e in episodes_jsonl:
        f.write(json.dumps(e)+"\n")

# tasks.jsonl

if "tasks" in metadata_ref:
    with open(f"{dst_folder}/meta/tasks.jsonl","w") as f:
        for t in metadata_ref["tasks"]:
            f.write(json.dumps(t)+"\n")

# stats.json (GLOBAL giống single version)

state_array = np.array(all_states)
action_array = np.array(all_actions)

stats = {
    "state":{
        "mean": state_array.mean(0).tolist(),
        "std": state_array.std(0).tolist(),
        "min": state_array.min(0).tolist(),
        "max": state_array.max(0).tolist()
    },
    "action":{
        "mean": action_array.mean(0).tolist(),
        "std": action_array.std(0).tolist(),
        "min": action_array.min(0).tolist(),
        "max": action_array.max(0).tolist()
    }
}

with open(f"{dst_folder}/meta/stats.json","w") as f:
    json.dump(stats,f,indent=2)

print("✅ Converted multi-episode dataset to LeRobot format.")