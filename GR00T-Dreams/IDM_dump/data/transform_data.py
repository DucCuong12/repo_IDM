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
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
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
        output_video, fps=fps, codec="libx264", format="FFMPEG"
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
    """Flatten dict/list to 1D list."""
    out = []
    for v in item.values():
        if isinstance(v, dict):
            out.extend(flatten_state_action(v))
        elif isinstance(v, (list, np.ndarray)):
            out.extend(list(v))
        else:
            out.append(v)
    return out


def build_modality_meta(state_dim, action_dim, num_branches=4):
    """
    Tự động build modality meta từ số chiều state/action thực tế.
    Chia đều cho 4 nhánh: left_arm, right_arm, left_hand, right_hand.
    """
    branch_names = ["left_arm", "right_arm", "left_hand", "right_hand"]

    def split_dim(total, n_branch):
        base = total // n_branch
        rem = total % n_branch
        splits = []
        start = 0
        for i in range(n_branch):
            end = start + base + (1 if i < rem else 0)
            splits.append((start, end))
            start = end
        return splits

    state_splits = split_dim(state_dim, num_branches)
    action_splits = split_dim(action_dim, num_branches)

    state_meta = {branch_names[i]: {"start": s, "end": e, "original_key": "observation.state", "dtype": "float32", "absolute": True, "rotation_type": None} for i, (s, e) in enumerate(state_splits)}
    action_meta = {branch_names[i]: {"start": s, "end": e, "original_key": "action", "dtype": "float32", "absolute": True, "rotation_type": None} for i, (s, e) in enumerate(action_splits)}

    return {
        "state": state_meta,
        "action": action_meta,
        "video": {
            "ego_view": {"original_key": "observation.images.cam_head"}
        },
        "annotation": {
            "human.annotation.task": {"original_key": "task_index"}
        }
    }
def build_info_meta(state_dim, action_dim, num_episodes, total_frames, fps):
    """
    Tự động build info.json meta từ số chiều thực tế.
    """
    return {
        "robot_type": "m2",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 1,
        "chunks_size": 1000,
        "total_chunks": 1,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [f"state_{i}" for i in range(state_dim)]
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": [f"action_{i}" for i in range(action_dim)]
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1]
            },
            "observation.images.cam_head": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False
                }
            }
        }
    }


def build_stats(all_states, all_actions):
    """Build stats.json từ toàn bộ data."""
    state_arr = np.array(all_states, dtype=np.float32)
    action_arr = np.array(all_actions, dtype=np.float32)

    def stat_dict(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }

    return {
        "observation.state": stat_dict(state_arr),
        "action": stat_dict(action_arr),
    }


# ===================== MAIN =====================
if __name__ == "__main__":
    src_root = os.getcwd()
    dst_folder = "m2_unified_output_test.data"

    episodes = sorted([
        f for f in os.listdir(src_root)
        if f.startswith("episode_") and os.path.isdir(f) and len(f.split('_')[-1]) == 4
    ])

    os.makedirs(f"{dst_folder}/data/chunk-000", exist_ok=True)
    os.makedirs(f"{dst_folder}/videos/chunk-000/observation.images.cam_head", exist_ok=True)
    os.makedirs(f"{dst_folder}/meta", exist_ok=True)

    all_states = []
    all_actions = []
    episodes_jsonl = []
    tasks_set = set()

    state_dim = None
    action_dim = None
    fps = 30

    for epi_idx, ep in enumerate(tqdm(episodes, desc="Episodes")):
        src_folder = ep

        # VIDEO
        images_dir = Path(f"{src_folder}/images")
        video_path = f"{dst_folder}/videos/chunk-000/observation.images.cam_head/episode_{epi_idx:06d}.mp4"
        images_to_video(images_dir, video_path)

        # METADATA
        with open(f"{src_folder}/metadata.json") as f:
            metadata = json.load(f)
        fps = metadata.get("fps", 30)

        # TRAJECTORY
        with open(f"{src_folder}/trajectory.json") as f:
            data = json.load(f)

        trajectory = []
        for frame in data["frames"]:
            if isinstance(frame, dict) and "states" in frame and "actions" in frame:
                trajectory.append(frame)

        if not trajectory:
            raise RuntimeError(f"{ep}: empty trajectory")

        states = [flatten_state_action(t["states"]) for t in trajectory]
        actions = [flatten_state_action(t["actions"]) for t in trajectory]

        # Tự động lấy số chiều từ data
        if state_dim is None:
            state_dim = len(states[0])
        if action_dim is None:
            action_dim = len(actions[0])

        all_states.extend(states)
        all_actions.extend(actions)

        num_frames = len(states)
        timestamps = (np.arange(num_frames) / fps).tolist()

        task_name = metadata.get("task", "unknown")
        tasks_set.add(task_name)
        task_index = sorted(list(tasks_set)).index(task_name)

        # PARQUET
        df = pd.DataFrame({
            "observation.state": states,
            "action": actions,
            "timestamp": timestamps,
            "episode_index": [epi_idx] * num_frames,
            "index": list(range(num_frames)),
            "task_index": [task_index] * num_frames,
        })
        pq = f"{dst_folder}/data/chunk-000/episode_{epi_idx:06d}.parquet"
        df.to_parquet(pq)

        episodes_jsonl.append({
            "episode_index": epi_idx,
            "length": num_frames
        })

    # ===================== META =====================
    total_frames = sum([e["length"] for e in episodes_jsonl])

    # modality.json (tự build từ data, không hardcode)
    modality = build_modality_meta(state_dim, action_dim)
    with open(f"{dst_folder}/meta/modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    # info.json (tự build từ data, không hardcode)
    info = build_info_meta(state_dim, action_dim, len(episodes), total_frames, fps)
    with open(f"{dst_folder}/meta/info.json", "w") as f:
        json.dump(info, f, indent=2)

    # episodes.jsonl
    with open(f"{dst_folder}/meta/episodes.jsonl", "w") as f:
        for e in episodes_jsonl:
            f.write(json.dumps(e) + "\n")

    # tasks.jsonl
    with open(f"{dst_folder}/meta/tasks.jsonl", "w") as f:
        for idx, task in enumerate(sorted(list(tasks_set))):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    # stats.json (tự build từ data, không hardcode)
    stats = build_stats(all_states, all_actions)
    with open(f"{dst_folder}/meta/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Converted {len(episodes)} episodes to LeRobot format.")
    print(f"   state_dim={state_dim}, action_dim={action_dim}, total_frames={total_frames}")