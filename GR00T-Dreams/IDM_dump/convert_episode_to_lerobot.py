#!/usr/bin/env python3
"""
Convert episode_0000 format (teleop data) to LeRobot format.

Input format:
- episode_0000/data.json: Contains states, actions, text descriptions
- episode_0000/colors/: Video frames

Output format:
- LeRobot Parquet + video structure
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import imageio
import argparse
from tqdm import tqdm
from datetime import datetime


def load_episode_data(episode_path):
    """Load data from episode JSON."""
    data_file = os.path.join(episode_path, "data.json")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    return data


def extract_video_frames(episode_path, color_key="color_0"):
    """Extract video frames from colors directory."""
    colors_dir = os.path.join(episode_path, "colors")
    
    frames = []
    frame_files = sorted([f for f in os.listdir(colors_dir) if f.endswith('.jpg')])
    
    for frame_file in tqdm(frame_files, desc="Loading frames"):
        frame_path = os.path.join(colors_dir, frame_file)
        frame = Image.open(frame_path)
        frames.append(np.array(frame))
    
    return frames


def extract_state_action_pairs(data):
    """Extract state-action pairs from episode data."""
    states = []
    actions = []
    
    for frame_data in data['data']:
        # Extract left_arm state (7 DOF)
        left_arm_state = frame_data['states']['left_arm']['qpos']
        
        # Extract right_arm state (7 DOF)
        right_arm_state = frame_data['states']['right_arm']['qpos']
        
        # Extract left_ee state (6 DOF - gripper)
        left_ee_state = frame_data['states']['left_ee']['qpos']
        
        # Extract right_ee state (6 DOF - gripper)
        right_ee_state = frame_data['states']['right_ee']['qpos']
        
        # Combine into 44-dim state vector (M2: 7+7+6+6 + 18 padding = 44)
        state_vector = (
            left_arm_state +           # [0:7]
            right_arm_state +          # [7:14]
            left_ee_state +            # [14:20]
            right_ee_state             # [20:26]
        )
        # Pad to 44 dims
        state_vector = list(state_vector) + [0.0] * (44 - len(state_vector))
        
        # Extract actions similarly
        left_arm_action = frame_data['actions']['left_arm']['qpos']
        right_arm_action = frame_data['actions']['right_arm']['qpos']
        left_ee_action = frame_data['actions']['left_ee']['qpos']
        right_ee_action = frame_data['actions']['right_ee']['qpos']
        
        action_vector = (
            left_arm_action +
            right_arm_action +
            left_ee_action +
            right_ee_action
        )
        # Pad to 44 dims
        action_vector = list(action_vector) + [0.0] * (44 - len(action_vector))
        
        states.append(state_vector)
        actions.append(action_vector)
    
    return np.array(states), np.array(actions)


def create_lerobot_structure(episode_path, output_dir, embodiment="m2"):
    """Create full LeRobot dataset structure."""
    # Load data
    print("Loading episode data...")
    data = load_episode_data(episode_path)
    
    # Extract frames
    print("Extracting frames...")
    frames = extract_video_frames(episode_path)
    
    # Extract states and actions
    print("Extracting states and actions...")
    states, actions = extract_state_action_pairs(data)
    
    num_frames = len(frames)
    print(f"Total frames: {num_frames}")
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_dir = output_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    video_dir = data_dir / "observation.images.cam_head"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Save video as mp4
    print("Saving video...")
    video_path = video_dir / "episode_000000.mp4"
    writer = imageio.get_writer(str(video_path), fps=30.0)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
    # Create Parquet file
    print("Creating Parquet file...")
    fps = data['info']['image']['fps']
    timestamps = np.arange(num_frames) / fps
    
    df = pd.DataFrame({
        'observation.state': [states[i].tolist() for i in range(num_frames)],
        'action': [actions[i].tolist() for i in range(num_frames)],
        'timestamp': timestamps.tolist(),
        'episode_index': [0] * num_frames,
        'index': list(range(num_frames)),
        'task_index': [0] * num_frames,
        'annotation.human.annotation.task': [data['text']['goal']] * num_frames,
    })
    
    parquet_path = data_dir / "episode_000000.parquet"
    df.to_parquet(str(parquet_path))
    print(f"Parquet: {num_frames} rows saved")
    
    # Create metadata files
    print("Creating metadata files...")
    
    # modality.json
    modality = {
        "state": {
            "left_arm": {"start": 0, "end": 7},
            "right_arm": {"start": 7, "end": 14},
            "left_hand": {"start": 14, "end": 20},
            "right_hand": {"start": 20, "end": 26},
        },
        "action": {
            "left_arm": {"start": 0, "end": 7},
            "right_arm": {"start": 7, "end": 14},
            "left_hand": {"start": 14, "end": 20},
            "right_hand": {"start": 20, "end": 26},
        },
        "video": {
            "cam_head": {"original_key": "observation.images.cam_head"}
        },
        "annotation": {
            "human.annotation.task": {"original_key": "task_index"}
        }
    }
    
    with open(meta_dir / "modality.json", 'w') as f:
        json.dump(modality, f, indent=2)
    
    # episodes.jsonl - in meta folder
    with open(meta_dir / "episodes.jsonl", 'w') as f:
        episode_info = {
            "episode_index": 0,
            "tasks_index": 0,
            "dataset_index": 0,
            "chunk_name": "chunk-000",
        }
        f.write(json.dumps(episode_info) + "\n")
    
    # tasks.jsonl - in meta folder
    with open(meta_dir / "tasks.jsonl", 'w') as f:
        task_info = {
            "task_index": 0,
            "task": data['text']['goal'],
            "task_description": data['text']['desc'],
        }
        f.write(json.dumps(task_info) + "\n")
    
    # info.json - get image size from actual frames
    first_frame = frames[0]
    height, width = first_frame.shape[:2]
    
    info = {
        "robot_type": "m2",
        "total_episodes": 1,
        "total_frames": num_frames,
        "total_tasks": 1,
        "total_videos": 1,
        "chunks_size": 1000,
        "total_chunks": 1,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "data/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [44],
                "names": [
                    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",
                    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
                    "left_ee_0", "left_ee_1", "left_ee_2", "left_ee_3", "left_ee_4", "left_ee_5",
                    "right_ee_0", "right_ee_1", "right_ee_2", "right_ee_3", "right_ee_4", "right_ee_5",
                    "padding_0", "padding_1", "padding_2", "padding_3", "padding_4", "padding_5", 
                    "padding_6", "padding_7", "padding_8", "padding_9", "padding_10", "padding_11",
                    "padding_12", "padding_13", "padding_14", "padding_15", "padding_16", "padding_17"
                ]
            },
            "action": {
                "dtype": "float32",
                "shape": [44],
                "names": [
                    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",
                    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
                    "left_ee_0", "left_ee_1", "left_ee_2", "left_ee_3", "left_ee_4", "left_ee_5",
                    "right_ee_0", "right_ee_1", "right_ee_2", "right_ee_3", "right_ee_4", "right_ee_5",
                    "padding_0", "padding_1", "padding_2", "padding_3", "padding_4", "padding_5",
                    "padding_6", "padding_7", "padding_8", "padding_9", "padding_10", "padding_11",
                    "padding_12", "padding_13", "padding_14", "padding_15", "padding_16", "padding_17"
                ]
            },
            "observation.images.cam_head": {
                "dtype": "uint8",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"]
            }
        }
    }

    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    # stats.json - calculate actual statistics from data
    state_array = states  # shape: (num_frames, 44)
    action_array = actions  # shape: (num_frames, 44)
    
    stats = {
        "state": {
            "mean": state_array.mean(axis=0).tolist(),
            "std": state_array.std(axis=0).tolist(),
            "min": state_array.min(axis=0).tolist(),
            "max": state_array.max(axis=0).tolist()
        },
        "action": {
            "mean": action_array.mean(axis=0).tolist(),
            "std": action_array.std(axis=0).tolist(),
            "min": action_array.min(axis=0).tolist(),
            "max": action_array.max(axis=0).tolist()
        }
    }
    with open(meta_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ LeRobot dataset created at: {output_dir}")
    print(f"   - Frames: {num_frames}")
    print(f"   - FPS: {fps}")
    print(f"   - Task: {data['text']['goal']}")


def main():
    parser = argparse.ArgumentParser(description="Convert episode_0000 to LeRobot format")
    parser.add_argument("--episode_path", default="episode_0000", help="Path to episode_0000 folder")
    parser.add_argument("--output_dir", default="IDM_dump/data/episode_0000_lerobot.data", help="Output directory")
    parser.add_argument("--embodiment", default="m2", help="Embodiment type")
    
    args = parser.parse_args()
    
    create_lerobot_structure(args.episode_path, args.output_dir, args.embodiment)


if __name__ == "__main__":
    main()
