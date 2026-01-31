#!/usr/bin/env python3
"""
Convert episodes folder to unified LeRobot format.

Input:
- episodes_folder/
  - episode_0000/data.json + colors/
  - episode_0001/data.json + colors/
  - ...

Output (LeRobot standard):
- output_dir/
  - data/chunk-000/
    - episode_000000.parquet
    - episode_000001.parquet
    - observation.images.cam_head/episode_000000.mp4, episode_000001.mp4
  - meta/modality.json, stats.json, episodes.jsonl, tasks.jsonl
  - info.json
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


def load_episode_data(episode_path):
    """Load data from episode JSON."""
    data_file = os.path.join(episode_path, "data.json")
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data


def extract_video_frames(episode_path):
    """Extract video frames from colors directory."""
    colors_dir = os.path.join(episode_path, "colors")
    frames = []
    frame_files = sorted([f for f in os.listdir(colors_dir) if f.endswith('.jpg')])
    
    for frame_file in frame_files:
        frame_path = os.path.join(colors_dir, frame_file)
        frame = Image.open(frame_path)
        frames.append(np.array(frame))
    
    return frames


def extract_state_action_pairs(data):
    """Extract state-action pairs from data.json."""
    states = []
    actions = []
    
    for frame_data in data['data']:
        # Get state components dynamically from data
        state_parts = []
        for joint_group in frame_data['states'].values():
            if isinstance(joint_group, dict) and 'qpos' in joint_group:
                state_parts.extend(joint_group['qpos'])
        
        state_vector = state_parts
        
        # Get action components dynamically from data
        action_parts = []
        for joint_group in frame_data['actions'].values():
            if isinstance(joint_group, dict) and 'qpos' in joint_group:
                action_parts.extend(joint_group['qpos'])
        
        action_vector = action_parts
        
        states.append(state_vector)
        actions.append(action_vector)
    
    return np.array(states), np.array(actions)


def main():
    parser = argparse.ArgumentParser(description="Convert episodes folder to LeRobot format")
    parser.add_argument("--episodes_dir", required=True, help="Path to folder containing episodes")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--embodiment", default="m2", help="Embodiment type")
    
    args = parser.parse_args()
    
    episodes_path = Path(args.episodes_dir)
    output_path = Path(args.output_dir)
    
    # Create directory structure (matching your format)
    data_dir = output_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    video_dir = output_path / "videos" / "chunk-000" / "observation.images.cam_head"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all episode folders
    episode_folders = sorted([d for d in episodes_path.iterdir() 
                             if d.is_dir() and (d / "data.json").exists()])
    
    if not episode_folders:
        print(f"❌ No episode folders found in {args.episodes_dir}")
        return
    
    print(f"Found {len(episode_folders)} episodes")
    
    # Process each episode
    all_episodes_info = []
    total_frames = 0
    first_fps = None
    
    for episode_idx, episode_folder in enumerate(tqdm(episode_folders, desc="Processing episodes")):
        try:
            # Load episode data
            data = load_episode_data(str(episode_folder))
            frames = extract_video_frames(str(episode_folder))
            states, actions = extract_state_action_pairs(data)
            
            num_frames = len(frames)
            fps = data['info']['image']['fps']
            
            if first_fps is None:
                first_fps = fps
            
            # Save video
            video_path = video_dir / f"episode_{episode_idx:06d}.mp4"
            writer = imageio.get_writer(str(video_path), fps=fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            
            # Create Parquet file
            timestamps = np.arange(num_frames) / fps
            df = pd.DataFrame({
                'observation.state': [states[i].tolist() for i in range(num_frames)],
                'action': [actions[i].tolist() for i in range(num_frames)],
                'timestamp': timestamps.tolist(),
                'episode_index': [episode_idx] * num_frames,
                'index': list(range(num_frames)),
                'task_index': [episode_idx] * num_frames,
                'annotation.human.annotation.task': [data['text']['goal']] * num_frames,
            })
            
            parquet_path = data_dir / f"episode_{episode_idx:06d}.parquet"
            df.to_parquet(str(parquet_path))
            
            all_episodes_info.append({
                'episode_index': episode_idx,
                'task': data['text']['goal'],
                'task_description': data['text']['desc'],
            })
            
            total_frames += num_frames
            
        except Exception as e:
            print(f"⚠️  SKIP {episode_folder.name}: {str(e)[:100]}")
            continue
    
    print(f"✅ Processed {len(all_episodes_info)} episodes, {total_frames} total frames")
    
    # Get state/action dimensions từ data thực tế
    if all_episodes_info:
        first_episode_path = episode_folders[0]
        first_data = load_episode_data(str(first_episode_path))
        _, _ = extract_state_action_pairs(first_data)
        
        # Calculate state/action dimensions
        first_frame = first_data['data'][0]
        state_dim = 0
        action_dim = 0
        state_names = []
        action_names = []
        
        for key in sorted(first_frame['states'].keys()):
            if isinstance(first_frame['states'][key], dict) and 'qpos' in first_frame['states'][key]:
                qpos_len = len(first_frame['states'][key]['qpos'])
                state_dim += qpos_len
                for i in range(qpos_len):
                    state_names.append(f"{key}_{i}")
        
        for key in sorted(first_frame['actions'].keys()):
            if isinstance(first_frame['actions'][key], dict) and 'qpos' in first_frame['actions'][key]:
                qpos_len = len(first_frame['actions'][key]['qpos'])
                action_dim += qpos_len
                for i in range(qpos_len):
                    action_names.append(f"{key}_{i}")
    else:
        state_dim = 44
        action_dim = 44
        state_names = [f"state_{i}" for i in range(44)]
        action_names = [f"action_{i}" for i in range(44)]
    
    # Create modality.json từ joint_names trong data
    modality = {
        "state": {},
        "action": {},
        "video": {
            "cam_head": {"original_key": "observation.images.cam_head"}
        },
        "annotation": {
            "human.annotation.task": {"original_key": "task_index"}
        }
    }
    
    # Get joint structure từ data.json
    if all_episodes_info:
        first_data = load_episode_data(str(episode_folders[0]))
        joint_names = first_data['info'].get('joint_names', {})
        
        start_idx = 0
        for joint_group in sorted(joint_names.keys()):
            names = joint_names[joint_group]
            if names:
                end_idx = start_idx + len(names)
                modality['state'][joint_group] = {"start": start_idx, "end": end_idx}
                modality['action'][joint_group] = {"start": start_idx, "end": end_idx}
                start_idx = end_idx
    
    # Save modality.json
    with open(meta_dir / "modality.json", 'w') as f:
        json.dump(modality, f, indent=2)
    
    with open(meta_dir / "episodes.jsonl", 'w') as f:
        for idx in range(len(all_episodes_info)):
            episode_info = {
                "episode_index": idx,
                "tasks_index": idx,
                "dataset_index": 0,
                "chunk_name": "chunk-000",
            }
            f.write(json.dumps(episode_info) + "\n")
    
    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", 'w') as f:
        for info in all_episodes_info:
            task_info = {
                "task_index": info['episode_index'],
                "task": info['task'],
                "task_description": info['task_description'],
            }
            f.write(json.dumps(task_info) + "\n")
    
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
    
    # info.json
    info = {
        "robot_type": args.embodiment,
        "total_episodes": len(all_episodes_info),
        "total_frames": total_frames,
        "total_tasks": len(all_episodes_info),
        "total_videos": len(all_episodes_info),
        "chunks_size": 1000,
        "total_chunks": 1,
        "fps": first_fps or 30.0,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": state_names
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": action_names
            },
            "observation.images.cam_head": {
                "dtype": "uint8",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"]
            }
        }
    }
    
    with open(output_path / "info.json", 'w') as f:
        json.dump(info, f, indent=2)
    


    # stats.json - lấy đúng dimensions

    all_states = []
    all_actions = []

    for episode_idx, episode_folder in enumerate(episode_folders):
        data = load_episode_data(str(episode_folder))
        states, actions = extract_state_action_pairs(data)
        all_states.extend(states)
        all_actions.extend(actions)

    all_states = np.array(all_states)
    all_actions = np.array(all_actions)

    stats = {
        "state": {
            "mean": all_states.mean(axis=0).tolist(),
            "std": all_states.std(axis=0).tolist(),
            "min": all_states.min(axis=0).tolist(),
            "max": all_states.max(axis=0).tolist()
        },
        "action": {
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
            "min": all_actions.min(axis=0).tolist(),
            "max": all_actions.max(axis=0).tolist()
        }
    }

   
    with open(meta_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ LeRobot dataset created at: {args.output_dir}")
    print(f"   Episodes: {len(all_episodes_info)}")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {first_fps or 30.0}")


if __name__ == "__main__":
    main()
