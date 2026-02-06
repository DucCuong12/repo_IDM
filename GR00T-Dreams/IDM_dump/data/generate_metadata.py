#!/usr/bin/env python3
"""
Script tạo metadata.json cho các episode trong m2_zed từ data.json.

Usage:
    python generate_metadata.py --input_dir m2_zed --task_goal "Pick the bottle"
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def generate_metadata_from_data_json(data_json_path: str, task_goal: str = "Pick the bottle", target_fps: float = 30.0) -> dict:
    """
    Đọc data.json và tạo metadata.json.
    """
    with open(data_json_path, 'r') as f:
        data = json.load(f)
    
    frame_count = len(data.get("data", []))
    
    # Ước tính duration từ frame_count và target_fps
    duration_seconds = frame_count / target_fps if target_fps > 0 else 0
    
    metadata = {
        "actual_fps": target_fps,  # Không có thông tin actual fps, dùng target
        "duration_seconds": duration_seconds,
        "frame_count": frame_count,
        "target_frequency": target_fps,
        "task_goal": task_goal,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return metadata


def process_directory(input_dir: str, task_goal: str = "Pick the bottle", target_fps: float = 30.0):
    """
    Process toàn bộ directory chứa episodes.
    Tìm tất cả colors/data.json và tạo metadata.json trong episode folder.
    """
    input_path = Path(input_dir)
    
    # Tìm tất cả episode folders
    episode_dirs = sorted([
        d for d in input_path.iterdir() 
        if d.is_dir() and d.name.startswith("episode_")
    ])
    
    if not episode_dirs:
        print(f"Không tìm thấy episode folders trong {input_dir}")
        return
    
    print(f"Tìm thấy {len(episode_dirs)} episodes")
    
    success_count = 0
    for episode_dir in tqdm(episode_dirs, desc="Generating metadata"):
        data_json_path = episode_dir / "colors" / "data.json"
        metadata_path = episode_dir / "metadata.json"
        
        if not data_json_path.exists():
            print(f"  Skip {episode_dir.name}: không có colors/data.json")
            continue
        
        try:
            metadata = generate_metadata_from_data_json(
                str(data_json_path), 
                task_goal=task_goal,
                target_fps=target_fps
            )
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            success_count += 1
        except Exception as e:
            print(f"  Error {episode_dir.name}: {e}")
    
    print(f"\n✅ Đã tạo metadata cho {success_count}/{len(episode_dirs)} episodes")


def main():
    parser = argparse.ArgumentParser(description='Generate metadata.json for episodes from data.json')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory chứa các episode folders (vd: m2_zed)')
    parser.add_argument('--task_goal', type=str, default="Pick the bottle",
                        help='Task goal cho tất cả episodes')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Target FPS (default: 30)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Directory không tồn tại: {args.input_dir}")
        return
    
    process_directory(args.input_dir, args.task_goal, args.fps)


if __name__ == '__main__':
    main()
