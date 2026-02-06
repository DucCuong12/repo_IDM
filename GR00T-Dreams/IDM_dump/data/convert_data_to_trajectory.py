#!/usr/bin/env python3
"""
Script chuyển đổi từ data.json sang trajectory.json format.
Loại bỏ colors, depths và thêm qvel, torque (mảng rỗng) vào mỗi part.

Usage:
    python convert_data_to_trajectory.py --input <path/to/data.json> --output <path/to/trajectory.json>
    
Hoặc chuyển toàn bộ folder:
    python convert_data_to_trajectory.py --input_dir <folder_chứa_episodes> --recursive
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def convert_frame(frame_data: dict) -> dict:
    """
    Chuyển đổi một frame từ data.json format sang trajectory.json format.
    - Loại bỏ: colors, depths, idx
    - Giữ lại: states, actions
    - Thêm: qvel=[], torque=[] vào mỗi part trong states và actions
    """
    result = {}
    
    # Process actions
    if "actions" in frame_data:
        result["actions"] = {}
        for part_name, part_data in frame_data["actions"].items():
            result["actions"][part_name] = {
                "qpos": part_data.get("qpos", []),
                "qvel": [],
                "torque": []
            }
    
    # Process states
    if "states" in frame_data:
        result["states"] = {}
        for part_name, part_data in frame_data["states"].items():
            result["states"][part_name] = {
                "qpos": part_data.get("qpos", []),
                "qvel": [],
                "torque": []
            }
    
    return result


def convert_data_to_trajectory(data_json: dict) -> dict:
    """
    Chuyển đổi toàn bộ data.json sang trajectory.json format.
    """
    frames = []
    
    # data.json có key "data" chứa list các frame
    data_list = data_json.get("data", [])
    
    for frame_data in data_list:
        converted_frame = convert_frame(frame_data)
        frames.append(converted_frame)
    
    return {"frames": frames}


def process_single_file(input_path: Path, output_path: Path):
    """Process một file data.json -> trajectory.json"""
    print(f"Converting: {input_path}")
    
    with open(input_path, 'r') as f:
        data_json = json.load(f)
    
    trajectory_json = convert_data_to_trajectory(data_json)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(trajectory_json, f, indent=4)
    
    print(f"  -> Saved to: {output_path}")
    print(f"  -> Total frames: {len(trajectory_json['frames'])}")


def process_directory(input_dir: Path, recursive: bool = False):
    """
    Process toàn bộ directory chứa episodes.
    Tìm tất cả data.json và chuyển thành trajectory.json trong cùng episode folder.
    """
    pattern = "**/colors/data.json" if recursive else "*/colors/data.json"
    data_files = list(input_dir.glob(pattern))
    
    if not data_files:
        print(f"No data.json files found in {input_dir}")
        return
    
    print(f"Found {len(data_files)} data.json files")
    
    for data_file in tqdm(data_files, desc="Converting"):
        # data.json ở trong colors/, trajectory.json ở episode folder
        episode_dir = data_file.parent.parent  # colors/ -> episode_XXXX/
        output_path = episode_dir / "trajectory.json"
        
        try:
            process_single_file(data_file, output_path)
        except Exception as e:
            print(f"Error processing {data_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert data.json to trajectory.json format')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to single data.json file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for trajectory.json (only used with --input)')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory containing episode folders with colors/data.json')
    parser.add_argument('--recursive', action='store_true',
                        help='Search recursively in input_dir')
    
    args = parser.parse_args()
    
    if args.input:
        # Single file mode
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return
        
        if args.output:
            output_path = Path(args.output)
        else:
            # Default: put trajectory.json in same folder as data.json's parent's parent
            output_path = input_path.parent.parent / "trajectory.json"
        
        process_single_file(input_path, output_path)
        
    elif args.input_dir:
        # Directory mode
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            return
        
        process_directory(input_dir, recursive=args.recursive)
        
    else:
        print("Error: Please provide either --input or --input_dir")
        parser.print_help()


if __name__ == '__main__':
    main()
