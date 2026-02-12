#!/usr/bin/env python3
"""
Script to convert trajectory.json format to concatenated 26-dim action vectors.

Input format (trajectory.json):
{
    "frames": [
        {
            "actions": {
                "left_arm": {"qpos": [7 values]},
                "right_arm": {"qpos": [7 values]},
                "left_ee": {"qpos": [6 values]},
                "right_ee": {"qpos": [6 values]}
            },
            "states": {...}
        },
        ...
    ]
}

Output format:
{
    "video_name": "episode_XXXXXX.mp4",
    "num_frames": N,
    "num_actions": N-1 or N,
    "action_dim": 26,
    "actions": [
        [26 values],  # left_arm(7) + right_arm(7) + left_ee(6) + right_ee(6)
        ...
    ]
}
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional


def extract_action_vector(frame_actions: Dict[str, Any]) -> Optional[List[float]]:
    """
    Extract and concatenate action vector from a frame's actions.
    
    Order: left_arm(7) + right_arm(7) + left_ee(6) + right_ee(6) = 26 dim
    
    Returns None if actions are empty.
    """
    try:
        left_arm = frame_actions.get("left_arm", {}).get("qpos", [])
        right_arm = frame_actions.get("right_arm", {}).get("qpos", [])
        left_ee = frame_actions.get("left_ee", {}).get("qpos", [])
        right_ee = frame_actions.get("right_ee", {}).get("qpos", [])
        
        # Check if all parts have data
        if not left_arm or not right_arm or not left_ee or not right_ee:
            return None
        
        # Validate dimensions
        if len(left_arm) != 7:
            print(f"Warning: left_arm has {len(left_arm)} values, expected 7")
        if len(right_arm) != 7:
            print(f"Warning: right_arm has {len(right_arm)} values, expected 7")
        if len(left_ee) != 6:
            print(f"Warning: left_ee has {len(left_ee)} values, expected 6")
        if len(right_ee) != 6:
            print(f"Warning: right_ee has {len(right_ee)} values, expected 6")
        
        # Concatenate: left_arm + right_arm + left_ee + right_ee
        action_vector = left_arm + right_arm + left_ee + right_ee
        
        return action_vector
    
    except Exception as e:
        print(f"Error extracting action vector: {e}")
        return None


def extract_state_vector(frame_states: Dict[str, Any]) -> Optional[List[float]]:
    """
    Extract and concatenate state vector from a frame's states.
    
    Order: left_arm(7) + right_arm(7) + left_ee(6) + right_ee(6) = 26 dim
    
    Returns None if states are empty.
    """
    try:
        left_arm = frame_states.get("left_arm", {}).get("qpos", [])
        right_arm = frame_states.get("right_arm", {}).get("qpos", [])
        left_ee = frame_states.get("left_ee", {}).get("qpos", [])
        right_ee = frame_states.get("right_ee", {}).get("qpos", [])
        
        # Check if all parts have data
        if not left_arm or not right_arm or not left_ee or not right_ee:
            return None
        
        # Concatenate: left_arm + right_arm + left_ee + right_ee
        state_vector = left_arm + right_arm + left_ee + right_ee
        
        return state_vector
    
    except Exception as e:
        print(f"Error extracting state vector: {e}")
        return None


def convert_trajectory(
    input_path: str,
    output_path: str,
    video_name: Optional[str] = None,
    include_states: bool = False
) -> Dict[str, Any]:
    """
    Convert trajectory.json to concatenated action format.
    
    Args:
        input_path: Path to input trajectory.json
        output_path: Path to output json file
        video_name: Optional video name, defaults to episode name from path
        include_states: Whether to also include states in output
    
    Returns:
        The converted data dictionary
    """
    # Load input file
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    frames = data.get("frames", [])
    
    if not frames:
        raise ValueError("No frames found in trajectory file")
    
    # Extract actions from each frame
    actions = []
    states = []
    
    for i, frame in enumerate(frames):
        frame_actions = frame.get("actions", {})
        action_vector = extract_action_vector(frame_actions)
        
        if action_vector is not None:
            actions.append(action_vector)
        
        if include_states:
            frame_states = frame.get("states", {})
            state_vector = extract_state_vector(frame_states)
            if state_vector is not None:
                states.append(state_vector)
    
    # Determine video name from path if not provided
    if video_name is None:
        input_path_obj = Path(input_path)
        episode_dir = input_path_obj.parent.name
        video_name = f"{episode_dir}.mp4"
    
    # Calculate action dimension
    action_dim = len(actions[0]) if actions else 26
    
    # Build output data
    output_data = {
        "video_name": video_name,
        "num_frames": len(frames),
        "num_actions": len(actions),
        "action_dim": action_dim,
        "actions": actions
    }
    
    if include_states and states:
        output_data["states"] = states
        output_data["num_states"] = len(states)
    
    # Save output
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Converted {len(actions)} actions from {input_path}")
    print(f"Output saved to {output_path}")
    print(f"Action dimension: {action_dim}")
    
    return output_data


def batch_convert(
    input_dir: str,
    output_dir: str,
    include_states: bool = False
) -> None:
    """
    Batch convert all trajectory.json files in a directory.
    
    Args:
        input_dir: Directory containing episode folders with trajectory.json
        output_dir: Directory to save converted files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all trajectory.json files
    trajectory_files = list(input_path.glob("**/trajectory.json"))
    
    if not trajectory_files:
        print(f"No trajectory.json files found in {input_dir}")
        return
    
    print(f"Found {len(trajectory_files)} trajectory files")
    
    for traj_file in sorted(trajectory_files):
        episode_name = traj_file.parent.name
        output_file = output_path / f"{episode_name}_actions.json"
        
        try:
            convert_trajectory(
                str(traj_file),
                str(output_file),
                video_name=f"{episode_name}.mp4",
                include_states=include_states
            )
        except Exception as e:
            print(f"Error converting {traj_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert trajectory.json to 26-dim action vectors"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input trajectory.json file or directory containing episode folders"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output json file or directory"
    )
    
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Batch convert all trajectory.json files in input directory"
    )
    
    parser.add_argument(
        "--video-name", "-v",
        type=str,
        default=None,
        help="Video name for output (single file mode only)"
    )
    
    parser.add_argument(
        "--include-states", "-s",
        action="store_true",
        help="Also include states in output"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        batch_convert(args.input, args.output, args.include_states)
    else:
        convert_trajectory(args.input, args.output, args.video_name, args.include_states)


if __name__ == "__main__":
    main()
