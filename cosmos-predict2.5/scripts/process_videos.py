#!/usr/bin/env python3
import os
import pandas as pd
import subprocess
from pathlib import Path

def ensure_h264(input_video, output_video):
    """Convert video to H.264 format if it's not already in that format."""
    # Check if video is already in H.264
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_video
    ]
    
    try:
        codec = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8').strip()
        if codec == 'h264':
            # If already H.264, just copy the file
            subprocess.run(['cp', input_video, output_video], check=True)
            return True
    except subprocess.CalledProcessError:
        pass
    
    # Convert to H.264 using ffmpeg
    cmd = [
        'ffmpeg',
        '-i', input_video,
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '23',
        '-c:a', 'aac',
        '-strict', 'experimental',
        output_video,
        '-y'  # Overwrite output file if it exists
    ]
    
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_video} to H.264: {e}")
        return False

def process_dataset(csv_path, root_dir):
    """Process the dataset based on the CSV file."""
    # Create necessary directories
    videos_dir = os.path.join(root_dir, 'videos')
    metas_dir = os.path.join(root_dir, 'metas')
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(metas_dir, exist_ok=True)
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Ensure required columns exist
    required_columns = ['video_name', 'description']
    if not all(col in df.columns for col in required_columns):
        print("Error: CSV must contain 'video_name' and 'description' columns")
        return
    
    # Process each video
    for _, row in df.iterrows():
        video_name = str(row['video_name']).strip()
        description = str(row['description']).strip()
        
        # Skip if video name or description is empty
        if not video_name or not description:
            print(f"Skipping row with empty video name or description: {row}")
            continue
        
        # Ensure video has .mp4 extension
        video_base = os.path.splitext(video_name)[0]
        input_video = video_name if video_name.endswith('.mp4') else f"{video_name}.mp4"
        output_video = os.path.join(videos_dir, f"{video_base}.mp4")
        
        # Process video (convert to H.264 if needed)
        if os.path.exists(input_video):
            if ensure_h264(input_video, output_video):
                print(f"Processed video: {input_video} -> {output_video}")
            else:
                print(f"Failed to process video: {input_video}")
                continue
        else:
            print(f"Video not found: {input_video}")
            continue
        
        # Create corresponding text file
        text_file = os.path.join(metas_dir, f"{video_base}.txt")
        with open(text_file, 'w') as f:
            f.write(description)
        print(f"Created text file: {text_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process video dataset with descriptions.')
    parser.add_argument('--csv', required=True, help='Path to CSV file with video names and descriptions')
    parser.add_argument('--output', default='./output', help='Output root directory (default: ./output)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Processing dataset from {args.csv}")
    print(f"Output will be saved to: {args.output}")
    print("-" * 50)
    
    process_dataset(args.csv, args.output)
    
    print("\nProcessing complete!")
