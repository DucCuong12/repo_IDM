#!/bin/bash

# Input directory
INPUT_DIR="/mnt/ssd/project/data-pipeline/GR00T-Dreams/data/m1_pick_and_place/videos/chunk-000/observation.images.cam_head"
# Output directory
OUTPUT_DIR="/mnt/ssd/project/data-pipeline/GR00T-Dreams/data/m1_pick_and_place/h264"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Change to the input directory
cd "$INPUT_DIR" || { echo "Failed to change to directory: $INPUT_DIR"; exit 1; }

# Process each video file
for video_file in *.mp4; do
    # Skip if no files found
    [ -e "$video_file" ] || continue
    
    # Output file path
    out_file="$OUTPUT_DIR/$video_file"
    
    echo "Converting: $video_file"
    
    # Convert to H.264 with FFmpeg
    ffmpeg -i "$video_file" \
        -c:v libx264 \
        -preset medium \
        -crf 23 \
        -pix_fmt yuv420p \
        -movflags +faststart \
        -y \
        -loglevel warning \
        -stats \
        "$out_file"
    
    # Check if conversion was successful
    if [ $? -eq 0 ]; then
        echo "✅ Success: $video_file"
    else
        echo "❌ Error converting: $video_file"
    fi
done

echo "All conversions complete. Output saved to: $OUTPUT_DIR"