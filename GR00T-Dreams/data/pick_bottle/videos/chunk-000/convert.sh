#!/bin/bash
# Convert all episode_*.avi or episode_*.mp4 to H.264 format and replace originals
# Works with CUDA (NVIDIA GPU) if available

# Set your video directory (optional: change this)
VIDEO_DIR="./observation.images.cam_head"

cd "$VIDEO_DIR" || { echo "❌ Folder not found!"; exit 1; }

# Loop through all video files
for f in episode_*.mp4 episode_*.avi; do
  # Skip if file doesn't exist (avoid 'no match' errors)
  [ -e "$f" ] || continue

  echo "🎬 Converting: $f"

  # Temporary output file
  tmp="${f%.*}_h264.mp4"

  # Try GPU first (if fails, fallback to CPU)
  if ffmpeg -y -hwaccel cuda -i "$f" -c:v h264_nvenc -c:a copy "$tmp"; then
    echo "✅ GPU encode success: $f"
  else
    echo "⚙️  GPU failed, switching to CPU encoding..."
    ffmpeg -y -i "$f" -c:v libx264 -preset fast -crf 23 -c:a copy "$tmp"
  fi

  # Replace old file
  mv -f "$tmp" "$f"
  echo "🔁 Replaced: $f"
done

echo "✅ All videos converted to H.264 successfully!"
