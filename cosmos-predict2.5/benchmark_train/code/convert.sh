#!/bin/bash
# Rename videos from episode_000000.mp4 -> 0.mp4, episode_000001.mp4 -> 1.mp4, etc.

# Folder containing videos (optional: change this)
VIDEO_DIR="./videos"

cd "$VIDEO_DIR" || { echo "❌ Folder not found!"; exit 1; }

# Loop through all episode_*.mp4 files
for f in episode_*.mp4; do
  # Skip if no files found
  [ -e "$f" ] || continue

  # Extract numeric part (e.g., 000012 -> 12)
  num=$(echo "$f" | sed -E 's/episode_0*([0-9]+)\.mp4/\1/')

  # New name
  new_name="${num}.mp4"

  echo "🔁 Renaming $f -> $new_name"
  mv -f "$f" "$new_name"
done

echo "✅ All files renamed successfully!"
