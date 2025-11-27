#!/bin/bash
# Reindex video files: 0.mp4 -> 101.mp4, 1.mp4 -> 102.mp4, etc.

VIDEO_DIR="./videos"

cd "$VIDEO_DIR" || { echo "❌ Folder not found!"; exit 1; }

# Use a temporary folder to avoid overwrite conflicts
TMP_DIR="${VIDEO_DIR}/tmp_rename"
mkdir -p "$TMP_DIR"

for f in *.mp4; do
  [ -e "$f" ] || continue

  # Extract the base number (remove .mp4)
  base=$(basename "$f" .mp4)

  # Make sure it's a number
  if [[ "$base" =~ ^[0-9]+$ ]]; then
    new_num=$((base + 101))
    new_name="${new_num}.mp4"
    echo "🔁 $f -> $new_name"
    mv "$f" "$TMP_DIR/$new_name"
  fi
done

# Move files back
mv "$TMP_DIR"/*.mp4 "$VIDEO_DIR"/
rmdir "$TMP_DIR"

echo "✅ Reindexing complete!"
