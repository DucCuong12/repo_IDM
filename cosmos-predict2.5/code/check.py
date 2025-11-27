import cv2
import os

# Folder containing your mp4 videos
video_folder = "../gr1/videos"

# List to store videos with fewer than 93 frames
short_videos = []

# Loop through all mp4 files in the folder
for file in sorted(os.listdir(video_folder)):
    if file.endswith(".mp4"):
        video_path = os.path.join(video_folder, file)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_count < 93:
            short_videos.append((file, frame_count))

# Print results
if short_videos:
    print("📉 Videos with fewer than 93 frames:")
    for name, count in short_videos:
        print(f"{name}: {count} frames")
else:
    print("✅ All videos have at least 93 frames.")

# Optionally save to a text file
with open("short_videos.txt", "w") as f:
    for name, count in short_videos:
        f.write(f"{name}: {count} frames\n")

print("📝 Saved results to short_videos.txt")
