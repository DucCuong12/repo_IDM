# IDM_dump Pipeline - Chi Tiết Input/Output

## 📋 Tổng Quan Folder Structure

```
IDM_dump/
├── base.yaml                          # Config cơ bản cho IDM model
├── split_video_instruction.py         # Script 1: Tách video + extract instruction
├── preprocess_video.py                # Script 2: Xử lý video (crop/resize)
├── raw_to_lerobot.py                  # Script 3: Convert to LeRobot format
├── dump_idm_actions.py                # Script 4: Generate actions với IDM model
├── global_metadata/                   # Metadata cho từng embodiment
│   ├── franka/
│   ├── g1/
│   ├── gr1/
│   ├── robocasa/
│   └── so100/
│       ├── modality.json              # Loại dữ liệu có trong dataset
│       └── stats.json                 # Thống kê normalize values
└── scripts/preprocess/
    ├── m2.sh                          # Pipeline cho M2 embodiment
    ├── franka.sh
    ├── gr1.sh
    ├── robocasa.sh
    ├── g1.sh
    └── so100.sh
```

---

## 🔄 Pipeline Chi Tiết (M2 Example)

### **Step 1: split_video_instruction.py**
**Input:**
```
/mnt/ssd/project/data-pipeline/GR00T-Dreams/data/pick_bottle/videos/chunk-000/observation.images.cam_head/
├── 1_pick_up_object.mp4
├── 2_place_down_carefully.mp4
└── 3_rotate_left_side.mp4
```

**Output:**
```
IDM_dump/data/m2/
├── labels/
│   ├── 1.txt (nội dung: "pick up object")
│   ├── 2.txt (nội dung: "place down carefully")
│   └── 3.txt (nội dung: "rotate left side")
└── videos/
    ├── 1.mp4 (copy của 1_pick_up_object.mp4)
    ├── 2.mp4
    └── 3.mp4
```

**Xử lý:**
- Trích xuất instruction từ filename
- Rename video thành số thứ tự
- Tách instruction vào file .txt riêng

---

### **Step 2: preprocess_video.py**
**Input:**
```
IDM_dump/data/m2/
├── labels/
│   ├── 1.txt
│   ├── 2.txt
│   └── 3.txt
└── videos/
    ├── 1.mp4 (832x480, chứa 3 camera views)
    ├── 2.mp4
    └── 3.mp4
```

**Output:**
```
IDM_dump/data/m2_split/
├── labels/
│   ├── 1.txt (copy)
│   ├── 2.txt
│   └── 3.txt
└── videos/
    └── observation.images.cam_head/
        ├── 1.mp4 (256x256, đã xử lý/normalize)
        ├── 2.mp4
        └── 3.mp4
```

**Xử lý:**
- Chia frame 832x480 thành 3 camera views
- Resize & pad mỗi view → 256x256
- Trích xuất subimages theo dataset type (m2, franka, robocasa, etc.)

---

### **Step 3: raw_to_lerobot.py**
**Input:**
```
IDM_dump/data/m2_split/
├── labels/
│   └── 1.txt
└── videos/
    └── observation.images.cam_head/
        └── 1.mp4
```

**Output:**
```
IDM_dump/data/m2_unified.data/
├── meta/
│   ├── info.json          # Metadata tổng hợp (tổng episodes, frames, tasks, fps)
│   ├── tasks.jsonl        # Danh sách task (instruction)
│   ├── episodes.jsonl     # Chi tiết từng episode
│   ├── modality.json      # Copy từ global_metadata/m2/modality.json
│   └── stats.json         # Copy từ global_metadata/m2/stats.json
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  # Frame data (state, action, timestamp, task_index)
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        └── observation.images.cam_head/
            ├── episode_000000.mp4
            ├── episode_000001.mp4
            └── ...
```

**Xử lý:**
- Convert video frame/annotation thành LeRobot format
- Tạo Parquet files với:
  - `observation.state`: Robot state (44 dims)
  - `action`: Robot action (44 dims)
  - `timestamp`: Thời gian frame
  - `task_index`: Index của instruction
- Tạo metadata files (tasks.jsonl, episodes.jsonl, info.json)
- Copy video vào cấu trúc LeRobot
- Copy modality.json & stats.json từ global_metadata

**Tham số quan trọng:**
- `--fps 16`: Frame rate (16 fps cho cosmos_predict2)
- `--cosmos_predict2`: Mode cố định 93 frames/video
- `--embodiment m2`: Xác định embodiment (robot type)

---

### **Step 4: dump_idm_actions.py**
**Input:**
```
1. IDM_dump/data/m2_unified.data/    # Dataset ở format LeRobot
2. /mnt/ssd/project/GR00T-Dreams/idm/m2/checkpoint-10000  # Pre-trained IDM model
```

**Output:**
```
IDM_dump/data/m2_unified.data/
└── meta/
    └── actions.jsonl  # Generated actions từ IDM model
```

**Xử lý:**
- Load pre-trained IDM (Inverse Dynamics Model) checkpoint
- Load dataset từ LeRobot format
- Inference: Video frame → Predicted actions
- Save actions vào actions.jsonl

**Tham số:**
- `--checkpoint`: Path đến model checkpoint
- `--dataset`: Path đến LeRobot dataset
- `--video_indices "0 8"`: Video delta indices để model sử dụng
- `--num_gpus 8`: Số GPU để inference
- `--output_dir`: Output (ghi vào cùng dataset)

---

## 📦 Requirements

### **Python Dependencies:**
```
opencv-python (cv2)
numpy
pandas
torch
tqdm
decord                 # Video reading
imageio               # Video writing
omegaconf / hydra     # Config management
tianshou              # Batch processing
huggingface_hub       # Download models
```

### **System Requirements:**
- **Video codec support**: ffmpeg, ffprobe (để lấy metadata video)
- **GPU**: Khuyến khích cho step 4 (dump_idm_actions)
- **Storage**: 
  - Input video (m2) ≈ 100GB
  - Preprocessed (m2_split) ≈ 50GB  
  - LeRobot format (m2_unified.data) ≈ 100GB (include video + parquet files)

### **Model Checkpoints:**
- `/mnt/ssd/project/GR00T-Dreams/idm/m2/checkpoint-10000` (cho M2)
- Tương tự cho các embodiment khác (franka, gr1, robocasa, etc.)

### **Metadata Files:**
- `IDM_dump/global_metadata/{embodiment}/modality.json`
- `IDM_dump/global_metadata/{embodiment}/stats.json`

---

## 🎯 Input/Output Summary

| Script | Input | Output | Dependencies |
|--------|-------|--------|--------------|
| **split_video_instruction.py** | Raw video files (MP4) | Organized videos + labels | ffprobe |
| **preprocess_video.py** | Organized videos | Preprocessed videos (256x256) | opencv, decord, imageio |
| **raw_to_lerobot.py** | Preprocessed videos | LeRobot format dataset | pandas, subprocess (ffprobe) |
| **dump_idm_actions.py** | LeRobot dataset + IDM checkpoint | actions.jsonl | torch, hydra, tianshou, huggingface_hub |

---

## 🚀 Cách Chạy

```bash
# Chạy toàn bộ pipeline cho M2
bash IDM_dump/scripts/preprocess/m2.sh

# Hoặc chạy từng step riêng
python IDM_dump/split_video_instruction.py \
    --source_dir "..." \
    --output_dir "IDM_dump/data/m2"

python IDM_dump/preprocess_video.py \
    --src_dir "IDM_dump/data/m2" \
    --dst_dir "IDM_dump/data/m2_split" \
    --dataset m2

python IDM_dump/raw_to_lerobot.py \
    --input_dir "IDM_dump/data/m2_split" \
    --output_dir "IDM_dump/data/m2_unified.data" \
    --embodiment m2 \
    --cosmos_predict2

python IDM_dump/dump_idm_actions.py \
    --checkpoint "path/to/checkpoint-10000" \
    --dataset "IDM_dump/data/m2_unified.data" \
    --output_dir "IDM_dump/data/m2_unified.data" \
    --num_gpus 8 \
    --video_indices "0 8"
```

---

## 📊 Data Flow Visualization

```
Raw Videos (832x480)
    ↓
[split_video_instruction.py]
    ↓
Labeled Videos + Instructions
    ↓
[preprocess_video.py]
    ↓
Preprocessed Videos (256x256, 3 views)
    ↓
[raw_to_lerobot.py]
    ↓
LeRobot Format Dataset
├── Parquet files (state, action, timestamp)
├── Video files (reorganized)
├── Metadata (tasks, episodes, info)
└── Stats (modality, normalization)
    ↓
[dump_idm_actions.py] (Optional - IDM inference)
    ↓
Final Dataset with Predicted Actions
```

---

## ✅ Checklist Trước Khi Chạy

- [ ] Có raw video files tại đúng path
- [ ] ffmpeg & ffprobe được cài
- [ ] Python dependencies đã install
- [ ] Có đủ disk space (~250GB cho M2)
- [ ] IDM checkpoint tồn tại (cho step 4)
- [ ] global_metadata files tồn tại
- [ ] GPU available (cho step 4)

