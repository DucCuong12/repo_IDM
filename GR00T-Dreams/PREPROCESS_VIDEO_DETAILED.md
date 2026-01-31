# preprocess_video.py - Chi Tiết Phân Tích

## 📋 Mục Đích Chính
Script này **chia nhỏ video (frames) thành nhiều camera views** và **chuẩn hóa kích thước** (256x256) với padding để giữ aspect ratio.

---

## 🎬 Input/Output

### **Input:**
```
IDM_dump/data/m2/
├── labels/
│   └── 1.txt
└── videos/
    └── observation.images.cam_head/
        └── 1.mp4  (832x480, chứa 3 camera views ghép vào 1 frame)
```

**Frame gốc (832x480):**
```
┌──────────────────────────────┐
│  Camera 1  │  Camera 2       │  ← Chiều cao: 240px
├──────────────────────────────┤
│  Camera 3  │  (phần rỗng)    │  ← Chiều cao: 240px
└──────────────────────────────┘
 Mỗi camera: 416px chiều rộng
```

### **Output:**
```
IDM_dump/data/m2_split/
├── labels/
│   └── 1.txt  (copy từ input)
└── videos/
    └── observation.images.cam_head/
        └── 1.mp4  (256x256, đã normalize)
```

---

## 🔍 Chi Tiết Các Hàm

### **1. `extract_subimages(frame, ratio)`**

**Mục đích:** Chia frame 832x480 thành 3 camera views, mỗi view 240x416

```python
def extract_subimages(frame, ratio):
    h, w = frame.shape[:2]  # h=480, w=832
    
    half_width = w // 2   # 416
    half_height = h // 2  # 240
    
    # Extract 3 subimages
    image_side_0 = frame[:half_height, :half_width]      # [0:240, 0:416]   (top-left)
    image_side_1 = frame[:half_height, half_width:]      # [0:240, 416:832] (top-right)
    wrist_image = frame[half_height:, :half_width]       # [240:480, 0:416] (bottom-left)
```

**Visualization:**
```
Frame 832x480:
┌─────────────┬─────────────┐
│  [0:240,    │  [0:240,    │
│   0:416]    │   416:832]  │  ← image_side_0, image_side_1
│             │             │
├─────────────┼─────────────┤
│  [240:480,  │  (ignored)  │
│   0:416]    │             │  ← wrist_image
│             │             │
└─────────────┴─────────────┘
```

**Tiếp theo:** Resize mỗi view với padding để giữ aspect ratio:

```python
image_side_0 = resize_with_padding(image_side_0, ratio)  # 240x416 → 256x256 (with padding)
image_side_1 = resize_with_padding(image_side_1, ratio)  # 240x416 → 256x256 (with padding)
wrist_image = resize_with_padding(wrist_image, ratio)    # 240x416 → 256x256 (with padding)

return image_side_0, image_side_1, wrist_image
```

---

### **2. `resize_with_padding(img, ratio=1.0, target_size=(256, 256))`**

**Mục đích:** Resize hình ảnh với padding để giữ aspect ratio

```python
def resize_with_padding(img, ratio=1.0, target_size=(256, 256)):
    h, w = img.shape[:2]  # h=240, w=416
    target_ratio = ratio  # Aspect ratio cần duy trì
    
    if target_ratio >= 1:  # Width-based limiting
        # Resize theo width
        new_w = target_size[0]  # 256
        new_h = int(new_w / target_ratio)  # Tính height dựa trên aspect ratio
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Thêm padding trên/dưới để fill 256x256
        pad_top = (target_size[1] - new_h) // 2
        pad_bottom = target_size[1] - new_h - pad_top
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0, 
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:  # Height-based limiting
        # Resize theo height
        new_h = target_size[1]  # 256
        new_w = int(new_h * target_ratio)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Thêm padding trái/phải để fill 256x256
        pad_left = (target_size[0] - new_w) // 2
        pad_right = target_size[0] - new_w - pad_left
        padded = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, 
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded
```

**Ví dụ:**
```
Input: 240x416 (aspect ratio = 416/240 ≈ 1.73)
Target: 256x256

Vì 1.73 > 1 → Width-limited:
  new_w = 256
  new_h = int(256 / 1.73) ≈ 148
  
  Resize → 256x148
  
  Thêm padding:
  pad_top = (256 - 148) / 2 = 54
  pad_bottom = 256 - 148 - 54 = 54
  
  Result: 256x256 với hình ảnh ở giữa, 54px đen trên/dưới
```

---

### **3. `extract_subimages_franka(frame, original_width, original_height)`**

**Mục đích:** Tương tự `extract_subimages()` nhưng resize theo original_width/height thay vì padding

```python
def extract_subimages_franka(frame, original_width, original_height):
    h, w = frame.shape[:2]  # 480, 832
    
    half_width = w // 2   # 416
    half_height = h // 2  # 240
    
    # Extract
    image_side_0 = frame[:half_height, :half_width]      # 240x416
    image_side_1 = frame[:half_height, half_width:]      # 240x416
    wrist_image = frame[half_height:, :half_width]       # 240x416
    
    # Resize KHÔNG dùng padding, chỉ scale trực tiếp
    image_side_0 = cv2.resize(image_side_0, (original_width, original_height), 
                              interpolation=cv2.INTER_LINEAR)  # 1280x800
    image_side_1 = cv2.resize(image_side_1, (original_width, original_height), ...)
    wrist_image = cv2.resize(wrist_image, (original_width, original_height), ...)
    
    return image_side_0, image_side_1, wrist_image
```

**Khác biệt so với `resize_with_padding()`:**
- `resize_with_padding()`: Giữ aspect ratio + thêm padding (đen) để fill 256x256
- `extract_subimages_franka()`: Resize trực tiếp không giữ aspect ratio (bóp méo hình)

---

### **4. `custom_crop_pad_resize_gr1(img, target_size=(256, 256))`**

**Mục đích:** Xử lý đặc biệt cho dataset GR1 - crop + pad + resize

```python
def custom_crop_pad_resize_gr1(img, target_size=(256, 256)):
    original_height, original_width = img.shape[:2]
    
    # Step 1: CROP dựa trên tỷ lệ
    # Original crop cho 1280x800: (310, 770, 110, 1130) - (top, bottom, left, right)
    crop_top_ratio = 310 / 800      # Tỷ lệ top
    crop_bottom_ratio = 770 / 800   # Tỷ lệ bottom
    crop_left_ratio = 110 / 1280    # Tỷ lệ left
    crop_right_ratio = 1130 / 1280  # Tỷ lệ right
    
    # Apply ratios
    crop_top = int(original_height * crop_top_ratio)
    crop_bottom = int(original_height * crop_bottom_ratio)
    crop_left = int(original_width * crop_left_ratio)
    crop_right = int(original_width * crop_right_ratio)
    
    # Crop
    img_cropped = img[crop_top:crop_bottom, crop_left:crop_right]
    
    # Step 2: RESIZE đến intermediate size
    intermediate_height = 480
    intermediate_width = 720
    img_resized = cv2.resize(img_cropped, (intermediate_width, intermediate_height), cv2.INTER_AREA)
    
    # Step 3: PAD để thành square
    if intermediate_width > intermediate_height:  # 720 > 480 → width larger
        height_pad = (intermediate_width - intermediate_height) // 2  # (720-480)/2 = 120
        img_pad = np.pad(img_resized, ((height_pad, height_pad), (0, 0), (0, 0)), 
                        mode="constant", constant_values=0)  # Pad top/bottom 120px
    
    # Step 4: RESIZE đến target size (256x256)
    final_img = cv2.resize(img_pad, target_size, cv2.INTER_AREA)
    
    return final_img
```

**Flow:**
```
Input frame (832x480)
    ↓ (CROP theo tỷ lệ)
Cropped image
    ↓ (RESIZE → 720x480)
Intermediate 720x480
    ↓ (PAD → 720x720)
Padded square 720x720
    ↓ (RESIZE → 256x256)
Final output 256x256
```

---

### **5. `process_batch_frames(frames, output_videos, ...)`**

**Mục đích:** Xử lý batch frame và ghi vào output video writers

```python
def process_batch_frames(frames, output_videos, src_path, dataset, 
                         original_width, original_height):
    ratio = original_width / original_height  # Aspect ratio
    
    for frame in frames:
        if dataset == 'robocasa':
            # Chia thành 3 views với padding
            image_side_0, image_side_1, wrist_image = extract_subimages(frame, ratio)
            output_videos['observation.images.left_view'].append_data(image_side_0)
            output_videos['observation.images.right_view'].append_data(image_side_1)
            output_videos['observation.images.wrist_view'].append_data(wrist_image)
            
        elif dataset == 'gr1':
            # Custom crop+pad+resize
            image = custom_crop_pad_resize_gr1(frame)
            output_videos['observation.images.ego_view'].append_data(image)
            
        elif dataset == 'franka':
            # Chia thành 3 views SỰ DỤNG franka extraction
            image_side_0, image_side_1, wrist_image = extract_subimages_franka(
                frame, original_width, original_height
            )
            output_videos['observation.images.exterior_image_1_left_pad_res256_freq15'].append_data(image_side_0)
            output_videos['observation.images.exterior_image_2_left_pad_res256_freq15'].append_data(image_side_1)
            output_videos['observation.images.wrist_image_left_pad_res256_freq15'].append_data(wrist_image)
```

---

### **6. `process_video(args)`**

**Mục đích:** Xử lý 1 video từ đầu đến cuối

```python
def process_video(args):
    src_path, dst_dir, video_name, dataset, original_width, original_height = args
    
    # Step 1: Tạo output directories dựa trên dataset type
    if dataset == 'robocasa':
        output_dirs = {
            'observation.images.left_view': ...,
            'observation.images.right_view': ...,
            'observation.images.wrist_view': ...,
        }
    elif dataset == 'gr1':
        output_dirs = {
            'observation.images.ego_view': ...,
        }
    # ... (khác dataset khác dirs)
    
    # Step 2: Mở video với decord
    vr = decord.VideoReader(src_path)
    fps = vr.get_avg_fps()
    frame_count = len(vr)
    
    # Step 3: Tạo video writers cho mỗi output
    output_videos = {}
    for name, dir_path in output_dirs.items():
        output_videos[name] = imageio.get_writer(os.path.join(dir_path, f"{video_name}.mp4"), fps=fps)
    
    # Step 4: Đọc frames thành batch
    batch_size = 32
    frames_batch = []
    pbar = tqdm(total=frame_count, desc=f"Processing {video_name}", leave=False)
    
    for frame in vr:
        frames_batch.append(frame.asnumpy())
        
        if len(frames_batch) >= batch_size:
            # Xử lý batch này
            process_batch_frames(frames_batch, output_videos, src_path, dataset, 
                                original_width, original_height)
            frames_batch = []
            pbar.update(batch_size)
    
    # Step 5: Xử lý frames còn lại
    if frames_batch:
        process_batch_frames(frames_batch, output_videos, src_path, dataset, 
                            original_width, original_height)
        pbar.update(len(frames_batch))
    
    pbar.close()
    
    # Step 6: Close writers
    for writer in output_videos.values():
        writer.close()
```

**Trong 1 video:**
1. Mở video input (src_path)
2. Tạo output video files (1, 3 hoặc N tùy dataset)
3. Đọc frames thành batch (batch_size=32)
4. Xử lý batch (crop/resize)
5. Ghi vào output videos
6. Đóng tất cả files

---

### **7. `copy_labels(src_dir, dst_dir)`**

**Mục đích:** Copy file .txt từ input labels → output labels

```python
def copy_labels(src_dir, dst_dir):
    src_labels_dir = os.path.join(src_dir, 'labels')
    dst_labels_dir = os.path.join(dst_dir, 'labels')
    
    if os.path.exists(src_labels_dir):
        os.makedirs(dst_labels_dir, exist_ok=True)
        for label_file in os.listdir(src_labels_dir):
            if label_file.endswith('.txt'):
                shutil.copy2(
                    os.path.join(src_labels_dir, label_file),
                    os.path.join(dst_labels_dir, label_file)
                )
```

---

### **8. `process_subdirectory()` & `process_directory()`**

**Mục đích:** Xử lý nhiều videos trong parallel

```python
def process_subdirectory(subdir, src_dir, dst_dir, num_workers, ...):
    # Copy labels
    copy_labels(src_subdir, dst_subdir)
    
    # Lấy tất cả .mp4 files
    video_files = [f for f in os.listdir(src_videos_dir) if f.endswith('.mp4')]
    
    # Tạo args list cho mỗi video
    args_list = [
        (mp4_path, dst_subdir, video_name, dataset, original_width, original_height)
        for video in video_files
    ]
    
    # Process với multiprocessing
    with mp.Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_video, args_list), total=len(args_list)))

def process_directory(src_dir, dst_dir, num_workers=None, ...):
    # Xác định subdirectories
    if recursive:
        subdirs = [d for d in os.listdir(src_dir) if os.path.isdir(...)]
    else:
        subdirs = ['']  # Chỉ root
    
    # Process subdirs trong parallel (ThreadPoolExecutor)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        list(executor.map(process_subdir_fn, subdirs))
```

**Parallelization:**
- **Level 1:** `ThreadPoolExecutor` để process subdirectories (sequential by default)
- **Level 2:** `multiprocessing.Pool` để process videos trong parallel

---

## 📊 Data Flow

```
Input video (832x480, 93 frames, 1 MP4)
    ↓
[Video Reader - decord]
    ↓
Đọc frames batch (batch_size=32)
    ↓
[extract_subimages / custom_crop_pad_resize_gr1 / extract_subimages_franka]
    ↓
3 processed frames (256x256) hoặc 1 processed frame (tùy dataset)
    ↓
[Video Writer - imageio]
    ↓
Output: 1-3 MP4 files (256x256, 93 frames mỗi file)

┌─ robocasa  → 3 videos (left, right, wrist)
├─ gr1       → 1 video (ego_view)
├─ franka    → 3 videos (exterior1, exterior2, wrist)
├─ so100     → 1 video (webcam)
└─ g1        → 1 video (cam_head)
```

---

## ✅ Summary

| Hàm | Input | Output | Mục đích |
|-----|-------|--------|---------|
| `extract_subimages()` | 832x480 frame | 3x 256x256 frames (with padding) | Chia 3 views + resize với padding |
| `resize_with_padding()` | 240x416 frame | 256x256 frame | Resize giữ aspect ratio + padding |
| `custom_crop_pad_resize_gr1()` | 832x480 frame | 256x256 frame | Crop + pad + resize cho GR1 |
| `extract_subimages_franka()` | 832x480 frame | 3x 1280x800 frames | Chia 3 views + resize direct (bóp méo) |
| `process_batch_frames()` | Batch frames | Append data to writers | Xử lý batch frames |
| `process_video()` | 1 MP4 file | 1-3 MP4 files (processed) | Toàn bộ flow 1 video |
| `copy_labels()` | Source labels dir | Destination labels dir | Copy .txt files |
| `process_directory()` | Source dir | Destination dir | Orchestrate toàn bộ pipeline |

---

## 🎯 Dataset Types

```
robocasa:  3 cameras → left_view, right_view, wrist_view (256x256 each)
gr1:       1 camera  → ego_view (256x256)
franka:    3 cameras → exterior_image_1, exterior_image_2, wrist_image (custom size)
so100:     1 camera  → webcam (256x256)
g1:        1 camera  → cam_head (256x256)
```

