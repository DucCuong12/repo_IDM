# Data Pipeline for Vision Language Action model

![Teaser Image](./teaser/image-20251031-082429.png)

A comprehensive pipeline for generating and processing synthetic motion data using natural language processing and computer vision techniques.
For more details about Cosmos, please visit: https://nvidia-cosmos.github.io/cosmos-cookbook/get_started.html

## Table of Contents
- [1. Overview](#1-overview)
- [2. Features](#2-features)
- [3. Quick Start](#3-quick-start)
- [4. Installation](#4-installation)
- [5. Usage](#5-usage)
  - [5.1 S3 Data Management](#51-s3-data-management)
  - [5.2 Cosmos Model](#52-cosmos-model)
  - [5.3 Cosmos Transfer 2.5](#53-cosmos-transfer-25)
  - [5.4 Cosmos Reason 1](#54-cosmos-reason-1)
  - [5.5 IDM Model](#55-idm-model)
  - [5.6 Cosmos-Curate](#56-cosmos-curate)
- [6. Contribution](#6-contribution)
- [7. Contact](#7-contact)

## 1. Overview

This repository contains tools and models for generating synthetic motion data, including:
- Natural language-driven motion generation
- Video synthesis and processing
- Intelligent data management and analysis

## 2. Features

Data storage
- **S3 data sstorage**: Upload & download data to S3 bucket

Data-pipeline modules

- **Cosmos model**: Advanced video-to-world prediction

- **Cosmos Reason 1**: Synthetic motion validation and reasoning

- **IDM model**: Pseudo label for generated video

- **Cosmos Transfer 2.5**: Video-to-video transfer for motion augmentation

## 3. Quick Start

### Base Installation

1. Clone the repository with submodules:
   ```bash
   git clone git@bitbucket.org:vinmotion/data-pipeline.git
   cd data-pipeline
   ```

2. Set up each module by following their respective README files:

### Module Installation

| Module | Path | Installation Guide |
|--------|------|-------------------|
| **Cosmos Model** | `cosmos-predict2.5/` | [README.md](cosmos-predict2.5/README.md) |
| **Cosmos Reason 1** | `cosmos-reason1/` | [README.md](cosmos-reason1/README.md) |
| **IDM Model** | `GR00T-Dreams/` | [README.md](GR00T-Dreams/README.md) |
| **Cosmos Transfer 2.5** | `cosmos-transfer2.5/` | [README.md](cosmos-transfer2.5/README.md) |
| **Cosmos curate** | `cosmos-curate/` | [README.md](cosmos-curate/README.md) | 

Each module has its own README with specific installation requirements and instructions. Please refer to them for detailed setup.

## 5. Usage

### 5.1 S3 Data Management

Upload data to S3:
```bash
aws s3 sync ./ s3://vmo-ai-manipulation/sentk1/server_code/cosmos-predict2.5/datasets
```

### 5.2 Cosmos Model

#### Setup noted 

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co)
2. Generate an access token in your account settings
3. Set up the access token on your server:
   ```bash
   huggingface-cli login
   # Enter your access token when prompted
   ```
4. Accept the license terms for the Cosmos models: [nvidia/Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)


#### Data Preparation

##### Directory Structure

```M1 example
M1/                 # Root directory for your dataset
├── videos/                   # Directory containing video files
│   ├── 0001.mp4             # Video file (H.264 encoded)
│   ├── 0002.mp4             # Sequential numbering
│   └── ...
└── metas/                    # Directory containing metadata
    ├── 0001.txt             # Text description of the action
    └── 0002.txt             # Matches video filename
```

### Example Files

1. **Video Files**:
   - Format: MP4 (H.264 codec)
   - Naming: Sequential numbers (e.g., `0001.mp4`, `0002.mp4`)
   - Resolution: 1280x720 (recommended)

2. **Metadata Files**:
   - Naming: Same as corresponding video but with `.txt` extension
   - Content: Single line description of the action
   - Example (`0001.txt`):
     ```
     The robot's left arm picks up the red cube and places it in the box
     ```

```bash
python process_videos.py --csv datasets/benchmark_train/gr1/metadata.csv --output datasets/benchmark_train/gr1
```

#### Single-View Training

Train the model with a single camera view using the following command:

```bash
CUDA_VISIBLE_DEVICES=2,3 WANDB_MODE=disabled torchrun \
    --nproc_per_node=2 \
    --master_port=12341 \
    -m scripts train \
    --config=cosmos_predict2/_src/cosmos_predict2/configs/video2world/config.py \
    --job.wandb_mode=disabled \
    ~trainer.callbacks.wandb \
    experiment=predict2_video2world_training_2b_groot_gr1_480
```

**Parameters**:
- `CUDA_VISIBLE_DEVICES`: Specifies which GPUs to use (2,3 in this case)
- `--nproc_per_node`: Number of processes per node (2 for data parallelism)
- `--master_port`: Port for process communication
- `--config`: Path to the training configuration file

#### Multi-View Training

For multi-view training scenarios, use the multiview training script:

```bash
python /mnt/ssd/project/data-pipeline/cosmos-predict2.5/examples/multiview.py \
    --config path/to/your/multiview_config.yaml \
    --views front top side  # Specify your camera views
```

**Requirements**:
- Synchronized video streams from multiple cameras
- Camera calibration data (intrinsics and extrinsics)
- Consistent naming convention for corresponding views (e.g., `video_001_front.mp4`, `video_001_top.mp4`)
- Configuration file specifying view-specific parameters

#### Checkpoint Conversion (DCP to PT)
After training, convert the distributed checkpoint to PyTorch format:

```bash
# Get path to the latest checkpoint
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_predict_v2p5/video2world/2b_cosmos_nemo_assets/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

#### Inference
```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=12341 examples/inference.py \
  -i ./prompt/2.json \
  -o outputs/gr00t_gr1_sample \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_video2world_training_2b_groot_gr1_480
```
### 5.3 Cosmos Transfer 2.5

Cosmos Transfer is a video-to-video transfer model that enables motion transfer between objects, serving as a powerful data augmentation technique.

** Acceptance criteria**:  https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B

### 5.4 Cosmos Reason 1

Validate synthetic motion video:

![Reasoning Diagram](./teaser/reasoning.png)

** Acceptance criteria**: https://huggingface.co/nvidia/Cosmos-Reason1-7B

#### Example Output
<video width="640" height="360" controls>
  <source src="./teaser/failure_case.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### IDM Model

1. **Data Conversion**
   ```bash
   python GR00T-Dreams/scripts/generate_stats.py \
     --dataset-path GR00T-Dreams/data/m1_pick_and_place \
     --output-path GR00T-Dreams/IDM_dump/global_metadata/m1/stats.json
   ```

2. **Model Training**
   *Training commands will be added soon*

## 6. Contribution

We welcome contributions! Please follow these guidelines:

- Write clear, well-documented code

- Include unit tests for new features

- Follow the existing code style

- Submit pull requests with detailed descriptions



## 7. Contact

For questions or support, please contact:

- Repository Owner: [SenTran]

- Email: [sentk1@vingroup.net]

- Team: [AI manipulation]

---
*Last updated: December 2025*
* Other community or team contact