# Data Pipeline for Vision Language Action model

![Teaser Image](./teaser/image-20251031-082429.png)

A comprehensive pipeline for generating and processing synthetic motion data using natural language processing and computer vision techniques.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [S3 Data Management](#s3-data-management)
  - [Cosmos Model](#cosmos-model)
  - [Cosmos Reason 1](#cosmos-reason-1)
  - [IDM Model](#idm-model)
- [Contribution](#contribution)
- [Contact](#contact)

## Overview

This repository contains tools and models for generating synthetic motion data, including:
- Natural language-driven motion generation
- Video synthesis and processing
- Intelligent data management and analysis

## Features

- **Cosmos Model**: Advanced video-to-world prediction
- **Cosmos Reason 1**: Synthetic motion validation and reasoning
- **IDM Integration**: Pseudo label for generated video
- **S3 Integration**: Seamless data storage and retrieval

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd data-pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### S3 Data Management

Upload data to S3:
```bash
aws s3 sync ./ s3://vmo-ai-manipulation/sentk1/server_code/cosmos-predict2.5/datasets
```

### Cosmos Model

#### Training
```bash
CUDA_VISIBLE_DEVICES=2,3 WANDB_MODE=disabled torchrun --nproc_per_node=2 --master_port=12341 -m scripts train \
    --config=cosmos_predict2/_src/cosmos_predict2/configs/video2world/config.py \
    --job.wandb_mode=disabled ~trainer.callbacks.wandb experiment=predict2_video2world_training_2b_groot_gr1_480
```

#### Inference
```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=12341 examples/inference.py \
  -i ./prompt/2.json \
  -o outputs/gr00t_gr1_sample \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_video2world_training_2b_groot_gr1_480
```

### Cosmos Reason 1

Validate synthetic motion generation:

![Reasoning Diagram](./teaser/reasoning.png)

#### Example Output
<video width="640" height="360" controls>
  <source src="./teaser/failure.mp4" type="video/mp4">
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

## Contribution

We welcome contributions! Please follow these guidelines:
- Write clear, well-documented code
- Include unit tests for new features
- Follow the existing code style
- Submit pull requests with detailed descriptions

## Contact

For questions or support, please contact:
- Repository Owner: [Your Name]
- Email: [Your Email]
- Team: [Team Name]

---
*Last updated: December 2025*
* Other community or team contact