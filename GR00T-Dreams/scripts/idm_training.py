
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
import tyro
from transformers import TrainingArguments, TrainerCallback

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP
from gr00t.experiment.runner_idm import TrainRunner
from gr00t.model.idm import IDM


class LossLoggerCallback(TrainerCallback):
    """Custom callback to log loss to a text file during training."""
    
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_file = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        # Open log file and write header
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        self.log_file = open(self.log_file_path, "w")
        self.log_file.write("step,loss,learning_rate,epoch\n")
        self.log_file.flush()
        print(f"[LossLoggerCallback] Logging to: {self.log_file_path}")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Called every logging_steps
        if logs is not None and self.log_file is not None:
            step = state.global_step
            loss = logs.get("loss", "N/A")
            lr = logs.get("learning_rate", "N/A")
            epoch = logs.get("epoch", "N/A")
            
            # Write to file
            self.log_file.write(f"{step},{loss},{lr},{epoch}\n")
            self.log_file.flush()
            
            # Also print to console
            print(f"[Step {step}] loss: {loss:.6f}, lr: {lr:.2e}, epoch: {epoch:.2f}" if isinstance(loss, float) else f"[Step {step}] loss: {loss}, lr: {lr}, epoch: {epoch}")
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.log_file is not None:
            self.log_file.close()
            print(f"[LossLoggerCallback] Training finished. Log saved to: {self.log_file_path}")

@dataclass
class Config:
    """Configuration for idm training."""

    # Dataset parameters
    dataset_path: str
    """Path to the dataset directory."""

    output_dir: str = "./idm/m2_training"
    os.makedirs(output_dir, exist_ok=True)
    """Directory to save model checkpoints."""

    data_config: str = "m2"
    """Data configuration name from DATA_CONFIG_MAP."""

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 500
    """Number of steps between saving checkpoints."""\

    tune_action_head: bool = True
    """Whether to fine-tune the action head."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    dataloader_num_workers: int = 16
    """Number of workers for data loading."""

    report_to: str = "tensorboard"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard')."""

    # Data loading parameters
    embodiment_tag: str = "m2"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: str = "decord"
    """Video backend to use for training. [decord, torchvision_av]"""

    random_init: bool = False
    """Whether to random init the model except action_head_cfg.siglip_model_cfg"""


#####################################################################################
# main training function
#####################################################################################


def main(config: Config):
    """Main training function."""
    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader
    train_dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
        video_backend=config.video_backend,
    )
    train_dataset.print_dataset_info()
    print((train_dataset[0].keys()))
    print((train_dataset[0]['actions'][0]))
    exit(0)
    exit(0)

    # ------------ step 2: load model ------------
    # model = GR00T_N1.from_pretrained(
    #     pretrained_model_name_or_path=config.base_model_path,
    #     tune_llm=config.tune_llm,  # backbone's LLM
    #     tune_visual=config.tune_visual,  # backbone's vision tower
    #     tune_projector=config.tune_projector,  # action head's projector
    #     tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
    # )
    
    print("Loading base model from IDM_dump/base.yaml")
    model = instantiate(OmegaConf.load("IDM_dump/base.yaml"))

    if config.random_init:
        # random init the model except action_head_cfg.siglip_model_cfg
        for name, param in model.named_parameters():
            if "action_head.siglip_model" not in name:
                param.data.normal_(0, 0.02)

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=True,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,  # Log every 10 steps for real-time monitoring
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=8,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 create loss logger callback
    log_dir = Path(config.output_dir) / "logs"
    log_file_path = log_dir / "training_loss.csv"
    loss_logger_callback = LossLoggerCallback(str(log_file_path))

    # 2.3 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )
    
    # Add custom callback to trainer
    experiment.trainer.add_callback(loss_logger_callback)
    
    # Print logging info
    print(f"\n{'='*50}")
    print("LOGGING CONFIGURATION:")
    print(f"  - Loss CSV: {log_file_path}")
    print(f"  - TensorBoard: {Path(config.output_dir) / 'runs'}")
    print(f"  - Checkpoints: {config.output_dir}")
    print(f"{'='*50}\n")

    # 2.4 start training
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(Config)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Use subprocess.run instead of os.system
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
            ]

            # Convert config to command line arguments
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    # For boolean values, use --flag or --no-flag format
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    # For non-boolean values, use --key value format
                    cmd.append(f"--{key.replace('_', '-')}")
                    cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)


# # Train như bình thường
# bash IDM_dump/scripts/train_idm/m2_visualize.sh

# # Sau khi train, visualize:
# python scripts/visualize_loss.py --log_file <output_dir>/logs/training_loss.csv

# # Hoặc xem TensorBoard:
# tensorboard --logdir <output_dir>/runs