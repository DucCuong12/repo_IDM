# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP
from gr00t.experiment.runner_idm import TrainRunner

import torch.distributed as dist


###############################################################################
# CONFIG
###############################################################################

@dataclass
class Config:
    dataset_path: str

    output_dir: str = "./idm/m2_new_update_train"

    data_config: str = "m2"

    batch_size: int = 32
    max_steps: int = 10000
    num_gpus: int = 1
    save_steps: int = 500

    tune_action_head: bool = True
    resume: bool = False

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05

    dataloader_num_workers: int = 8

    report_to: str = "tensorboard"

    embodiment_tag: str = "m2"
    video_backend: str = "decord"

    random_init: bool = False


###############################################################################
# MAIN
###############################################################################

def main(config: Config):

    os.makedirs(config.output_dir, exist_ok=True)

    is_rank0 = not dist.is_initialized() or dist.get_rank() == 0

    if is_rank0:
        print("Running on rank0 — TensorBoard + checkpoints enabled")

    ################ DATA ################

    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    train_dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=config.video_backend,
    )

    ################ MODEL ################

    if is_rank0:
        print("Loading base model from IDM_dump/base.yaml")

    model = instantiate(OmegaConf.load("IDM_dump/base.yaml"))

    if config.random_init:
        for name, param in model.named_parameters():
            if "action_head.siglip_model" not in name:
                param.data.normal_(0, 0.02)

    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    ################ TRAINING ARGS ################

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        remove_unused_columns=False,
        bf16=True,
        tf32=True,

        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,

        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=True,

        optim="adamw_torch",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",

        logging_steps=10,
        report_to=config.report_to,

        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=8,

        seed=42,
        do_eval=False,

        ddp_find_unused_parameters=False,
    )

    ################ RUNNER ################

    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    ################ TRAIN ################

    # IMPORTANT: let HuggingFace Trainer handle everything
    experiment.train()

    if is_rank0:
        print("✅ Training finished")


###############################################################################
# ENTRY
###############################################################################

if __name__ == "__main__":

    config = tyro.cli(Config)

    print("\n========== IDM CONFIG ==========")
    for k, v in vars(config).items():
        print(f"{k}: {v}")
    print("================================\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    assert config.num_gpus <= available_gpus
    assert config.num_gpus > 0

    if config.num_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        main(config)

    else:

        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)

        else:
            script_path = Path(__file__).absolute()

            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",
                str(script_path),
            ]

            for k, v in vars(config).items():
                cmd.append(f"--{k.replace('_','-')}")
                cmd.append(str(v))

            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"

            print("Launching torchrun:\n", " ".join(cmd))
            sys.exit(subprocess.run(cmd, env=env).returncode)
