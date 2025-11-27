#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag

def compute_stats(dataset_path, output_path):
    """
    Compute statistics for a given dataset and save to a JSON file.
    
    Args:
        dataset_path: Path to the dataset directory
        output_path: Path to save the stats.json file
    """
    import gr00t.experiment.data_config_idm
    import inspect
    
    # Print the actual file being imported
    print("Loading module from:", inspect.getfile(gr00t.experiment.data_config_idm))
    
    # Get the actual DATA_CONFIG_MAP
    DATA_CONFIG_MAP = gr00t.experiment.data_config_idm.DATA_CONFIG_MAP
    
    # Debug: Print available keys and their types
    print("Available config keys:", list(DATA_CONFIG_MAP.keys()))
    print("Module file:", inspect.getfile(gr00t.experiment.data_config_idm))
    
    # Try to get the config class
    try:
        data_config_cls = DATA_CONFIG_MAP["m1"]
    except KeyError as e:
        print(f"Error: 'm1' not found in DATA_CONFIG_MAP. Available keys: {list(DATA_CONFIG_MAP.keys())}")
        raise
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    
    # Initialize dataset
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        video_backend="decord"
    )
    
    # Initialize data loader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        shuffle=False,
        drop_last=False
    )
    
    # Initialize accumulators for each state and action component
    state_components = {
        'left_arm': [],
        'right_arm': [],
        'left_hand': [],
        'right_hand': []
    }
    action_components = {
        'left_arm': [],
        'right_arm': [],
        'left_hand': [],
        'right_hand': []
    }
    
    # Process dataset
    for batch in tqdm(dataloader, desc="Processing dataset"):
        # Process state components
        for component in state_components.keys():
            state_key = f'state.{component}'
            if state_key in batch:
                state_components[component].append(batch[state_key].numpy())
        
        # Process action components
        for component in action_components.keys():
            action_key = f'action.{component}'
            if action_key in batch:
                action_components[component].append(batch[action_key].numpy())
    
    # Concatenate all batches for each component
    all_states = {}
    all_actions = {}
    
    for component in state_components:
        if state_components[component]:
            all_states[component] = np.concatenate(state_components[component], axis=0)
    
    for component in action_components:
        if action_components[component]:
            all_actions[component] = np.concatenate(action_components[component], axis=0)
    
    # Compute statistics for states and actions
    stats = {
        'state': {},
        'action': {}
    }
    
    # Compute statistics for each state component
    for component, data in all_states.items():
        stats['state'][component] = {
            'mean': data.mean(axis=0).tolist(),
            'std': data.std(axis=0).tolist(),
            'min': data.min(axis=0).tolist(),
            'max': data.max(axis=0).tolist(),
            'q01': np.quantile(data, 0.01, axis=0).tolist(),
            'q99': np.quantile(data, 0.99, axis=0).tolist(),
        }
    
    # Compute statistics for each action component
    for component, data in all_actions.items():
        stats['action'][component] = {
            'mean': data.mean(axis=0).tolist(),
            'std': data.std(axis=0).tolist(),
            'min': data.min(axis=0).tolist(),
            'max': data.max(axis=0).tolist(),
            'q01': np.quantile(data, 0.01, axis=0).tolist(),
            'q99': np.quantile(data, 0.99, axis=0).tolist(),
        }
    
    # Save statistics to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Statistics saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate statistics for a dataset')
    parser.add_argument('--dataset-path', type=str, required=True,
                      help='Path to the dataset directory')
    parser.add_argument('--output-path', type=str, required=True,
                      help='Path to save the stats.json file')
    
    args = parser.parse_args()
    compute_stats(args.dataset_path, args.output_path)
