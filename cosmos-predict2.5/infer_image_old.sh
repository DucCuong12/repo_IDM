#!/bin/bash

# Set GPU devices and other common parameters
export CUDA_VISIBLE_DEVICES=2,3
OUTPUT_DIR="outputs_v2/gr00t_gr1_sample"
EXPERIMENT="predict2_video2world_training_2b_groot_gr1_480"
MASTER_PORT=12341

# Loop from 1 to 7
for i in {1..8}; do
  echo "Running inference for ./prompt_ring/${i}.json ..."
  
  torchrun --nproc_per_node=2 --master_port=$MASTER_PORT examples/inference.py \
    -i ./prompt_ring/${i}.json \
    -o ${OUTPUT_DIR}_${i} \
    --checkpoint-path "./model_ema_fp32.pt" \
    --experiment ${EXPERIMENT}

  echo "✅ Finished ./promp_ringt/${i}.json"
done
