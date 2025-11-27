#!/bin/bash

# Set GPU devices and other common parameters
export CUDA_VISIBLE_DEVICES=2,3
OUTPUT_DIR="outputs/gr00t_gr1_sample"
EXPERIMENT="predict2_video2world_training_2b_groot_gr1_480"
MASTER_PORT=12344

# Loop from 1 to 7
for i in {1..2}; do
  echo "Running inference for ./prompt/${i}.json ..."
  
  torchrun --nproc_per_node=2 --master_port=$MASTER_PORT examples/inference.py \
    -i ./prompt_video/${i}.json \
    -o ${OUTPUT_DIR}_${i} \
    --checkpoint-path "/tmp/imaginaire4-output/cosmos_predict_v2p5/video2world/2b_groot_gr1_480/checkpoints/iter_000005000/model_ema_bf16.pt" \
    --experiment ${EXPERIMENT}

  echo "✅ Finished ./prompt/${i}.json"
done
