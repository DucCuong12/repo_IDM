python IDM_dump/split_video_instruction.py \
    --source_dir "/mnt/ssd/project/data-pipeline/GR00T-Dreams/data/pick_bottle/videos/chunk-000/observation.images.cam_head" \
    --output_dir "IDM_dump/data/g1"

python IDM_dump/preprocess_video.py \
    --src_dir "IDM_dump/data/g1" \
    --dst_dir "IDM_dump/data/g1_split" \
    --dataset g1

python IDM_dump/raw_to_lerobot.py \
    --input_dir "IDM_dump/data/g1_split" \
    --output_dir "IDM_dump/data/g1_unified.data" \
    --embodiment g1 \
    --cosmos_predict2

python IDM_dump/dump_idm_actions.py \
    --checkpoint "/mnt/ssd/project/GR00T-Dreams/idm/g1/checkpoint-10000" \
    --dataset "IDM_dump/data/g1_unified.data" \
    --output_dir "IDM_dump/data/g1_unified.data" \
    --num_gpus 8 \
    --video_indices "0 8" 