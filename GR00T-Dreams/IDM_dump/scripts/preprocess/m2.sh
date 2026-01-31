# python IDM_dump/split_video_instruction.py \
#     --source_dir "data/pick_bottle/videos/chunk-000/" \
#     --output_dir "IDM_dump/data/m2"
python IDM_dump/split_video_instruction.py \
    --source_dir "data/picktest/" \
    --output_dir "IDM_dump/data/m2_check"


# # python IDM_dump/preprocess_video.py \
# #     --src_dir "IDM_dump/data/m2" \
# #     --dst_dir "IDM_dump/data/m2_split" \
# #     --dataset m2

# python IDM_dump/raw_to_lerobot.py \
#     --input_dir "IDM_dump/data/m2_split" \
#     --output_dir "IDM_dump/data/m2_unified.data" \
#     --embodiment m2 \
#     --cosmos_predict2

# python IDM_dump/dump_idm_actions.py \
#     # --checkpoint "/mnt/ssd/project/GR00T-Dreams/idm/m2/checkpoint-10000" \
#     --dataset "IDM_dump/data/m2_unified.data" \
#     --output_dir "IDM_dump/data/m2_unified.data" \
#     --num_gpus 8 \
#     --video_indices "0 8" 