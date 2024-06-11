# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export DEEPSPEED_LOG_LEVEL=DEBUG
# export OMPI_MCA_btl_base_verbose=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_CPP_LOG_LEVEL=INFO



OUTPUT_DIR=./checkpoints_new/llava-lora_debug #

# max_model_length=2048 # 2048 + 576 * 4 = 2048 + 2304 = 4352
# LR=0.001
# EPOCH=1
# PRBATCH=32
# gradient_accumulation_steps=1

deepspeed \
    --master_port=12322 \
    --include localhost:5 \
    llava/train/llava_uhd/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path /home/guozonghao/pretrained_models/vicuna-13b-v1.5 \
    --version plain \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --vision_tower /home/guozonghao/pretrained_models/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

# OUTPUT_DIR=./checkpoints_new/llava-uhd_adapt_1_schema_split_resample #

# max_model_length=2048 # 2048 + 576 * 4 = 2048 + 2304 = 4352
# LR=0.001
# EPOCH=1
# PRBATCH=32
# gradient_accumulation_steps=1

# deepspeed \
#     --master_port=12322 \
#     llava/train/llava_uhd/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path /home/guozonghao/pretrained_models/vicuna-13b-v1.5 \
#     --version plain \
#     --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
#     --image_folder ./playground/data/LLaVA-Pretrain/images \
#     --vision_tower /home/guozonghao/pretrained_models/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb


#     # --include localhost:7 \
# # #     # --include localhost:1,2,3,4,5,6,7 \
#     # --data_path ./playground/data/llava_v1_5_mix665k.json \

# deepspeed \
#     --master_port=12302 \
#     llava/train/llava_uhd/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path /home/guozonghao/pretrained_models/vicuna-13b-v1.5 \
#     --version v1 \
#     --data_path ./playground/data/llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower /home/guozonghao/pretrained_models/clip-vit-large-patch14-336 \
#     --pretrain_mm_mlp_adapter $OUTPUT_DIR/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb

# # fix144query QWenSampler 0.0002 5ep plain adapt
# OUTPUT_DIR=./checkpoints_new/llava-v1.5-13b-pretrain_fix144_QWsampler_5ep0002_uhd_adpat #
# mm_projector_type=tokenizerQW
# process_mode=anyres
# max_model_length=2048 # 2048 + 576 * 4 = 2048 + 2304 = 4352
# LR=0.0002
# EPOCH=5
# PRBATCH=32
# gradient_accumulation_steps=1
# deepspeed \
#     --master_port=12302 \
#     llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path /home/guozonghao/pretrained_models/vicuna-13b-v1.5 \
#     --version plain \
#     --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
#     --image_folder ./playground/data/LLaVA-Pretrain/images \
#     --vision_tower /home/guozonghao/pretrained_models/clip-vit-large-patch14-336 \
#     --mm_projector_type $mm_projector_type \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio $process_mode \
#     --bf16 True \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs $EPOCH \
#     --per_device_train_batch_size $PRBATCH \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps $gradient_accumulation_steps \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate $LR \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length $max_model_length \
#     --report_to wandb \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --crop_image_size 336 \
#     --num_query 144 \
#     --spatial_cross False \
#     --adapt True

# # # fix144query QWenSampler 0.0002 5ep plain adapt sft
# # OUTPUT_DIR=./checkpoints_new/llava-v1.5-13b-pretrain_fix144_QWsampler_5ep0002_multi_spatial_cross #
# LR=0.00002
# EPOCH=1
# FTBATCH=16
# max_model_length=2048 # 2048 + 4 * 256
# gradient_accumulation_steps=1
# deepspeed \
#     --include localhost:1 \
#     --master_port=12302 \
#     llava/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#     --model_name_or_path /home/guozonghao/pretrained_models/vicuna-13b-v1.5 \
#     --version v1 \
#     --data_path ./playground/data/llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower /home/guozonghao/pretrained_models/clip-vit-large-patch14-336 \
#     --pretrain_mm_mlp_adapter $OUTPUT_DIR/mm_projector.bin \
#     --mm_projector_type $mm_projector_type \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio $process_mode \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs $EPOCH \
#     --per_device_train_batch_size $FTBATCH \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps $gradient_accumulation_steps \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --learning_rate $LR \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length $max_model_length \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --crop_image_size 336 \
#     --num_query 144 \
#     --spatial_cross False \
#     --adapt True


# # # fix144query QWenSampler 0.001 1ep spatial cross
# OUTPUT_DIR=./checkpoints_new/llava-v1.5-13b-pretrain_fix144_QWsampler_1ep001_spatial_cross #
# mm_projector_type=tokenizerQW
# max_model_length=2048 # 2048 + 576 * 4 = 2048 + 2304 = 4352
# LR=0.001
# EPOCH=1
# PRBATCH=32
# gradient_accumulation_steps=1
# deepspeed \
#     --master_port=12302 \
#     llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path /home/guozonghao/pretrained_models/vicuna-13b-v1.5 \
#     --version plain \
#     --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
#     --image_folder ./playground/data/LLaVA-Pretrain/images \
#     --vision_tower /home/guozonghao/pretrained_models/clip-vit-large-patch14-336 \
#     --mm_projector_type $mm_projector_type \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs $EPOCH \
#     --per_device_train_batch_size $PRBATCH \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps $gradient_accumulation_steps \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate $LR \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length $max_model_length \
#     --report_to wandb \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --crop_image_size 336 \
#     --num_query 144 \
#     --spatial_cross True \
#     --ratio "2"

# # # qw sampler fix 144 sft 0.001 1ep spatial cross
# # OUTPUT_DIR=./checkpoints_new/llava-v1.5-13b-pretrain_fix144_QWsampler_5ep0002_multi_spatial_cross #
# LR=0.00002
# EPOCH=1
# FTBATCH=16
# max_model_length=2048 # 2048 + 4 * 256
# gradient_accumulation_steps=1
# mm_projector_type=tokenizerQW
# deepspeed \
#     --master_port=12302 \
#     llava/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path /home/guozonghao/pretrained_models/vicuna-13b-v1.5 \
#     --version v1 \
#     --data_path ./playground/data/llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower /home/guozonghao/pretrained_models/clip-vit-large-patch14-336 \
#     --pretrain_mm_mlp_adapter $OUTPUT_DIR/mm_projector.bin \
#     --mm_projector_type $mm_projector_type \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs $EPOCH \
#     --per_device_train_batch_size $FTBATCH \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps $gradient_accumulation_steps \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --learning_rate $LR \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length $max_model_length \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --crop_image_size 336 \
#     --num_query 144 \
#     --spatial_cross True \
#     --ratio "2"



# # # fix144query QWenSampler 0.0005 2ep spatial cross
# OUTPUT_DIR=./checkpoints_new/llava-v1.5-13b-pretrain_fix144_QWsampler_2ep0005_spatial_cross #
# mm_projector_type=tokenizerQW
# max_model_length=2048 # 2048 + 576 * 4 = 2048 + 2304 = 4352
# LR=0.0005
# EPOCH=2
# PRBATCH=32
# gradient_accumulation_steps=1
# deepspeed \
#     --master_port=12302 \
#     llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path /home/guozonghao/pretrained_models/vicuna-13b-v1.5 \
#     --version plain \
#     --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
#     --image_folder ./playground/data/LLaVA-Pretrain/images \
#     --vision_tower /home/guozonghao/pretrained_models/clip-vit-large-patch14-336 \
#     --mm_projector_type $mm_projector_type \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs $EPOCH \
#     --per_device_train_batch_size $PRBATCH \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps $gradient_accumulation_steps \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate $LR \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length $max_model_length \
#     --report_to wandb \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --crop_image_size 336 \
#     --num_query 144 \
#     --spatial_cross True \
#     --ratio "2"

# # # qw sampler fix 144 sft 0.001 1ep spatial cross
# # OUTPUT_DIR=./checkpoints_new/llava-v1.5-13b-pretrain_fix144_QWsampler_5ep0002_multi_spatial_cross #
# LR=0.00002
# EPOCH=1
# FTBATCH=16
# max_model_length=2048 # 2048 + 4 * 256
# gradient_accumulation_steps=1
# mm_projector_type=tokenizerQW
# deepspeed \
#     --master_port=12302 \
#     llava/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path /home/guozonghao/pretrained_models/vicuna-13b-v1.5 \
#     --version v1 \
#     --data_path ./playground/data/llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower /home/guozonghao/pretrained_models/clip-vit-large-patch14-336 \
#     --pretrain_mm_mlp_adapter $OUTPUT_DIR/mm_projector.bin \
#     --mm_projector_type $mm_projector_type \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs $EPOCH \
#     --per_device_train_batch_size $FTBATCH \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps $gradient_accumulation_steps \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --learning_rate $LR \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length $max_model_length \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --crop_image_size 336 \
#     --num_query 144 \
#     --spatial_cross True \
#     --ratio "2"