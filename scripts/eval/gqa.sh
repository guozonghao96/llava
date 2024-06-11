#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-13b"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"

        # --model-path ./checkpoints/llava-v1.5-13b-pretrain_pretrained_pos_avgpool4 \

        
        # --model-path ./checkpoints/llava-v1.5-13b-pretrain_tokenizer_global_336_v2_lr2e_4_5ep \
        # --is_sam True \

        
        # --model-path ./checkpoints/llava-v1.5-13b-pretrain_sft_pos_inter_tune_glabol_local_672 \
        # --moe_models True \

        # --model-path ./checkpoints/llava-v1.5-13b-pretrain_tokenizer_global_336_v2_ms112_fs24_up576query_lr2e_4_5ep \

        # --model-path ./checkpoints/llava-v1.5-13b-pretrain_tokenizer_global_336_v2_ms112_fs24_lr2e_4_5ep \
        # --model-path ./checkpoints/llava-v1.5-13b-pretrain_tokenizer_global_336_v2_ms112_fs24_up576query_lr2e_4_5ep \
        # llava-v1.5-13b-tokenizer_v2_336_ms24_fs24_on_672_split_hd_low_sft
        # llava-v1.5-13b-tokenizer_v2_336_ms24_fs24_on_672_split_hd_sft
        
        # llava-v1.5-13b-pretrain_tokenizer_global_max256query_336_v2_ms24_fs24_low_hd_hd+
        # llava-v1.5-13b-pretrain_tokenizer_global_336_v2_fix384query_max384_ms112_fs24_sft
        # llava-v1.5-13b-pretrain_tokenizer_global_336_v2_fix256query_max256_ms56_fs24_sft
        # llava-v1.5-13b-pretrain_tokenizer_global_336_v2_fix512query_fix512_ms112_fs24_sft
        # llava-v1.5-13b-pretrain_tokenizer_global_336_v2_fix256query_max256_ms56_fs24_baseline_sft
        # llava-v1.5-13b-pretrain_fix256ms112fs24_sort3_sft
        # --is_sam True \
        # llava-v1.5-13b-pretrain_vary_t_sam_b_clip_tokenizer_336_ms56_fs24_fix256query_1_sft

        # --model-path ./checkpoints_new/llava-v1.5-13b-pretrain_sam_l_proj_clip_l_save \

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.train.llava_uhd.model_vqa_loader_uhd \
        --model-path ./checkpoints_new/llava-uhd_adapt_1_schema_5ep \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &

#     # CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#     #     --model-path ./checkpoints/llava-v1.5-13b-pretrain_tokenizer_global_336_v2_fix256query_max256_ms56_fs24_sft \
#     #     --is_sam True \
#     #     --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
#     #     --image-folder ./playground/data/eval/gqa/data/images \
#     #     --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#     #     --num-chunks $CHUNKS \
#     #     --chunk-idx $IDX \
#     #     --temperature 0 \
#     #     --conv-mode vicuna_v1 &

#     # CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#     #     --model-path ./checkpoints/llava-v1.5-13b-pretrain_sft_glabol_local_672 \
#     #     --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
#     #     --image-folder ./playground/data/eval/gqa/data/images \
#     #     --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#     #     --num-chunks $CHUNKS \
#     #     --chunk-idx $IDX \
#     #     --temperature 0 \
#     #     --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
