#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-13b"
SPLIT="llava_rec_val"
GQADIR="./playground/data/eval/rec_rsvg/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints_new/llava-v1.5-13b-pretrain_sam_l_proj_clip_l_save_sft_ft_rsvg \
        --question-file ./playground/data/eval/rec_rsvg/llava_rsvg_test.jsonl \
        --image-folder ./playground/data/eval/rec_rsvg/JPEGImages \
        --answers-file ./playground/data/eval/rec_rsvg/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/rec_rsvg/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/rec_rsvg/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python llava/eval/eval_rec_rsvg.py \
    --annotation-file ./playground/data/eval/rec_rsvg/answers/llava_rsvg_labels.jsonl \
    --question-file ./playground/data/eval/rec_rsvg/llava_rsvg_test.jsonl \
    --result-file ./playground/data/eval/rec_rsvg/answers/$SPLIT/$CKPT/merge.jsonl