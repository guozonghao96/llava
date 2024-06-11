#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-13b"
SPLIT="llava_rec_val"
GQADIR="./playground/data/eval/rec/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints_new/llava-v1.5-13b-pretrain_fix144_QWsampler_5ep0002_spatial_cross \
        --question-file ./playground/data/eval/rec/llava_ref3_test.jsonl \
        --image-folder ./playground/data/eval/rec/train2014 \
        --answers-file ./playground/data/eval/rec/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/rec/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/rec/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python llava/eval/eval_rec.py \
    --annotation-file ./playground/data/eval/rec/answers/llava_ref3_labels.jsonl \
    --question-file ./playground/data/eval/rec/llava_ref3_test.jsonl \
    --result-file ./playground/data/eval/rec/answers/$SPLIT/$CKPT/merge.jsonl