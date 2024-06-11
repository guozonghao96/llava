#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-13b-pretrain_sft_glabol_local_672 \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-v1.5-13b.json